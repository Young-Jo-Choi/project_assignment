import torch
import torch.nn as nn
import numpy as np
import copy


class FL_net(nn.Module):
    def __init__(self, input_dim, hidden_input_dim=[128,64,32], hidden_output_dim=None, output_dim=2):
        super().__init__()
        if hidden_output_dim is None:
            hidden_output_dim = hidden_input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_output_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_input_dim[0], hidden_output_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_input_dim[1], hidden_output_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_input_dim[2], output_dim)
        )
    def forward(self, x):
        return self.layers(x)
    
class PPFL_net(nn.Module) : 
    def __init__(self, horizontal_model, vertical_model, common_idx, specific_idx, device, ppfl_hidden_dim=[128,64,32]) : 
        super().__init__()
        self.horizontal_model = horizontal_model
        self.vertical_model = vertical_model
        self.common_idx = common_idx
        self.specific_idx = specific_idx
        self.ppfl_hidden_dim = ppfl_hidden_dim
        self.device = device
        self.setup_horizontal()
        self.setup_vertical()
        self.setup_ppfl()

    def setup_horizontal(self):
        linear_idx, activations_idx, in_features = [],[],[]
        for i,layer in enumerate(self.horizontal_model.layers) : 
            if isinstance(layer, nn.Linear) : 
                linear_idx.append(i)
                in_features.append(layer.in_features)
            elif isinstance(layer, nn.ReLU) : 
                activations_idx.append(i)
        self.linear_idx_HFL = linear_idx
        self.activations_idx_HFL = activations_idx
        self.in_features_HFL = in_features

        for param in self.horizontal_model.parameters() : 
            param.requires_grad = False
        
    def setup_vertical(self) : 
        linear_idx, activations_idx, in_features = [],[],[]
        for i,layer in enumerate(self.vertical_model.layers) : 
            if isinstance(layer, nn.Linear) : 
                linear_idx.append(i)
                in_features.append(layer.in_features)
            elif isinstance(layer, nn.ReLU) : 
                activations_idx.append(i)
        self.linear_idx_VFL = linear_idx
        self.activations_idx_VFL = activations_idx
        self.in_features_VFL = in_features
        

    def setup_ppfl(self):
        if len(self.activations_idx_HFL) != len(self.activations_idx_VFL) : 
            raise ValueError("The number of ReLU layers in the horizontal model and the vertical model must be the same.")
        else:
            self.activations_idx = self.activations_idx_HFL
            self.linear_idx = self.linear_idx_HFL
            self.last_layer_num = len(self.horizontal_model.layers)-1
        
        input_dim = self.in_features_HFL[0] + self.in_features_VFL[0]
        hidden_output_dim = self.ppfl_hidden_dim
        hidden_input_dim = [h1+h2+h3 for h1,h2,h3 in zip(self.in_features_HFL[1:], self.in_features_VFL[1:], hidden_output_dim)]
        self.ppfl_model = FL_net(input_dim=input_dim, hidden_input_dim=hidden_input_dim, hidden_output_dim=hidden_output_dim)
        self.horizontal_model.to(self.device)
        self.vertical_model.to(self.device)
        self.ppfl_model.to(self.device)

    def forward(self, X):
        hfl_hidden = X[:,self.common_idx]
        vfl_hidden = X[:,self.specific_idx]
        for i in range(self.last_layer_num+1):
            if i in self.activations_idx : 
                ppfl_hidden = self.ppfl_model.layers[i](ppfl_hidden)
                hfl_hidden = self.horizontal_model.layers[i](hfl_hidden)
                vfl_hidden = self.vertical_model.layers[i](vfl_hidden)
            else : 
                if i==0:
                    ppfl_hidden = torch.cat([hfl_hidden,vfl_hidden],dim=1)
                else:
                    ppfl_hidden = torch.cat([hfl_hidden,vfl_hidden,ppfl_hidden],dim=1)
                ppfl_hidden = self.ppfl_model.layers[i](ppfl_hidden)
                hfl_hidden = self.horizontal_model.layers[i](hfl_hidden)
                vfl_hidden = self.vertical_model.layers[i](vfl_hidden)
                
        return ppfl_hidden