import sys
import os
import copy
import time
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import OrderedDict

import pickle

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def get_loader():
    client1 = pd.read_csv('./data/client1.csv')
    X_1 = client1.drop('fraud_bool',axis=1)
    y_1 = client1['fraud_bool']
    client2 = pd.read_csv('./data/client2.csv')
    X_2 = client2.drop('fraud_bool',axis=1)
    y_2 = client2['fraud_bool']
    client3 = pd.read_csv('./data/client3.csv')
    X_3 = client3.drop('fraud_bool',axis=1)
    y_3 = client3['fraud_bool']
    client4 = pd.read_csv('./data/client4.csv')
    X_4 = client4.drop('fraud_bool',axis=1)
    y_4 = client4['fraud_bool']
    client5 = pd.read_csv('./data/client5.csv')
    X_5 = client5.drop('fraud_bool',axis=1)
    y_5 = client5['fraud_bool']

    test = pd.read_csv('./data/client_test_under.csv')
    X_test = test.drop('fraud_bool',axis=1)
    y_test = test['fraud_bool']

    common =  ['prev_address_months_count', 'current_address_months_count',
       'customer_age', 'days_since_request', 'intended_balcon_amount',
       'payment_type', 'bank_branch_count_8w',
       'date_of_birth_distinct_emails_4w', 'employment_status',
       'credit_risk_score', 'housing_status', 'phone_mobile_valid',
       'bank_months_count', 'proposed_credit_limit', 'foreign_request',
       'source', 'session_length_in_minutes', 'device_os',
       'device_distinct_emails_8w', 'month']

    # labelencoding

    object_col = {'payment_type':{'AE':0, 'AD':1, 'AC':2, 'AA':3, 'AB':4},
                    'employment_status':{'CE':0, 'CA':1, 'CB':2, 'CC':3, 'CG':4, 'CD':5, 'CF':6},
                    'housing_status':{'BE':0, 'BF':1, 'BC':2, 'BG':3, 'BA':4, 'BD':5, 'BB':6},
                    'source':{'INTERNET':0, 'TELEAPP':1},
                    'device_os':{'other':0, 'windows':1, 'x11':2, 'linux':3, 'macintosh':4}}

    for col,vals in object_col.items():
        X_1[col] = X_1[col].replace(vals)
        X_2[col] = X_2[col].replace(vals)
        X_3[col] = X_3[col].replace(vals)
        X_4[col] = X_4[col].replace(vals)
        X_5[col] = X_5[col].replace(vals)
        X_test[col] = X_test[col].replace(vals)
    
    X_1 = X_1[common]
    X_2 = X_2[common]
    X_3 = X_3[common]
    X_4 = X_4[common]
    X_5 = X_5[common]
    X_test = X_test[common]

    X_5 = scaler.fit_transform(X_5)
    X_4 = scaler.transform(X_4)
    X_3 = scaler.transform(X_3)
    X_2 = scaler.transform(X_2)
    X_1 = scaler.transform(X_1)
    X_test = scaler.transform(X_test)

    DS5 = TensorDataset(torch.Tensor(X_5), torch.LongTensor(y_5.values))
    DS4 = TensorDataset(torch.Tensor(X_4), torch.LongTensor(y_4.values))
    DS3 = TensorDataset(torch.Tensor(X_3), torch.LongTensor(y_3.values))
    DS2 = TensorDataset(torch.Tensor(X_2), torch.LongTensor(y_2.values))
    DS1 = TensorDataset(torch.Tensor(X_1), torch.LongTensor(y_1.values))
    DS_test = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test.values))

    loader5 = DataLoader(DS5, batch_size=64)
    loader4 = DataLoader(DS4, batch_size=64)
    loader3 = DataLoader(DS3, batch_size=64)
    loader2 = DataLoader(DS2, batch_size=64)
    loader1 = DataLoader(DS1, batch_size=64)
    local_loader_list = [loader1, loader2, loader3, loader4, loader5]
    loader_test = DataLoader(DS_test, batch_size=64)

    return local_loader_list, loader_test

class client():
    def __init__(self, model, local_dataloader, common_dataloader, client_name, device):
        
        self.name = client_name
        self.local_dataloader = local_dataloader
        self.common_dataloader = common_dataloader
        self.sample_size = len(self.common_dataloader.dataset)
        self.device = device

        self.model = nn.DataParallel(model).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, weight_decay=0.0004)
        self.criterion = nn.CrossEntropyLoss()

    def common_update(self):
        
        self.model.train()
        

        for data, label in self.common_dataloader:

            data = data.to(self.device)
            label = label.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            label = label
            
            loss = self.criterion(output, label)
            
            loss.backward()
            
            self.optimizer.step()
    
    def local_update(self):
        
        self.model.train()
        

        for data, label in self.local_dataloader:

            data = data.to(self.device)
            label = label.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            label = label
            
            loss = self.criterion(output, label)
            loss.backward()
            
            self.optimizer.step()

    def local_test(self, test_loader):
        
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
              
        for batch_idx, (data, labels) in enumerate(test_loader):

            data = data.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(data)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, pred_labels = torch.max(outputs, 1)
            
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        
        self.accuracy = correct/total

        return self.accuracy, loss/total
    
    def final_test(self, test_loader):
        
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        outputs_list = []
        labels_list = []

        for batch_idx, (data, labels) in enumerate(test_loader):

            data = data.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(data)
            outputs_list.append(outputs)
            labels_list.append(labels)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, pred_labels = torch.max(outputs, 1)
            
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        
        self.accuracy = correct/total

        return self.accuracy, loss/total, outputs_list, labels_list


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.layers(x)

def create_agent(local_loader_list, common_loader_list, num_node):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP()
    client_list = []
    for i in range(1,num_node + 1):
        setattr(mod, f'client{i}', eval(f'client(copy.deepcopy(model), local_loader_list[{i-1}], common_loader_list[{i-1}], "client{i}", device)'))
        eval(f'client_list.append(client{i})')
    return client_list



def global_avg(sampled_list):

    model = OrderedDict()
    total_size = sum([i.sample_size for i in sampled_list])

    for i, cli in enumerate(sampled_list,1):
        update_ratio = cli.sample_size / total_size

        if i == 1:
            for key in cli.model.state_dict().keys():
                model[key] = copy.deepcopy(cli.model.state_dict()[key])*update_ratio
        else:
            for key in cli.model.state_dict().keys():
                model[key] += copy.deepcopy(cli.model.state_dict()[key])*update_ratio 

    return model

def fed_avg(client_list, total_round, epoch_per_round, sample_rate):

    # Make empty lists to save global accuracy & loss
    num_sampled_clients = max(int(sample_rate * len(client_list)), 1)
    
    for r in range(1, total_round + 1):

        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(len(client_list))], size=num_sampled_clients, replace=False).tolist())

        # Load global model to each client after the first round
        if r > 1:
            for idx in sampled_client_indices:
                client_list[idx].model.load_state_dict(global_model)
        
        # Local update        
        for idx in tqdm(sampled_client_indices, desc='local_update'):
            for _ in range(epoch_per_round):
                client_list[idx].common_update()

        sampled_list = []
        for idx in sampled_client_indices:
            sampled_list.append(client_list[idx])

        # Global update
        global_model = global_avg(sampled_list)

    for cli in client_list:
        cli.model.load_state_dict(global_model)
    
    # save global weight to pt
    # torch.save(global_model,f'result/model/weight.pt')
    # temp=torch.load('./result/model/weight.pt')

    for k in global_model.keys():
        global_model[k]=global_model[k].cpu()

    # save global weight to pickle
    with open('./weight.pickle','wb') as f:
        pickle.dump(global_model,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_node', type = int, default=5)
    parser.add_argument('--total_round', type=int, default=50)
    parser.add_argument('--epoch_per_round', type = int, default=1)
    parser.add_argument('--num_class', type=int, default=20)
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    mod = sys.modules[__name__]

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    now = time.strftime('%Y%m%d%H%M%S')

    common_loader_list, test_loader = get_loader()

    # local loader (VFL Part)
    local_loader_list = [0,0,0,0,0]

    agent_list = create_agent(local_loader_list, common_loader_list, args.num_node)

    fed_avg(agent_list, args.total_round, args.epoch_per_round, args.sample_rate)
