import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

df = pd.read_csv('translated_df.csv')
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")


# model = SentenceTransformer('bert-base-multilingual-cased')  
model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')


embedding = model.encode(df['번역'].tolist())
pd.DataFrame(embedding).to_csv('embedding_biobert.csv',index=False)