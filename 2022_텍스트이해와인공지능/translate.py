import pandas as pd
import numpy as np
from googletrans import Translator
# import pickle

translator = Translator()

@np.vectorize 
def translate(x):
    try:
        y = translator.translate(x, dest='en')
    except:
        y = np.nan
    return y

df = pd.read_csv('./중간결과.csv')
data = translate(df['IMEX_OPN_CONT'].values)


# with open('translated.pickle','wb') as f:
#     pickle.dump(data, f)
pd.Series(data).to_csv('translate.py')
