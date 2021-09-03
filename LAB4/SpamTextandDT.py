import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

dataset = pd.read_csv('spam1.csv') 
print("\nData :\n",dataset)
print("\nData statistics\n",dataset.info())


dataset.dropna(how='any',inplace=True)
print(dataset.isnull().sum())
dataset.drop_duplicates(inplace=True)
print(dataset.duplicated().any())

# Preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
import string
import re
import random


def stemWords(list3):
    stemmer=nltk.stem.SnowballStemmer('english')
    words=[]
    for text in list3:
        for word in re.findall(r"[a-z]+",text):
            if (len(word)>2 and word not in string.punctuation):
                words.append(stemmer.stem(word))
    return words