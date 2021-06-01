# -*- coding: utf-8 -*-
"""
Created on Wed May 19 02:11:16 2021

@author: GulAyik
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


data=pd.read_csv('yorumlar.csv',delimiter=(';'))

# veri onisleme
yorum = data.copy()
yorum['veri'] = yorum['veri'].str.lower()
for i in range(yorum.shape[0]) :
    yorum['veri'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|([^\w\s])|(\d)", " ", str(yorum['veri'][i])).split())


#stopwords
stop_word_list = open('stop_words_turkish.txt','r').read().split()
docs = yorum['veri']
def token(values):
    filtered_words = [word for word in values.split() if word not in stop_word_list]
    not_stopword_doc = " ".join(filtered_words)
    return not_stopword_doc
docs = docs.map(lambda x: token(x))
yorum['stop_veri'] = docs


#işaretlenenler 
tagged_yorum = yorum.loc[yorum.duygu.isin(['Olumlu', 'Olumsuz'])]
yorum_datas = tagged_yorum['stop_veri'].values.tolist()
yorum_sentiment = tagged_yorum['duygu'].values.tolist()
yorum_all = yorum['stop_veri'].iloc[1599:].values.tolist()


#test ve train olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(yorum_datas, yorum_sentiment, test_size = 0.25, random_state = 69)
print('Eğitim seti: {}'.format(len(X_train)))
print('Test seti: {}'.format(len(X_test)))
print("***")

#tfidf
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df = 5)
x_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
x_test_tfidf = tfidf_vectorizer.transform(X_test)
yorum_all_tfidf = tfidf_vectorizer.transform(yorum_all)


# Naive Bayes Classifier
from sklearn.naive_bayes import BernoulliNB  
from sklearn.metrics import accuracy_score

model_naive = BernoulliNB().fit(x_train_tfidf, y_train) 
predicted_naive = model_naive.predict(x_test_tfidf)

score_naive = accuracy_score(predicted_naive, y_test)
print("Accuracy with Naive-bayes: ",score_naive)

yorum_all_pred = model_naive.predict(yorum_all_tfidf)
yorum['duygu'].iloc[1599:]=yorum_all_pred

list = yorum['duygu'].ravel()
d= dict(Counter(list))
print(d)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predicted_naive)
sns.heatmap(cm, annot=True, fmt=".0f")
plt.xlabel('Öngörülen değerler')
plt.ylabel('Gerçek değerler')
plt.title('Başarı Skoru: {}'.format(accuracy_score(y_test,  predicted_naive)), size = 13)
plt.show()


