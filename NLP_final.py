import re
import spacy
import string
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score


df = pd.read_csv('IMDB_Dataset.csv',encoding='latin-1')
df.head()

df['review'].value_counts()
nlp = spacy.load('en_core_web_lg')

def data_prep(x):
    x = str(x).lower().replace('\\','').replace('_',' ')
    x = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)  # removes emails
    x = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x) #remove url
    x = re.sub(r'[^\w ]+', "", x)   # removes special characters
    x = BeautifulSoup(x, 'lxml').get_text().strip()  # removes html tags
    x = ' '.join([t for t in x.split() if t not in stopwords])  #removes stopwords
    return x

df['review'] = df['review'].apply(lambda x:data_prep(x))


tfidf = TfidfVectorizer(max_features = 10000)
x = df['review']
y = df['sentiment']


x = tfidf.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
x_train.shape, x_test.shape

lsvc = LinearSVC()
lsvc.fit(x_train,y_train)
y_pred = lsvc.predict(x_test)
print(y_pred)
print(classification_report(y_test, y_pred))

print("Sample review 1")
x ='the movie is worst to wtch in my entire life, the acting is too good and u can watch for the performances of the cast'
print(x)
x = data_prep(x)
tap = tfidf.transform([x])


a = lsvc.predict(tap)
print(a)

print("Sample review 2")
x = 'film was good'
print(x)
x = data_prep(x)
tip = tfidf.transform([x])

b = lsvc.predict(tip)
print(b)


def review():
    refBool = True
    while refBool:
        try:
            a = input("Enter your review \n")  # takes review from user
            # print(a)
            a = data_prep(a)
            tip = tfidf.transform([a])
            c = lsvc.predict(tip)
            print("the movie review is ",c)
            refBool = False
        except:
            print("Kindly input proper input values")
    return

review()
