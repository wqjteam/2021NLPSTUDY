import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

# Reading the Dataset (ISEAR Dataset)
data = pd.read_csv('ISEAR.csv', header=None)

data.head()

labels = data[0].values.tolist()
sents = data[1].values.tolist()
X_train, X_test, y_train, y_test = train_test_split(sents, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 其他特征可以提取吗？
# 1. 词性的特征
# 2. n-gram


parameters = {'C': [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]}
lr = LogisticRegression()
lr.fit(X_train, y_train).score(X_test, y_test)

clf = GridSearchCV(lr, parameters, cv=10)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(clf.best_params_)

confusion_matrix(y_test, clf.predict(X_test))
