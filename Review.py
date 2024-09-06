Sentiment Analysis of Restaurant Reviews

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import zipfile
import os

df = pd.read_csv('/content/Restaurant_Reviews.csv', delimiter=',')

df.head()

print(df.columns)
sns.countplot(x='Liked', data=df)
plt.title('Sentiment Distribution (0 = Negative, 1 = Positive)')
plt.show()
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Review'])
y = df['Liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print(X_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


#link

#-- https://colab.research.google.com/drive/1QfloDZKx2z5AtwYLREpXU660mjASNfDU#scrollTo=7ZG8ZnBFAbvT
