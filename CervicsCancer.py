from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

#1.veri seti
df = pd.read_csv('kag_risk_factors_cervical_cancer.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.head())

df = df.replace(to_replace=r'^\s*\?\s*$', value=np.nan, regex=True)
df = df.replace(r'^\s*$', np.nan, regex=True)

df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(df.median())

print(df.info())
print(df.head())


X = df.drop("Dx:Cancer", axis=1)  #target dışındaki tüm sütunlar
y = df["Dx:Cancer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf1 = LogisticRegression(max_iter=1000, class_weight='balanced')
clf2 = KNeighborsClassifier()
clf3 = DecisionTreeClassifier(random_state=42, class_weight='balanced')

voting = VotingClassifier(
    estimators=[('lr', clf1), ('knn', clf2), ('dt', clf3)],
    voting='hard'
)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report: \n", classification_report(y_test, y_pred))
