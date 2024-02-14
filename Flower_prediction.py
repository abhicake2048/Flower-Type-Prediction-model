import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


 
d= load_iris()
x = d.data
y = d.target
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train,y_train)
p = model.score(X_test,y_test)
print(p)
yper = model.predict(X_test)
cm= confusion_matrix(y_test,yper)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()