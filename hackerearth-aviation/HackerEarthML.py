
# coding: utf-8

# In[305]:


import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[306]:


train['Severity'].unique()


# In[307]:


test.head()


# In[308]:


import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[309]:


train.shape


# In[310]:


plt.figure(figsize=(15,5))
plt.plot(train['Severity'],train['Safety_Score'],'bo')
plt.show()


# In[311]:


mapping = {'Minor_Damage_And_Injuries': 1, 'Significant_Damage_And_Fatalities': 2,
           'Significant_Damage_And_Serious_Injuries': 3,'Highly_Fatal_And_Damaging' : 4}
T=train.replace({'Severity': mapping})


# In[312]:


sns.heatmap(T.corr(), vmin=-0.8, vmax=1, center= 0,cmap= 'coolwarm')


# In[313]:


U=T.drop(['Severity','Accident_ID'],axis=1)


# In[314]:


y=T.Severity


# In[315]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(U, y, test_size=0.25, random_state=0)


# In[290]:





# In[316]:


from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# In[317]:


knn = KNeighborsClassifier(n_neighbors=20, metric='euclidean')
knn.fit(X_train, y_train)


# In[318]:


y_pred = knn.predict(X_test)


# In[319]:


y_pred


# In[320]:


confusion_matrix(y_test, y_pred)


# In[321]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[322]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[323]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, Y_pred))


# In[324]:


svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print(accuracy_score(y_test, Y_pred))


# In[299]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
print(accuracy_score(y_test, Y_pred))


# In[325]:


perceptron = Perceptron()
perceptron.fit(X_train, y_train)
Y_pred = perceptron.predict(X_test)
print(accuracy_score(y_test, Y_pred))


# In[326]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
print(accuracy_score(y_test, Y_pred))


# In[327]:


sgd = SGDClassifier()
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
print(accuracy_score(y_test, Y_pred))


# In[328]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
print(accuracy_score(y_test, Y_pred))


# In[329]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)

print(accuracy_score(y_test, Y_pred))


# In[330]:


TestT = test.drop('Accident_ID',axis=1)


# In[331]:


Result = random_forest.predict(TestT)


# In[332]:


R=pd.DataFrame(Result, columns=['Severity'])


# In[333]:


mapping = {1 :'Minor_Damage_And_Injuries', 2: 'Significant_Damage_And_Fatalities',
           3: 'Significant_Damage_And_Serious_Injuries',4 :'Highly_Fatal_And_Damaging' }
Ans=R.replace({'Severity': mapping})


# In[334]:


submission = pd.DataFrame({ 'Accident_ID': test.Accident_ID.values, 'Severity': Ans.Severity })
submission.to_csv("my_submission.csv", index=False)

