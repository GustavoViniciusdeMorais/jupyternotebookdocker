#!/usr/bin/env python
# coding: utf-8

# In[11]:


#sudo pip install xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


dataframe = pd.read_csv('water_potability.csv',',')


# In[4]:


dataframe.fillna(dataframe.mean(), inplace=True)


# In[5]:


dataframe.head(2)


# In[27]:


x = dataframe.iloc[:,0:5].values
y = dataframe.iloc[:,9].values


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35,random_state=0)


# In[29]:


# fit model no training data
model = XGBClassifier()
model.fit(x_train, y_train)


# In[30]:


print(model)


# In[31]:


# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]


# In[32]:


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

