#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
import seaborn as sn
import matplotlib.pyplot as plt
import joblib 
import pickle


# In[2]:


dataframe = pd.read_csv('water_potability.csv',',')


# In[3]:


dataframe.head(10)


# In[4]:


dataframe.fillna(dataframe.mean(), inplace=True)


# In[5]:


dataframe.head(10)


# In[11]:


predictors = dataframe.iloc[:,0:9].values
classes = dataframe.iloc[:,9].values


# In[17]:


# Create an SelectKBest object to select features with two best ANOVA F-Values
selector = SelectKBest(f_classif, k=3)

# Choose the best attributes to the model
selector.fit(predictors, classes)

# Show the name of the columns in the data set that are the best attributes to the model
cols = selector.get_support(indices=True)
features_df_new = dataframe.iloc[:,cols]

# show the columns that best contribute to the model
features_df_new.head()


# In[18]:


newdataframe = dataframe.filter(['Solids','Chloramines','Organic_carbon'])
newdataframe.head(10)


# In[22]:


x_training, x_test, y_training, y_test = train_test_split(predictors, classes, test_size=0.3, random_state=0)


# In[26]:


y_training


# In[27]:


# training the model
naive_bayes = GaussianNB()
naive_bayes.fit(x_training, y_training)


# In[30]:


y_predicted = naive_bayes.predict(x_test)


# In[34]:


mlaccuracy = accuracy_score(y_test,y_predicted)
mlaccuracy


# In[36]:


# confusion matrix
cf_matrix = confusion_matrix(y_test,y_predicted)
df_cm = pd.DataFrame(cf_matrix, range(2), range(2))
sn.heatmap(df_cm, annot=True, fmt="d")
plt.show()


# In[41]:


# save model
joblib.dump(naive_bayes,'./naive_bayes.pkl')


# In[42]:


f = open('naive_bayes.pkl', 'wb')
pickle.dump(naive_bayes, f)
f.close()
print ("Export the model to model.pkl")

