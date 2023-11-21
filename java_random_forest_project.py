#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# In[20]:


data=pd.read_csv("Book1.csv", sep=';',on_bad_lines='skip')
labels = data['Column1']
symptoms = data['Column2']
df=pd.DataFrame(data)
df


# In[21]:


data.isnull().sum()


# In[22]:


data.columns


# In[23]:


data.value_counts()


# In[24]:


df.drop_duplicates(inplace=True)


# In[25]:


#Extraction of Independent and Dependent Variable in variable VAR1,VAR2
Y=df.iloc[:,0:1]
X=df.iloc[:,-1:]


# In[26]:


# Use TF-IDF vectorizer for text data
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_transformed = vectorizer.fit_transform(X['Column2'])


# In[27]:


# Assuming Y is one-hot encoded
Y_original = Y.idxmax(axis=1)


# In[28]:


print("X_transformed shape:", X_transformed.shape)
print("Y shape:", Y_original.shape)
Y


# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, test_size=0.2, random_state=0)


# In[30]:


# Create and train the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classi = RandomForestClassifier(n_estimators=180, criterion='entropy', random_state=0)
classi.fit(X_train, Y_train)




# In[ ]:





# In[ ]:





# In[32]:


Y_pred=classi.predict(X_test)


# In[33]:


# Remove "Column1_" prefix from predictions
Y_pred = [pred.replace('Column1_', '') for pred in Y_pred]
# Display the modified predictions
Y_pred


# In[34]:


# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))


# In[16]:


# Take user input
#tedst = input()
# Transform the user input using the same vectorizer
#tedst_transformed = vectorizer.transform([tedst])
#user_prediction = classi.predict(tedst_transformed)
#print("Predicted Disease:", user_prediction)


# In[ ]:
# Save preprocessed data and fitted vectorizer
joblib.dump(X_transformed, "X_transformed.joblib")
joblib.dump(vectorizer, "fitted_vectorizer.joblib")




