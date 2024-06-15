#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE,mean_absolute_error as MAE,r2_score as RS


# In[5]:


# Load dataset 
df=pd.read_csv(r'D:\Intern\SkillCraft Intern\House pred.csv')


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[11]:


# Split into Indepent and Dependent Values
x=df.drop(['price'],axis=1)
y=df['price']


# In[12]:


# Data Split Into Train And Test
x_train,x_test,y_train,y_test=TTS(x,y,test_size=0.3,random_state=203)


# In[13]:


lm=LR()
lm.fit(x_train,y_train)


# In[14]:


y_pred=lm.predict(x_test)


# In[20]:


# Visualize 
plt.plot(y_pred,y_test)
plt.title('Actual Vs Predict')
plt.show()


# In[22]:


# evalute its Perfomance Using Metrics
mse=MSE(y_pred,y_test)
mae=MAE(y_pred,y_test)
rs=RS(y_pred,y_test)
print(f"Mean Squared Error = {mse}")
print(f"Mean Absolute Error = {mae}")
print(f"R2 Score = {rs}")


# In[23]:


#Enter the New Data to Predict 
import numpy as np


# In[27]:


new_data=np.array([[3,2,3400,7910,2,1,0,3,1300,200,2000]])
print(f"Predicted House Price = {lm.predict(new_data)[0]}")

