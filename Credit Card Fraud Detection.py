#!/usr/bin/env python
# coding: utf-8

# In[59]:


# kaggle competition - credit card detection 
import pandas as pd
import numpy as np
import os 

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.metrics import classification_report
from sklearn import metrics

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.layers import Activation, Embedding, Flatten, LeakyReLU, BatchNormalization 
from keras.activations import relu, sigmoid
from keras.layers import LeakyReLU

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


os.getcwd()


# In[3]:


os.chdir("C:\\Users\\Seamus Laptop\\Desktop")


# In[4]:


df_fraud = pd.read_csv("creditcard.csv")


# In[5]:


df_fraud


# In[6]:


# total number of no entries, "empty space"
df_fraud.isna().sum()


# In[7]:


# analysing duplicate values that need to be removed 
df_fraud.duplicated().sum()


# In[8]:


# drop duplicate entries 
df_fraud.drop_duplicates(keep=False,inplace=True)


# Investigate dataset
# 
# Credit card fraud is a form of identity theft in which criminals make purchases or obtain cash advances using a credit card account assigned to you. This can occur through one of your existing accounts, via theft of your physical credit card or your account numbers and PINs, or by means of new credit card accounts being opened in your name without your knowledge. Once they're in, thieves then run up charges and stick you and your credit card company with the bill.
# Credit card issuers are acutely aware of this scourge, and are continually developing new methods to thwart unauthorized card usage. At the same time, however, resourceful fraudsters (including international organized crime syndicates) keep finding work-arounds for new security measures.
# Because card issuers are well-versed in dealing with card fraud, it's unlikely that being defrauded will cost you money out-of-pocket over the long haul, but necessary investigations can take months and, as discussed at greater length below, unaddressed credit card fraud can do major damage to your credit reports and scores.

# In[9]:


df_fraud['Class'].value_counts(normalize=True) * 100
# dataset is highly unblanaced in favour of the non-fraud detections
# need a more balanced dataset 


# In[10]:


X=df_fraud.drop("Class",axis=1)
y=df_fraud["Class"]


# In[17]:


# imbalanced classification
# randomly resample the df using under/over samplying from the imblearn api
over = RandomOverSampler(sampling_strategy=0.1)


# In[19]:


X, y = over.fit_resample(X, y)


# In[22]:


under = RandomUnderSampler(sampling_strategy=0.5)
X, y = under.fit_resample(X,y)


# In[28]:


y


# In[29]:


y.value_counts(normalize=True) * 100
# newly balanced dataset with bias removed


# Training

# In[32]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[34]:


# create sequential model
# single stack of layers connected sequentially 
# first layer
model = Sequential()


# In[36]:


# first layer 
model.add(Dense(units=20,kernel_initializer='he_normal',activation='relu',input_dim=30))


# In[37]:


# 2nd layer
model.add(Dense(units=15,kernel_initializer='he_normal',activation='relu'))


# In[38]:


#3rd layer
model.add(Dense(units=1,kernel_initializer='he_normal',activation='sigmoid'))


# In[39]:


# compiling!
model.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])


# In[40]:


# Train ANN
history = model.fit(X_train, y_train, batch_size = 32, epochs = 20,validation_split=0.2)


# Evaluating

# In[41]:


output=pd.DataFrame(history.history)
output


# In[44]:


fig,ax=plt.subplots(figsize=(10,6))
fig.patch.set_facecolor('#f6f5f5')
ax.set_facecolor('#f6f5f5')


plt.plot(output.loss,color="grey")
plt.plot(output.val_loss,color="#b20710")

ax.text(1,42,"Loss and Validation Loss",{'font':'serif','size':20,'weight':'bold','color':'black'})
ax.text(10,38,'LOSS',{'font':'serif','size':20,'weight':'bold','color':'grey'})
ax.text(13,38,'|',{'font':'serif','size':20,'weight':'bold','color':'black'})
ax.text(14,38,'VAL-LOSS',{'font':'serif','size':20,'weight':'bold','color':'#b20710'})



ax.axes.get_xaxis().set_visible(False)
#ax.axes.get_yaxis().set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


fig.show()


# In[43]:


fig,ax=plt.subplots(figsize=(10,6))
fig.patch.set_facecolor('#f6f5f5')
ax.set_facecolor('#f6f5f5')


plt.plot(output.accuracy,color="grey")
plt.plot(output.val_accuracy,color="#b20710")

ax.text(1,1.02,"ACCURACY AND VALIDATION ACCURACY",{'font':'serif','size':20,'weight':'bold','color':'black'})
ax.text(6,1,'ACCURACY',{'font':'serif','size':20,'weight':'bold','color':'grey'})
ax.text(12,1,'|',{'font':'serif','size':20,'weight':'bold','color':'black'})
ax.text(13,1,'VAL-ACCURACY',{'font':'serif','size':20,'weight':'bold','color':'#b20710'})



ax.axes.get_xaxis().set_visible(False)
#ax.axes.get_yaxis().set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# In[45]:


# Predicting the test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)


# In[47]:


sns.heatmap(metrics.confusion_matrix(y_test,y_pred),annot=True)


# In[48]:


print(classification_report(y_test, y_pred))


# Hyper Parameter Tuning

# In[50]:


def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(1)) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)


# In[51]:


model


# In[52]:


# hyper param tuning using grid search 
# pass parameters and grid search returns optimal param settings
layers = [[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid)


# In[53]:


# fitting params to training 
grid_result = grid.fit(X_train, y_train)


# In[54]:


# best params from training 
[grid_result.best_score_,grid_result.best_params_]


# In[55]:


pred_y = grid.predict(X_test)

y_pred = (pred_y > 0.5)

y_pred


# In[60]:


cm = confusion_matrix(y_test, y_pred)


# In[61]:


score=accuracy_score(y_test,y_pred)


# In[62]:


score


# In[ ]:




