
# coding: utf-8

# # Importing Data

# In[1]:


import pandas as pd


# In[25]:


housing_data = pd.read_csv('housing.csv')


# In[26]:


housing_data.head()


# # Splitting The Dataset 

# In[4]:


labels = housing_data['medianHouseValue']


# In[5]:


features = housing_data.drop('medianHouseValue',axis=1)


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.3,random_state=0)


# # Feature Scaling the data

# In[8]:


from sklearn.preprocessing import MinMaxScaler


# In[9]:


scaler = MinMaxScaler()


# In[10]:


scaler.fit(X_train)


# In[11]:


X_train = pd.DataFrame(data=scaler.transform(X_train),columns=X_train.columns,index=X_train.index)


# In[12]:


X_test = pd.DataFrame(data=scaler.transform(X_test),columns=X_test.columns,index=X_test.index)


# # Creating Feature Columns

# In[13]:


housing_data.columns


# In[14]:


import tensorflow as tf


# In[15]:


age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')


# In[16]:


feat_cols = [age,rooms,bedrooms,population,households,income]


# # Creating Input Function

# In[17]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)


# # Creating The Estimator model

# In[21]:


model = tf.estimator.DNNRegressor(hidden_units=[5,5,5],feature_columns=feat_cols)


# # Training the model

# In[19]:


model.train(input_fn=input_func,steps=10000)


# # Create Predict input function

# In[49]:


predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)


# In[50]:


pred = model.predict(input_fn=predict_input_func)


# In[51]:


predictions = list(pred)


# # Calculate Root Mean Squared Errors - RMSE

# In[52]:


final_pred = []
for pred in predictions:
    final_pred.append(pred['predictions'])


# In[53]:


from sklearn.metrics import mean_squared_error


# In[54]:


mean_squared_error(y_test,final_pred)**0.5

