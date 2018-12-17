
# coding: utf-8

# # Importing Data

# In[1]:


import pandas as pd


# In[3]:


census = pd.read_csv('census_data.csv')


# In[4]:


census.head()


# In[6]:


census['income_bracket'].unique()


# In[7]:


# Fixing the class column to 0 - 1
def label_fix(label):
    if label == ' <=50K':
        return 0
    else:
        return 1


# In[8]:


census['income_bracket'] = census['income_bracket'].apply(label_fix)


# In[10]:


census.head(10)


# # Splitting the dataset

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x_data = census.drop('income_bracket', axis=1)


# In[13]:


y_labels = census['income_bracket']


# In[14]:


X_train,X_test,y_train,y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=0)


# # Creating TensorFlow Columns

# In[15]:


census.columns


# In[16]:


import tensorflow as tf


# In[20]:


# Create features with categorical values
gender = tf.feature_column.categorical_column_with_vocabulary_list('gender',['Female','Male'])
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation',hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship',hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket('education',hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass',hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country',hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status',hash_bucket_size=1000)


# In[22]:


# Create features with continiuous values
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')


# In[23]:


feat_cols = [gender,occupation,relationship,education,workclass,native_country,marital_status,age,education_num,
            capital_gain,capital_loss,hours_per_week]


# # Create Input Function

# In[27]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)


# # Create Model

# In[25]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols)


# # Train the Model

# In[26]:


model.train(input_fn=input_func, steps=10000)


# # Evaluation Phase

# In[28]:


pred_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), num_epochs=1, shuffle=False)


# In[29]:


pred_gen = model.predict(input_fn=pred_func)


# In[30]:


predictions = list(pred_gen)


# In[31]:


predictions


# In[35]:


final_preds = [pred['class_ids'][0] for pred in predictions]


# In[36]:


final_preds


# In[37]:


from sklearn.metrics import classification_report, confusion_matrix


# In[38]:


print(classification_report(y_test,final_preds))

