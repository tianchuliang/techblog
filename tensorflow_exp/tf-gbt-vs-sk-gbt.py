#!/usr/bin/env python
# coding: utf-8

# In[19]:


# imports
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder

def one_hot_cat_column(feature_name, vocab):
  return tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                 vocab))

def make_input_fn(X, y, num_examples, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
    if shuffle:
      dataset = dataset.shuffle(num_examples)
    # For training, cycle thru dataset as many times as need (n_epochs=None).    
    dataset = (dataset
      .repeat(n_epochs)
      .batch(num_examples)) 
    return dataset
  return input_fn

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

CATEGORICAL_COLUMNS = ['sex', 
                       'n_siblings_spouses', 
                       'parch', 
                       'class', 
                       'deck', 
                       'embark_town', 
                       'alone']
NUMERIC_COLUMNS = ['age', 'fare']



# In[5]:


# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tfbt/titanic_train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tfbt/titanic_eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')


# In[6]:


# feature engineering 
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
    # one hot categoricals
    if feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(one_hot_cat_column(feature_name, vocabulary))
    # force numerical dtype to float16
    elif feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                               dtype=tf.float16))        


# # Prepare Comparison Params

# In[11]:


ntrees = 50
maxdepth = 3


# # Train & Evaluate TF BoostedTrees

# In[ ]:


params = {
  'n_trees': ntrees,
  'max_depth': maxdepth,
  'n_batches_per_layer': 1,
  'center_bias': True
}

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
est.train(make_input_fn(dftrain, y_train, len(y_train)), max_steps=100)
results = est.evaluate(make_input_fn(dfeval, y_eval, len(y_train), shuffle=False, n_epochs=1))
pd.Series(results).to_frame()


# # Train & Evaluate SKlean BoostedTrees

# In[12]:


clf = ensemble.GradientBoostingClassifier(n_estimators = ntrees,                                          max_depth = maxdepth)


# In[18]:


skdftrain = dftrain.copy()


# In[20]:


enc = OneHotEncoder(handle_unknown='ignore')


# In[21]:


skdftrain = enc.fit_transform(skdftrain)


# In[40]:


tfds = make_input_fn(dftrain, y_train, len(y_train))()


# In[45]:


tfds.batch(100)


# In[ ]:




