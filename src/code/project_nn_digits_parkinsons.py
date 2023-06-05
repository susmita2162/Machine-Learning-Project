#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
from nn import NeuralNetwork as NeuralNet
from matplotlib import pyplot as plt
from my_utils import get_crossValidation_datasets, transform_datasets


# In[2]:


digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
labels = digits.target
cols = digits.feature_names

data_df = pd.DataFrame(data, columns = cols)
labels_df = pd.DataFrame(labels, columns = ['target'])

dataset = pd.concat([data_df, labels_df], axis = 1)
dataset[cols] = (dataset[cols] - dataset[cols].min())/(dataset[cols].max() - dataset[cols].min())
dataset.fillna(0, inplace=True)

X_train, X_test = get_crossValidation_datasets(10, dataset)


# In[3]:


def run_nn(setup, lamda, epochs, alpha, X_train, X_test):
    a = f = p = r = 0
    for k in range(10):
        train, test = transform_datasets(X_train[k], X_test[k])
        nn = NeuralNet(len(setup), setup)
        nn.reg_lambda = lamda
        nn.learning_rate = alpha
        nn.epochs = epochs
        nn.gradientDescent(train)
        accuracy, precision, recall, f1 = nn.evaluate(test)
        a += accuracy
        p += precision
        r += recall
        f += f1
    print("Average accuracy    = ", a/10)
    print("Average precision   = ", p/10)
    print("Average recall      = ", r/10)
    print("Average F1 score    = ", f/10)


# In[12]:


run_nn([64,8,10], 0.25, 300, 1, X_train, X_test)


# In[13]:


run_nn([64,16,10], 0.25, 300, 1, X_train, X_test)


# In[14]:


run_nn([64,8,8,10], 0.25, 300, 1, X_train, X_test)


# In[15]:


run_nn([64,16,8,10], 0.25, 300, 1, X_train, X_test)


# In[16]:


run_nn([64,16,16,10], 0.25, 400, 1, X_train, X_test)


# In[17]:


run_nn([64,8,8,8,10], 0.25, 400, 1, X_train, X_test)


# In[18]:


run_nn([64,16,8,8,10], 0.25, 400, 1, X_train, X_test)


# In[4]:


run_nn([64,8,10], 0.1, 300, 1, X_train, X_test)


# In[5]:


run_nn([64,16,10], 0.1, 300, 1, X_train, X_test)


# In[6]:


run_nn([64,16,10], 0.1, 400, 1, X_train, X_test)


# In[ ]:





# In[24]:


df = pd.read_csv('./data/parkinsons.csv')
df.rename(columns={"Diagnosis":"target"}, inplace=True)
df = (df-df.min())/(df.max()-df.min())
df.fillna(0, inplace=True)

X_train_p, X_test_p = get_crossValidation_datasets(10, df)


# In[27]:


run_nn([22,8,2], 0.25, 360, 1, X_train_p, X_test_p)
print("-------------------------------------------------------")
run_nn([22,16,2], 0.25, 360, 1, X_train_p, X_test_p)
print("-------------------------------------------------------")
run_nn([22,8,8,2], 0.25, 360, 1, X_train_p, X_test_p)
print("-------------------------------------------------------")
run_nn([22,16,8,2], 0.25, 360, 1, X_train_p, X_test_p)
print("-------------------------------------------------------")
run_nn([22,16,8,4,2], 0.25, 360, 1, X_train_p, X_test_p)


# In[28]:


run_nn([22,16,8,4,2], 0.25, 400, 1, X_train_p, X_test_p)


# In[13]:


df = pd.read_csv('./data/parkinsons.csv')
df.rename(columns={"Diagnosis":"target"}, inplace=True)
df = (df-df.min())/(df.max()-df.min())
df.fillna(0, inplace=True)

x = []
y = []
train_df, test_df = get_crossValidation_datasets(10, df)
for k in range(10):
    train, test = transform_datasets(train_df[k], test_df[k])
    nn1 = NeuralNet(4, [22,8,8,2])
    nn1.reg_lambda = 0.25
    nn1.epochs = 360
    nn1.learning_rate = 1.0

    size = len(train)
    print("Epoch size : ", size)
    
    epochs = 1
    while epochs <= 360:
        nn1.update(train)
        x.append(epochs)
        y.append(nn1.compute_cost(test))
        epochs = epochs + 1
    break


# In[17]:


plt.plot(x,y)
plt.xlabel("epochs")
plt.ylabel("Cost (J)")
plt.title('Parkinsons ([8,8],0.25, alpha:1)')
plt.show()


# In[18]:


digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
labels = digits.target
cols = digits.feature_names

data_df = pd.DataFrame(data, columns = cols)
labels_df = pd.DataFrame(labels, columns = ['target'])

dataset = pd.concat([data_df, labels_df], axis = 1)
dataset[cols] = (dataset[cols] - dataset[cols].min())/(dataset[cols].max() - dataset[cols].min())
dataset.fillna(0, inplace=True)

x = []
y = []
train_df, test_df = get_crossValidation_datasets(10, dataset)
for k in range(10):
    train, test = transform_datasets(train_df[k], test_df[k])
    nn1 = NeuralNet(3, [64,16,10])
    nn1.reg_lambda = 0.1
    nn1.epochs = 400
    nn1.learning_rate = 1.0

    size = len(train)
    print("Epoch size : ", size)
    
    epochs = 1
    while epochs <= 400:
        nn1.update(train)
        x.append(epochs)
        y.append(nn1.compute_cost(test))
        epochs = epochs + 1
    break

plt.plot(x,y)
plt.xlabel("epochs")
plt.ylabel("Cost (J)")
plt.title('Digits ([16], 0.1, alpha:1)')
plt.show()

