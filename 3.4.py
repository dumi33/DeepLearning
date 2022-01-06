#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('hello world!')


# In[2]:


from keras.datasets import imdb


# In[3]:


(train_data,train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)


# In[4]:


train_labels[0]


# In[5]:


train_data[0]


# In[6]:


import numpy as np


# In[16]:


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results


# In[17]:


x_train = vectorize_sequences(train_data)


# In[18]:


x_test = vectorize_sequences(test_data)


# In[19]:


x_train[0]


# In[20]:


y_train = np.asarray(train_labels).astype('float32')


# In[21]:


y_test = np.asarray(test_labels).astype('float32')


# In[22]:


from keras import models


# In[23]:


from keras import layers


# In[24]:


model=models.Sequential()


# In[25]:


model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))


# In[27]:


model.add(layers.Dense(16,activation='relu'))


# In[29]:


model.add(layers.Dense(1,activation='sigmoid'))


# In[30]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[31]:


x_val = x_train[:10000]


# In[32]:


partial_x_train = x_train[10000:]


# In[33]:


y_val = y_train[:10000]


# In[34]:


partial_y_train = y_train[10000:]


# In[35]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])


# In[36]:


history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,validation_data=(x_val,y_val))


# In[37]:


import matplotlib.pyplot as plt


# In[38]:


history_dict = history.history


# In[39]:


loss = history_dict['loss']


# In[40]:


val_loss = history_dict['val_loss']


# In[41]:


epochs = range(1,len(loss)+1)


# In[49]:


plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[50]:


plt.clf()


# In[51]:


acc = history_dict['acc']


# In[52]:


val_acc=history_dict['val_acc']


# In[53]:


plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[54]:


model = models.Sequential()


# In[56]:


model.add(layers.Dense(16,activation='relu',input_shape = (10000,)))


# In[58]:


model.add(layers.Dense(16,activation='relu'))


# In[59]:


model.add(layers.Dense(1,activation='sigmoid'))


# In[60]:


model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])


# In[61]:


model.fit(x_train,y_train,epochs=4, batch_size=512)


# In[63]:


result = model.evaluate(x_test,y_test)


# In[64]:


result


# In[ ]:




