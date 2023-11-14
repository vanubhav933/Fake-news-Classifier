#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[6]:


import numpy as np


# In[7]:


df=pd.read_csv('C:/Users/HP/Downloads/train.csv/train.csv')


# In[8]:


df.head()


# In[9]:


df=df.dropna()


# In[10]:


X=df.drop('label',axis=1)


# In[11]:


y=df['label']


# In[12]:


y.value_counts()


# In[13]:


import matplotlib.pyplot as plt


# In[15]:


X.shape


# In[16]:


y.shape


# In[20]:


pip install tensorflow


# In[21]:


import tensorflow as tf


# In[22]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout


# In[23]:


voc_size=5000


# # Onehot Representation

# In[32]:


messages=X.copy()


# In[34]:


messages['title'][1]


# In[26]:


messages.reset_index(inplace=True)


# In[27]:


pip install nltk


# In[28]:


import nltk
import re
from nltk.corpus import stopwords


# In[29]:


nltk.download('stopwords')


# # DataSet Preprocessing

# In[31]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[36]:


corpus


# In[37]:


onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr


# # Embedding Representation

# In[38]:


sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[39]:


embedded_docs[0]


# In[40]:


embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[41]:


embedding_vector_features=40
model1=Sequential()
model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())


# In[42]:


len(embedded_docs),y.shape


# In[43]:


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


# In[44]:


X_final.shape,y_final.shape


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


# # Model Training

# In[46]:


model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


# # Performance Metrics And Accuracy

# In[47]:


y_pred1=model1.predict_classes(X_test)


# In[48]:


from sklearn.metrics import confusion_matrix


# In[49]:


confusion_matrix(y_test,y_pred1)


# In[1]:


import matplotlib.pyplot as plt

# Create the data
x = [1, 2, 3, 4, 5,6,7,8,9,10]
y = [0.8562, 0.9457, 0.9624, 0.9740, 0.9812, 0.9896, 0.9918, 0.9929, 0.9962, 0.9976]

# Plot the data
plt.plot(x, y)

# Add a title
plt.title('Accuracy Graph')

# Add labels to the axes
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Show the plot
plt.show()


# In[2]:


import matplotlib.pyplot as plt

# Create the data
x = [1, 2, 3, 4, 5,6,7,8,9,10]
y = [0.3066, 0.1404, 0.1004, 0.0707, 0.0527, 0.0325, 0.0254, 0.0221, 0.0130, 0.0098]

# Plot the data
plt.plot(x, y)

# Add a title
plt.title('Loss Graph')

# Add labels to the axes
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Show the plot
plt.show()


# In[3]:


fig = plt.figure()

# Create two lists for the accuracy and loss values
accuracy_values = [0.8562, 0.9457, 0.9624, 0.9740, 0.9812, 0.9896, 0.9918, 0.9929, 0.9962, 0.9976]
loss_values = [0.3066, 0.1404, 0.1004, 0.0707, 0.0527, 0.0325, 0.0254, 0.0221, 0.0130, 0.0098]

# Plot the accuracy values
plt.bar(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], height=accuracy_values, label='Accuracy')

# Plot the loss values
plt.bar(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], height=loss_values, label='Loss')

# Add a title to the figure
plt.title('Accuracy and Loss Bar Graph')

# Add labels to the x-axis
plt.xlabel('Epochs')

# Add labels to the y-axis
plt.ylabel('Values')

# Show the figure
plt.show()


# In[4]:


import wordcloud
import matplotlib.pyplot as plt


# In[5]:


pip install wordcloud


# In[3]:


import wordcloud
import matplotlib.pyplot as plt

text = open("C:/Users/HP/OneDrive/Documents/minor.txt").read()

wc = wordcloud.WordCloud(width=800, height=400, background_color="white")

wc.generate(text)

plt.imshow(wc)
plt.show()


# In[ ]:




