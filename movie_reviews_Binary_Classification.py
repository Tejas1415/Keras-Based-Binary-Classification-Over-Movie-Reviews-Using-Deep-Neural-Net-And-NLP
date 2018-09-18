# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:57:26 2018

@author: Tejas_K_Reddy

Binary classification using NN on reviews based on IMDB dataset.(Keras)
"""

from keras.datasets import imdb

#load the top 10k words. IMDB dataset has already label encoded the words. 
(train_Data, train_labels), (test_Data, test_labels)  = imdb.load_data(num_words = 10000)

print(train_Data[0]) # len train data is diff for diff cases
print(len(train_Data[24999])) # train data has values ranging from 0 to 10k
print(train_labels[-10:]) # print last 10 in the sequence


# A quick way to decode any review back ###### Not required in the program
# Printing the first [0] review
word_index = imdb.get_word_index()
reverse_word_index = dict(
        [(value, key) for (key,value) in word_index.items()]) # Reversing the indices to get words
decoded_review = ' '.join(
        reverse_word_index.get(i-3,'?') for i in train_Data[0]) # 0,1,2 are padded unknown sequences, hence cut them out
print(decoded_review)
############


### Now one Hot encode the train data. 
# We can use the inbuilt command, but lets go with premitive method
# U might face a problem if, laptop ram < 4GB.
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension)) #create a null vector with same dimension to store the results
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results
X_train = vectorize_sequences(train_Data)
X_test = vectorize_sequences(test_Data)
#since y is already interms of 0,1 just converting elements to dtype float
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


### Build the Network
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation ='relu')) #Input shape auto understood
model.add(layers.Dense(1, activation ='sigmoid')) # if its binary classification, sigmoid is preferred and last output=1

#compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])

############# or if u want to configure the optimizer
from keras import optimizers, losses, metrics
model.compile(optimizer = optimizers.RMSprop(lr=0.001),
loss=losses.binary_crossentropy,
metrics = [metrics.binary_accuracy])


### Validate your output
# Create a test andtrain split dataset
X_val = X_train[:10000]
partial_X_train = X_train[10000:]

y_val =y_train[:10000]
partial_y_train = y_train[10000:]


#train on the partial datasets
# if xval, yval = np.array(xval/yval) is used then use 'sprse_categorical_crossentropy' as loss function
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['acc'])
history = model.fit(partial_X_train, partial_y_train, epochs=20, batch_size=512, 
                    validation_data=(x_val, y_val))

###############IMPORTANT##############

# Plotting validation loss and training loss
# page 97 in DL with Python book.
import matplotlib.pyplot as plt
history_dict = history.history # histdict= [u'acc',u'loss', u'val_acc',u'val_loss']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc)+1)

plt.plot(epochs, loss_values,'bo', label'Trainig loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting training and validation accuracy

plt.clf # clearing the figure
acc_val = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_val, 'bo', label = 'training accuracy')
plt.plot(epochs, val_acc_values, 'b', label=' validation accuracy')
plt.title('Traning and validation accuracy')
plt.legend()
plt.show()
###########################################

# To predict the answers:
model.predict(X_test)
#[Outputs an array with probabilities (0 to 1)] -  due to the sigmoid activation function
# softmax activation key also does the same of mapping probability to all the end layers nodes
# softmax is for multicclass classification problems




