import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array,load_img
from keras_preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.datasets import cifar10
directory = 'F:\\New folder\\archive (7)\\Img'
files=os.listdir(directory)
datafile=[]
data=[]
for file in files:
    image=load_img(os.path.join(directory,file),grayscale=False,color_mode='rgb',target_size=(100,100))
    image=img_to_array(image)
    image=image/255.0
    data+=[image]
    datafile+=[file]
data1=np.array(data)
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

engl=pd.read_csv('F:\\New folder\\archive (7)\\english.csv')
factlabel=pd.factorize(engl['label'])
labelfile=[]
for item in engl['image']:
    labelfile+=[item[4:]]
engl['file']=labelfile
engl['labeln']=factlabel[0]

engl2=[]
for item in datafile:
    engl2+=[engl['labeln'][engl['file']==item].values[0]]
labels1=to_categorical(engl2)
labels2=np.array(labels1)

xtr = data1
xt = data1

from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(data1,labels2,test_size=0.2,random_state=44)

datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                    width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")
yt = data1
ytr = data1

import tensorflow as tf
from keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

pred = model.predict(test_images)
print(pred)