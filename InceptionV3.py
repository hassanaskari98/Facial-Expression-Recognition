from keras.applications import InceptionV3
from ctypes import c_void_p
import keras
import numpy as np
#import pandas as pd
import glob
import cv2
import tensorflow as tf
from sklearn.model_selection import KFold 

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from sklearn.metrics import log_loss
import keras.backend as K
#from keras.layers import Input
from keras.applications.imagenet_utils import _obtain_input_shape


main_dir = '/home/hasan/Desktop/new111/jaffe/'
filenames = []
filenames += glob.glob(main_dir+"/*"+".tiff")

n_row = 164
n_col = 164
img = np.zeros((n_row,n_col,3))
images = []
for file in filenames:
    img = np.zeros((n_row,n_col,3))
    temp = np.asarray(cv2.imread(file, 0))
    temp = cv2.resize(temp,(n_row,n_col))
    img[:,:,0] = temp
    img[:,:,1] = temp
    img[:,:,2] = temp
    images.append(img)
images = np.asanyarray(images)
#print(images)

labels_count = 7
TRAINING_SIZE = 170
VALIDATION_SIZE = len(images)- TRAINING_SIZE
train_images = images[:TRAINING_SIZE,:,:]
test_images = images[TRAINING_SIZE:,:,:]

train_images = train_images.reshape((170,n_row,n_col,3))
test_images = test_images.reshape((43,n_row,n_col,3))
num_classes = 7
labels = []

images = []
for file in filenames:
 #   print(file)
    if file.find('NE')!=-1:
        labels.append(0)
    if file.find('HA')!=-1:
        labels.append(1)
    if file.find('SA')!=-1:
        labels.append(2)
    if file.find('SU')!=-1:
        labels.append(3)
    if file.find('AN')!=-1:
        labels.append(4)
    if file.find('DI')!=-1:
        labels.append(5)
    if file.find('FE')!=-1:
        labels.append(6)
    img = np.array(cv2.imread(file))
    images.append(img)

labels = np.array(labels)
#print(labels)
train_labels = np.asanyarray(labels[:TRAINING_SIZE],dtype=int)
test_labels = np.asarray(labels[170:], dtype=int)

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

# Convert type to float32
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255
train_images[:,:,:,1] = train_images[:,:,:,0]
train_images[:,:,:,2] = train_images[:,:,:,0]



conv_base = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(n_row,n_col, 3))
conv_base.summary()

conv_base.summary()

add_model = Sequential()
add_model.add(Flatten(input_shape=conv_base.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(7, activation='sigmoid'))
model = Model(inputs=conv_base.input, outputs=add_model(conv_base.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
batch_size = 24
epochs = 100
k = 10
    
train_datagen = ImageDataGenerator(
            rotation_range=30, 
            width_shift_range=0.1,
            height_shift_range=0.1, 
            horizontal_flip=True)
kf = KFold(n_splits=k)
kf.get_n_splits(train_images)

pred = np.zeros((len(train_images),7))
test_pred = np.zeros((len(test_images),7))
for train_index, test_index in kf.split(train_images):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_images[train_index], train_images[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]
    train_datagen.fit(X_train)
    gmodel = model.fit_generator(
           train_datagen.flow(X_train, y_train, batch_size=batch_size),
           steps_per_epoch=24,
           verbose=1,
           epochs=epochs,
           validation_data=(X_test, y_test),
           callbacks=[EarlyStopping('val_loss', patience=3, mode="min")])
    pred[test_index,:] = model.predict(X_test)
    test_pred += model.predict(test_images)
   # gmodel = model.fit(X_train,y_train,batch_size=1,epochs=1,
    # callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)])


print(log_loss(train_labels,pred))
test_pred /= k

predictions = model.predict(test_images)
###-------------------------------------
scores=model.evaluate(train_images,train_labels)
print("%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))



score = model.evaluate(test_images, test_labels)
print("%s:%.2f%%"%(model.metrics_names[1],score[1]*100))

#============save result===========================
import pickle
filename = 'train_image_InceptionV3.csv'
pickle.dump(pred, open(filename, 'wb'))

filename1 = 'test_image_InceptionV3.csv'
pickle.dump(predictions, open(filename1, 'wb'))
