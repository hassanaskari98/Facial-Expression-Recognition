import numpy as np
import pickle
import pandas as pd
#import tensorflow as tf
import cv2
#import keras
import glob
#from keras.layers.core import Dense, Dropout, Activation
#from keras.models import Sequential


main_dir = 'C:/Users/hasan/Desktop/new111/jaffe/'
filenames = []
filenames += glob.glob(main_dir+"/*"+".tiff")

n_row = 64
n_col = 64
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
#train_labels=train_labels.ravel()
#train_labels = train_labels.T
#train_labels = np.tile(train_labels,(5,1)).T

test_labels = np.asarray(labels[170:], dtype=int)
#test_labels = np.tile(test_labels,(5,1)).T
#train_labels = keras.utils.to_categorical(train_labels, num_classes)
#test_labels = keras.utils.to_categorical(test_labels, num_classes)
#================================================
#matrix_train_labels = np.zeros((170,7))
#matrix_train_labels[0:170] = train_labels
#matrix_train_labels[170:340] = train_labels
#matrix_train_labels[340:510] = train_labels
#matrix_train_labels[510:680] = train_labels
#matrix_train_labels[680:850] = train_labels
#matrix_train_labels = matrix_train_labels
#-----------------------------------------------
#matrix_test_labels = np.zeros((43,7))
#matrix_test_labels[0:43] = test_labels
#matrix_test_labels[43:86] = test_labels
#matrix_test_labels[86:129] = test_labels
#matrix_test_labels[129:172] = test_labels
#matrix_test_labels[172:215] = test_labels

#==========================================================
#face_matrix_train=np.zeros((170,28))

x1 = pickle.load(open('save.train/train_image_vgg16.csv', 'rb'))
x2 = pickle.load(open('save.train/train_image_vgg19.csv', 'rb'))
#x8 = pickle.load(open('save.train/train_image_VGGFace.csv', 'rb'))
x3 = pickle.load(open('save.train/train_image_ResNet.csv', 'rb'))
x4 = pickle.load(open('save.train/train_image_DenseNet201.csv', 'rb'))
x5 =  pickle.load(open('save.train/train_image_InceptionV3.csv', 'rb'))
x6 =  pickle.load(open('save.train/train_image_DenseNet121.csv', 'rb'))
x7 =  pickle.load(open('save.train/train_image_MobileNet.csv', 'rb'))


face_matrix_train =np.concatenate((x1,x2,x3,x4,x5,x6,x7),axis = 1)


#face_matrix_train[680:850] = x5
#face_matrix_train = face_matrix_train.T
#-----------------------------------------------
face_matrix_test =np.zeros((172,7))



y1 = pickle.load(open('save.test/test_image_vgg16.csv', 'rb'))
y2 = pickle.load(open('save.test/test_image_vgg19.csv', 'rb'))
#y8 = pickle.load(open('save.test/test_image_VGGFace.csv', 'rb'))
y3 = pickle.load(open('save.test/test_image_ResNet.csv', 'rb'))
y4 = pickle.load(open('save.test/test_image_DenseNet201.csv', 'rb'))
y5 = pickle.load(open('save.test/test_image_InceptionV3.csv', 'rb'))
y6 = pickle.load(open('save.test/test_image_DenseNet121.csv', 'rb'))
y7 = pickle.load(open('save.test/test_image_MobileNet.csv', 'rb'))

face_matrix_test =np.concatenate((y1,y2,y3,y4,y5,y6,y7),axis = 1)

#face_matrix_test[0:43] = y1
#face_matrix_test[43:86] = y2
#face_matrix_test[86:129] = y3
#face_matrix_test[129:172] = y4
#face_matrix_test[172:215] = y5
#---------------------------------------------------------------
# Feature Scaling
import lightgbm as lgb
d_train = lgb.Dataset(face_matrix_train, label=train_labels)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'multiclass'
params['num_class']=7
params['metric'] = 'multi_logloss'
params['sub_feature'] = .5
#params['num_leaves'] = 10
params['min_data'] = 5
params['max_depth'] = 3

clf = lgb.train(params, d_train, 200)

#Prediction
y_pred=clf.predict(face_matrix_test)

from sklearn.metrics import accuracy_score
argpred = np.argmax(y_pred,axis=1)
accuracy = accuracy_score(argpred,test_labels)
accuracy2 = accuracy_score(np.argmax(y2,axis=1),test_labels)
accuracy3 = accuracy_score(np.argmax(y3,axis=1),test_labels)
accuracy4 = accuracy_score(np.argmax(y4,axis=1),test_labels)
accuracy5 = accuracy_score(np.argmax(y5,axis=1),test_labels)
accuracy6 = accuracy_score(np.argmax(y6,axis=1),test_labels)
accuracy7 = accuracy_score(np.argmax(y7,axis=1),test_labels)
#accuracy8 = accuracy_score(np.argmax(y8,axis=1),test_labels)

#accuracy2=accuracy_score(y2,test_labels)
#accuracy3=accuracy_score(y3,test_labels)
#accuracy4=accuracy_score(y4,test_labels)


