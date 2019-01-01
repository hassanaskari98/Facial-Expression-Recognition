

conv_base = MobileNet(weights='imagenet',
                  include_top=False,
                  input_shape=(224,224, 3))
conv_base.summary()

add_model = Sequential()
add_model.add(Flatten(input_shape=conv_base.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(7, activation='sigmoid'))
model = Model(inputs=conv_base.input, outputs=add_model(conv_base.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
batch_size = 7
epochs = 50
k = 5
    
train_datagen = ImageDataGenerator(
            rotation_range=30, 
            width_shift_range=0.1,
            height_shift_range=0.1, 
            vartical_flip=True)
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
           callbacks=[EarlyStopping('val_loss', patience=2, mode="min")])
    pred[test_index,:] = model.predict(X_test)
    test_pred += model.predict(test_images)
print(log_loss(train_labels,pred))
test_pred /= k

predictions = model.predict(test_images)
###-------------------------------------
scores=model.evaluate(train_images,train_labels)
print("%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))



score = model.evaluate(test_images, test_labels)
print("%s:%.2f%%"%(model.metrics_names[1],score[1]*100))





