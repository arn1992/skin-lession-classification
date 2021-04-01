import os
from glob import glob
from sklearn.metrics import classification_report
import functools
import keras
import pandas as pd
import numpy as np
import numpy
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import MobileNet
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
import random
from keras.layers import Dense, Dropout, GlobalAveragePooling2D,GlobalMaxPooling2D,DepthwiseConv2D,Concatenate
import matplotlib.pyplot as plt
import tensorflow
from sklearn.metrics import confusion_matrix
X_train = np.load("D:/AI in medicine/final_project_mobilenet/256_192_train.npy")
y_train = np.load("D:/AI in medicine/final_project_mobilenet/train_labels.npy")
X_val = np.load("D:/AI in medicine/final_project_mobilenet/256_192_val.npy")

y_val = np.load("D:/AI in medicine/final_project_mobilenet/val_labels.npy")

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
print(y_train.shape, y_val.shape)
print(y_val)

base_model=MobileNet(weights='imagenet',alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
#x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
#x = Dropout(0.5)(x)
x=Dense(512,activation='relu')(x) #dense layer 2
x = Dropout(0.5)(x)
#x=Dense(256,activation='relu')(x) #dense layer 3
#x = Dropout(0.5)(x)
preds=Dense(7,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)
# Configure and compile the model
for i,layer in enumerate(model.layers):
  print(i,layer.name)
print(i)

top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

top3_acc.__name__ = 'top3_acc'

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy','top_k_categorical_accuracy',top3_acc])

model.summary()


train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest')

train_datagen.fit(X_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(X_val)

batch_size = 32
epochs = 1
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),
                              validation_steps=(X_val.shape[0] // batch_size))

for layer in model.layers:
    layer.trainable = True
'''
for layer in model.layers[20:]:
    layer.trainable = True'''
'''
for layer in model.layers[:12]:
    layer.trainable=True
for layer in model.layers[12:13]:
    layer.trainable=DepthwiseConv2D(64,strides=(2, 2),dilation_rate=(1,1))
for layer in model.layers[13:25]:
    layer.trainable=True
for layer in model.layers[25:26]:
    layer.trainable=DepthwiseConv2D(128,strides=(2, 2),dilation_rate=(1,1))
for layer in model.layers[26:38]:
    layer.trainable=True
for layer in model.layers[38:39]:
    layer.trainable=DepthwiseConv2D(256,strides=(2, 2),dilation_rate=(2,2))

for layer in model.layers[39:75]:
    layer.trainable=True
for layer in model.layers[75:76]:
    layer.trainable=DepthwiseConv2D(512,strides=(2, 2),dilation_rate=(2,2))
for layer in model.layers[76:81]:
    layer.trainable=True
for layer in model.layers[81:82]:
    a=DepthwiseConv2D(1024, strides=(2, 2),dilation_rate=(4,4))
    b=DepthwiseConv2D(1024, strides=(2, 2),dilation_rate=(8,8))
    c=DepthwiseConv2D(1024, strides=(2, 2),dilation_rate=(16,16))
    d=Concatenate([a,b])
    layer.trainable=Concatenate([d,c])
for layer in model.layers[82:]:
    layer.trainable=True
'''
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc','top_k_categorical_accuracy',top3_acc])
model.summary()

save_best = ModelCheckpoint(filepath='D:/AI in medicine/final_project_mobilenet/mobilenet/checkpoint-{val_acc:.4f}.h5',monitor='val_acc',mode='auto',save_best_only=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=7, verbose=1, factor=np.sqrt(0.1),
                                            min_lr=0.5e-6, cooldown=0,mode='auto')
callbacks=[learning_rate_reduction,save_best]

batch_size = 32
epochs = 200
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),
                              validation_steps=(X_val.shape[0] // batch_size), callbacks=callbacks)

loss_val, acc_val,top5,top3 = model.evaluate(X_val, y_val, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))
print('top5 validate accuracy: ',top5)
print('top3 validate accuracy: ',top3)

X_test = np.load("D:/AI in medicine/final_project_mobilenet/256_192_test.npy")
y_test = np.load("D:/AI in medicine/final_project_mobilenet/test_labels.npy")
#print("1st y_test",y_test)
target_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign_keratosis-like_lesions', 'Dermatofibroma', 'Melanocytic Nevi', 'Vascular_lesions', 'Melanoma']
print("1st y_test",y_test, y_test.shape)
y_pred = model.predict(X_test)
print("1 y_pred",y_pred,y_pred.shape)
y_classes = y_pred.argmax(axis=-1)
print("2  y_pred",y_classes,y_classes.shape)
print(classification_report(y_test, y_classes, target_names=target_names))




y_test = to_categorical(y_test)
print("2nd y_test",y_test, y_test.shape)
loss_test, acc_test,top5,top3 = model.evaluate(X_test, y_test, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))
print("top5 test accuracy: ",top5)
print("top3 test accuracy: ",top3)
pre=model.predict(X_test,verbose=1)

print("pre",pre, pre.shape,pre.ndim)
n_values=7
c = np.eye(n_values, dtype=int)[np.argmax(pre, axis=1)]
print('one-hot-encoding',c,c.shape)


cm = confusion_matrix(y_test.argmax(axis=1), c.argmax(axis=1))
print("confusion matrix",cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('new_cm',cm)
print('acc',cm.diagonal())
model.save("VGG16.h5")
# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, label = "training")
plt.plot(epochs, val_acc, label = "validation")
plt.legend(loc="upper left")
plt.title('Training and validation accuracy')
plt.show()
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, label = "training")
plt.plot(epochs, val_loss, label = "validation")
plt.legend(loc="upper right")
plt.title('Training and validation loss')
plt.show()
