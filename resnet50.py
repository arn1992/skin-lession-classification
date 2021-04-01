import os
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import Dense, Dropout, GlobalAveragePooling2D,GlobalMaxPooling2D
import matplotlib.pyplot as plt

X_train = np.load("D:/AI in medicine/final_project_mobilenet/256_192_train.npy")
y_train = np.load("D:/AI in medicine/final_project_mobilenet/train_labels.npy")
X_val = np.load("D:/AI in medicine/final_project_mobilenet/256_192_val.npy")

y_val = np.load("D:/AI in medicine/final_project_mobilenet/val_labels.npy")

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
print(y_train.shape, y_val.shape)

base_model=ResNet50(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dropout(0.5)(x)
x=Dense(512,activation='relu')(x) #dense layer 2
x = Dropout(0.5)(x)
x=Dense(256,activation='relu')(x) #dense layer 3
x = Dropout(0.5)(x)
preds=Dense(7,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)
# Configure and compile the model
for i,layer in enumerate(model.layers):
  print(i,layer.name)
print(i)

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()


train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest')

train_datagen.fit(X_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(X_val)

batch_size = 32
epochs = 3
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),
                              validation_steps=(X_val.shape[0] // batch_size))

for layer in model.layers[:20]:
    layer.trainable = False

for layer in model.layers[20:]:
    layer.trainable = True

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
                                            min_lr=0.000001, cooldown=3)

batch_size = 32
epochs = 30
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),
                              validation_steps=(X_val.shape[0] // batch_size), callbacks=[learning_rate_reduction])

loss_val, acc_val = model.evaluate(X_val, y_val, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))

X_test = np.load("D:/AI in medicine/final_project_mobilenet/256_192_test.npy")
y_test = np.load("D:/AI in medicine/final_project_mobilenet/test_labels.npy")
y_test = to_categorical(y_test)
loss_test, acc_test = model.evaluate(X_test, y_test, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))
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
