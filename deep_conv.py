import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape,Dropout
from keras import optimizers
import tensorflow as tf
import keras



DATADIR='D:/AI in medicine/final_project_mobilenet/input'

CATEGORIES = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
category="HAM10000_images_part_1"






IMG_SIZE = 148



training_data = []

def create_training_data():
    path = os.path.join(DATADIR, category)  # create path to dogs and cats


    for img in (os.listdir(path)):  # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(os.path.join(path, img))
            b, g, r = cv2.split(img_array)
            rgb_img = cv2.merge([r, g, b])

            new_array = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            training_data.append(new_array)  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass



            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
#print(len(training_data))



random.shuffle(training_data)
'''for sample in training_data:
    print(sample)'''


x_train = np.array(training_data, dtype="float")

max_value = float(x_train.max())
x_train =x_train / max_value




#test

category="HAM10000_images_part_2"



testing_data = []

def create_testing_data():
    path = os.path.join(DATADIR, category)  # create path to dogs and cats


    for img in (os.listdir(path)):  # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(os.path.join(path, img))  # convert to array
            b, g, r = cv2.split(img_array)
            rgb_img = cv2.merge([r, g, b])
            new_array = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            testing_data.append(new_array)  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass



            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_testing_data()
#print(len(testing_data))



random.shuffle(testing_data)



x_test=np.array(testing_data, dtype="float")
max_value = float(x_test.max())
x_test =x_test/ max_value

x_train = x_train.reshape((len(x_train), 148, 148, 3))
x_test = x_test.reshape((len(x_test), 148, 148, 3))



autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))
autoencoder.summary()
# Flatten encoding for visualization
autoencoder.add(Flatten())
autoencoder.add(Reshape((19, 19, 8)))

# Decoder Layers
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

autoencoder.summary()

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)
encoder.summary()

opt=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
autoencoder.compile(optimizer=opt, loss='mean_squared_error')
autoencoder.fit(x_train, x_train,
                epochs=330,
                batch_size=64,
                validation_data=(x_test, x_test))

num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(256,140))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)

    plt.imshow(x_test[image_idx].reshape(148, 148,-1))





    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(76,38))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(148, 148,-1))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()