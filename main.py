import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names=["0: airplane", "1: automobile", "2: bird", "3: cat", "4: deer",
             "5: dog", "6: frog", "7: horse", "8: ship", "9: truck"]

# Normalising images
x_train=x_train/255.0
x_test=x_test/255.0
# x_train.shape= (50000, 32, 32, 3), y_train.shape=(50000, 1)
# x_test.shape= (10000, 32, 32, 3), y_test.shape=(10000, 1)

# plt.imshow(x_train[0])
# plt.show()

# Defining object
model=tf.keras.models.Sequential()

# Adding first CCN layer:
# 1) filters (kernels)=32 (number of features the layer will learn to detect)
# 2) kernel size= 3 (dimension of filter which determines area of input data)
# 3) padding= same (for non sqr matrix, same when all data covered
#             after adding 0s, valid when some data left out)
# 4) activation= ReLU   5) input shape= (32,32,3) (pixels, rgb (3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same",
                                 activation="relu", input_shape=[32,32,3]))

# Adding 2nd CCN layer
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                 padding="same", activation="relu"))
# Adding Maxpool layer: (to find important features and ignore small changes)
# 1) pool size=2  2) Strides=2 (step size)
# 3) padding=valid (ccn: details imp so same, maxpool: reduction so valid)
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))

# Adding 3rd and 4th CCN layer and maxpool layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                 padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                 padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))

# Adding dropout and flattening layer
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Flatten())

# Adding 2 dense layers (second is output layer)
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

# Compiling model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size=10, epochs=10)

# evaluate model performance;
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

y_pred=np.argmax(model.predict(x_test),axis=-1)
print(y_pred[0], y_test[0])
print(y_pred[200], y_test[200])
print(y_pred[86], y_test[86])

# Confusion Matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)
acc_cm=accuracy_score(y_test, y_pred)
print(acc_cm)