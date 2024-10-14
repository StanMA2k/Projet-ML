import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.datasets import mnist
from sklearn.model_selection import train_test_split

#%% Task 1
#load dataset and output dataset shape
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#%% Task 2
#class label
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_random_image(X_data, Y_data, num_img=None):
    for i in range(num_img):
        random_index = np.random.randint(0, X_data.shape[0])

        random_image = X_data[random_index]
        random_label = Y_data[random_index][0]

        plt.imshow(random_image)
        plt.title(f"Label: {class_names[random_label]}")
        plt.show()

show_random_image(X_train, Y_train, num_img=3)