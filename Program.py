import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.datasets import mnist
from sklearn.model_selection import train_test_split


#%% Task 1
#load dataset and output dataset shape
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()


#%% Task 2
#class label
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_example(X_data, Y_data, num_img=None):
    for i in range(num_img):
        random_index = np.random.randint(0, X_data.shape[0])

        random_image = X_data[random_index]
        random_label = Y_data[random_index][0]

        plt.imshow(random_image)
        plt.title(f"Label: {class_names[random_label]}")
        plt.show()

show_example(X_train, Y_train, num_img=3)

#%% Task 3
def normalize(X_data):
    X_data = X_data.astype('float32')
    X_data = X_data/255.0
    return X_data

X_train = normalize(X_train)
X_test = normalize(X_test)

#%% Task 4
def to_categorical(X, n_col = None):
    if not n_col:
        n_col = np.max(X) + 1
    one_hot = np.zeros((X.shape[0], n_col))
    one_hot[np.arange(X.shape[0]), X] = 1
    return one_hot

def transform_label(class_names):
    labels = [index for index in range(len(class_names))]
    return np.array(labels)

labels = transform_label(class_names)
one_hot = to_categorical(labels)

#%% Task 5
# Count the number of samples for each class in Y_train
class_counts = np.bincount(Y_train.flatten(), minlength=len(class_names))

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(class_names, class_counts, color='skyblue')

# Add titles and labels
plt.title('Class Distribution in CIFAR-10 Training Set')
plt.xlabel('Class')
plt.ylabel('Number of Samples')

plt.show()





















