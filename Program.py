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



#%% Model construction

#------------ Parameters -----------#
learningRate = 0.01
maxIter = 5

nHidden = 128        #Number of neurones in hidden layer
ConvKernel_size = (3,3)       #Size of filters in convolution layer
filter = 32                 #number of filters in convolutional layer
Pool_kernel = (2,2)   #Size of filters in pooling layer

class CrossEntropy:
    def __init__(self): pass

    def loss(self, y, p):
        '''Cross-Entropy Loss function for multiclass predictions'''
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(p))

    def acc(self, y, p):
        ''' Accuracy between One-hot encoding : target value 'y' and predicted 'p' '''
        # np.argmax translates to nominal values, for each entry.
        # the whole values are given to %accuracy function
        accuracy = np.argmax(y, axis=1), np.argmax(p, axis=1)
        return accuracy

    def gradient(self, y, p):
        '''Gradient of Cross-Entropy function with respect to the input of softmax, not the softmax output itself'''
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return p - y  # This is the gradient for the input to softmax



# Gradient descending algo
class GDA():
    def __call__(self, X,y):
        n_samples,n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0
        self.lr = learningRate

        for i in range (maxIter):
            y_pred = np.dot(X,y) + y
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters using the gradients
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

#  Sigmoid activation function
class Sigmoid():
    def __call__(self, X):
        return 1 / (1 + np.exp(-X))
    def gradient(self, X):
        return self.__call__(X) * (1 - self.__call__(X))

# ReLU activation function
class ReLU():
    def __call__(self, X):
        return np.maximum(0, X)

    def gradient(self, X):
        return 1. * (X > 0)

# Softmax activation function
class Softmax():
    def __call__(self, X):
        e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def gradient(self, X):
        return self.__call__(X) * (1 - self.__call__(X))
























