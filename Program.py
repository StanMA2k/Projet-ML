import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam




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


#%% LeNet 5
def LeNet5(X_train, y_train, X_test, y_test, opt="SGD"):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3)))

    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
    model.add(tf.keras.layers.AvgPool2D())

    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(tf.keras.layers.AvgPool2D())

    model.add(tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation='relu'))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=84, activation='relu'))

    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model

X_train = X_train.reshape(-1, 32, 32, 3)
X_test  = X_test.reshape(-1, 32, 32, 3)

y_train = tf.keras.utils.to_categorical(Y_train, 10)
y_test  = tf.keras.utils.to_categorical(Y_test, 10)

cnn = LeNet5(X_train, y_train, X_test, y_test, opt="SGD")


#%% VGG1

y_train_onehot = to_categorical(Y_train, num_classes=10)
y_test_onehot = to_categorical(Y_test, num_classes=10)

def create_vgg1_one_conv_adam():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

vgg1_one_conv = create_vgg1_one_conv_adam()
history_vgg1_one_conv = vgg1_one_conv.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=10, batch_size=64)

#%% VGG2

def build_vgg2():
    model = Sequential()

    # first VGG Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # second VGG Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # hidden layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # output layer 
    model.add(layers.Dense(10, activation='softmax'))

    return model

# build and compile the model
vgg2_model = build_vgg2()
vgg2_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



#%% VGG3
from tensorflow.keras import layers, models

def create_vgg3_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()

    # first VGG bloc 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # second VGG bloc 
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # third VGG bloc
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


vgg3_model = create_vgg3_model()

# Compile model
vgg3_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



















