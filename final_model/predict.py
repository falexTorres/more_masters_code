import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn import variables as var
from tflearn.data_utils import to_categorical

X_train = np.load('../data/train_data/fer_X_train.npy')
y_train = np.load('../data/train_data/fer_y_train.npy')
X_test = np.load('../data/public_test/fer_X_public_test.npy')
y_test = np.load('../data/public_test/fer_y_public_test.npy')
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32')
y_train = to_categorical(y_train.astype(int), 7)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32')
y_test = to_categorical(y_test.astype(int), 7) 

network = input_data(shape = [None, 48, 48, 1])
network = conv_2d(network, 64, 5, activation = 'relu')
network = max_pool_2d(network, 3, strides = 2)
network = conv_2d(network, 64, 5, activation = 'relu')
network = max_pool_2d(network, 3, strides = 2)
network = conv_2d(network, 128, 4, activation = 'relu')
network = dropout(network, 0.3)
network = fully_connected(network, 3072, activation = 'relu')
network = fully_connected(network, 7, activation = 'softmax')
network = regression(network, optimizer = 'momentum', loss = 'categorical_crossentropy')
model = tflearn.DNN(network)

model.load('./Gudi_model_100_epochs_20000_faces')

with model.session.as_default():
	score = model.evaluate(X_test, y_test, batch_size=50)
	predictions = model.predict(X_test)
	np.save('../data/predictions/public_test_predictions', predictions)
	#score = model.evaluate(X_train, y_train, batch_size=50)
	#predictions = model.predict(X_train)
	#np.save('../data/predictions/train_predictions', predictions)
	print score