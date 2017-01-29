import numpy as np
from tflearn.data_utils import to_categorical

predictions = np.load('./data/predictions/public_test_predictions.npy')
labels = np.load('./data/public_test/fer_y_public_test.npy')
labels = to_categorical(labels.astype(int), 7)

correct = 0.0
total = 0.0

for i in xrange(labels.shape[0]):
	pred = np.argmax(predictions[i])
	label = np.argmax(labels[i])
	if pred == label:
		correct += 1
	total += 1

acc = correct / total
print "accuracy = " + str(acc)