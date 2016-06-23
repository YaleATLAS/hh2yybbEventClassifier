import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import random

X1=np.asarray([random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10)], dtype=np.float)
X2=np.asarray([random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10)], dtype=np.float)
X3=np.asarray([random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10)], dtype=np.float)
Y=np.asarray([random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10)], dtype=np.float)
W=np.asarray([random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10), random.sample(xrange(1000), 10)], dtype=np.float)


def data_processing(X1, X2, X3, Y, W):
	#shuffle events & split into testing and training
	X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, Y_train, Y_test, W_train, W_test=train_test_split(X1, X2, X3, Y, W, test_size=0.4)
	#scale training set & apply transformation to the test data
	scaler=preprocessing.StandardScaler()
	X1_train = scaler.fit_transform(X1_train)
	X1_test = scaler.transform(X1_test)
	X2_train = scaler.fit_transform(X2_train)
	X2_test = scaler.transform(X2_test)		
	X3_train = scaler.fit_transform(X2_train)
	X3_test = scaler.transform(X2_test)
	Y_train = scaler.fit_transform(Y_train)
	Y_test = scaler.transform(Y_test)
	W_train = scaler.fit_transform(W_train)
	W_test = scaler.transform(W_test)
	return X1_test, X2_test, X3_test, Y_test, W_test
