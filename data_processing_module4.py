#importing libraries 
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import random

#shuffling events and spliting into testing and training sets, scaling X_jet, X_photon, and X_muon test sets based on training sets. This function takes in the read X_jet, X_photon, X_muon, Y and W arrays and returns the scaled test arrays of each variable.
def scale_shuffle_split(X_jets, X_photons, X_muons, Y, W):
	#shuffle events & split into testing and training sets
	X_jets_train, X_jets_test, X_photons_train, X_photons_test, X_muons_train, X_muons_test, Y_train, Y_test, W_train, W_test=train_test_split(X1, X2, X3, Y, W, test_size=0.4)
	#fit a transformation to the training set of the X_Jet, X_Photon, and X_Muon data and thus apply a transformation to the corresponding test data 
	scaler=preprocessing.StandardScaler()
	X_jets_train = scaler.fit_transform(X_jets_train)
	X_jets_test = scaler.transform(X_jets_test)
	X_photons_train = scaler.fit_transform(X_photons_train)
	X_photons_test = scaler.transform(X_photons_test)		
	X_muons_train = scaler.fit_transform(X_muons_train)
	X_muons_test = scaler.transform(X_muons_test)
	return X_jets_test, X_photons_test, X_muons_test, Y_test, W_test

#testing the scaling of the X_jet, X_photon, and X_muon sets so each variable has a mean of about 0 and a standard deviation of about 1. This function takes in the test arrays of each variable and returns the mean for each element of each variable through all events.
def test_scale(X_jets_test, X_photons_test, X_muons_test):
	mean=[]
	standev=[]
	for i in range (X_jets_test.shape[1]):
		mean.append(round(sum(X_jets_test[:, i])/float(len(X_jets_test[:,i])), 0))
		standev.append(round(np.std(X_jets_test[:,i]), 0))
	print 'X_Jet Mean: {}, X_Jet Standard Deviation'.format(mean, standev)
	mean=[]
	standev=[]
	for i in range (X_photons_test.shape[1]):
		mean.append(round(sum(X_photons_test[:, i])/float(len(X_photons_test[:,i])), 0))
		standev.append(round(np.std(X_photons_test[:,i]), 0))
	print 'X_Photon Mean: {}, X_Photon Standard Deviation'.format(mean, standev)
	mean=[]
	standev=[]
	for i in range (X_muons_test.shape[1]):
		mean.append(round(sum(X_muons_test[:, i])/float(len(X_muons_test[:,i])), 0))
		standev.append(round(np.std(X_muons_test[:,i]), 0))
	print 'X_Muon Mean: {}, X_Muon Standard Deviation'.format(mean, standev)

#checking to make sure that the test/ train data is split into appropriate proportions (test data (40%) and training data (other 60%)). This function takes in the read arrays of each variables and the test arrays for each variable and returns True for each variable if the test set is approximately 40% of the size of the input set.
def test_split(X_jets, X_jets_test, X_photons, X_photons_test, X_muons, X_muons_test, Y, Y_test, W, W_test):
	if round (float(X_jets_test.shape[0])/float(X_jets.shape[0]), 1)==0.4:
		print True 
	if round (float(X_photons_test.shape[0])/float(X_photons.shape[0]), 1)==0.4:
		print True
	if round (float(X_muons_test.shape[0])/float(X_muons.shape[0]), 1)==0.4:
		print True 
	if round (float(Y_test.shape[0])/float(Y.shape[0]), 1)==0.4:
		print True 
	if round (float(W_test.shape[0])/float(W.shape[0]), 1)==0.4:
		print True  