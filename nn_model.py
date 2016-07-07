from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Masking, GRU, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def train(X_jets_train, X_photons_train, X_muons_train, y_train, w_train):
	jet_channel = Sequential()
	photon_channel = Sequential()

	JET_SHAPE = X_jets_train.shape[1:]
	PHOTON_SHAPE = X_photons_train.shape[1:]

	jet_channel.add(Masking(mask_value=-999, input_shape=JET_SHAPE, name='jet_masking'))
	jet_channel.add(GRU(25, name='jet_gru'))
	jet_channel.add(Dropout(0.3, name='jet_dropout'))

	photon_channel.add(Masking(mask_value=-999, input_shape=PHOTON_SHAPE, name='photon_masking'))
	photon_channel.add(GRU(10, name='photon_gru'))
	photon_channel.add(Dropout(0.3, name='photon_dropout'))

	combined_rnn = Sequential()
	combined_rnn.add(Merge([jet_channel, photon_channel], mode='concat'))
	combined_rnn.add(Dense(12, activation='relu'))
	combined_rnn.add(Dropout(0.3))
	combined_rnn.add(Dense(6, activation='relu'))
	combined_rnn.add(Dropout(0.3))
	combined_rnn.add(Dense(3, activation='relu'))
	combined_rnn.add(Dropout(0.3))
	combined_rnn.add(Dense(2, activation='softmax'))

	combined_rnn.compile('adam', 'sparse_categorical_crossentropy')

	print 'Training:'
	try:
		combined_rnn.fit([X_jets_train, X_photons_train], 
    		y_train, batch_size=16, class_weight={
                0 : 0.5 * (float(len(y_train)) / (y_train == 0).sum()),
                1 : 0.5 * (float(len(y_train)) / (y_train == 1).sum()),
     		},
        	callbacks = [
            	EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
          		ModelCheckpoint('./models/combinedrnn-progress',
          		monitor='val_loss', verbose=True, save_best_only=True)
        	],
    	nb_epoch=30, validation_split = 0.2) 

	except KeyboardInterrupt:
		print 'Training ended early.'

	return combined_rnn

def test(net, X_jets_test, X_photons_test, X_muons_test, y_test, w_test):
	print y_test.shape
	print w_test.shape

	yhat_rnn = net.predict([X_jets_test, X_photons_test], verbose = True, batch_size = 512) 
	
	return yhat_rnn