from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Masking, GRU, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

def train(data, mode):
	'''
	Args:
		data: an OrderedDict containing all X, y, w ndarrays for all particles (both train and test), e.g.:
              data = {
                "X_jet_train" : X_jet_train,
                "X_jet_test" : X_jet_test,
                "X_photon_train" : X_photon_train,
                "X_photon_test" : X_photon_test,
                "y_train" : y_train,
                "y_test" : y_test,
                "w_train" : w_train,
                "w_test" : w_test
              }
        mode: a string specifying the type of task, either 'regression' or 'classification'
    Returns:
    	combine_rnn: a Sequential trained on data
    '''

	X_jet_train = data['X_jet_train']
	X_photon_train = data['X_photon_train']
	y_train = data['y_train']

	jet_channel = Sequential()
	photon_channel = Sequential()

	JET_SHAPE = X_jet_train.shape[1:]
	PHOTON_SHAPE = X_photon_train.shape[1:]

	jet_channel.add(Masking(mask_value=-999, input_shape=JET_SHAPE, name='jet_masking'))
	jet_channel.add(GRU(25, name='jet_gru'))
	jet_channel.add(Dropout(0.3, name='jet_dropout'))

	photon_channel.add(Masking(mask_value=-999, input_shape=PHOTON_SHAPE, name='photon_masking'))
	photon_channel.add(GRU(10, name='photon_gru'))
	photon_channel.add(Dropout(0.3, name='photon_dropout'))

	combined_rnn = Sequential()
	combined_rnn.add(Merge([jet_channel, photon_channel], mode='concat'))
	combined_rnn.add(Dense(24, activation='relu'))
	combined_rnn.add(Dropout(0.3))
	combined_rnn.add(Dense(12, activation='relu'))
	combined_rnn.add(Dropout(0.3))
	if mode == 'classification':
		combined_rnn.add(Dense(6, activation='softmax'))
		combined_rnn.compile('adam', 'sparse_categorical_crossentropy')

	elif mode == 'regression':
		combined_rnn.add(Dense(1))
		combined_rnn.compile('adam', 'mae')

	try:
		weights_path = os.path.join('weights', 'combinedrnn-progress.h5')
		combined_rnn.load_weights(weights_path)
	except IOError:
		print 'Pre-trained weights not found'

	print 'Training:'
	try:
		combined_rnn.fit([X_jet_train, X_photon_train], 
    		y_train, batch_size=16, class_weight={
                k : (float(len(y_train)) / float(len(np.unique(y_train)) * 
                	(len(y_train[y_train == k])))) for k in np.unique(y_train)
     		},
        	callbacks = [
            	EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
          		ModelCheckpoint(weights_path,
          		monitor='val_loss', verbose=True, save_best_only=True)
        	],
    	nb_epoch=30, validation_split = 0.2) 

	except KeyboardInterrupt:
		print 'Training ended early.'

	# -- load best weights back into the net
	combined_rnn.load_weights(weights_path)

	return combined_rnn