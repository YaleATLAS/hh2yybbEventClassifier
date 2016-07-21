from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.layers import Masking, GRU, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import numpy as np

def train(data):

	# -- extract training matrices from `data`
	X_jets_train = data['X_jet_train']
	X_photons_train = data['X_photon_train']
	X_event_train = data['X_event_train']
	y_train = data['y_train']

	# -- build net
	jet_channel = Sequential()
	photon_channel = Sequential()
	event_level = Sequential()

	JET_SHAPE = X_jets_train.shape[1:]
	PHOTON_SHAPE = X_photons_train.shape[1:]
	EVENT_SHAPE = X_event_train.shape[1]

	jet_channel.add(Masking(mask_value=-999, input_shape=JET_SHAPE, name='jet_masking'))
	jet_channel.add(GRU(25, name='jet_gru'))
	jet_channel.add(Dropout(0.3, name='jet_dropout'))

	photon_channel.add(Masking(mask_value=-999, input_shape=PHOTON_SHAPE, name='photon_masking'))
	photon_channel.add(GRU(10, name='photon_gru'))
	photon_channel.add(Dropout(0.3, name='photon_dropout'))

	event_level.add(Lambda(lambda x: x, input_shape=(EVENT_SHAPE, )))

	combined_rnn = Sequential()
	combined_rnn.add(Merge([jet_channel, photon_channel, event_level], mode='concat'))	
	combined_rnn.add(Dense(32, activation='relu'))
	# combined_rnn.add(Dropout(0.3))
	# combined_rnn.add(Dense(32, activation='relu'))
	# combined_rnn.add(Dropout(0.3))
	# combined_rnn.add(Dense(16, activation='relu'))
	combined_rnn.add(Dropout(0.3))
	combined_rnn.add(Dense(8, activation='relu'))
	combined_rnn.add(Dropout(0.3))
	combined_rnn.add(Dense(len(np.unique(y_train)), activation='softmax'))

	combined_rnn.compile('adam', 'sparse_categorical_crossentropy')

	logger = logging.getLogger('Train')
	logger.info('Compiling the net')
	try:
		combined_rnn.fit([X_jets_train, X_photons_train, X_event_train], 
    		y_train, batch_size=16, 
        	callbacks = [
            	EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
          		ModelCheckpoint('./combinedrnn-progress',
          		monitor='val_loss', verbose=True, save_best_only=True)
        	],
    	nb_epoch=30, validation_split = 0.2) 

	except KeyboardInterrupt:
		print 'Training ended early.'

	return combined_rnn