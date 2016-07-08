from keras.layers import Masking, GRU, Input, merge, Activation, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import numpy as np
import deepdish.io as io
import os

def train(data):
	'''
	'''
	logger = logging.getLogger('Train')

	# -- extract training matrices from `data`
	X_jets_train = data['X_jet_train']
	X_photons_train = data['X_photon_train']
	X_event_train = data['X_event_train']
	y_train = data['y_train']

	# -- class weight needed if we want equal class representation
	#class_weight = {k : (1./len(np.unique(y_train))) / (float(len(y_train[y_train == k])) / float(len(y_train))) for k in np.unique(y_train)}
	class_weight = {k : (float(len(y_train)) / float(len(np.unique(y_train)) * (len(y_train[y_train == k])))) for k in np.unique(y_train)}


	# -- placeholders for matrix shapes
	JET_SHAPE = X_jets_train.shape[1:]
	PHOTON_SHAPE = X_photons_train.shape[1:]
	EVENT_SHAPE = (X_event_train.shape[1], )

	# -- input layers (3 streams)
	jet_inputs = Input(shape=JET_SHAPE)
	photon_inputs = Input(shape=PHOTON_SHAPE)
	event_inputs = Input(shape=EVENT_SHAPE)

	# -- jet RNN
	jet_outputs = GRU(25, name='jet_gru')(
				  	  Masking(mask_value=-999, name='jet_masking')(
					  	  jet_inputs
				  	  )
			  	  )

	# -- photon RNN
	photon_outputs = GRU(10, name='photon_gru')(
					 	Masking(mask_value=-999, name='photon_masking')(
					    	photon_inputs
					  	)
				  	 )
			  	  
	# -- merge (jet, photon) RNNs
	rnn_outputs = merge([jet_outputs, photon_outputs], mode='concat')
	rnn_outputs = Dense(16, activation='relu')(
					#Dropout(0.3)(
						#Dense(32, activation='relu')(
							#Dropout(0.3)(
								rnn_outputs
							#)
						#)
					#)
				  )

	# -- merge event level info as well
	merged_streams = merge([rnn_outputs, event_inputs], mode='concat')

	# -- final feed-forward processing of info up to output layer with softmax
	yhat = Dense(len(np.unique(y_train)), activation='softmax')(
			Dropout(0.2)(
				#Dense(8, activation='relu')(
					#Dropout(0.2)(
						Dense(8, activation='relu')(
							Dropout(0.2)(
								merged_streams
							)
						)
					#)
				#)
			)
		   )

	# -- build Model from symbolic structure above
	net = Model(input=[jet_inputs, photon_inputs, event_inputs], output=yhat)
	net.summary()
	net.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

	try:
		weights_path = os.path.join('weights', 'functional_combined_rnn-progress.h5')
		logger.info('Trying to load weights from ' + weights_path)
		net.load_weights(weights_path)
		logger.info('Weights found and loaded from ' + weights_path)
	except IOError:
		logger.info('Pre-trained weights not found in ' + weights_path)

	logger.info('Compiling the net')
	# -- train!
	try:
		net.fit([X_jets_train, X_photons_train, X_event_train], 
    		y_train, batch_size=16, 
    		class_weight=class_weight,
        	callbacks = [
            	EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
          		ModelCheckpoint(weights_path,
          		monitor='val_loss', verbose=True, save_best_only=True)
        	],
    	nb_epoch=100, validation_split=0.2) 

	except KeyboardInterrupt:
		print 'Training ended early.'

	# -- load best weights back into the net
	net.load_weights(weights_path)
	return net

def test(net, data):
	'''
	Args:
	-----
		net: a trained keras Model instance
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
    Returns:
    --------
    	yhat: numpy array of dim [n_ev, n_classes] with the net predictions on the test data 
	'''
	yhat = net.predict([data['X_jet_test'], data['X_photon_test'], data['X_event_test']], 
		verbose = True, batch_size = 512)
	io.save(open(os.path.join('output','yhat.h5'), 'wb'), yhat)
	# -- example of other quantities that can be evaluated from yhat
	#class_predictions = [np.argmax(ev) for ev in yhat])
	return yhat