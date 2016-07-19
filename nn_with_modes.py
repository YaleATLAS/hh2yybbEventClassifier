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

def test(net, data):
	'''
	Args:
		net: a Sequential instance trained on data
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
		yhat_rnn: a numpy array containing the predicted values for each event
				In the case of regression:
				[[  28.82653809]
 				[ 332.62536621]
 				[ 343.72662354]
 				...,
 				[ 290.94213867]
 				[ 311.36965942]
				[ 325.11975098]]

 				In the case of classification:
				[[  2.98070186e-03   1.02684367e-03   6.20509265e-04   5.31344442e-04 
				   4.20760407e-05   9.94798541e-01]
 				[  1.43380761e-01   2.02934369e-01   2.18192190e-01   2.09208429e-01
    			   1.84640139e-01   4.16441038e-02]
 				[  1.91159040e-01   2.36048207e-01   2.16798335e-01   1.83185950e-01
    			   1.12408176e-01   6.04002886e-02]
				...,
 				[  8.16606451e-03   5.52139431e-02   1.69157043e-01   2.80651450e-01
    			   3.87061536e-01   9.97499675e-02]
 				[  3.25843632e-01   2.48317569e-01   1.64540142e-01   1.18563063e-01
    			   5.40928766e-02   8.86427015e-02]
 				[  3.07332397e-01   2.48623013e-01   1.71252742e-01   1.26610160e-01
    			   6.08449057e-02   8.53367895e-02]]
	'''
	X_jet_test = data['X_jet_test']
	X_photon_test = data['X_photon_test']
	y_test= data ['y_test']

	yhat_rnn = net.predict([X_jet_test, X_photon_test], verbose = True, batch_size = 512) 

	return yhat_rnn