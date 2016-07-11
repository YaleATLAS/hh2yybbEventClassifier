from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Masking, GRU, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time

def NN_train(data):
    '''
    Args:
        data: dictionary containing relevant data
     Returns:
        recurrent neural network: A combined recurrent neural network trained on the different classes of the data
    '''
    
    #defines training sets of different classes
    X_jets_train=data['X_jet_train']
    X_photons_train=data['X_photon_train']
    y_train=data['y_train']

    #set up sequential neural networks for the jet and photon classes 
    jet_channel = Sequential()
    photon_channel = Sequential()

    #declaring the shape of the first row of each class matrix
    JET_SHAPE = X_jets_train.shape[1:]
    PHOTON_SHAPE = X_photons_train.shape[1:]

    #adding layers to the jet and photon class neural networks
    jet_channel.add(Masking(mask_value=-999, input_shape=JET_SHAPE, name='jet_masking'))
    jet_channel.add(GRU(25, name='jet_gru'))
    jet_channel.add(Dropout(0.3, name='jet_dropout'))

    photon_channel.add(Masking(mask_value=-999, input_shape=PHOTON_SHAPE, name='photon_masking'))
    photon_channel.add(GRU(10, name='photon_gru'))
    photon_channel.add(Dropout(0.3, name='photon_dropout'))

    #combining the jet and photon classes to make a combined recurrent neural network
    combined_rnn = Sequential()
    combined_rnn.add(Merge([jet_channel, photon_channel], mode='concat'))
    combined_rnn.add(Dense(36, activation='relu'))
    combined_rnn.add(Dropout(0.3))
    combined_rnn.add(Dense(24, activation='relu'))
    combined_rnn.add(Dropout(0.3))
    combined_rnn.add(Dense(12, activation='relu'))
    combined_rnn.add(Dropout(0.3))
    combined_rnn.add(Dense(6, activation='softmax'))

    combined_rnn.compile('adam', 'sparse_categorical_crossentropy')

    print 'Training:'
    try:
        combined_rnn.fit([X_jets_train, X_photons_train], 
            y_train, batch_size=16, class_weight={
                k : (float(len(y_train)) / float(len(np.unique(y_train)) * (len(y_train[y_train == k])))) for k in np.unique(y_train)
            },
            callbacks = [
                EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
                ModelCheckpoint('./models/combinedrnn-progress',
                monitor='val_loss', verbose=True, save_best_only=True)
            ],
            nb_epoch=1, validation_split = 0.2) 

    except KeyboardInterrupt:
        print 'Training ended early.'

    #saving the combined recurrent neural network
    setType=raw_input("What set is this?")
    combined_rnn.save_weights('TestModel'+setType+'.H5')
    combined_rnn_json=combined_rnn.to_json()
    open('TestModel.json','w').write(combined_rnn_json)

    return combined_rnn


def NN_test(net, data):
    '''
    Args:
        net: a combined recurrent neural network
        data: dictionary containing relevant data
     Returns:
        yhat: an ndarray of the probability of each event for each class
    '''
    X_jets_test=data['X_jet_test']
    X_photons_test=data['X_photon_test']
    y_test=data['y_test']
    w_test=data['w_test']

    print y_test.shape
    print w_test.shape

    yhat_rnn = net.predict([X_jets_test, X_photons_test], verbose = True, batch_size = 512) 
    
    return yhat_rnn