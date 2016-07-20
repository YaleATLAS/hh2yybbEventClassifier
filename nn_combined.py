from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Masking, GRU, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time

def NN_train(data, model_name):
    '''
    Args:
        data: dictionary containing relevant data
     Returns:
        recurrent neural network: A combined recurrent neural network trained on the different classes of the data
    '''
    
    #defines training sets of different classes
    X_jets_train = data['X_jet_train']
    X_photons_train = data['X_photon_train']
    X_event_train = data['X_event_train']
    y_train = data['y_train']
    X_muons_train=data['X_muon_train']
    X_electrons_train=data['X_electron_train']

    #set up sequential neural networks for the jet and photon classes 
    jet_channel = Sequential()
    photon_channel = Sequential()
    event_level = Sequential()
    muon_channel=Sequential()
    electron_channel=Sequential()

    #declaring the shape of the first row of each class matrix
    JET_SHAPE = X_jets_train.shape[1:]
    PHOTON_SHAPE = X_photons_train.shape[1:]
    EVENT_SHAPE = X_event_train.shape[1]
    MUON_SHAPE = X_muons_train.shape[1:]
    ELECTRON_SHAPE = X_electrons_train.shape[1:]

    #adding layers to the jet and photon class neural networks
    jet_channel.add(Masking(mask_value=-999, input_shape=JET_SHAPE, name='jet_masking'))
    jet_channel.add(GRU(25, name='jet_gru'))
    jet_channel.add(Dropout(0.3, name='jet_dropout'))

    photon_channel.add(Masking(mask_value=-999, input_shape=PHOTON_SHAPE, name='photon_masking'))
    photon_channel.add(GRU(10, name='photon_gru'))
    photon_channel.add(Dropout(0.3, name='photon_dropout'))

    event_level.add(Lambda(lambda x: x, input_shape=(EVENT_SHAPE, )))

    muon_channel.add(Masking(mask_value=-999, input_shape=MUON_SHAPE, name='muon_masking'))
    muon_channel.add(GRU(10, name='muon_gru'))
    muon_channel.add(Dropout(0.3, name='muon_dropout'))

    electron_channel.add(Masking(mask_value=-999, input_shape=ELECTRON_SHAPE, name='electron_masking'))
    electron_channel.add(GRU(10, name='electron_gru'))
    electron_channel.add(Dropout(0.3, name='electron_dropout'))


    #combining the jet and photon classes to make a combined recurrent neural network
    combined_rnn = Sequential()
    combined_rnn.add(Merge([jet_channel, photon_channel], mode='concat'))
    combined_rnn.add(Dense(72, activation='relu'))
    combined_rnn.add(Dropout(0.3))   
    combined_rnn.add(Dense(36, activation='relu'))
    combined_rnn.add(Dropout(0.3))
    combined_rnn.add(Dense(24, activation='relu'))
    combined_rnn.add(Dropout(0.3))
    combined_rnn.add(Dense(12, activation='relu'))
    combined_rnn.add(Dropout(0.3))
    combined_rnn.add(Dense(6, activation='softmax'))

    combined_rnn.compile('adam', 'sparse_categorical_crossentropy')

    logger = logging.getLogger('Train')
    logger.info('Compiling the net')
    try:
        combined_rnn.fit([X_jets_train, X_photons_train], 
            y_train, batch_size=100, class_weight={
                k : (float(len(y_train)) / float(len(np.unique(y_train)) * (len(y_train[y_train == k])))) for k in np.unique(y_train)
            },
            callbacks = [
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint('./models/combinedrnn-progress'+model_name,
                monitor='val_loss', verbose=True, save_best_only=True)
            ],
            nb_epoch=1, validation_split = 0.2) 

    except KeyboardInterrupt:
        print 'Training ended early.'

    #saving the combined recurrent neural network
    combined_rnn.save_weights('TestModel_'+model_name+'.H5')
    combined_rnn_json=combined_rnn.to_json()
    open('TestModel'+model_name+'.json','w').write(combined_rnn_json)

    return combined_rnn
