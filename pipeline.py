import json
from data_processing import read_in, shuffle_split_scale, zero_padding
import pandautils as pup
import cPickle
from plotting import plot_inputs, plot_NN
import utils
import logging
from neural_network import NN_train, NN_test
import deepdish.io as io
#from plotting import plot_inputs, plot_performance
#from nn_model import train, test

def main(json_config, exclude_vars):
    '''
    Args:
    -----
        json_config: a JSON file, containing a dictionary that links the names of the different
                     classes in the classification problem to the paths of the ROOT files
                     associated with each class; for example:

                     {
                        "ttbar" :
                        [
                            "/path/to/file1.root",
                            "/path/to/file2.root",
                        ],
                        "qcd" :
                        [
                            "/path/to/file3.root",
                            "/path/to/file4.root",
                        ],
                        ...
                     }
         exclude_vars: list of strings of names of branches not to be used for training   
    Saves:
    ------
        'processed_data.h5': dictionary with processed ndarrays (X, y, w) for all particles for training and testing
    '''
    logger = logging.getLogger('Main')

    # -- load in the JSON file
    logger.info('Loading JSON config')
    class_files_dict = json.load(open(json_config))

    # -- hash the config dictionary to check if the pickled data exists
    from hashlib import md5
    def sha(s):
        '''Get a unique identifier for an object'''
        m = md5()
        m.update(s.__repr__())
        return m.hexdigest()[:5]

    # -- if the pickle exists, use it
    try:
        data = cPickle.load(open('processed_data_' + sha(class_files_dict) + '.pkl', 'rb'))
        logger.info('Preprocessed data found in pickle')
        X_jets_train = data['X_jets_train']
        X_jets_test = data['X_jets_test']
        X_photons_train = data['X_photons_train']
        X_photons_test = data['X_photons_test']
        X_muons_train = data['X_muons_train']
        X_muons_test = data['X_muons_test']
        y_train = data['y_train']
        y_test = data['y_test']
        w_train = data['w_train']
        w_test = data['w_test']
        varlist = data['varlist']

    # -- otherwise, process the new data
    except IOError:
        logger.info('Preprocessed data not found')
        logger.info('Processing data')
        # -- transform ROOT files into standard ML format (ndarrays) 
        X_jets, X_photons, X_muons, y, w, varlist = read_in(class_files_dict, exclude_vars)
        # -- shuffle, split samples into train and test set, scale features
        X_jets_train, X_jets_test, \
        X_photons_train, X_photons_test, \
        X_muons_train, X_muons_test, \
        y_train, y_test, \
        w_train, w_test = shuffle_split_scale(X_jets, X_photons, X_muons, y, w)
        # -- save out to pickle
        logger.info('Saving processed data to pickle')
        cPickle.dump({
            'X_jets_train' : X_jets_train,
            'X_jets_test' : X_jets_test,
            'X_photons_train' : X_photons_train,
            'X_photons_test' : X_photons_test,
            'X_muons_train' : X_muons_train,
            'X_muons_test' : X_muons_test,
            'y_train' : y_train,
            'y_test' : y_test,
            'w_train' : w_train,
            'w_test' : w_test,
            'varlist' : varlist
            }, 
            open('processed_data_' + sha(class_files_dict) + '.pkl', 'wb'),
            protocol=cPickle.HIGHEST_PROTOCOL)

    # -- plot distributions:
    '''
    This should produce normed, weighted histograms of the input distributions for all variables
    The train and test distributions should be shown for every class
    Plots should be saved out a pdf with informative names
    
    logger.info('Plotting input distributions')
    plot_inputs(
        X_jets_train, X_jets_test, 
        X_photons_train, X_photons_test, 
        X_muons_train, X_muons_test, 
        y_train, y_test, 
        w_train, w_test,
        varlist 
        )
    '''
    X_jets_train, X_jets_test, \
    X_photons_train, X_photons_test, \
    X_muons_train, X_muons_test = map(zero_padding, 
        [
            X_jets_train, X_jets_test, 
            X_photons_train, X_photons_test, 
            X_muons_train, X_muons_test
        ],
        [5, 5, 3, 3, 2, 2]
    )

    # # -- train
    # # design a Keras NN with three RNN streams (jets, photons, muons)
    io.save(('X_jets_NN.h5'), NN(X_jets_train, X_jets_test, y_train))
    X_jets_NN_h5 = io.load('X_jets_NN.h5')
    io.save(('X_photons_NN.h5'), NN(X_photons_train, X_photons_test, y_train))
    X_photons_NN_h5 = io.load('X_photons_NN.h5')
    io.save(('X_muons_NN.h5'), NN(X_muons_train, X_muons_test, y_train))
    X_muons_NN_h5 = io.load('X_muons_NN.h5')
  
    plot_NN(NN_test(X_jets_test, NN_train(X_jets_train, y_train)), y_test, w_test)
    plot_NN(NN_test(X_photons_test, NN_train(X_photons_train, y_train)), y_test, w_test)
    plot_NN(NN_test(X_muons_test, NN_train(X_muons_train, y_train)), y_test, w_test)

    # # combine the outputs and process them through a bunch of FF layers
    # # use a validation split of 20%
    # # save out the weights to hdf5 and the model to yaml
    # net = train(X_jets_train, X_photons_train, X_muons_train, y_train, w_train)

    # # -- test
    # # evaluate performance on the test set
    # yhat = test(net, X_jets_test, X_photons_test, X_muons_test, y_test, w_test)

    # # -- plot performance
    # # produce ROC curves to evaluate performance
    # # save them out to pdf
    # plot_performance(yhat, y_test, w_test)

if __name__ == '__main__':
    
    import sys
    import argparse

    utils.configure_logging()

    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="JSON file that specifies classes and corresponding ROOT files' paths")
    parser.add_argument('--exclude', help="names of branches to exclude from training", nargs="*", default=[])
    args = parser.parse_args()

    # -- pass arguments to main
    sys.exit(main(args.config, args.exclude))
