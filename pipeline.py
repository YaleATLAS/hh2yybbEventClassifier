import json
from data_processing import read_in, shuffle_split_scale, padding
import numpy as np
import pandautils as pup
import cPickle
import utils
import logging
from plotting import plot_inputs, plot_confusion, plot_regression#, plot_performance
from nn_with_modes import train, test 

def main(json_config, mode, tree_name):
    '''
    Args:
    -----
        json_config: path to a JSON file, containing a dictionary that links the names of the different
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
         tree_name: string, name of the tree that contains the correct branches
    Saves:
    ------
        'processed_data_<hash>.pkl': dictionary with processed ndarrays (X, y, w) for all particles for training and testing
    '''
    logger = logging.getLogger('Main')

    # -- load in the JSON file
    logger.info('Loading information from ' + json_config)
    config = utils.load_config(json_config) # check config has expected structure
    class_files_dict = config['classes']
    particles_dict = config['particles']

    # -- hash the config dictionary to check if the pickled data exists
    from hashlib import md5
    def sha(s):
        '''Get a unique identifier for an object'''
        m = md5()
        m.update(s.__repr__())
        return m.hexdigest()[:5]

    #-- if the pickle exists, use it
    pickle_name = 'processed_data_' + sha(config) + '_' + sha(mode) + '.pkl'
    try:
        logger.info('Attempting to read from {}'.format(pickle_name))
        data = cPickle.load(open(pickle_name, 'rb'))
        logger.info('Pre-processed data found and loaded from pickle')
    # -- otherwise, process the new data 
    except IOError:
        logger.info('Pre-processed data not found in {}'.format(pickle_name))
        logger.info('Processing data')
        # -- transform ROOT files into standard ML format (ndarrays) 
        X, y, w, le = read_in(class_files_dict, tree_name, particles_dict, mode)

        # -- shuffle, split samples into train and test set, scale features
        data = shuffle_split_scale(X, y, w) 
  
        data.update({
            'varlist' : [
                branch 
                for particle_info in particles_dict.values() 
                for branch in particle_info['branches']
            ],
            'LabelEncoder' : le
        })

        # -- plot distributions:
        '''
        This should produce normed, weighted histograms of the input distributions for all variables
        The train and test distributions should be shown for every class
        Plots should be saved out a pdf with informative names
        '''
        logger.info('Saving input distributions in ./plots/')
        plot_inputs(data, particles_dict)

        logger.info('Padding')
        for key in data:
            if ((key.startswith('X_')) and ('event' not in key)): # no padding for `event` matrix
                data[key] = padding(data[key], particles_dict[key.split('_')[1]]['max_length']) 
                # ^ assuming naming convention: X_<particle>_train, X_<particle>_test 

        # -- save out to pickle
        logger.info('Saving processed data to {}'.format(pickle_name))
        cPickle.dump(data, 
            open(pickle_name, 'wb'),
            protocol=cPickle.HIGHEST_PROTOCOL)

    # # -- train
    # # design a Keras NN with three RNN streams (jets, photons, muons)
    # # combine the outputs and process them through a bunch of FF layers
    # # use a validation split of 20%
    # # save out the weights to hdf5 and the model to yaml
    net = train(data, mode)

    # # -- test
    # # evaluate performance on the test set
    yhat = test(net, data)
    
    # # -- plot performance by mode
    if mode == 'regression':
        plot_regression(yhat, data)
    if mode == 'classification':
        plot_confusion(yhat, data)

    # # -- plot performance
    # # produce ROC curves to evaluate performance
    # # save them out to pdf
    # plot_performance(yhat, data['y_test'], data['w_test'])

if __name__ == '__main__':
    
    import sys
    import argparse

    utils.configure_logging()

    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="path to JSON file that specifies classes and corresponding ROOT files' paths")
    parser.add_argument('mode', help="classification or regression")
    parser.add_argument('--tree', help="name of the tree to open in the ntuples", default='mini')
    args = parser.parse_args()

    if args.mode != 'classification' and args.mode != 'regression':
        raise ValueError('Mode must be classification or regression')

    # -- pass arguments to main
    sys.exit(main(args.config, args.mode, args.tree))