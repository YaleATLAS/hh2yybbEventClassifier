import json
from data_processing import read_in, shuffle_split_scale, padding
import pandautils as pup
import cPickle
from plotting import plot_inputs
import utils
import logging
#from plotting import plot_inputs, plot_performance
#from nn_model import train, test

def main(json_config, tree_name):
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
         tree_name:    
    Saves:
    ------
        'processed_data.h5': dictionary with processed ndarrays (X, y, w) for all particles for training and testing
    '''
    logger = logging.getLogger('Main')

    # -- load in the JSON file
    logger.info('Loading JSON config')
    config = utils.load_config(json_config)
    class_files_dict = config['classes']
    particles = config['particles']

    # -- hash the config dictionary to check if the pickled data exists
    from hashlib import md5
    def sha(s):
        '''Get a unique identifier for an object'''
        m = md5()
        m.update(s.__repr__())
        return m.hexdigest()[:5]

    #-- if the pickle exists, use it
    try:
        data = cPickle.load(open('processed_data_' + sha(class_files_dict) + '.pkl', 'rb'))
        logger.info('Preprocessed data found in pickle')

    # -- otherwise, process the new data 
    except IOError:
        logger.info('Preprocessed data not found')
        logger.info('Processing data')
        # -- transform ROOT files into standard ML format (ndarrays) 
        X, y, w = read_in(class_files_dict, tree_name, particles)

        # -- shuffle, split samples into train and test set, scale features
        data = shuffle_split_scale(X, y, w) #X_muons, y, w)
  
        data.update({
            'varlist' : [
                branch 
                for particle_info in particles.values() 
                for branch in particle_info['branches']
            ]
        })
        # -- save out to pickle
        logger.info('Saving processed data to pickle')
        cPickle.dump(data, 
            open('processed_data_' + sha(class_files_dict) + '.pkl', 'wb'),
            protocol=cPickle.HIGHEST_PROTOCOL)

    # -- plot distributions:
    '''
    This should produce normed, weighted histograms of the input distributions for all variables
    The train and test distributions should be shown for every class
    Plots should be saved out a pdf with informative names
    '''
    logger.info('Plotting input distributions')
    plot_inputs(data, particles.keys())

    logger.info('Padding')
    for key in data:
        if key.startswith('X_'):
            data[key] = padding(data[key], particles[key.split('_')[1]]['max_length'])

    # # -- train
    # # design a Keras NN with three RNN streams (jets, photons, muons)
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
    parser.add_argument('--tree', help="name of the tree to open in the ntuples", default='mini')
    args = parser.parse_args()

    # -- pass arguments to main
    sys.exit(main(args.config, args.tree))
