import json
from data_processing import read_in, shuffle_split_scale
from plotting import plot_inputs, plot_performance
from nn_model import train, test

def main(json_config):
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

    '''
    # -- load in the JSON file
    class_files_dict = json.load(open(json_config))

    # -- transform ROOT files into standard ML format (ndarrays) 
    X_jets, X_photons, X_muons, y, w, varlist = read_in(class_files_dict)
    
    # -- shuffle, split samples into train and test set, scale features
    X_jets_train, X_jets_test, \
    X_photons_train, X_photons_test, \
    X_muons_train, X_muons_test, \
    y_train, y_test, \
    w_train, w_test = shuffle_split_scale(X_jets, X_photons, X_muons, y, w)

    # -- plot distributions:
    # this should produce weighted histograms of the input distributions for all variables
    # on each plot, the train and test distributions should be shown for every class
    # plots should be saved out a pdf with informative names
    plot_inputs(
        X_jets_train, X_jets_test, 
        X_photons_train, X_photons_test, 
        X_muons_train, X_muons_test, 
        y_train, y_test, 
        w_train, w_test,
        varlist 
        )

    # -- train
    # design a Keras NN with three RNN streams (jets, photons, muons)
    # combine the outputs and process them through a bunch of FF layers
    # use a validation split of 20%
    # save out the weights to hdf5 and the model to yaml
    net = train(X_jets_train, X_photons_train, X_muons_train, y_train, w_train)

    # -- test
    # evaluate performance on the test set
    yhat = test(net, X_jets_test, X_photons_test, X_muons_test, y_test, w_test)

    # -- plot performance
    # produce ROC curves to evaluate performance
    # save them out to pdf
    plot_performance(yhat, y_test, w_test)

if __name__ == '__main__':
    
    import sys
    import argparse

    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="JSON file that specifies classes and corresponding ROOT files' paths")
    args = parser.parse_args()

    # -- pass arguments to main
    sys.exit(main(args.config))