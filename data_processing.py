import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
import pandautils as pup
import warnings
import logging
from collections import OrderedDict
from itertools import izip

logger = logging.getLogger('data_processing')

def _pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return izip(a, a)

def _build_X(events, particle_branches):
    '''slices related branches into a numpy array
    Args:
        events: a pandas DataFrame containing the complete data by event
        phrase: a string like 'Jet' corresponding to the related branches wanted
    Returns:
        output_array: a numpy array containing data only pertaining to the related branches
    '''
    sliced_events = events[particle_branches].values
    return sliced_events


def read_in(class_files_dict, tree_name, streams):
    '''
    takes in dict mapping class names to list of root files, loads them and slices them into ML format
    Args:
        class_files_dict: dictionary that links the names of the different classes
                          in the classification problem to the paths of the ROOT files
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
                                "/path/to/file4*",
                            ],
                            ...
                          } 
        tree_name: string, name of the tree to open in the ntuples
        
    Returns:
        X_jets: ndarray [n_ev, n_jet_feat] containing jet related branches
        X_photons: ndarray [n_ev, n_photon_feat] containing photon related branches
        X_muons: ndarray [n_ev, n_muon_feat] containing muon related branches
        y: ndarray [n_ev, 1] containing the truth labels
        w: ndarray [n_ev, 1] containing EventWeights
        jet_branches + photon_branches + muon_branches = list of strings that concatenates the individual 
                lists of variables for each particle type, e.g.:
                ['Jet_Px', 'Jet_E', 'Muon_ID', 'Photon_Px']
    '''
    
    #convert files to pd data frames, assign key to y, concat all files
    def _make_df(val, key):
        df = pup.root2panda(val, tree_name)
        df['y'] = key
        return df

    all_events = pd.concat([_make_df(val, key) for key, val in class_files_dict.iteritems()], ignore_index=True)
    
    X = OrderedDict()
    for stream_name, stream_info in streams.iteritems():
        logger.info('building X_{}'.format(stream_name))
        X[stream_name] = _build_X(all_events, stream_info["branches"])

    #transform string labels to integer classes
    le = LabelEncoder()
    y = le.fit_transform(all_events['y'].values)
    
    #w = all_events['eventWeight'].values
    w = all_events['yybb_weight'].values
    
    return X, y, w
    # return X_jets, X_photons, X_muons, y, w, jet_branches + photon_branches + muon_branches


def _scale(matrix_train, matrix_test):
    '''
    Use scikit learn to sclae features to 0 mean, 1 std. 
    Because of event-level structure, we need to flatten X, scale, and then reshape back into event format.
    Args:
        matrix_train: X_train [n_ev_train, n_particle_features], numpy ndarray of unscaled features of events allocated for training
        matrix_test: X_test [n_ev_test, n_particle_features], numpy ndarray of unscaled features of events allocated for testing
    Returns:
        the same matrices after scaling
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ref_test = matrix_test[:, 0]
        ref_train = matrix_train[:, 0]
        for col in xrange(matrix_train.shape[1]):
            scaler = StandardScaler()
            matrix_train[:, col] = pup.match_shape(
                scaler.fit_transform(pup.flatten(matrix_train[:, col]).reshape(-1, 1)).ravel(), ref_train)
            matrix_test[:, col] = pup.match_shape(
                scaler.transform(pup.flatten(matrix_test[:, col]).reshape(-1, 1)).ravel(), ref_test)

    return matrix_train, matrix_test


#def shuffle_split_scale(X_jets, X_photons, X_muons, y, w):
def shuffle_split_scale(X, y, w):
    '''
    takes in X_jets, X_photons, X_Muons, y and w nd arrays, shuffles them, splits them into test (40%) and training (60%) sets
    Args:
        X_jets: ndarray [n_ev, n_jet_feat] containing jet related branches
        X_photons: ndarray [n_ev, n_photon_feat] containing photon related branches
        X_muons: ndarray [n_ev, n_muon_feat] containing muon related branches
        y: ndarray [n_ev, 1] containing the truth labels
        w: ndarray [n_ev, 1] containing EventWeights
    Returns:

    '''
    #shuffle events & split into testing and training sets
    logger.info('shuffling, splitting and scaling X')

    data_tuple = train_test_split(*(X.values() + [y, w]), test_size=0.4)

    data = OrderedDict()
    for particle, (train, test) in zip(X.keys(), _pairwise(data_tuple[:(2 * len(X))])):
        data['X_' + particle + '_train'], data['X_' + particle+ '_test'] = _scale(train, test)

    data['y_train'], data['y_test'], data['w_train'], data['w_test'] = data_tuple[-4:]

    return data


def padding(X, max_length, value=-999):
    '''
    Transforms X to a 3D array where the dimensions correspond to [n_ev, n_particles, n_features].
    n_particles is now fixed and equal to max_length.
    If the number of particles in an event was < max_length, the missing particles will be filled with default values
    If the number of particles in an event was > max_length, the excess particles will be removed
    Args:
        X: ndarray [n_ev, n_features] with an arbitrary number of particles per event
        max_length: int, the number of particles to keep per event 
        value (optional): the value to input in case there are not enough particles in the event, default=-999
    Returns:
        data: ndarray [n_ev, n_particles, n_features], padded version of X with fixed number of particles
    Note: 
        Use Masking to avoid the particles with artificial entries = -999
    '''
    data = value * np.ones((X.shape[0], max_length, X.shape[1]), dtype='float32')
    for i, row in enumerate(X):
        data[i, :min(len(row[0]), max_length), :] = np.array(row.tolist()).T[:min(len(row[0]), max_length), :]

    return data
