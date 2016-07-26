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
import tqdm

logger = logging.getLogger('data_processing')

def _pairwise(iterable):
    '''s -> (s0, s1), (s2, s3), (s4, s5), ...'''
    a = iter(iterable)
    return izip(a, a)


def read_in(class_files_dict, tree_name, particles, mode):
    '''
    takes in dict mapping class names to list of root files, loads them and slices them into ML format
    Args:
        class_files_dict: dictionary that links the names of the different classes
                          in the classification problem to the paths of the ROOT files
                          associated with each class; for example:

                          {
                            "ttbar" :
                            { 
                                "filenames":
                                [
                                    "/path/to/file1.root",
                                    "/path/to/file2.root"
                                ],
                                "lumiXsecWeight" :
                                [
                                    0.1239,
                                    1.2283
                                ]
                            }
                            ...
                          } 
        tree_name: string, name of the tree to open in the ntuples
        particles: dictionary that provides various informations about the different streams in the events,
                   for example:
                   {
                    "jet" :
                        {
                            "branches" :
                                [
                                    "jet_pt",
                                    "jet_eta"
                                ],
                            "max_length" : 5
                        },
                    "photon" :
                        {
                            "branches" :
                                [
                                    "photon_pt",
                                    "photon_eta"
                                ],
                            "max_length" : 3
                        }
                   }
    Returns:
        X: an OrderedDict containing the feature matrices for the different particle types, e.g.:
           X = {
                    "jet" : X_jet,
                    "photon" : X_photon,
                    "muon" : X_muon
           }
           where each X_<particle> is an ndarray of dimensions [n_ev, n_<particle>features]
        y: ndarray [n_ev, 1] containing the truth labels
        w: ndarray [n_ev, 1] containing the event weights
        le: LabelEncoder to transform numerical y back to its string values
    '''
    
    branches = []
    for particle_name, particle_info in particles.iteritems():
        branches += particle_info["branches"]

    #convert files to pd data frames, assign key or mass to y, concat all files
    def _make_df(fname, key, branches, fweight):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pup.root2panda(fname, tree_name, branches = branches + ['HGamEventInfoAuxDyn.yybb_weight'])
        df['lumiXsecWeight'] = fweight
        if mode == 'classification':
            df['y'] = key
        elif mode == 'regression':
            if key == 'bkg':
                df['y'] = 0
        else:
            df['y'] = int(key[1:])
        return df

    all_events = pd.concat([_make_df(fname, key, branches, fweight) 
        for key, val in class_files_dict.iteritems() 
            for fname, fweight in zip(val['filenames'], val['lumiXsecWeight'])], 
        ignore_index=True)

    X = OrderedDict()
    for particle_name, particle_info in particles.iteritems():
        logger.info('Building X_{}'.format(particle_name))
        X[particle_name] = all_events[particle_info["branches"]].values

    #transform string labels to integer classes for classification or set y for regression
    if mode == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(all_events['y'].values)
    elif mode == 'regression':
        le = None
        y = all_events['y'].values
    
    #w = all_events['HGamEventInfoAuxDyn.yybb_weight'].values
    #w = np.ones(len(y))
    w = all_events['lumiXsecWeight'].values
    
    return X, y, w, le


def _scale(matrix_train, matrix_test):
    '''
    Use scikit learn to scale features to 0 mean, 1 std. 
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


def shuffle_split_scale(X, y, w):
    '''
    Shuffle data, split it into test (40%) and training (60%) sets, scale X
    Args:
        X: an OrderedDict containing the feature matrices for the different particle types, e.g.:
           X = {
                    "jet" : X_jet,
                    "photon" : X_photon,
                    "muon" : X_muon
           }
           where each X_<particle> is an ndarray of dimensions [n_ev, n_<particle>features]
        y: ndarray [n_ev, 1] containing the truth labels
        w: ndarray [n_ev, 1] containing the event weights
    Returns:
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
    '''
    logger.info('Shuffling, splitting and scaling')

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
        X_pad: ndarray [n_ev, n_particles, n_features], padded version of X with fixed number of particles
    Note: 
        Use Masking to avoid the particles with artificial entries = -999
    '''
    X_pad = value * np.ones((X.shape[0], max_length, X.shape[1]), dtype='float32')
    for i, row in enumerate(X):
        X_pad[i, :min(len(row[0]), max_length), :] = np.array(row.tolist()).T[:min(len(row[0]), max_length), :]

    return X_pad
