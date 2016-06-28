import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array
import pandas as pd
import glob
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
import pandautils as pup

# def _root2pandas(file_paths, tree_name, **kwargs):
#     '''converts files from .root to pandas DataFrame
#     Args:
#         file_paths: a string like './data/*.root', or
#                     a list of strings with multiple files to open
#         tree_name: a string like 'Collection_Tree' corresponding to the name of the folder inside the root file that we want to open
#         kwargs: arguments taken by root2array, such as branches to consider, start, stop, step, etc
#     Returns:
#         output_panda: a pandas dataframe like allbkg_df in which all the info from the root file will be stored
#     Note:
#         if you are working with .root files that contain different branches, you might have to mask your data
#         in that case, return pd.DataFrame(ss.data)
#     '''
    
#     if isinstance(file_paths, basestring):
#         files = glob.glob(file_paths)
#     else:
#         files = [matched_f for f in file_paths for matched_f in glob.glob(f)]

#     ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])
#     try:
#         return pd.DataFrame(ss)
#     except Exception:
#         return pd.DataFrame(ss.data)

def _build_X(events, phrase, exclude_vars):
    '''slices related branches into a numpy array
    Args:
        events: a pandas DataFrame containing the complete data by event
        phrase: a string like 'Jet' corresponding to the related branches wanted
    Returns:
        output_array: a numpy array containing data only pertaining to the related branches
    '''
    branch_names = [key for key in events.keys() if (key.startswith(phrase) and (key not in exclude_vars))]
    sliced_events = events[branch_names].as_matrix()
    return sliced_events, branch_names

def read_in(class_files_dict, exclude_vars):
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
        exclude_vars: list of strings of names of branches not to be used for training
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
        df = _root2pandas(val, 'events')
        df['y'] = key
        return df

    all_events = pd.concat([_make_df(val, key) for key, val in class_files_dict.iteritems()], ignore_index=True)
        
    #slice related branches
    X_jets, jet_branches = _build_X(all_events, 'Jet', exclude_vars)
    X_photons, photon_branches = _build_X(all_events, 'Photon', exclude_vars)
    X_muons, muon_branches = _build_X(all_events, 'Muon', exclude_vars)
    
    #transform string labels to integer classes
    le = LabelEncoder()
    y = le.fit_transform(all_events['y'].values)
    
    w = all_events['EventWeight'].values
    
    return X_jets, X_photons, X_muons, y, w, jet_branches + photon_branches + muon_branches


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
    from sklearn.preprocessing import StandardScaler
    ref_test = matrix_test[:, 0]
    ref_train = matrix_train[:, 0]
    for col in xrange(matrix_train.shape[1]):
        scaler = StandardScaler()
        matrix_train[:, col] = pup.match_shape(
            scaler.fit_transform(pup.flatten(matrix_train[:, col]).reshape(-1, 1)).ravel(), ref_train)
        matrix_test[:, col] = pup.match_shape(
            scaler.transform(pup.flatten(matrix_test[:, col]).reshape(-1, 1)).ravel(), ref_test)

    return matrix_train, matrix_test


def shuffle_split_scale(X_jets, X_photons, X_muons, y, w):
    '''
    takes in X_jets, X_photons, X_Muons, y and w nd arrays, shuffles them, splits them into test (40%) and training (60%) sets, and scales X_jet, \
    X_photon and X_muon test sets based on training sets
    Args:
        X_jets: ndarray [n_ev, n_jet_feat] containing jet related branches
        X_photons: ndarray [n_ev, n_photon_feat] containing photon related branches
        X_muons: ndarray [n_ev, n_muon_feat] containing muon related branches
        y: ndarray [n_ev, 1] containing the truth labels
        w: ndarray [n_ev, 1] containing EventWeights
    Returns:
        X_jets_train: ndarray [n_ev_train, n_jet_feat] containing the scaled shuffled events of jet related branches allocated for training
        X_jets_test: ndarray [n_ev_test, n_jet_feat] containing the scaled shuffled events of jet related branches allocated for testing
        X_photons_train: ndarray [n_ev_train, n_photon_feat] containing the scaled shuffled events of photon related branches allocated for training
        X_photons_test: ndarray [n_ev_test, n_photon_feat] containing the scaled shuffled events of photon related branches allocated for testing
        X_muons_train: ndarray [n_ev_train, n_muon_feat] containing the scaled shuffled events of muon related branches allocated for training
        X_muons_test: ndarray [n_ev_test, n_muon_feat] containing the scaled shuffled events of muon related branches allocated for testing
        Y_train: ndarray [n_ev_train, 1] containing the shuffled truth labels for training
        Y_test: ndarray [n_ev_test, 1] containing the shuffled truth labels allocated for testing
        W_train: ndarray [n_ev_train, 1] containing the shuffled EventWeights allocated for training
        W_test: ndarray [n_ev_test, 1] containing the shuffled EventWeights allocated for testing
    '''
    
    #shuffle events & split into testing and training sets
    X_jets_train, X_jets_test, \
    X_photons_train, X_photons_test, \
    X_muons_train, X_muons_test, \
    Y_train, Y_test, \
    W_train, W_test = train_test_split(X_jets, X_photons, X_muons, y, w, test_size=0.4)

    X_jets_train, X_jets_test = _scale(X_jets_train, X_jets_test)
    X_photons_train, X_photons_test = _scale(X_photons_train, X_photons_test)
    X_muons_train, X_muons_test = _scale(X_muons_train, X_muons_test)

    return X_jets_train, X_jets_test, X_photons_train, X_photons_test, X_muons_train, X_muons_test, Y_train, Y_test, W_train, W_test
