import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array
import pandas as pd
import glob
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from compiler.ast import flatten
import pandautils as pup
import pandas as pd 

def _root2pandas(file_paths, tree_name, **kwargs):
    '''converts files from .root to pandas DataFrame
    Args:
        file_paths: a string like './data/*.root', or
                    a list of strings with multiple files to open
        tree_name: a string like 'Collection_Tree' corresponding to the name of the folder inside the root file that we want to open
        kwargs: arguments taken by root2array, such as branches to consider, start, stop, step, etc
    Returns:
        output_panda: a pandas dataframe like allbkg_df in which all the info from the root file will be stored
    Note:
        if you are working with .root files that contain different branches, you might have to mask your data
        in that case, return pd.DataFrame(ss.data)
    '''
    
    if isinstance(file_paths, basestring):
        files = glob.glob(file_paths)
    else:
        files = [matched_f for f in file_paths for matched_f in glob.glob(f)]

    ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])
    try:
        return pd.DataFrame(ss)
    except Exception:
        return pd.DataFrame(ss.data)


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
    all_events = False  
    for key in class_files_dict.keys():
        df = _root2pandas(class_files_dict[key], 'events')
        df['y'] = key
        if all_events is False:
            all_events = df
        else:
            all_events = pd.concat([all_events, df], ignore_index=True)
        
    #slice related branches
    X_jets, jet_branches = _build_X(all_events, 'Jet', exclude_vars)
    X_photons, photon_branches = _build_X(all_events, 'Photon', exclude_vars)
    X_muons, muon_branches = _build_X(all_events, 'Muon', exclude_vars)
    
    #transform string labels to integer classes
    le = LabelEncoder()
    y = le.fit_transform(all_events['y'].values)
    
    w = all_events['EventWeight'].values
    print jet_branches + photon_branches + muon_branches
    
    return X_jets, X_photons, X_muons, y, w, jet_branches + photon_branches + muon_branches

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
    
    #define a set with ideal shape for later match-shape
    X_jets_train_set=X_jets_train[:,0]
    X_jets_test_set=X_jets_test[:,0]
    X_photons_train_set=X_photons_train[:,0]
    X_photons_test_set=X_photons_test[:,0]
    X_muons_train_set=X_muons_train[:,0]
    X_muons_test_set=X_muons_test[:,0]


    #for i in range (X_jets_train.shape[1]):
       # a=pup.flatten(X_jets_train[:,i]) 
       # scaler = StandardScaler()
      #  b = pup.match_shape(scaler.fit_transform(a), X_jets_train_set)
        #c = scaler.fit_transform(a)
        #d = a.shape
        #print b
  

    #print "X_Jets_train {}".format(X_jets_train)

    #flattens X_jets, X_photons, and X_muons test and training arrays, scales the arrays and then regives them their original shape based on the previously declared test set
    for i in range (X_jets_train.shape[1]):
        a=pup.flatten(X_jets_train[:,i])
        c=pup.flatten(X_jets_test[:,i])
        scaler = StandardScaler()
        X_jets_train_final = pup.match_shape(scaler.fit_transform(a), X_jets_train_set)
        X_jets_test_final = pup.match_shape(scaler.transform(c), X_jets_test_set)

    for i in range (X_photons_train.shape[1]):
        a=pup.flatten(X_photons_train[:,i])
        c=pup.flatten(X_photons_test[:,i])
        scaler = StandardScaler()
        X_photons_train_final = pup.match_shape(scaler.fit_transform(a), X_photons_train_set)
        X_photons_test_final = pup.match_shape(scaler.transform(c), X_photons_test_set)

    for i in range (X_muons_train.shape[1]):
        a=pup.flatten(X_muons_train[:,i])
        c=pup.flatten(X_muons_test[:,i])
        scaler = StandardScaler()
        X_muons_train_final = pup.match_shape(scaler.fit_transform(a), X_muons_train_set)
        X_muons_test_final = pup.match_shape(scaler.transform(c), X_muons_test_set)


        #X_jets_train[:,i]=pup.match_shape(scaler.fit_transform(flatten(X_jets_train[:,i])), X_jets_train_set)
        #X_jets_test[:,i]=pup.match_shape(scaler.transform(flatten(X_jets_test[:,i])), X_jets_test_set)
       # y=X_jets_train
       # z=X_jets_train[:,i]
       # a = pup.flatten(X_jets_train[:,i])
        #b = scaler.fit_transform(a)
        #c = pup.match_shape(b, X_jets_train_set)
        #X_jets_train[:,i] = c
        #print "Z is {}.".format(z) 
   # print "y is {}",format(y)
   # print "z is {}".format(z)
   # print "a is {}".format(a) 
    
    """
    for i in range (X_photons_test.shape[1]):
        scaler = StandardScaler()
        X_photons_train[:,i]=pup.match_shape(scaler.fit_transform(flatten(X_photons_train[:,i])), X_photons_train_set)
        X_photons_test[:,i]=pup.match_shape(scaler.transform(flatten(X_photons_test[:,i])), X_photons_train_set)

    for i in range (X_muons_train.shape[1]):
        scaler = StandardScaler()
        X_muons_train[:,i]=pup.match_shape(scaler.fit_transform(flatten(X_muons_train[:,i])), X_muons_train_set)
        X_muons_test[:,i]=pup.match_shape(scaler.transform(flatten(X_muons_test[:,i])), X_muons_train_set)
  
    """


   # print X_jets_test[:,0]

    """X_jets_test_flat=[]
    for i in range (X_jets_test.shape[1]):
        X_jets_test_flat.append(flatten(X_jets_test[:,i]))

    X_photons_train_flat=[]
    for i in range (X_photons_train.shape[1]):
        X_photons_train_flat.append(flatten(X_jets_train[:,i]))

    X_photons_test_flat=[]
    for i in range (X_photons_test.shape[1]):
        X_photons_test_flat.append(flatten(X_photons_test[:,i]))

    X_muons_train_flat=[]
    for i in range (X_muons_train.shape[1]):
        X_muons_train_flat.append(flatten(X_muons_train[:,i]))

    X_muons_test_flat=[]
    for i in range (X_muons_test.shape[1]):
        X_muons_test_flat.append(flatten(X_muons_test[:,i]))
"""

    #X_jets_train=pd.DataFrame(X_jets_train)
    #X_jets_test=pd.DataFrame(X_jets_test)
    #X_photons_train=pd.DataFrame(X_photons_train)
    #X_photons_test=pd.DataFrame(X_photons_test)
    #X_muons_train=pd.DataFrame(X_muons_train)
    #X_muons_test=pd.DataFrame(X_muons_test)
    
    #flatten each array by first turning it into a dataframe
    #X_jets_train_flat={k: flatten(c) for k, c in X_jets_train.iterkv()}
    #X_jets_test_flat={k: flatten(c) for k, c in X_jets_test.iterkv()}
    #X_photons_train_flat={k: flatten(c) for k, c in X_photons_train.iterkv()}
    #X_photons_test_flat={k: flatten(c) for k, c in X_photons_test.iterkv()}
    #X_muons_train_flat={k: flatten(c) for k, c in X_muons_train.iterkv()}
    #X_muons_test_flat={k: flatten(c) for k, c in X_muons_test.iterkv()}


    #turn from dataframe back into array
   # X_jets_train_flat=np.array(X_jets_train)
   # X_jets_test_flat=np.array(X_jets_test)
   # X_photons_train_flat=np.array(X_photons_train)
   # X_photons_test_flat=nd.array(X_photons_test)
  #  X_muons_train_flat=nd.array(X_muons_train)
   # X_muons_test_flat=nd.array(X_muons_test)

    #scale each flattened array
    
    #X_jets_train = scaler.fit_transform(X_jets_train)
    #X_jets_test = scaler.transform(X_jets_test)
    #X_photons_train = scaler.fit_transform(X_photons_train)
    #X_photons_test = scaler.transform(X_photons_test)       
    #X_muons_train = scaler.fit_transform(X_muons_train)
    #X_muons_test = scaler.transform(X_muons_test)

    #get scaled array with original shape
    #X_jets_train_final=pup.match_shape(X_jets_train, X_jets_train_set)
    #X_jets_test_final=pup.match_shape(X_jets_test, X_jets_test_set)
    #X_photons_train_final=pup.match_shape(X_photons_train, X_photons_train_set)
    #X_photons_test_final=pup.match_shape(X_photons_test, X_photons_test_set)
    #X_muons_train_final=pup.match_shape(X_muons_train, X_muons_train_set)
    #X_muons_test_final=pup.match_shape(X_muons_test, X_muons_test_set)
   
    return X_jets_train_final, X_jets_test_final, X_photons_train_final, X_photons_test_final, X_muons_train_final, X_muons_test_final, Y_train, Y_test, W_train, W_test
    
    #return X_jets_train, X_jets_test, X_photons_train, X_photons_test, X_muons_train, X_muons_test, Y_train, Y_test, W_train, W_test
     
