import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array
import pandas as pd
import glob
import argparse
from sklearn.preprocessing import LabelEncoder

def root2pandas(file_paths, tree_name, **kwargs):
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


def build_X(events, phrase):
    '''slices related branches into a numpy array
    Args:
        events: a pandas DataFrame containing the complete data by event
        phrase: a string like 'Jet' corresponding to the related branches wanted
    Returns:
        output_array: a numpy array containing data only pertaining to the related branches
    '''
    branch_names = [key for key in events.keys() if key.startswith(phrase)]
    sliced_events = events[varlist].as_matrix()
    return sliced_events, branch_names

def read_in(class_files_dict):
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
        df = root2pandas(class_files_dict[key], 'events')
        df['y'] = key
        if all_events is False:
            all_events = df
        else:
            all_events = pd.concat([all_events, df], ignore_index=True)
        
    #slice related branches
    X_jets, jet_branches = build_X(all_events, 'Jet')
    X_photons, photon_branches = build_X(all_events, 'Photon')
    X_muons, muon_branches = build_X(all_events, 'Muon')
    
    #transform string labels to integer classes
    le = LabelEncoder()
    y = le.fit_transform(all_events['y'].values)
    
    w = all_events['EventWeight'].values
    
    return X_jets, X_photons, X_muons, y, w, jet_branches + photon_branches + muon_branches
