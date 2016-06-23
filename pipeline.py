from data_processing import read_in, shuffle_split_scale

def main(class_files_dict):
    '''
    Args:
    -----
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
                                "/path/to/file4.root",
                            ],
                            ...
                          }

    '''
    # -- transform ROOT files into standard ML format (ndarrays) 
    X_jets, X_photons, X_muons, y, w = read_in(class_files_dict)
    
    # -- shuffle, split samples into train and test set, scale features
    X_jets_train, \
    X_jets_test, \
    X_photons_train, \
    X_photons_test, \
    X_muons_train, \
    X_muons_test, \
    y_train, \
    y_test, \
    w_train, \
    w_test = shuffle_split_scale(X_jets, X_photons, X_muons, y, w)

    # -- plot distributions
    # -- train
    # -- test
    # -- plot performance

if __name__ == '__main__':
    
    import sys
    import argparse

    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="JSON file that specifies classes and corresponding ROOT files' paths", required=True)
    args = parser.parse_args()

    # -- pass arguments to main
    sys.exit(main(args.config))