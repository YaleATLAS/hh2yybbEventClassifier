import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import pandautils as pup
import os

def _plot_X(train, test, y_train, y_test, w_train, w_test, varlist, feature):
	'''
	Args:
		train: ndarray [n_ev_train, n_muon_feat] containing the events allocated for training
        test: ndarray [n_ev_test, n_muon_feat] containing the events allocated for testing
       	y_train: ndarray [n_ev_train, 1] containing the shuffled truth labels for training
        y_test: ndarray [n_ev_test, 1] containing the shuffled truth labels allocated for testing
        w_train: ndarray [n_ev_train, 1] containing the shuffled EventWeights allocated for training
        w_test: ndarray [n_ev_test, 1] containing the shuffled EventWeights allocated for testing
        varlist: list of names of branches like 'Jet_px', 'Photon_E', 'Muon_Iso'
		feature: a string like 'Jet', 'Muon', 'Photon'
	Returns:
		Saves .pdf histograms for each feature-related branch plotting the training and test sets for each class
	'''
	# -- extend w and y arrays to match the total number of particles per event
	w_train_ext = np.array(pup.flatten([[w] * (len(train[i, 0])) for i, w in enumerate(w_train)]))
	w_test_ext = np.array(pup.flatten([[w] * (len(test[i, 0])) for i, w in enumerate(w_test)]))
	y_train_ext = np.array(pup.flatten([[y] * (len(train[i, 0])) for i, y in enumerate(y_train)]))
	y_test_ext = np.array(pup.flatten([[y] * (len(test[i, 0])) for i, y in enumerate(y_test)]))
	
	# -- we keep a column counter because `varlist` contains all variables for all particles,
	# but each X matrix only contains as many columns as the number of variables 
	# related to that specific paritcle type
	column_counter = 0
	# -- loop through the variables
	for key in varlist:
		if key.startswith(feature):
			flat_train = pup.flatten(train[:, column_counter])
			flat_test = pup.flatten(test[:, column_counter])
			matplotlib.rcParams.update({'font.size': 16})
			fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
			bins = np.linspace(
				min(min(flat_train), min(flat_test)), 
				max(max(flat_train), max(flat_test)), 
				30)
			color = iter(cm.rainbow(np.linspace(0, 1, 2)))
			# -- loop through the classes
			for k in range(len(np.unique(y_train))):
				c = next(color)
				_ = plt.hist(flat_train[y_train_ext == k], 
					bins=bins, 
					histtype='step', 
					normed=True, 
					label='Train - class: '+str(k),
					weights=w_train_ext[y_train_ext == k],
					color=c, 
					linewidth=1)
				_ = plt.hist(flat_test[y_test_ext == k], 
					bins=bins, 
					histtype='step', 
					normed=True,
					label='Test  - class: ' + str(k),
					weights=w_test_ext[y_test_ext == k], 
					color=c,
					linewidth=2, 
					linestyle='dashed')	
			plt.xlabel(key)
			plt.yscale('log')
			plt.ylabel('Weighted Events')
			plt.legend()
			try:
				plt.savefig(os.path.join('plots', key + '.pdf'))
			except IOError:
				os.makedirs('plots')
				plt.savefig(os.path.join('plots', key + '.pdf'))
			#plt.show()
			column_counter += 1

def plot_inputs(X_jets_train, X_jets_test, X_photons_train, X_photons_test, 
	X_muons_train, X_muons_test, y_train, y_test, w_train, w_test, varlist):
	'''
	Args:
		X_jets_train: ndarray [n_ev_train, n_jet_feat] containing the 
				events of jet related branches allocated for training
        X_jets_test: ndarray [n_ev_test, n_jet_feat] containing the 
        		events of jet related branches allocated for testing
        X_photons_train: ndarray [n_ev_train, n_photon_feat] containing 
        		the events of photon related branches allocated for training
        X_photons_test: ndarray [n_ev_test, n_photon_feat] containing 
        		the events of photon related branches allocated for testing
        X_muons_train: ndarray [n_ev_train, n_muon_feat] containing the 
        		events of muon related branches allocated for training
        X_muons_test: ndarray [n_ev_test, n_muon_feat] containing the 
        		events of muon related branches allocated for testing
        Y_train: ndarray [n_ev_train, 1] containing the shuffled truth 
        		labels for training
        Y_test: ndarray [n_ev_test, 1] containing the shuffled truth labels 
        		allocated for testing
        W_train: ndarray [n_ev_train, 1] containing the shuffled EventWeights 
        		allocated for training
        W_test: ndarray [n_ev_test, 1] containing the shuffled EventWeights 
        		allocated for testing
        varlist: list of strings that concatenates the individual 
                lists of variables for each particle type, e.g.:
                ['Jet_Px', 'Jet_E', 'Muon_ID', 'Photon_Px']
	Returns:
		Saves .pdf histograms plotting the training and test 
		sets of each class for each feature 
	'''
	
	_plot_X(X_jets_train, X_jets_test, y_train, y_test, w_train, w_test, varlist, 'Jet')
	_plot_X(X_photons_train, X_photons_test, y_train, y_test, w_train, w_test, varlist, 'Photon')
	_plot_X(X_muons_train, X_muons_test, y_train, y_test, w_train, w_test, varlist, 'Muon')
