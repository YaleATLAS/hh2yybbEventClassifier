import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import pandautils as pup
import os
from sklearn.preprocessing import LabelEncoder

def _plot_X(train, test, y_train, y_test, w_train, w_test, varlist, le, feature):
	'''
	Args:
		train: ndarray [n_ev_train, n_muon_feat] containing the events allocated for training
        test: ndarray [n_ev_test, n_muon_feat] containing the events allocated for testing
       	y_train: ndarray [n_ev_train, 1] containing the shuffled truth labels for training in numerical format
        y_test: ndarray [n_ev_test, 1] containing the shuffled truth labels allocated for testing in numerical format
        w_train: ndarray [n_ev_train, 1] containing the shuffled EventWeights allocated for training
        w_test: ndarray [n_ev_test, 1] containing the shuffled EventWeights allocated for testing
        varlist: list of names of branches like 'jet_px', 'photon_E', 'muon_Iso'
        le: LabelEncoder to transform numerical y back to its string values
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
			color = iter(cm.rainbow(np.linspace(0, 1, len(np.unique(y_train)))))
			# -- loop through the classes
			for k in range(len(np.unique(y_train))):
				c = next(color)
				_ = plt.hist(flat_train[y_train_ext == k], 
					bins=bins, 
					histtype='step', 
					normed=True, 
					label='Train - ' + le.inverse_transform(k),
					weights=w_train_ext[y_train_ext == k],
					color=c, 
					linewidth=1)
				_ = plt.hist(flat_test[y_test_ext == k], 
					bins=bins, 
					histtype='step', 
					normed=True,
					label='Test  - ' + le.inverse_transform(k),
					weights=w_test_ext[y_test_ext == k], 
					color=c,
					linewidth=2, 
					linestyle='dashed')	
			plt.title(key)
			plt.yscale('log')
			plt.ylabel('Weighted Events')
			plt.legend(prop={'size': 10}, fancybox=True, framealpha=0.5)
			try:
				plt.savefig(os.path.join('plots', key + '.pdf'))
			except IOError:
				os.makedirs('plots')
				plt.savefig(os.path.join('plots', key + '.pdf'))
			column_counter += 1

def plot_inputs(data, particle_names):
	'''
	Args:
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
        particle_names: list of strings, names of particle streams
	Returns:
		Saves .pdf histograms plotting the training and test 
		sets of each class for each feature 
	'''
	
	for particle in particle_names:
		_plot_X(
			data['X_' + particle + '_train'], 
			data['X_' + particle + '_test'], 
			data['y_train'],
			data['y_test'], 
			data['w_train'], 
			data['w_test'], 
			data['varlist'],
			data['LabelEncoder'],
			particle
			)
