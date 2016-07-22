import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import pandautils as pup
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from viz import ROC_plotter, add_curve, calculate_roc
import cPickle

def _plot_X(train, test, y_train, y_test, w_train, w_test, le, particle, particles_dict):
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
		particle: a string like 'jet', 'muon', 'photon', ...
		particles_dict:
	Returns:
		Saves .pdf histograms for each feature-related branch plotting the training and test sets for each class
	'''
	# -- extend w and y arrays to match the total number of particles per event
	try:
		w_train = np.array(pup.flatten([[w] * (len(train[i, 0])) for i, w in enumerate(w_train)]))
		w_test = np.array(pup.flatten([[w] * (len(test[i, 0])) for i, w in enumerate(w_test)]))
		y_train = np.array(pup.flatten([[y] * (len(train[i, 0])) for i, y in enumerate(y_train)]))
		y_test = np.array(pup.flatten([[y] * (len(test[i, 0])) for i, y in enumerate(y_test)]))
	except TypeError: # `event` has a different structure that does not require all this
		pass
	
	varlist = particles_dict[particle]['branches']
	
	# -- loop through the variables
	for column_counter, key in enumerate(varlist):
		
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
		for k in np.unique(y_train):
			c = next(color)

			# -- in regression, le is None and we want to keep the original key
			try:
				transformed_k=le.inverse_transform(k)
			except AttributeError:
				transformed_k=k

			_ = plt.hist(flat_train[y_train == k], 
				bins=bins, 
				histtype='step', 
				normed=True, 
				label='Train - ' + str(transformed_k),
				weights=w_train[y_train == k],
				color=c, 
				linewidth=1)
			_ = plt.hist(flat_test[y_test == k], 
				bins=bins, 
				histtype='step', 
				normed=True,
				label='Test  - ' + str(transformed_k),
				weights=w_test[y_test == k], 
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

def plot_inputs(data, particles_dict):
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
        #particle_names: list of strings, names of particle streams
        particles_dict:
	Returns:
		Saves .pdf histograms plotting the training and test 
		sets of each class for each feature 
	'''
	
	for particle in particles_dict.keys():
		_plot_X(
			data['X_' + particle + '_train'], 
			data['X_' + particle + '_test'], 
			data['y_train'],
			data['y_test'], 
			data['w_train'], 
			data['w_test'], 
			data['LabelEncoder'],
			particle,
			particles_dict
			)

def plot_confusion(yhat, data):
	'''
	Args:
		yhat: numpy array of dim [n_ev, n_classes] with the net predictions on the test data 
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
	Returns:
		Saves confusion.pdf confusion matrix
	'''
	
	y_test = data['y_test']
	le = data['LabelEncoder']

	def _plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title(title)
	    plt.colorbar()
	    tick_marks = np.arange(len(np.unique(y_test)))
	    plt.xticks(tick_marks, sorted(np.unique(y_test)))
	    plt.yticks(tick_marks, sorted(np.unique(y_test)))
	    plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')

	cm = confusion_matrix(y_test, np.argmax(yhat, axis=1))
	# Normalize the confusion matrix by row (i.e by the number of samples
	# in each class)
	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	_plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
	plt.savefig('confusion.pdf')


def plot_regression(yhat, data):
	'''
	Args:
		yhat: numpy array of dim [n_ev, n_classes] with the net predictions on the test data 
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
	Saves:
		'regression_test.pdf': a histogram plotting yhat containing the predicted masses
	'''
	
	y_test = data['y_test'].values
	w_test = data['w_test']

	color = iter(cm.rainbow(np.linspace(0, 1, len(np.unique(y_test)))))
	matplotlib.rcParams.update({'font.size': 16})
	fig = plt.figure(figsize=(11.69, 8.27), dpi=100)

	bins = np.linspace(
		min(min(yhat), min(y_test)), 
		max(max(yhat), max(y_test)), 
		30)

	for k in np.unique(y_test):
		c = next(color)
		_ = plt.hist(yhat[y_test == k], 
			bins=bins, 
			histtype='step', 
			normed=True, 
			label=str(k),
			weights=w_test[y_test == k],
			color=c, 
			linewidth=1)

	plt.ylabel('Weighted Events')
	plt.legend(prop={'size': 10}, fancybox=True, framealpha=0.5)
	plt.savefig('regression.pdf')


def save_roc_curves(yhat, data, model_name):
	'''
	Args:
	    yhat: an ndarray of the probability of each event for each class
	    data: dictionary containing X, y, w ndarrays
	    model_name:
	Returns:
		plot: 
		pickle file: pkl file dictionary with each curve
	'''


	cutflow_eff = [0.0699191919192, 0.0754639175258, 0.08439, 0.0921212121212, 0.110275510204, 0.00484432269559]

	y_test = data['y_test']
	w_test = data['w_test']
	le = data['LabelEncoder']
	bkg_col = np.argwhere(le.classes_ == 'bkg')[0][0]

	pkl_dict = {}
	for k in np.unique(y_test)[np.unique(y_test) != bkg_col]:
		k_string = le.inverse_transform(k)
		selection = (y_test == k) | (y_test == bkg_col)
		finite = np.isfinite(np.log(yhat[selection][:, k] / yhat[selection][:, bkg_col]))
		curves = {}
		add_curve('DNN', 'black', 
			calculate_roc(
				y_test[selection][finite] == k, 
				np.log(yhat[selection][finite][:, k] / yhat[selection][finite][:, bkg_col]), 
				weights=w_test[selection][finite]
				),
			curves
			)
		pkl_dict.update(curves)
		fig = ROC_plotter(curves, 
			title=k_string + r' vs. Sherpa $\gamma \gamma$ Background', 
			min_eff=0.05, max_eff=1.0, ymax=500, 
			logscale=False)
		plt.scatter(cutflow_eff[k], 1. / cutflow_eff[bkg_col], label='Cutflow -' + k_string)
		plt.legend()
		matplotlib.rcParams.update({'font.size': 16})
		fig.savefig('roc_' + k_string + '_' + model_name +'.pdf')
	cPickle.dump(pkl_dict, open(model_name + '.pkl', 'wb'))



