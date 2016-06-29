import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import pandautils as pup       

def _plot_X(train, test, y_train, y_test, w_train, w_test, varlist, feature):
	'''Args:
		train:
		test:
		feature: a string like 'Jet', 'Muon', 'Photon'
	'''

	w_train_ext = np.array(pup.flatten([[w]*(len(train[i,0])) for i, w in enumerate(w_train)]))
	w_test_ext = np.array(pup.flatten([[w]*(len(test[i,0])) for i, w in enumerate(w_test)]))
	y_train_ext = np.array(pup.flatten([[y]*(len(train[i,0])) for i, y in enumerate(y_train)]))
	y_test_ext = np.array(pup.flatten([[y]*(len(test[i,0])) for i, y in enumerate(y_test)]))

	column_counter=0
	for key in varlist:
		if key.startswith(feature):
			flat_train = pup.flatten(train[:,column_counter])
			flat_test = pup.flatten(test[:,column_counter])
			matplotlib.rcParams.update({'font.size': 16})
			fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
			bins = np.linspace(min(min(pup.flatten(train[:,column_counter])), 
				min(pup.flatten(test[:,column_counter]))), 
				max(max(pup.flatten(train[:,column_counter])), 
				max(pup.flatten(test[:,column_counter]))), 30)
			column_counter=column_counter+1
			color=iter(cm.rainbow(np.linspace(0,1,2)))
			for k in range(len(np.unique(y_train))):
				c=next(color)
				_ = plt.hist(flat_train[y_train_ext==k], bins=bins, histtype='step', normed=True, 
					label='Xtr class:'+str(k),weights=w_train_ext[y_train_ext==k],
					color = c, linewidth=1)
				_ = plt.hist(flat_test[y_test_ext==k], bins=bins, histtype='step', normed=True,
					label='Xte class:'+str(k),weights=w_test_ext[y_test_ext==k], 
					linewidth=2, color = c, linestyle='dashed')	
			plt.xlabel(key)
			plt.yscale('log')
			plt.ylabel('Weighted Events')
			plt.legend()
			plt.plot()
			plt.show()
			plt.savefig(key+'.pdf')

def plot_inputs(X_jets_train, X_jets_test, X_photons_train, X_photons_test, 
	X_muons_train, X_muons_test, y_train, y_test, w_train, w_test, varlist):
	
	_plot_X(X_jets_train, X_jets_test, y_train, y_test, w_train, w_test, varlist, 'Jet')
	_plot_X(X_photons_train, X_photons_test, y_train, y_test, w_train, w_test, varlist, 'Photon')
	_plot_X(X_muons_train, X_muons_test, y_train, y_test, w_train, w_test, varlist, 'Muon')
