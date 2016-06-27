import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

def plot_X(train, test, y_train, y_test, w_train, w_test, varlist, feature):
	'''
	Args:
		train:
		test:
		feature: a string like 'Jet', 'Muon', 'Photon'
	'''
	color=iter(cm.rainbow(np.linspace(0,1,n)))

	for key in varlist:
		if key.startswith(feature):
			matplotlib.rcParams.update({'font.size': 16})
			fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
			bins = np.linspace(min(min(train[key]),min(test[key])), 
				max(max(train[key]),max(test[key])), 30)
			for i in range(len(np.unique(y_train))):
				c=next(color)
				_ = plt.hist([train[key][y_train==i], test[key][y_test==i]], bins=bins, histtype=['stepfilled', 'step'], 
					label=['Xtr', 'Xte'], weights=[w_train[y_train==i], w_test[y_test==i]], linewidth=2, color = [c], alpha=0.5)
			plt.xlabel(key)
			plt.ylabel('Weighted Events')
			plt.legend()
			plt.show()
			plt.savefig(key+'.pdf')

def plot_inputs(X_jets_train, X_jets_test, X_photons_train, X_photons_test, 
	X_muons_train, X_muons_test, y_train, y_test, w_train, w_test, varlist):
	plot_X(X_jets_train, X_jets_test, y_train, y_test, w_train, w_test, varlist, 'Jet')
	plot_X(X_photons_train, X_photons_test, y_train, y_test, w_train, w_test, varlist, 'Photon')
	plot_X(X_muons_train, X_muons_test, y_train, y_test, w_train, w_test, varlist, 'Muon')