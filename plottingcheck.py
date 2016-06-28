import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

def flatten(column):
    '''
    Args:
    -----
        column: a column of a pandas df whose entries are lists (or regular entries -- in which case nothing is done)
                e.g.: my_df['some_variable'] 

    Returns:
    --------    
        flattened out version of the column. 

        For example, it will turn:
        [1791, 2719, 1891]
        [1717, 1, 0, 171, 9181, 537, 12]
        [82, 11]
        ...
        into:
        1791, 2719, 1891, 1717, 1, 0, 171, 9181, 537, 12, 82, 11, ...
    '''
    try:
        return np.array([v for e in column for v in e])
    except (TypeError, ValueError):
        return column

def _plot_X(train, test, y_train, y_test, w_train, w_test, varlist, feature):
	'''
	Args:
		train:
		test:
		feature: a string like 'Jet', 'Muon', 'Photon'
	'''
	color=iter(cm.rainbow(np.linspace(0,1,len(varlist)*2)))

	for i, key in enumerate(varlist):
		if key.startswith(feature):
			matplotlib.rcParams.update({'font.size': 16})
			fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
			bins = np.linspace(min(min(flatten(train[:,i])), min(flatten(test[:,i]))), 
				max(max(flatten(train[:,i])), max(flatten(test[:,i]))), 30)
			print bins
			for k in range(len(np.unique(y_train))):
				c=next(color)
				_ = plt.hist(train[y_train==k,i], bins=bins, histtype='stepfilled', 
					label='Xtr', weights=w_train[y_train==k], linewidth=2, color = c, alpha=0.5)
				_ = plt.hist(test[y_test==k,i], bins=bins, histtype='step', 
					label='Xte', weights=w_test[y_test==k], linewidth=2, color = c, alpha=0.5)
			plt.xlabel(key)
			plt.ylabel('Weighted Events')
			plt.legend()
			plt.show()
			plt.savefig(key+'.pdf')

def plot_inputs(X_jets_train, X_jets_test, X_photons_train, X_photons_test, 
	X_muons_train, X_muons_test, y_train, y_test, w_train, w_test, varlist):
	_plot_X(X_jets_train, X_jets_test, y_train, y_test, w_train, w_test, varlist, 'Jet')
	_plot_X(X_photons_train, X_photons_test, y_train, y_test, w_train, w_test, varlist, 'Photon')
	_plot_X(X_muons_train, X_muons_test, y_train, y_test, w_train, w_test, varlist, 'Muon')