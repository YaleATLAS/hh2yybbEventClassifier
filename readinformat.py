'''Given files with data
   Concatenates all events
   Creates (number of events)x(number of related branches) np array
   Under the names of:
   jet_branches
   muon_branches
   photon_branches

'''


import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array
import pandas as pd
import glob

#convert root files to pandas
def root2pandas(files_path, tree_name, **kwargs):
	files = glob.glob(files_path)
	ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])
	try:
        	return pd.DataFrame(ss)
    	except Exception:
        	return pd.DataFrame(ss.data)

ttbar = root2pandas('/home/ubuntu/jenny/ttbar.root', 'events')
qcd = root2pandas('/home/ubuntu/jenny/qcd.root', 'events')
wjets = root2pandas('/home/ubuntu/jenny/wjets.root', 'events')


#create column indicating background(0)/signal(1)
ttbar['y'] = 1
qcd['y'] = 0
wjets['y'] = 0


#join the arrays together
all_events = pd.concat([ttbar, qcd, wjets], ignore_index=True)


#slice related branches
def xjets(events):
	return events[[key for key in events.keys() if 'Jet' in key or key=='y']]
def xmuons(events):
	return events[[key for key in events.keys() if 'Muon' in key or key=='y']]
def xphotons(events):
        return events[[key for key in events.keys() if 'Photon' in key or key=='y']]

all_events_jets = xjets(all_events)
all_events_muons = xmuons(all_events)
all_events_photons = xphotons(all_events)


#convert to np arrays
jet_branches = all_events_jets.as_matrix()
muon_branches = all_events_muons.as_matrix()
photon_branches = all_events_jets.as_matrix()
