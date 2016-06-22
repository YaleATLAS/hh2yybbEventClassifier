'''Given files with data (in this example: ttbar, qcd, and wjets)
   Concatenates all events
   Creates Xjets, Xmuons, Xphotons, w, and y   

'''


import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array
import pandas as pd
import glob
import argparse

def root2pandas(files_path, tree_name, **kwargs):
	files = glob.glob(files_path)
	ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])
	try:
        	return pd.DataFrame(ss)
    	except Exception:
        	return pd.DataFrame(ss.data)

def x(events, phrase):
        return events[[key for key in events.keys() if key.startswith(phrase)]]

def readin(ttbar_path, qcd_path, wjets_path):
	#convert root files to pandas
	ttbar = root2pandas(ttbar_path, 'events')
	qcd = root2pandas(qcd_path, 'events')
	wjets = root2pandas(wjets_path, 'events')
	
	
	#create column indicating background(0)/signal(1)
	ttbar['y'] = 1
	qcd['y'] = 0
	wjets['y'] = 0
	
	
	#join the arrays together
	all_events = pd.concat([ttbar, qcd, wjets], ignore_index=True)
	
	
	#slice related branches
	all_events_jets = x(all_events, 'Jet')
	all_events_muons = x(all_events, 'Muon')
	all_events_photons = x(all_events, 'Photon')
	all_events_weights = all_events['EventWeight']
	all_events_y = all_events['y']
	
	
	#convert to np arrays
	jet_branches = all_events_jets.as_matrix()
	muon_branches = all_events_muons.as_matrix()
	photon_branches = all_events_jets.as_matrix()
	w_branch = all_events_weights.values
	y_branch = all_events_y.values
	
	return jet_branches, muon_branches, photon_branches, w_branch, y_branch

#test
print readin('/home/ubuntu/jenny/ttbar.root', '/home/ubuntu/jenny/qcd.root','/home/ubuntu/jenny/ttbar.root')
