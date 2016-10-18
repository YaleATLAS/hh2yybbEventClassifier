# hh2yybb Event Classifier: RNN-based Event Level Classifier for ATLAS Analysis (hh-->yybb) 
Event level classifier for the hh-->yybb analysis using multi-stream RNNs

## Introduction
### Purpose
'Cut-based' methods are the prevalent data analysis strategy in ATLAS, which is also currently being used in the hh-->yybb analysis group. To improve on this, a ML algorithm is being trained to select the correct “second b jet” in single-tagged events ([bbyy jet classifier](https://github.com/jemrobinson/bbyy_jet_classifier)). The project in this repo will take this a step further and develop an event classifier. 
 
### Current Status
![Pipeline](images/UGsWorkflow.png)

### Deep Learning Module
![Net](images/MultiStreamRNN.png)
Our goal is to train an event-level classifier, in order to be able to distinguish di-Higgs resonant decays from photons+jets backgrounds. To describe an event, we combine the pre-calculated event-level information with a series of RNN streams that extract information about the event from the properties of the different types of particles in it (jets, electrons, photons, muons, ...).

We phrase our project both as a classification and a regression. In the classification task, we consider every mass hypothesis for the di-Higgs parent particle as a separate class, and we attempt to solve a multi-class classification problem. In the regression task, we use the nominal mass of the parent particle as the continuous variable to predict.

## Software Structure
### Dependencies
The necessary packages are listed in the `requirements.txt` file. They can be installed as follows.
* With God's help:
   * ROOT ([instructions](https://root.cern.ch/downloading-root))
* With `pip`:
   * `pip install numpy` for all the awesomeness of Python
   * `pip install matplotlib` for decent plotting
   * `pip install pandas` for simple data handling
   * `pip install deepdish` for saving and loadding .h5 files ([instructions](http://deepdish.readthedocs.io/en/latest/)); requires PyTables, HDF5
   * `pip install keras` for deep learning ([instructions](https://keras.io/#installation)); requires Theano or TensorFlow
   * `pip install rootpy` for pythonic ROOT ([instructions](http://www.rootpy.org/install.html)); requires ROOT
   * `pip install scikit-learn` for data processing and machine learning 
   * `root_numpy` can be installed with `pip` ([instructions](https://rootpy.github.io/root_numpy/install.html)) but be careful with paths
   * `pip install tqdm` for visual progress bars for loop
* If I ever package this up nicely:
   * `viz`

### Structure
`pipeline.py` is the main script which connects all other modules together. It uses functions from the modules `utils`, `data_processing` and `plotting`. Various neural net models are defined in `./nets/`. We currently choose which one to use by specifying it in the imports at the beginning of `pipeline.py` (not ideal).
### Usage of `pipeline.py`
```
usage: pipeline.py [-h] [--tree TREE] config model_name mode

positional arguments:
  config       Path to JSON file that specifies classes and corresponding ROOT
               files' paths
  model_name   Neural net identifier
  mode         Choose: classification or regression

optional arguments:
  -h, --help   show this help message and exit
  --tree TREE  Name of the tree to open in the ntuples. Default:
               CollectionTree
```
#### Example:
`python pipeline.py config.json abcxyz classification --tree CollectionTree`

## To-do list:
 1. Testing module to check performance and produce ROC curves. Plot ROC curve as a function of mu (pile-up), pt of largest jet, Njets, etc. 
 2. Much needed code improvements, safety checks
 3. CI
 
---
This project has been assigned to [@gstark12](https://github.com/gstark12) and [@jennyailin](https://github.com/jennyailin) as part of their Summer 2016 internship at CERN. They will work under the supervision of [@mickypaganini](https://github.com/mickypaganini) and Prof. Paul Tipton.
