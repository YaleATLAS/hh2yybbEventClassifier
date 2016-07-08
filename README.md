# hh2yybb Event Classifier
Event level classifier for the hh-->yybb analysis using multi-stream RNNs

## RNN-based Event Level Classifier for ATLAS Analysis (hh-->yybb) 
 
### Purpose: 
ATLAS has a ‘cut based’ hh data analysis. To improve on this, a ML algoorithm is being trained to select the correct “second b jet” in single-tagged events ([bbyy jet classifier](https://github.com/jemrobinson/bbyy_jet_classifier)). The project proposed here will take this a step further and develop an event-level tagger to distinguish di-Higgs events from various background sources. In the resonant analysis, we will use regression to determine the most likely mass of the resonance. 
 
### Current Status:
![Pipeline](images/UGsWorkflow.png)

### To-do list:
 
* Produce the ntuples using `HGamAnalysisFramework`:  
     Decide what info to include (jets and photons, but also leptons, pileup info?)   
     Apply the pre-selection </br>
     Assign truth labels using b-quark-from-Higgs labeling scheme </br>
     Actually make the ntuples on grid – run on signal and bkg events
 
* Analysis Coding Tasks -- Modules needed: 
   1. Training module that uses `Keras` (design RNN, test different NN architectures, etc.) 
   2. Testing module to check performance and produce ROC curves. Plot ROC curve as a function of mu (pile-up), pt of largest jet, Njets, etc. 
  
* Write presentations

---
This project has been assigned to [@gstark12](https://github.com/gstark12) and [@jennyailin](https://github.com/jennyailin) as part of their Summer 2016 internship at CERN. They will work under the supervision of [@mickypaganini](https://github.com/mickypaganini) and Prof. Paul Tipton.
