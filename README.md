# hh2yybb Event Classifier
Event level classifier for the hh-->yybb analysis using multi-stream RNNs

## RNN-based Event Level Classifier for ATLAS Analysis (hh-->yybb) 
 
### Purpose: 
'Cut-based' methods are the prevalent data analysis strategy in ATLAS, which is also currently being used in the hh-->yybb analysis group. To improve on this, a ML algorithm is being trained to select the correct “second b jet” in single-tagged events ([bbyy jet classifier](https://github.com/jemrobinson/bbyy_jet_classifier)). The project in this repo will take this a step further and develop an event classifier. 
 
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
   ![Net](images/MultiStreamRNN.png)
   2. Testing module to check performance and produce ROC curves. Plot ROC curve as a function of mu (pile-up), pt of largest jet, Njets, etc. 
  
* Write presentations

---
This project has been assigned to [@gstark12](https://github.com/gstark12) and [@jennyailin](https://github.com/jennyailin) as part of their Summer 2016 internship at CERN. They will work under the supervision of [@mickypaganini](https://github.com/mickypaganini) and Prof. Paul Tipton.
