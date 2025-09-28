# Project to create a simple ML Cloud classifaction program 

## data 
For this project we used a data set from Kaggle https://www.kaggle.com/datasets/mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database . 
Containing 2543 images of clouds distributed over 11 cloud classes. 
Ci = cirrus; Cs = cirrostratus; Cc = cirrocumulus; Ac = altocumulus; As = altostratus; Cu = cumulus; Cb = cumulonimbus; Ns = nimbostratus; Sc = stratocumulus; St = stratus; Ct = contrail.

## code
The code is divided into two jupyter notebooks and one streamlit application. 

The EDA_cloud_classifier contains our prepratory analysis for the creation of the model. 

The cloud_class_model_creation contains the workflow for creation of our machine learning model 

and app.py contains the code for a simple streamlit app that uses the created model to classify cloud types.