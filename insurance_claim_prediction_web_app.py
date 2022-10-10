# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 01:22:21 2022

@author: surya_pc
"""
# NumPy for numerical computing
import numpy as np

# Pickle for reading model files
import pickle

# scikitlearn
import sklearn
sklearn.set_config(print_changed_only = False)

# Ignore sklearn's FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#Ignore Pandas SettingWithCopyWarning 
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#streamlit
import streamlit as st


# loading the saved model
clf = pickle.load(open('enet_model.sav', 'rb'))

# creating a function for Prediction
def fn_insurance_claim_prediction(input_data):   
    
    input_array = np.asarray(input_data)
    
    input_array_reshaped = input_array.reshape(1,38)
    
    prediction = (clf.predict_proba(input_array_reshaped)[:,1] >= 0.035).astype(int)
    print(prediction[0])
    
    if (prediction[0] == 0):
        return 'The driver will not file an insurance claim next year'
    else:
      return 'The driver will file an insurance claim next year'
    
  
def main(): 
    
    # giving a title
    st.title('Insurace Claim Prediction Web App')
    
    
    # getting the input data from the user
    user_input = st.text_input('Enter all the feature values separated by comma')
    
    #string to list
    user_input = user_input.split(",")
    
    # Get the required field values to separate variables
    ps_calc_02=user_input[39]
    ps_ind_12_bin=user_input[12]
    ps_ind_07_bin=user_input[7]
    ps_ind_15=user_input[15]
    ps_car_08_cat=user_input[29]
    ps_car_02_cat=user_input[23]
    ps_car_13=user_input[35]
    ps_car_07_cat=user_input[28]
    ps_car_04_cat=user_input[25]
    ps_car_14=user_input[36]
    ps_ind_02_cat=user_input[2]
    ps_ind_17_bin=user_input[17]
    ps_car_11=user_input[33]
    ps_ind_18_bin=user_input[18]
    ps_car_12=user_input[34]
    ps_ind_04_cat=user_input[4]
    ps_reg_03=user_input[21]
    ps_reg_01=user_input[19]
    ps_car_15=user_input[37]
    ps_calc_01=user_input[38]
    ps_car_09_cat=user_input[30]
    ps_ind_14=user_input[14]
    ps_ind_16_bin=user_input[16]
    ps_ind_05_cat=user_input[5]
    ps_ind_11_bin=user_input[11]
    ps_ind_01=user_input[1]
    ps_reg_02=user_input[20]
    ps_ind_08_bin=user_input[8]
    ps_car_10_cat=user_input[31]
    ps_car_03_cat=user_input[24]
    ps_car_05_cat=user_input[26]
    ps_car_01_cat=user_input[22]
    ps_car_06_cat=user_input[27]
    ps_calc_03=user_input[40]
    ps_ind_10_bin=user_input[10]
    ps_ind_06_bin=user_input[6]
    ps_ind_03=user_input[3]
    ps_ind_09_bin=user_input[9]
    
    # code for Prediction
    claim_prediciton = ''
    
    # creating a button for Prediction
    if st.button('Claim Prediction Result'):
        claim_prediciton = fn_insurance_claim_prediction([ps_calc_02	,ps_ind_12_bin	,ps_ind_07_bin	,ps_ind_15	,ps_car_08_cat	,ps_car_02_cat	,ps_car_13	,
                                                          ps_car_07_cat	,ps_car_04_cat	,ps_car_14	,ps_ind_02_cat	,ps_ind_17_bin	, ps_car_11	,ps_ind_18_bin	,
                                                          ps_car_12	,ps_ind_04_cat	,ps_reg_03	,ps_reg_01	,ps_car_15	,ps_calc_01	,ps_car_09_cat	,ps_ind_14	,
                                                          ps_ind_16_bin	,ps_ind_05_cat	,ps_ind_11_bin	,ps_ind_01	,ps_reg_02	,ps_ind_08_bin	,ps_car_10_cat	,
                                                          ps_car_03_cat	,ps_car_05_cat	,ps_car_01_cat	,ps_car_06_cat	,ps_calc_03	,ps_ind_10_bin	,ps_ind_06_bin	,
                                                          ps_ind_03	,ps_ind_09_bin] )
        
    st.success(claim_prediciton)
    
    
if __name__ == '__main__':
    main()