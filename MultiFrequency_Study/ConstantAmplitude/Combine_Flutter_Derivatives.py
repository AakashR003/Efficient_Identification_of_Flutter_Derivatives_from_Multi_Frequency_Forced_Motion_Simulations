import os
from MultiFrequency_Data import *

path_base = "_ConstantAmplitude"
velocity = 11.46
n = 4074 # number of lines you want to remove from top
sample_frequency = 2500

frequency = [frequency_set_1, frequency_set_2, frequency_set_3]

for i in frequency:
  
    iteration_variable1 = 1
    
    for j in i:    

        current_path = str(len(j)) + path_base
        combined_flutter_derivatives = os.path.join(current_path + "Flutter_Derivatives")
        current_path = os.path.join(current_path, str(len(j)) + str(iteration_variable1))
        path_CFD_results = os.path.join(current_path, "CFD_Results")
        path_results = os.path.join(current_path ,"Results")         

        for k in j:

            path_flutter_derivatives = os.path.join(path_results, current_path + "_"+ str(k) + "_Flutter_Derivatives.txt")
