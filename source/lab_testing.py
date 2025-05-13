import numpy as np
import matplotlib.pyplot as plt
import functions as f
import tx
import parameters as p
import performance_evaluation as pe
import os
from scipy.io import savemat
import datetime

timestamp = datetime.datetime.now().strftime("%a_%Y%m%d_%H%M")

if(p.toggle.toggle_PAS==False):
    np.random.seed(1)
    original_bits = f.generate_original_bits(p.Mod_param.num_symbols, p.Mod_param.Modbits, p.Mod_param.NPol) #NPol-dimensional array
else:
    original_bits = np.random.randint(0, 2, size= p.PAS_param.k*p.PAS_param.blocks*2*p.Mod_param.NPol)
    original_bits = original_bits.reshape((2,len(original_bits)//2))

run = p.run
if(p.lab_testing==True and p.save_run==True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_bits_save_dir = os.path.join(script_dir, f"data/original_bits/{run}_{timestamp}")
    source_symbols_save_dir = os.path.join(script_dir, f"data/source_symbols/{run}_{timestamp}")
    channel_output_symbols_save_dir = os.path.join(script_dir, f"data/channel_output/{run}_{timestamp}")
    matlab_objects_save_dir = os.path.join(script_dir, f"data/matlab_objects/{run}_{timestamp}")
    raw_rx_save_dir = os.path.join(script_dir, f"data/raw_rx/{run}_{timestamp}")

    os.makedirs(original_bits_save_dir, exist_ok=True)
    os.makedirs(source_symbols_save_dir, exist_ok=True)
    os.makedirs(channel_output_symbols_save_dir, exist_ok=True)
    os.makedirs(matlab_objects_save_dir, exist_ok=True)
    os.makedirs(raw_rx_save_dir, exist_ok=True)
 
 
    #  SAVING ORIGINAL BITS
    if(p.Mod_param.NPol==1):
        np.savetxt(os.path.join(original_bits_save_dir, f"original_bits_0403_{run}_{timestamp}.csv"), original_bits, delimiter=",", fmt="%d")
    else:
        np.savetxt(os.path.join(original_bits_save_dir, f"original_bits_0403_Pol0_{run}_{timestamp}.csv"), original_bits[0], delimiter=",", fmt="%d")
        np.savetxt(os.path.join(original_bits_save_dir, f"original_bits_0403_Pol1_{run}_{timestamp}.csv"), original_bits[1], delimiter=",", fmt="%d")
    print('######### SAVED BITS #########')

pulse_shaped_symbols, source_symbols = tx.tx(original_bits)

if(p.lab_testing==True and p.save_run==True):
    #SAVE SENT SYMBOLS
    source_symbols_dict = { "source": source_symbols}
    savemat(os.path.join(source_symbols_save_dir, f"source_symbols_{run}_{timestamp}.mat"), source_symbols_dict)
    print('######### SAVED SOURCE SYMBOLS #########')

#timestamp is {run}_{HHMM}