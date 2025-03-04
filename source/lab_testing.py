import numpy as np
import matplotlib.pyplot as plt
import functions as f
import tx
import channel
import rx
import parameters as p
import performance_evaluation as pe
import os
from scipy.io import savemat

if(p.toggle.toggle_PAS==False):
    original_bits = f.generate_original_bits(p.Mod_param.num_symbols, p.Mod_param.Modbits, p.Mod_param.NPol) #NPol-dimensional array
else:
    np.random.seed(1)
    original_bits = np.random.randint(0, 2, size= p.PAS_param.k*p.PAS_param.blocks*2*p.Mod_param.NPol)
    original_bits = original_bits.reshape((2,len(original_bits)//2))

run=2
script_dir = os.path.dirname(os.path.abspath(__file__))
original_bits_save_dir = os.path.join(script_dir, f"data/original_bits/{run}")
source_symbols_save_dir = os.path.join(script_dir, f"data/source_symbols/{run}")
 
# SAVING ORIGINAL BITS
if(p.Mod_param.NPol==1):
    np.savetxt(os.path.join(original_bits_save_dir, f"original_bits_0403_{run}.csv"), original_bits, delimiter=",", fmt="%d")
else:
    np.savetxt(os.path.join(original_bits_save_dir, f"original_bits_0403_Pol0_{run}.csv"), original_bits[0], delimiter=",", fmt="%d")
    np.savetxt(os.path.join(original_bits_save_dir, f"original_bits_0403_Pol1_{run}.csv"), original_bits[1], delimiter=",", fmt="%d")

pulse_shaped_symbols, source_symbols = tx.tx(original_bits)


#SAVE SENT SYMBOLS
# source_symbols0_real = source_symbols[0].real
# source_symbols0_imag = source_symbols[0].imag
# source_symbols1_real = source_symbols[1].real
# source_symbols1_imag = source_symbols[1].imag

# np.savetxt(os.path.join(source_symbols_save_dir, f"source_symbols_0403_Pol0_real_{run}.csv"), source_symbols0_real, delimiter=",",fmt="%d")
# np.savetxt(os.path.join(source_symbols_save_dir, f"source_symbols_0403_Pol0_imag_{run}.csv"), source_symbols0_imag, delimiter=",",fmt="%d")
# np.savetxt(os.path.join(source_symbols_save_dir, f"source_symbols_0403_Pol1_real_{run}.csv"), source_symbols1_real, delimiter=",",fmt="%d")
# np.savetxt(os.path.join(source_symbols_save_dir, f"source_symbols_0403_Pol1_imag_{run}.csv"), source_symbols1_imag, delimiter=",",fmt="%d")

source_symbols_dict = { "source": source_symbols}
savemat(os.path.join(source_symbols_save_dir, f"source_symbols_{run}.mat"), source_symbols_dict)