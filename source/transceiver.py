import numpy as np
import matplotlib.pyplot as plt
import functions as f
import tx
import channel
import rx
import parameters as p
import performance_evaluation as pe
import os
import scipy

if(p.toggle.toggle_PAS==False):
    original_bits = f.generate_original_bits(p.Mod_param.num_symbols, p.Mod_param.Modbits, p.Mod_param.NPol) #NPol-dimensional array
else:
    np.random.seed(1)
    original_bits = np.random.randint(0, 2, size= p.PAS_param.k*p.PAS_param.blocks*2*p.Mod_param.NPol)
    original_bits = original_bits.reshape((2,len(original_bits)//2))

pulse_shaped_symbols, source_symbols = tx.tx(original_bits)

# channel_output = channel.channel(pulse_shaped_symbols)

# LOAD IN SYMBOLS
channel_output_dict = scipy.io.loadmat("/Users/henrywall/Desktop/IIB Project/Python/source/PAS/ldpc_jossy/rx_trace_16QAM50G_1.mat")
channel_output0 = channel_output_dict["X_payload"].squeeze()
channel_output1 = channel_output_dict["Y_payload"].squeeze()
channel_output0 = channel_output0[2048:]
channel_output1 = channel_output1[2048:]
channel_output = np.array([channel_output0, channel_output1])

demodulated_bits, processed_symbols, demodulated_symbols = rx.rx(channel_output)

# BER, AIR, AIR_theoretical = pe.performance_metrics(original_bits, demodulated_bits, source_symbols, processed_symbols)

# print('BER:', BER)

if(p.Mod_param.NPol==1):
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    f.plot_constellation(axs, processed_symbols, title='processed', lim=4)
    # if(p.toggle.toggle_PAS):
    #     erroneous_indexes = np.where(np.abs(source_symbols*np.sqrt(p.PAS_param.PAS_normalisation) - demodulated_symbols) > 1e-9)[0]
    # else:
    #     erroneous_indexes = np.where(np.abs(source_symbols-demodulated_symbols) > 1e-9)
    # axs.scatter(processed_symbols[erroneous_indexes].real, processed_symbols[erroneous_indexes].imag, color='red', label='Errors', alpha=0.5)
elif(p.Mod_param.NPol==2):
    fig, axs = plt.subplots(1,2, figsize=(15,6.5))
    f.plot_constellation(axs[0], processed_symbols[0], title='processed V', lim=2)
    f.plot_constellation(axs[1], processed_symbols[1], title='processed H', lim=2)
    # if(p.toggle.toggle_PAS):
    #     erroneous_indexesV = np.where(np.abs(source_symbols[0]*np.sqrt(p.PAS_param.PAS_normalisation) - demodulated_symbols[0]) > 1e-9)[0]
    #     erroneous_indexesH = np.where(np.abs(source_symbols[1]*np.sqrt(p.PAS_param.PAS_normalisation) - demodulated_symbols[1]) > 1e-9)[0]
    # else:
    #     erroneous_indexesV = np.where(np.abs(source_symbols[0]-demodulated_symbols[0]) > 1e-9)
    #     erroneous_indexesH = np.where(np.abs(source_symbols[1]-demodulated_symbols[1]) > 1e-9)
    # axs[0].scatter(processed_symbols[0][erroneous_indexesV].real, processed_symbols[0][erroneous_indexesV].imag, color='red', label='Errors', alpha=0.5)
    # axs[1].scatter(processed_symbols[1][erroneous_indexesH].real, processed_symbols[1][erroneous_indexesH].imag, color='red', label='Errors', alpha=0.5)

plt.show()
