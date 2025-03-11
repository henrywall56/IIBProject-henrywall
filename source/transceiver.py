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
from scipy.fft import fft, ifft


if(p.lab_testing==False):
    if(p.toggle.toggle_PAS==False):
        original_bits = f.generate_original_bits(p.Mod_param.num_symbols, p.Mod_param.Modbits, p.Mod_param.NPol) #NPol-dimensional array
    else:
        original_bits = np.random.randint(0, 2, size= p.PAS_param.k*p.PAS_param.blocks*2*p.Mod_param.NPol)
        original_bits = original_bits.reshape((2,len(original_bits)//2))

    pulse_shaped_symbols, source_symbols = tx.tx(original_bits)

    channel_output = channel.channel(pulse_shaped_symbols)

    demodulated_bits, processed_symbols, demodulated_symbols = rx.rx(channel_output)
    
    # source_symbols, processed_symbols, demodulated_symbols = f.align_symbols_2Pol(source_symbols, processed_symbols, demodulated_symbols)

    BER, AIR, AIR_theoretical = pe.performance_metrics(original_bits, demodulated_bits, source_symbols, processed_symbols)

    print('BER:', BER)

    if(p.Mod_param.NPol==1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        f.plot_constellation(axs, processed_symbols, title='processed', lim=2)
        if(p.toggle.toggle_PAS):
            erroneous_indexes = np.where(np.abs(source_symbols*np.sqrt(p.PAS_param.PAS_normalisation) - demodulated_symbols) > 1e-9)[0]
        else:
            norm_dict = {2: np.sqrt(2),
                         4: np.sqrt(10),
                         6: np.sqrt(42)}
            norm = norm_dict[p.Mod_param.Modbits]

            erroneous_indexes = np.where(np.abs((source_symbols/norm)-demodulated_symbols) > 1e-9)
        axs.scatter(processed_symbols[erroneous_indexes].real, processed_symbols[erroneous_indexes].imag, color='red', label='Errors', s=3, alpha=0.5)
    elif(p.Mod_param.NPol==2):
        fig, axs = plt.subplots(1,2, figsize=(15,6.5))
        f.plot_constellation(axs[0], processed_symbols[0], title='processed V', lim=2)
        f.plot_constellation(axs[1], processed_symbols[1], title='processed H', lim=2)
        if(p.toggle.toggle_PAS):
            erroneous_indexesV = np.where(np.abs(source_symbols[0]*np.sqrt(p.PAS_param.PAS_normalisation) - demodulated_symbols[0]) > 1e-9)[0]
            erroneous_indexesH = np.where(np.abs(source_symbols[1]*np.sqrt(p.PAS_param.PAS_normalisation) - demodulated_symbols[1]) > 1e-9)[0]
        else:
            norm_dict = {2: np.sqrt(2),
                         4: np.sqrt(10),
                         6: np.sqrt(42)}
            norm = norm_dict[p.Mod_param.Modbits]
            erroneous_indexesV = np.where(np.abs((source_symbols[0]/norm)-demodulated_symbols[0]) > 1e-9)
            erroneous_indexesH = np.where(np.abs((source_symbols[1]/norm)-demodulated_symbols[1]) > 1e-9)
        axs[0].scatter(processed_symbols[0][erroneous_indexesV].real, processed_symbols[0][erroneous_indexesV].imag, color='red', label='Errors', s=3, alpha=0.5)
        axs[1].scatter(processed_symbols[1][erroneous_indexesH].real, processed_symbols[1][erroneous_indexesH].imag, color='red', label='Errors', s=3, alpha=0.5)


else: #processing channel output

    # LOAD IN SYMBOLS AND ORIGINAL BITS
    run = p.run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    channel_output_save_dir = os.path.join(script_dir, f"data/channel_output/{run}")
    original_bits_save_dir = os.path.join(script_dir, f"data/original_bits/{run}")
    os.makedirs(channel_output_save_dir, exist_ok=True)
    os.makedirs(original_bits_save_dir, exist_ok=True)

    # if(p.Mod_param.NPol==1):
    #     original_bits = np.loadtxt(os.path.join(original_bits_save_dir, f"original_bits_0403_Pol0_{run}.csv"))
    # else:
    #     original_bits0 = np.loadtxt(os.path.join(original_bits_save_dir, f"original_bits_0403_Pol0_{run}.csv"))
    #     original_bits1 = np.loadtxt(os.path.join(original_bits_save_dir, f"original_bits_0403_Pol1_{run}.csv"))
    #     original_bits = np.array([original_bits0,original_bits1])

    channel_output_dict = scipy.io.loadmat(os.path.join(channel_output_save_dir, f"rx_trace_16QAM50G"))
    if(p.Mod_param.NPol==1):
        channel_output = np.array(channel_output_dict["X_payload"].squeeze())
        channel_output = channel_output[2048:]
    else:
        channel_output0 = channel_output_dict["X_payload"].squeeze()
        channel_output1 = channel_output_dict["Y_payload"].squeeze()
        channel_output0 = channel_output0[2048:]
        channel_output1 = channel_output1[2048:] 
        channel_output = np.array([channel_output0, channel_output1])

    print('####### EXPERIMENTAL CHANNEL OUTPUT SYMBOLS LOADED #######')

    demodulated_bits, processed_symbols, demodulated_symbols = rx.rx(channel_output)

    if(p.Mod_param.NPol==1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        f.plot_constellation(axs, processed_symbols, title='processed', lim=4)
        
    elif(p.Mod_param.NPol==2):
        fig, axs = plt.subplots(1,2, figsize=(15,6.5))
        #only plot those not discarded after adaptive equalisation
        f.plot_constellation(axs[0], processed_symbols[0][p.AE_param.Ndiscard:], title='processed V', lim=2)
        f.plot_constellation(axs[1], processed_symbols[1][p.AE_param.Ndiscard:], title='processed H', lim=2)

    pol0 = channel_output[0]

    snr = np.sum(np.abs(pol0)**2)/len(pol0)




plt.show()

