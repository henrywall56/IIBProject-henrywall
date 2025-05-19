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
import intensity_plot as ip
import rx_final as rx_final
from matplotlib.ticker import MaxNLocator
from scipy.signal import resample_poly

if(p.lab_testing==False):
    if(p.toggle.toggle_PAS==False):
        original_bits = f.generate_original_bits(p.Mod_param.num_symbols, p.Mod_param.Modbits, p.Mod_param.NPol) #NPol-dimensional array
    else:
        np.random.seed(1)
        original_bits = np.random.randint(0, 2, size= p.PAS_param.k*p.PAS_param.blocks*2*p.Mod_param.NPol)
        original_bits = original_bits.reshape((2,len(original_bits)//2))

    pulse_shaped_symbols, source_symbols = tx.tx(original_bits)

    channel_output = channel.channel(pulse_shaped_symbols)

    demodulated_bits, processed_symbols, demodulated_symbols, H = rx.rx(channel_output, source_symbols)

    target_signal = np.array([resample_poly(source_symbols[0], up=2, down=1), resample_poly(source_symbols[1], up=2, down=1)])
    target_signal = np.array([target_signal[0]/(np.sqrt(np.mean(np.abs(target_signal[0])**2))), target_signal[1]/(np.sqrt(np.mean(np.abs(target_signal[1])**2)))])
    # demodulated_bits, processed_symbols, demodulated_symbols, source_symbols, original_bits, H = rx_final.rx_final(channel_output, target_signal, source_symbols, original_bits)
    
    if(p.Mod_param.NPol==1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        ip.get_color_constellation(processed_symbols,axs[0])
    elif(p.Mod_param.NPol==2):
        fig1, axs1 = plt.subplots(1,2, figsize=(15,6.5))
        ip.get_color_constellation(processed_symbols[0],axs1[0])
        ip.get_color_constellation(processed_symbols[1],axs1[1])

    if(p.Mod_param.NPol==1):
        source_symbols, processed_symbols, demodulated_symbols, demodulated_bits, original_bits, _,_ = f.align_symbols_1Pol(source_symbols, processed_symbols, demodulated_symbols, demodulated_bits, original_bits, p.Mod_param.Modbits)
    else:
        source_symbols, processed_symbols, demodulated_symbols, demodulated_bits, original_bits, _,_ = f.align_symbols_2Pol(source_symbols, processed_symbols, demodulated_symbols, demodulated_bits, original_bits, p.Mod_param.Modbits)

    if(p.Mod_param.NPol==2):
        snr_db0, snr_db1, _ = f.estimate_snr(processed_symbols[:,p.AE_param.NTaps:], p.Mod_param.Modbits, source_symbols[:,p.AE_param.NTaps:], p.toggle.toggle_PAS)
        # p.fibre_param.snr_db  = (snr_db0+snr_db1)/2
        if(p.toggle.toggle_PAS==False):
            H=[]
        BER, AIR, _ = pe.performance_metrics(original_bits, demodulated_bits, source_symbols, processed_symbols, H)

        print('BER:', BER)
        print('AIR:', AIR)


    if(p.Mod_param.NPol==1):
        fig, axs = plt.subplots(1, 1, figfze=(8, 8))
        f.plot_constellation(axs, processed_symbols, title='processed', lim=2)
        if(p.toggle.toggle_PAS):
            erroneous_indexes = np.where(np.abs(source_symbols*np.sqrt(p.PAS_param.PAS_normalisation) - demodulated_symbols) > 1e-9)[0]
        else:
            norm_dict = {2: np.sqrt(2),
                         4: np.sqrt(10),
                         6: np.sqrt(42),
                         8: np.sqrt(170)}
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
                         6: np.sqrt(42),
                         8: np.sqrt(170)}
            norm = norm_dict[p.Mod_param.Modbits]
            erroneous_indexesV = np.where(np.abs((source_symbols[0]/norm)-demodulated_symbols[0]) > 1e-9)
            erroneous_indexesH = np.where(np.abs((source_symbols[1]/norm)-demodulated_symbols[1]) > 1e-9)
        axs[0].scatter(processed_symbols[0][erroneous_indexesV].real, processed_symbols[0][erroneous_indexesV].imag, color='red', label='Errors', s=3, alpha=0.5)
        axs[1].scatter(processed_symbols[1][erroneous_indexesH].real, processed_symbols[1][erroneous_indexesH].imag, color='red', label='Errors', s=3, alpha=0.5)


    plt.tight_layout()
        
    plt.show()


else: #processing channel output

    # LOAD IN SYMBOLS AND ORIGINAL BITS
    run = p.run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    channel_output_save_dir = os.path.join(script_dir, f"data/channel_output/{run}")
    original_bits_save_dir = os.path.join(script_dir,  f"data/original_bits/{run}")
    matlab_objects_save_dir = os.path.join(script_dir, f"data/matlab_objects/{run}")
    source_symbols_save_dir = os.path.join(script_dir, f"data/source_symbols/{run}")


    if(p.Mod_param.NPol==1):
        original_bits = np.loadtxt(os.path.join(original_bits_save_dir, f"original_bits_Pol0_{run}.csv"))
    else:
        original_bits0 = np.loadtxt(os.path.join(original_bits_save_dir, f"original_bits_Pol0_{run}.csv"))
        original_bits1 = np.loadtxt(os.path.join(original_bits_save_dir, f"original_bits_Pol1_{run}.csv"))
        original_bits = np.array([original_bits0,original_bits1])

    #Weds/Thurs 14/15 Data:
    # channel_output_dict = scipy.io.loadmat(os.path.join(channel_output_save_dir, f"X_D_DualPol.mat"))
    # matlab_objects_dict = scipy.io.loadmat(os.path.join(matlab_objects_save_dir, f"DualPol.mat"))
    # source_symbols_dict = scipy.io.loadmat(os.path.join(source_symbols_save_dir, f"source_symbols_{run}.mat"))

    # Monday 19 Data:
    matlab_objects_dict = scipy.io.loadmat(os.path.join(matlab_objects_save_dir, f"matlab.mat"))
    source_symbols_dict = scipy.io.loadmat(os.path.join(source_symbols_save_dir, f"source_symbols_{run}.mat"))

    if(p.Mod_param.NPol==1):    
    #     #channel_output = np.array(channel_output_dict["X_payload"].squeeze())
    #     channel_output = np.array(channel_output_dict["X"][:,0].squeeze())
    #     channel_output = channel_output[2048:]
        nopremphasis_source_symbols = np.array(matlab_objects_dict["SignalX_nopre"][::2].squeeze())
        source_symbols = np.array(source_symbols_dict["source"][0,:].squeeze())
        # Target_Signal = np.array(channel_output_dict["D"][:,0].squeeze())

    else:
    #     # channel_output0 = channel_output_dict["X_payload"].squeeze()
    #     # channel_output1 = channel_output_dict["Y_payload"].squeeze()

        #Weds/Thurs 14/15 Data:
        # channel_output0 = channel_output_dict["X"][:,0].squeeze()
        # channel_output1 = channel_output_dict["X"][:,1].squeeze()
        # channel_output0 = channel_output0[2048:]
        # channel_output1 = channel_output1[2048:] 
        # channel_output = np.array([channel_output0, channel_output1])
        # source_symbols = np.array([source_symbols_dict["source"][0,:].squeeze(),source_symbols_dict["source"][1,:].squeeze()])
        # Target_Signal = np.array([channel_output_dict["D"][:,0].squeeze(),channel_output_dict["D"][:,1].squeeze()])

        nopremphasis_source_symbols = np.array([matlab_objects_dict["SignalX_nopre"][::2].squeeze(),matlab_objects_dict["SignalY_nopre"][::2].squeeze()])

        #Monday 19 Data:
        channel_output0 = matlab_objects_dict["X_payload"].squeeze()
        channel_output1 = matlab_objects_dict["Y_payload"].squeeze()
        channel_output0 = channel_output0[2048:]
        channel_output1 = channel_output1[2048:] 
        channel_output = np.array([channel_output0, channel_output1])
        source_symbols = np.array([source_symbols_dict["source"][0,:].squeeze(),source_symbols_dict["source"][1,:].squeeze()])
        if(p.toggle.toggle_PAS==False):
            Target_Signal = np.array([matlab_objects_dict["D"][:,0].squeeze(),matlab_objects_dict["D"][:,1].squeeze()])
        else:
            Target_Signal = np.array([matlab_objects_dict["TargetSignalX"].squeeze(),matlab_objects_dict["TargetSignalY"].squeeze()])

    
    # fig5, axs5 = plt.subplots(1,2, figsize=(15,6.5))

    if(p.Mod_param.NPol==1):
        nopremphasis_source_symbols = nopremphasis_source_symbols[1536:]/(np.sqrt(np.mean(np.abs(nopremphasis_source_symbols[1536:])**2)))
    if(p.Mod_param.NPol==2):
        nopremphasis_source_symbols0 = nopremphasis_source_symbols[0][1536:]/(np.sqrt(np.mean(np.abs(nopremphasis_source_symbols[0][1536:])**2)))
        nopremphasis_source_symbols1 = nopremphasis_source_symbols[1][1536:]/(np.sqrt(np.mean(np.abs(nopremphasis_source_symbols[1][1536:])**2)))
        nopremphasis_source_symbols = np.array([nopremphasis_source_symbols0, nopremphasis_source_symbols1])

    # if(p.Mod_param.NPol==1):    
    #     f.plot_constellation(axs5[0], nopremphasis_source_symbols, title='source symbols', lim=2)
    # else:
    #     f.plot_constellation(axs5[0], nopremphasis_source_symbols[0], title='source symbols', lim=2)
    #     f.plot_constellation(axs5[1], nopremphasis_source_symbols[1], title='source symbols', lim=2)


    print('####### EXPERIMENTAL CHANNEL OUTPUT SYMBOLS LOADED #######')

    # demodulated_bits, processed_symbols, demodulated_symbols, H = rx.rx(channel_output, Target_Signal)

    demodulated_bits, processed_symbols, demodulated_symbols, source_symbols, original_bits, H = rx_final.rx_final(channel_output, Target_Signal, source_symbols, original_bits)

    if(p.Mod_param.NPol==1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        ip.get_color_constellation(processed_symbols[p.AE_param.Ndiscard:],axs[0])
    #     f.plot_constellation(axs, processed_symbols, title='processed', lim=4)
        
    elif(p.Mod_param.NPol==2):
    #     fig, axs = plt.subplots(1,2, figsize=(15,6.5))
    #     #only plot those not discarded after adaptive equalisation
    #     f.plot_constellation(axs[0], processed_symbols[0][p.AE_param.Ndiscard:], title='processed V', lim=2)
    #     f.plot_constellation(axs[1], processed_symbols[1][p.AE_param.Ndiscard:], title='processed H', lim=2)
        
        fig1, axs1 = plt.subplots(1,2, figsize=(15,6.5))
        ip.get_color_constellation(processed_symbols[0][p.AE_param.Ndiscard:],axs1[0])
        ip.get_color_constellation(processed_symbols[1][p.AE_param.Ndiscard:],axs1[1])


    # plot_autocorr = True
    # if(plot_autocorr==True):
    #     if(p.Mod_param.NPol==1):
    #         autocorr = np.real(ifft(np.conjugate(fft(source_symbols))*fft(processed_symbols)))
    #         # plt.figure()
    #         # plt.plot(autocorr)
    #         # plt.title('autocorrelation')
    #         print('Max autocorrelation at index', np.argmax(autocorr), 'or', np.argmax(autocorr)-p.Mod_param.num_symbols)
    #     else:
    #         autocorrV = np.real(ifft(np.conjugate(fft(source_symbols[0]))*fft(processed_symbols[0])))
    #         autocorrH = np.real(ifft(np.conjugate(fft(source_symbols[1]))*fft(processed_symbols[1])))
    #         # plt.figure()
    #         # plt.plot(autocorrV)
    #         # plt.title('V autocorrelation')
    #         print('Max V autocorrelation at index', np.argmax(autocorrV), 'or', np.argmax(autocorrV)-p.Mod_param.num_symbols)
    #         # plt.figure()    
    #         # plt.plot(autocorrH)
    #         # plt.title('H autocorrelation')
    #         print('Max H autocorrelation at index', np.argmax(autocorrH), 'or', np.argmax(autocorrV)-p.Mod_param.num_symbols)

    
    if(p.Mod_param.NPol==1):
        source_symbols, processed_symbols, demodulated_symbols, demodulated_bits, original_bits, _,_ = f.align_symbols_1Pol(source_symbols, processed_symbols, demodulated_symbols, demodulated_bits, original_bits, p.Mod_param.Modbits)
    else:
        source_symbols, processed_symbols, demodulated_symbols, demodulated_bits, original_bits, _,_ = f.align_symbols_2Pol(source_symbols, processed_symbols, demodulated_symbols, demodulated_bits, original_bits, p.Mod_param.Modbits)

    if(p.Mod_param.NPol==2):
        if(p.toggle.toggle_PAS==False): #Truncate first Ndiscard symbols
            original_bits = original_bits[:,p.AE_param.Ndiscard*p.Mod_param.Modbits:]
            demodulated_bits = demodulated_bits[:,p.AE_param.Ndiscard*p.Mod_param.Modbits:]
            source_symbols =source_symbols[:,p.AE_param.Ndiscard:]
            demodulated_symbols = demodulated_symbols[:,p.AE_param.Ndiscard:]
            processed_symbols = processed_symbols[:,p.AE_param.Ndiscard:]
        snr_db0, snr_db1, _ = f.estimate_snr(processed_symbols, p.Mod_param.Modbits, source_symbols, p.toggle.toggle_PAS)
        p.fibre_param.snr_db  = (snr_db0+snr_db1)/2
        if(p.toggle.toggle_PAS==False):
            H=[]
        BER, AIR, _ = pe.performance_metrics(original_bits, demodulated_bits, source_symbols, processed_symbols, H)

        print('BER:', BER)
        print('AIR:', AIR)

    if(p.Mod_param.NPol==1):
        fig, axs = plt.subplots(1, 1, figfze=(8, 8))
        f.plot_constellation(axs, processed_symbols, title='processed', lim=2)
        if(p.toggle.toggle_PAS):
            erroneous_indexes = np.where(np.abs(source_symbols*np.sqrt(p.PAS_param.PAS_normalisation) - demodulated_symbols) > 1e-9)[0]
        else:
            norm_dict = {2: np.sqrt(2),
                         4: np.sqrt(10),
                         6: np.sqrt(42),
                         8: np.sqrt(170)}
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
                         6: np.sqrt(42),
                         8: np.sqrt(170)}
            norm = norm_dict[p.Mod_param.Modbits]
            erroneous_indexesV = np.where(np.abs((source_symbols[0]/norm)-demodulated_symbols[0]) > 1e-9)
            erroneous_indexesH = np.where(np.abs((source_symbols[1]/norm)-demodulated_symbols[1]) > 1e-9)
        axs[0].scatter(processed_symbols[0][erroneous_indexesV].real, processed_symbols[0][erroneous_indexesV].imag, color='red', label='Errors', s=3, alpha=0.5)
        axs[1].scatter(processed_symbols[1][erroneous_indexesH].real, processed_symbols[1][erroneous_indexesH].imag, color='red', label='Errors', s=3, alpha=0.5)


    plt.tight_layout()
        
    plt.show()





