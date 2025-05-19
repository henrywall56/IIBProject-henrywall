import numpy as np
import matplotlib.pyplot as plt
import functions as f
import PAS.PAS_architecture as pas
import DDPhaseRecoveryTesting as dd
import parameters as p
from scipy.fft import fft, ifft
import intensity_plot as ip

def rx_final(rx, target_signal, source_symbols, original_bits):
    if(p.Mod_param.NPol==1):
        N=len(rx)//2
        source_symbols = source_symbols[:N] #1sps
        target_signal = target_signal[:N*2] #2sps
        original_bits = original_bits[:N*p.Mod_param.Modbits] #1sps
    else:
        N = rx.shape[1]//2
        source_symbols = source_symbols[:,:N] #1sps
        target_signal = target_signal[:,:N*2] #2sps
        original_bits = original_bits[:,:N*p.Mod_param.Modbits] #1sps

    NPol = p.Mod_param.NPol
    Modbits = p.Mod_param.Modbits

    if(p.lab_testing==False):
        filtered_signal = f.matched_filter(rx, p.RRC_param.RRCimpulse, NPol, toggle=p.toggle.toggle_RRC) #if toggle is False, this function returns input
        ADC = f.downsample(filtered_signal, p.RRC_param.sps//2, NPol, toggle=p.toggle.toggle_RRC) #Simulate ADC downsampling to 2 sps
    else:
        filtered_signal = rx
        ADC = filtered_signal #at 2 sps

    #Chromatic Dispersion Compensation
    if(p.toggle.toggle_RRC==True):
        spsCD = 2
    else:
        spsCD=1

    CD_compensated_rx = f.CD_compensation(ADC, p.fibre_param.D, p.fibre_param.L, p.fibre_param.Clambda, p.Mod_param.Rs, NPol, spsCD, p.CD_param.NFFT, p.CD_param.NOverlap, p.toggle.toggle_CD_compensation)
    
    frequency_recovered_rx = f.frequency_recovery(CD_compensated_rx, p.Mod_param.Rs, NPol, p.toggle.toggle_frequencyrecovery)
    
    if(NPol==1):
        frequency_recovered_rx = frequency_recovered_rx/np.sqrt(np.mean(np.abs(frequency_recovered_rx)**2))
    else:
        frequency_recovered_rx = np.array([frequency_recovered_rx[0]/np.sqrt(np.mean(np.abs(frequency_recovered_rx[0])**2)), frequency_recovered_rx[1]/np.sqrt(np.mean(np.abs(frequency_recovered_rx[1])**2))])

    if(p.toggle.toggle_adaptive_equalisation == True and NPol == 2):
        #***adaptive equalisation (2x2 MIMO, LMS with known data)***
        MIMO_LMS_2x2_rx = f.MIMO_LMS_AEQ(frequency_recovered_rx,target_signal,p.AE_param.mu,p.AE_param.NTaps)
        
        print('--------------------------------------')
        print('LMS Adaptive Equalisation Parameters:')
        print('Number of taps:         ', p.AE_param.NTaps)
        print('μ:                      ', p.AE_param.mu)

        figAE, axsAE = plt.subplots(1,1, figsize=(8,8))
        axsAE.plot(abs(MIMO_LMS_2x2_rx[0]), linestyle='', marker='o', markersize='1', color='b', label='y1 mag.')
        axsAE.set_ylabel('Magnitude of symbols after Complex LMS AEQ')
        axsAE.plot(abs(MIMO_LMS_2x2_rx[1]), linestyle='', marker='o', markersize='1', color='r', label='y2 mag.')
        axsAE.set_ylim(0,3)
        axsAE.legend()

        MIMO_LMS_2x2_rx[:,:p.AE_param.NTaps] = frequency_recovered_rx[:,:p.AE_param.NTaps]

    else:
        MIMO_LMS_2x2_rx = frequency_recovered_rx
    
    #Still 2sps
    #Normalise:
    if(NPol==1):
        MIMO_LMS_2x2_rx = MIMO_LMS_2x2_rx/np.sqrt(np.mean(np.abs(MIMO_LMS_2x2_rx)**2))
    else:
        MIMO_LMS_2x2_rx = np.array([MIMO_LMS_2x2_rx[0]/np.sqrt(np.mean(np.abs(MIMO_LMS_2x2_rx[0])**2)), MIMO_LMS_2x2_rx[1]/np.sqrt(np.mean(np.abs(MIMO_LMS_2x2_rx[1])**2))])

    
    
    if(p.toggle.toggle_BPS==True):
        print('--------------------------------------')
        print('BPS parameters:')
        print('Test Angles:      ', p.BPS_param.B)
        print('Averaging number: ', p.BPS_param.N)
        
        Phase_Noise_compensated_rx, thetaHat = f.BPS(MIMO_LMS_2x2_rx, Modbits, p.BPS_param.N, p.BPS_param.B, NPol, p.toggle.toggle_phasenoisecompensation)
        if(p.toggle.toggle_phasenoisecompensation):
            plt.figure()
            plt.plot(thetaHat, color='blue')

        
    else:
        #Note DD algorithm currently only set up for NPol==1
        #Note this DD algorithm currently uses SNR per bit, which should be changed to per symbol
        frac = 0.05
        Phase_Noise_compensated_rx, thetaHat = dd.DD_phase_noise_compensation(MIMO_LMS_2x2_rx, p.RRC_param.sps, p.Mod_param.Rs, p.laser_param.Linewidth, Modbits, p.Mod_param.snr_db, frac, p.toggle.toggle_phasenoisecompensation)
    
    if(p.toggle.toggle_phasenoisecompensation==True and p.lab_testing==False):
        figPhase, axsPhase = plt.subplots(1,1, figsize=(8,8))
        axsPhase.plot(1e9*np.arange(p.Mod_param.num_symbols)/(p.Mod_param.Rs*p.RRC_param.sps), p.laser_param.theta[0::p.RRC_param.sps], label='Phase')
        #Note plotting theta before frequency recovery here

        print('--------------------------------------')
        axsPhase.set_title(f'Phase Noise at SNR = {p.fibre_param.snr_db}dB \n ∆νT = {p.laser_param.maxDvT}')
        axsPhase.grid(True)
        axsPhase.set_xlabel('Time (ns)')
        axsPhase.set_ylabel('Phase (rad)')

        if(NPol==1):
            axsPhase.plot(1e9*np.arange(p.Mod_param.num_symbols)/(p.Mod_param.Rs*p.RRC_param.sps), thetaHat, color='red', label='Phase Estimate')
        elif(NPol==2):
            axsPhase.plot(1e9*np.arange(p.Mod_param.num_symbols)/(p.Mod_param.Rs*p.RRC_param.sps), thetaHat[:,0], color='red', label='Phase Estimate (V)')
            axsPhase.plot(1e9*np.arange(p.Mod_param.num_symbols)/(p.Mod_param.Rs*p.RRC_param.sps), thetaHat[:,1], color='orange', label='Phase Estimate (H)')
        axsPhase.legend(loc='lower left')

    if(p.toggle.toggle_adaptive_equalisation==True and p.toggle.toggle_real_adaptive_equalisation==True):
        # if(p.Mod_param.NPol==1):
        #     autocorr = np.real(ifft(np.conjugate(fft(target_signal))*fft(Phase_Noise_compensated_rx)))
        #     print('Max autocorrelation pre real valued AEQ at index', np.argmax(autocorr), 'or', np.argmax(autocorr)-p.Mod_param.num_symbols)
        # else:
        #     autocorrV = np.real(ifft(np.conjugate(fft(target_signal[0]))*fft(Phase_Noise_compensated_rx[0])))
        #     autocorrH = np.real(ifft(np.conjugate(fft(target_signal[1]))*fft(Phase_Noise_compensated_rx[1])))
        #     plt.figure()
        #     plt.plot(autocorrV)
        #     plt.title('V autocorrelation')
        #     print('Max V autocorrelation pre real valued AEQ  at index', np.argmax(autocorrV), 'or', np.argmax(autocorrV)-p.Mod_param.num_symbols)
        #     plt.figure()    
        #     plt.plot(autocorrH)
        #     plt.title('H autocorrelation')
        #     print('Max H autocorrelation pre real valued AEQ  at index', np.argmax(autocorrH), 'or', np.argmax(autocorrV)-p.Mod_param.num_symbols)

        if(p.Mod_param.NPol==1):
            #need to align target_signal and processed symbols for Real 2x2 AEQ
            #Need to align source symbols and original bits too for performance metric
            _1 = np.zeros(len(target_signal))
            _2 = np.zeros(len(target_signal)*p.Mod_param.Modbits)
            target_signal, Phase_Noise_compensated_rx, _, _, _, shift= f.align_symbols_1Pol(target_signal, Phase_Noise_compensated_rx, _1, _2, _2, p.Mod_param.Modbits)
            shift = shift//2 #Shifting source and original bits which are 1sps
            source_symbols, original_bits = f.shift(source_symbols,original_bits,shift,0,1,p.Mod_param.Modbits)
        else:
            _1 = np.zeros((2,target_signal.shape[1]))
            _2 = np.zeros((2,target_signal.shape[1]*p.Mod_param.Modbits))
            target_signal, Phase_Noise_compensated_rx, _, _, _, shift0,shift1 = f.align_symbols_2Pol(target_signal, Phase_Noise_compensated_rx, _1, _2, _2, p.Mod_param.Modbits)
            shift0 = shift0//2
            shift1 = shift1//2
            source_symbols, original_bits = f.shift(source_symbols,original_bits,shift0,shift1,2,p.Mod_param.Modbits)
            #Shifting source and original bits which are 1sps

            print('--------------------------------------')
            print('Real Valued LMS Adaptive Equalisation Parameters:')
            print('Number of taps:         ', p.AE_param.NTaps_real)
            print('μ:                      ', p.AE_param.mu_real)

            Real_LMS_2x2_rx0 = f.MIMO_2x2_with_CPR(Phase_Noise_compensated_rx[0],target_signal[0],p.AE_param.mu_real,p.AE_param.NTaps_real)
            Real_LMS_2x2_rx1 = f.MIMO_2x2_with_CPR(Phase_Noise_compensated_rx[1],target_signal[1],p.AE_param.mu_real,p.AE_param.NTaps_real)
            Real_LMS_2x2_rx0[:p.AE_param.NTaps_real] = Phase_Noise_compensated_rx[0][:p.AE_param.NTaps_real]
            Real_LMS_2x2_rx1[:p.AE_param.NTaps_real] = Phase_Noise_compensated_rx[1][:p.AE_param.NTaps_real]
            Real_LMS_2x2_rx = np.array([Real_LMS_2x2_rx0,Real_LMS_2x2_rx1])
            Real_LMS_2x2_rx = np.array([Real_LMS_2x2_rx[0]/np.sqrt(np.mean(np.abs(Real_LMS_2x2_rx[0])**2)), Real_LMS_2x2_rx[1]/np.sqrt(np.mean(np.abs(Real_LMS_2x2_rx[1])**2))])

            figAE_R, axsAE_R = plt.subplots(1,1, figsize=(8,8))
            axsAE_R.plot(abs(Real_LMS_2x2_rx[0]), linestyle='', marker='o', markersize='1', color='b', label='y1 mag.')
            axsAE_R.set_ylabel('Magnitude of symbols after real LMS AEQ')
            axsAE_R.plot(abs(Real_LMS_2x2_rx[1]), linestyle='', marker='o', markersize='1', color='r', label='y2 mag.')
            axsAE_R.set_ylim(0,3)
            axsAE_R.legend()
    else: 
        Real_LMS_2x2_rx = Phase_Noise_compensated_rx      

    processed_rx = f.downsample(Real_LMS_2x2_rx, 2, NPol, toggle=p.toggle.toggle_RRC)

    if(p.toggle.toggle_PAS==True):
        if(p.Mod_param.NPol==1):
            _1 = np.zeros(len(source_symbols))
            _2 = np.zeros(len(source_symbols)*p.Mod_param.Modbits)
            source_symbols, processed_rx, _, _, original_bits, _,_ = f.align_symbols_1Pol(source_symbols, processed_rx, _1, _2, original_bits, p.Mod_param.Modbits)
        else:
            _1 = np.zeros((2,source_symbols.shape[1]))
            _2 = np.zeros((2,source_symbols.shape[1]*p.Mod_param.Modbits))

            source_symbols, processed_rx, _, _, original_bits, _,_ = f.align_symbols_2Pol(source_symbols, processed_rx, _1, _2, original_bits, p.Mod_param.Modbits)

        if(NPol==1):
            #Need to make sure that signal is an integer block length of signals
            #Then update number of blocks
            processed_rx, source_symbols, original_bits = pas.PAS_truncate(processed_rx, source_symbols, original_bits, p.PAS_param.N, p.Mod_param.NPol, p.PAS_param.k)
            p.PAS_param.blocks = len(processed_rx)//p.PAS_param.N
            _,_, p.PAS_param.sigma = f.estimate_snr(processed_rx, Modbits, source_symbols, p.toggle.toggle_PAS) #need to update this function for 1Pol
            demod_symbols, demod_bits, HI0, HQ0 = pas.PAS_decoder(processed_rx, Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.N, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)
            H = [HI0, HQ0]
        elif(NPol==2):
            if(p.lab_testing==True):
                processed_rx, source_symbols, original_bits = pas.PAS_truncate(processed_rx, source_symbols, original_bits, p.PAS_param.N, p.Mod_param.NPol, p.PAS_param.k)
            _,_, p.PAS_param.sigma = f.estimate_snr(processed_rx, Modbits, source_symbols, p.toggle.toggle_PAS)
            #Maybe check if estimating sigma correctly
            p.PAS_param.blocks = processed_rx.shape[1]//p.PAS_param.N

            demod_symbols0, demod_bits0, HI0, HQ0 = pas.PAS_decoder(processed_rx[0], Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.blocks, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)
            demod_symbols1, demod_bits1, HI1, HQ1 = pas.PAS_decoder(processed_rx[1], Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.blocks, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)
            demod_bits = np.array([demod_bits0,demod_bits1])
            demod_symbols = np.array([demod_symbols0,demod_symbols1])
            H = [[HI0, HQ0], [HI1, HQ1]]

    elif(p.toggle.toggle_DE==True):
        if(NPol==1):
            demod_bits = f.Differential_decode_symbols(processed_rx, Modbits)
        if(NPol==2):
            demod_bits0 = f.Differential_decode_symbols(processed_rx[0], Modbits)
            demod_bits1 = f.Differential_decode_symbols(processed_rx[1], Modbits)
            demod_bits = np.array([demod_bits0, demod_bits1]).flatten()
    else:
        if(NPol==1):
            demod_symbols = f.max_likelihood_decision(processed_rx, Modbits) #pass in Modbits which says QPSK, 16QAM or 64QAM
            demod_bits = f.decode_symbols(demod_symbols, Modbits, NPol) #pass in Modbits which says QPSK, 16QAM or 64QAM
        elif(NPol==2):
            demod_symbols0 = f.max_likelihood_decision(processed_rx[0], Modbits) #pass in Modbits which says QPSK, 16QAM or 64QAM
            demod_symbols1 = f.max_likelihood_decision(processed_rx[1], Modbits)
            demod_symbols = np.array([demod_symbols0, demod_symbols1])
            # SER only has meaning if Differential Encoding is NOT used 
            # Find erroneous symbol indexes
            demod_bits = f.decode_symbols(demod_symbols, Modbits, NPol) #pass in Modbits which says QPSK,16QAM or 64QAM
    if(p.toggle.toggle_PAS==False):
        H=[]
    return demod_bits, processed_rx, demod_symbols, source_symbols, original_bits, H