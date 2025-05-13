import numpy as np
import matplotlib.pyplot as plt
import functions as f
import PAS.PAS_architecture as pas
import DDPhaseRecoveryTesting as dd
import parameters as p

def rx_final(rx, target_signal):

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

    frequency_recovered_rx = np.array([frequency_recovered_rx[0]/np.sqrt(np.mean(np.abs(frequency_recovered_rx[0])**2)), frequency_recovered_rx[1]/np.sqrt(np.mean(np.abs(frequency_recovered_rx[1])**2))])

    if(p.toggle.toggle_adaptive_equalisation == True and NPol == 2):
        #***adaptive equalisation (2x2 MIMO, LMS with known data)***
        MIMO_LMS_2x2_rx = f.MIMO_LMS_AEQ(frequency_recovered_rx,target_signal,p.AE_param.mu,p.AE_param.NTaps)
        
        print('--------------------------------------')
        print('Adaptive Equalisation Parameters:')
        print('Number of taps:         ', p.AE_param.NTaps)
        print('μ:                      ', p.AE_param.mu)

        #downsampling done within adaptive equalisation
        figAE, axsAE = plt.subplots(1,1, figsize=(8,8))
        axsAE.plot(abs(MIMO_LMS_2x2_rx[0]), linestyle='', marker='o', markersize='1', color='b', label='y1 mag.')
        axsAE.set_ylabel('magnitude of symbols')
        axsAE.plot(abs(MIMO_LMS_2x2_rx[1]), linestyle='', marker='o', markersize='1', color='r', label='y2 mag.')
        axsAE.set_ylim(0,3)
        axsAE.legend()

    else:
        MIMO_LMS_2x2_rx = frequency_recovered_rx
    
    #Still 2sps
    #Normalise:
    MIMO_LMS_2x2_rx = np.array([MIMO_LMS_2x2_rx[0]/np.sqrt(np.mean(np.abs(MIMO_LMS_2x2_rx[0])**2)), MIMO_LMS_2x2_rx[1]/np.sqrt(np.mean(np.abs(MIMO_LMS_2x2_rx[1])**2))])

    if(p.toggle.toggle_BPS==True):
        print('--------------------------------------')
        print('BPS parameters:')
        print('Test Angles:      ', p.BPS_param.B)
        print('Averaging number: ', p.BPS_param.N)
        print('--------------------------------------')
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


    # if(p.toggle.toggle_adaptive_equalisation==True):
    #     Real_LMS_2x2_rx0 = f.MIMO_2x2_with_CPR(Phase_Noise_compensated_rx[0],target_signal[0],p.AE_param.mu,31)
    #     Real_LMS_2x2_rx1 = f.MIMO_2x2_with_CPR(Phase_Noise_compensated_rx[1],target_signal[1],p.AE_param.mu,31)
    #     Real_LMS_2x2_rx = np.array([Real_LMS_2x2_rx0,Real_LMS_2x2_rx1])

    # else:
    Real_LMS_2x2_rx = Phase_Noise_compensated_rx    

    processed_rx = f.downsample(Real_LMS_2x2_rx, 2, NPol, toggle=p.toggle.toggle_RRC)

    if(p.toggle.toggle_PAS==True):
        if(NPol==1):
            demod_symbols, demod_bits = pas.PAS_decoder(processed_rx, Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.blocks, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)
            
        elif(NPol==2):
            demod_symbols0, demod_bits0 = pas.PAS_decoder(processed_rx[0], Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.blocks, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)
            demod_symbols1, demod_bits1 = pas.PAS_decoder(processed_rx[1], Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.blocks, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)
            demod_bits = np.array([demod_bits0,demod_bits1])
            demod_symbols = np.array([demod_symbols0,demod_symbols1])

    elif(p.toggle.toggle_DE==True):
        if(NPol==1):
            demod_bits = f.Differential_decode_symbols(processed_rx, Modbits)
        if(NPol==2):
            demod_bits0 = f.Differential_decode_symbols(processed_rx[0], Modbits)
            demod_bits1 = f.Differential_decode_symbols(processed_rx[1], Modbits)
            demod_bits = np.array([demod_bits0, demod_bits1]).flatten()
    else:
        if(NPol==1):
            demod_symbols = f.max_likelihood_decision(processed_rx, Modbits) #pass in Modbits which says 16QAM or 64QAM
            demod_bits = f.decode_symbols(processed_rx, Modbits, NPol) #pass in Modbits which says 16QAM or 64QAM
        elif(NPol==2):
            demod_symbols0 = f.max_likelihood_decision(processed_rx[0], Modbits) #pass in Modbits which says 16QAM or 64QAM
            demod_symbols1 = f.max_likelihood_decision(processed_rx[1], Modbits)
            demod_symbols = np.array([demod_symbols0, demod_symbols1])
            # SER only has meaning if Differential Encoding is NOT used 
            # Find erroneous symbol indexes
            demod_bits = f.decode_symbols(demod_symbols, Modbits, NPol) #pass in Modbits which says 16QAM or 64QAM
    
    return demod_bits, processed_rx, demod_symbols

