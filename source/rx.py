import numpy as np
import matplotlib.pyplot as plt
import functions as f
import PAS.PAS_architecture as pas
import DDPhaseRecoveryTesting as dd
import parameters as p

def rx(rx, source_symbols):
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
    
    if(p.toggle.toggle_adaptive_equalisation == True and NPol == 2):
        if(p.AE_param.AE_type=="2x2"):
            if(Modbits==2):
                flag='CMA'
            else:   
                flag='CMA+RDE'
            print('--------------------------------------')
            print('Adaptive Equalisation Parameters:')
            print('Update used:            ', flag)
            print('Number of taps:         ', p.AE_param.NTaps)
            print('μ:                      ', p.AE_param.mu)
            print('2nd Filter starts at:   ', p.AE_param.N1)
            print('CMA to RDE switch at:   ', p.AE_param.N2)
            print('Samples discarded:      ', p.AE_param.Ndiscard)
            print('--------------------------------------')

            #2 sps in
            adaptive_eq_rx = f.adaptive_equalisation(CD_compensated_rx ,2, flag, p.AE_param.NTaps, p.AE_param.mu, True, p.AE_param.N1, p.AE_param.N2)
            #1sps out

            downsampled_CD_compensated_rx = f.downsample(CD_compensated_rx, 2, NPol, True)
            downsampled_rx = np.concatenate([downsampled_CD_compensated_rx[:,:p.AE_param.Ndiscard], adaptive_eq_rx[:, p.AE_param.Ndiscard:]], axis=1) #Discard first NOut symbols of adaptive equalisation
            
        elif(p.AE_param.AE_type=="4x4"):
            adaptive_eq_rx = f.AE_4x4(CD_compensated_rx,p.AE_param.mu,p.AE_param.NTaps, p.Mod_param.Modbits)
            downsampled_rx = f.downsample(adaptive_eq_rx, 2, NPol, True)

        #downsampling done within adaptive equalisation
        figAE, axsAE = plt.subplots(1,1, figsize=(8,8))
        axsAE.plot(abs(adaptive_eq_rx[0]), linestyle='', marker='o', markersize='1', color='b', label='y1 mag.')
        axsAE.set_ylabel('magnitude of symbols')
        axsAE.plot(abs(adaptive_eq_rx[1]), linestyle='', marker='o', markersize='1', color='r', label='y2 mag.')
        axsAE.set_ylim(0,3)
        if(p.AE_param.AE_type=="2x2"):
            axsAE.vlines(p.AE_param.N1, colors='purple', label='N1', ymin=0, ymax=3)
            axsAE.vlines(p.AE_param.N2, colors='green', label='N2', ymin=0, ymax=3)
        axsAE.vlines(p.AE_param.Ndiscard, colors='orange', label='Ndiscard', ymin=0, ymax=3)
        axsAE.legend()

    else:
        adaptive_eq_rx = CD_compensated_rx
        downsampled_rx = f.downsample(adaptive_eq_rx, 2, NPol, toggle=p.toggle.toggle_RRC) #Downsampled from 2 sps to 1 sps

    frequency_recovered = f.frequency_recovery(downsampled_rx, p.Mod_param.Rs, NPol, p.toggle.toggle_frequencyrecovery)
    
    if(p.toggle.toggle_BPS==True):
        print('--------------------------------------')
        print('BPS parameters:')
        print('Test Angles:      ', p.BPS_param.B)
        print('Averaging number: ', p.BPS_param.N)
        print('--------------------------------------')
        Phase_Noise_compensated_rx, thetaHat = f.BPS(frequency_recovered, Modbits, p.BPS_param.N, p.BPS_param.B, NPol, p.toggle.toggle_phasenoisecompensation)
        if(p.toggle.toggle_phasenoisecompensation):
            plt.figure()
            plt.plot(thetaHat, color='blue')

        # Testing real valued AEQ
        # if(p.Mod_param.Modbits==4): 
        #     real_adaptive_eq_rx0 = f.real_valued_2x2_AEQ(adaptive_eq_rx[0], p.AE_param.mu, p.AE_param.NTaps, source_symbols[0]) #testing for 16-QAM only
        #     real_adaptive_eq_rx1 = f.real_valued_2x2_AEQ(adaptive_eq_rx[1], p.AE_param.mu, p.AE_param.NTaps, source_symbols[1]) #testing for 16-QAM only
        #     Phase_Noise_compensated_rx = np.array([real_adaptive_eq_rx0,real_adaptive_eq_rx1])
        
    else:
        #Note DD algorithm currently only set up for NPol==1
        #Note this DD algorithm currently uses SNR per bit, which should be changed to per symbol
        frac = 0.05
        Phase_Noise_compensated_rx, thetaHat = dd.DD_phase_noise_compensation(downsampled_rx, p.RRC_param.sps, p.Mod_param.Rs, p.laser_param.Linewidth, Modbits, p.Mod_param.snr_db, frac, p.toggle.toggle_phasenoisecompensation)
    
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


    f.estimate_snr(Phase_Noise_compensated_rx, p.Mod_param.Modbits, source_symbols)
#   check that this is comparing the correct things

    if(p.toggle.toggle_PAS==True):
        if(NPol==1):
            demod_symbols, demod_bits = pas.PAS_decoder(Phase_Noise_compensated_rx, Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.blocks, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)
            
        elif(NPol==2):
            demod_symbols0, demod_bits0 = pas.PAS_decoder(Phase_Noise_compensated_rx[0], Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.blocks, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)
            demod_symbols1, demod_bits1 = pas.PAS_decoder(Phase_Noise_compensated_rx[1], Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.blocks, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)
            demod_bits = np.array([demod_bits0,demod_bits1])
            demod_symbols = np.array([demod_symbols0,demod_symbols1])

    elif(p.toggle.toggle_DE==True):
        if(NPol==1):
            demod_bits = f.Differential_decode_symbols(Phase_Noise_compensated_rx, Modbits)
        if(NPol==2):
            demod_bits0 = f.Differential_decode_symbols(Phase_Noise_compensated_rx[0], Modbits)
            demod_bits1 = f.Differential_decode_symbols(Phase_Noise_compensated_rx[1], Modbits)
            demod_bits = np.array([demod_bits0, demod_bits1]).flatten()
    else:
        if(NPol==1):
            demod_symbols = f.max_likelihood_decision(Phase_Noise_compensated_rx, Modbits) #pass in Modbits which says 16QAM or 64QAM
            demod_bits = f.decode_symbols(demod_symbols, Modbits, NPol) #pass in Modbits which says 16QAM or 64QAM
        elif(NPol==2):
            demod_symbols0 = f.max_likelihood_decision(Phase_Noise_compensated_rx[0], Modbits) #pass in Modbits which says 16QAM or 64QAM
            demod_symbols1 = f.max_likelihood_decision(Phase_Noise_compensated_rx[1], Modbits)
            demod_symbols = np.array([demod_symbols0, demod_symbols1])
            # SER only has meaning if Differential Encoding is NOT used 
            # Find erroneous symbol indexes
            demod_bits = f.decode_symbols(demod_symbols, Modbits, NPol) #pass in Modbits which says 16QAM or 64QAM
    
    return demod_bits, Phase_Noise_compensated_rx, demod_symbols

