import numpy as np
import matplotlib.pyplot as plt
import functions as f
import PAS.PAS_architecture as pas
import DDPhaseRecoveryTesting as dd
import parameters as p

def rx(rx):
    NPol = p.Mod_param.NPol
    Modbits = p.Mod_param.Modbits

    filtered_signal = f.matched_filter(rx, p.RRC_param.RRCimpulse, NPol, toggle=p.toggle.toggle_RRC) #if toggle is False, this function returns input
        
    ADC = f.downsample(filtered_signal, p.RRC_param.sps//2, NPol, toggle=p.toggle.toggle_RRC) #Simulate ADC downsampling to 2 sps

    #Chromatic Dispersion Compensation
    if(p.toggle.toggle_RRC==True):
        spsCD = 2
    else:
        spsCD=1

    CD_compensated_rx = f.CD_compensation(ADC, p.fibre_param.D, p.fibre_param.L, p.fibre_param.Clambda, p.Mod_param.Rs, NPol, spsCD, p.CD_param.NFFT, p.CD_param.NOverlap, p.toggle.toggle_CD_compensation)
    
    if(p.toggle.toggle_adaptive_equalisation == True and NPol == 2):
        if(Modbits==2):
            AE_Type='CMA'
        else:
            AE_Type='CMA+RDE'
        print('--------------------------------------')
        print('Adaptive Equalisation Parameters:')
        print('Update used:            ', AE_Type)
        print('Number of taps:         ', p.AE_param.NTaps)
        print('μ:                      ', p.AE_param.mu)
        print('2nd Filter starts at:   ', p.AE_param.N1)
        print('CMA to RDE switch at:   ', p.AE_param.N2)
        print('Samples discarded:      ', p.AE_param.Ndiscard)
        print('--------------------------------------')
        adaptive_eq_rx = f.adaptive_equalisation(CD_compensated_rx ,2, AE_Type, p.AE_param.NTaps, p.AE_param.mu, True, p.AE_param.N1, p.AE_param.N2)
        downsampled_CD_compensated_rx = f.downsample(CD_compensated_rx, 2, NPol, True)
        downsampled_rx = np.concatenate([downsampled_CD_compensated_rx[:,:p.AE_param.Ndiscard], adaptive_eq_rx[:, p.AE_param.Ndiscard:]], axis=1) #Discard first NOut symbols of adaptive equalisation

        #downsampling done within adaptive equalisation

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
    else:
        #Note DD algorithm currently only set up for NPol==1
        #Note this currently uses SNR per bit, which should be changed to per symbol
        frac = 0.05
        Phase_Noise_compensated_rx, thetaHat = dd.DD_phase_noise_compensation(downsampled_rx, p.RRC_param.sps, p.Mod_param.Rs, p.laser_param.Linewidth, Modbits, p.Mod_param.snrb_db, frac, p.toggle.toggle_phasenoisecompensation)
    
    if(p.toggle.toggle_PAS==True):
        if(NPol==1):
            demod_symbols, demod_bits = pas.PAS_decoder(Phase_Noise_compensated_rx, Modbits, p.PAS_param.λ, p.PAS_param.sigma, p.PAS_param.blocks, p.PAS_param.LDPC_encoder, p.PAS_param.k, p.PAS_param.C, p.PAS_param.PAS_normalisation)


    elif(p.toggle.toggle_DE==True):
        if(NPol==1):
            demod_bits = f.Differential_decode_symbols(Phase_Noise_compensated_rx, Modbits)
        if(NPol==2):
            demod_bits0 = f.Differential_decode_symbols(Phase_Noise_compensated_rx[0], Modbits)
            demod_bits1 = f.Differential_decode_symbols(Phase_Noise_compensated_rx[1], Modbits)
            demod_bits = np.array([demod_bits0, demod_bits1])
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
    
    return demod_bits, demod_symbols, Phase_Noise_compensated_rx
