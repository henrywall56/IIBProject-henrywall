import numpy as np
import functions as f
import matplotlib.pyplot as plt
import parameters as p
   
def channel(pulse_shaped_symbols):
    #mod_param is a Modulation_param class type
    #snr_db is the snr per symbol
    #sps is samples per symbol
    #IQ_Mod_param is IQ_Mod_param class type
    #laser_param is a laser_param class type
    #fibre_param is a fibre_param class type
    #t is a toggle class type

    Modbits = p.Mod_param.Modbits
    Rs = p.Mod_param.Rs
    NPol = p.Mod_param.NPol
    snr_db = p.fibre_param.snr_db #per symbol
    sps = p.RRC_param.sps

    #IQ Modulator Parameters (Mach-Zehnder Modulators Parameters):
    Vpi = p.IQ_Mod_param.Vpi
    Bias = p.IQ_Mod_param.Bias
    MinExc = p.IQ_Mod_param.MinExc
    MaxExc = p.IQ_Mod_param.MaxExc

    #Chromatic Dispersion Parameters
    D = p.fibre_param.D
    Clambda = p.fibre_param.Clambda
    L = p.fibre_param.L

    #Phase noise parameters
    #There is a max Linewidth * T = maxDvT
    #T = 1/(Rs*sps)
    Linewidth = p.laser_param.Linewidth #Linewidth of laser in Hz
    maxDvT = p.laser_param.maxDvT
    laser_power = p.laser_param.laser_power #Total laser power in dBm

    print('∆νT:                 ', maxDvT)
    print('Laser Linewidth:     ', Linewidth/1e3,'kHz')
    print('Laser Power:         ',laser_power, 'dBm')

    #Phase_Noise_rx, theta = f.add_phase_noise(tx, num_symbols, sps, Rs, Linewidth, toggle=t.toggle_phasenoise) #Phase noise added at transmitting laser

    Elaser, p.laser_param.theta = f.Laser(laser_power, Linewidth, sps, Rs, p.Mod_param.num_symbols, NPol, p.toggle.toggle_phasenoise) #Laser phase noise

    Laser_Eoutput = f.IQModulator(pulse_shaped_symbols, Elaser, Vpi, Bias, MaxExc, MinExc, NPol) #laser output E field with phase noise
    if(NPol==1):
        laser_norm = np.sum(abs(Laser_Eoutput)**2)/len(Laser_Eoutput)
    else:
        laser_norm = np.sum(abs(Laser_Eoutput)**2)/(2*Laser_Eoutput.shape[1])
    Laser_Eoutput = Laser_Eoutput/np.sqrt(laser_norm) #normal
        
    print(f'Processing SNR {snr_db}')

    Gaussian_noise_signal, p.PAS_param.sigma = f.add_noise(Laser_Eoutput, snr_db, sps, Modbits, NPol, p.toggle.toggle_AWGNnoise) 
    
    if(p.toggle.toggle_NL==False):
        
        CD_NL_signal = f.add_chromatic_dispersion(Gaussian_noise_signal, p.RRC_param.sps, Rs, D, Clambda, L, NPol, p.toggle.toggle_CD)

    else:   
        CD_NL_signal = f.SSFM(Gaussian_noise_signal, Rs, D, Clambda, L, 1, NPol)

    rx = CD_NL_signal #Skipping receiver front end for now

    #rx = f.mix_polarisation_signals(CD_NL_signal, 45)

    return rx


