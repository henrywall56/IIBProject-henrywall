import numpy as np
import matplotlib.pyplot as plt
import functions as f
import DDPhaseRecoveryTesting as dd
import performance_evaluation as p
from matplotlib.ticker import MaxNLocator
from scipy.fft import fft, ifft, fftshift, ifftshift

plt.rcParams['font.size'] = 12  # Change the font size
plt.rcParams['font.family'] = 'Times New Roman'  # Change the font family

"""
1) Generate random bits.
2) Generate QAM symbols from bits.
3) Upsampling by inserting zeros.
4) Pulse shaping with RRC. 
5) Add circular Gaussian noise.
6) Matched RRC filter.
7) Downsample.
8) Max Likelihood Decision on constellation.
9) Generate SER.
10) Decode symbols to bits.
11) Generate BER.

"""
def main():
    num_power = 16
    num_symbols = 2**num_power #Number of symbols in each polarisation
    Modbits = 2 #2 is QPSK, 4 is 16QAM, 6 is 64QAM

    NPol = 2 #Number of polarisations used
    
    #Generate RRC filter impulse response
    #base 20, 16, 0.1
    span = 20 #Span of filter
    sps = 16 #Samples per symbol
    rolloff = 0.1 #Roll-off of RRC

    #IQ Modulator Parameters (Mach-Zehnder Modulators Parameters):
    Vpi = 1.05
    Bias = -1*Vpi
    MinExc = -1.5*Vpi
    MaxExc = -0.5*Vpi

    #DD Phase Recovery Parameters:
    frac = 0.05

    #BPS Phase Recovery Parameters:
    N=20 #Number of symbols to average over
    #"N = 6,..,10 will be a fairly good choice" - Pfau paper

    if(Modbits==2): #Values given by Pfau paper
        B=64 #Number of trial angles
        plotsize = 2
        modulation_format='QPSK'
    elif(Modbits==4):
        B=32 #Number of trial angles
        plotsize = 2
        modulation_format='16-QAM'
    elif(Modbits==6):
        B=64 #Number of trial angles
        plotsize = 1.5
        modulation_format='64-QAM'

    Rs = 400e9 #Rs symbol rate symbols/second (Baud)

    #Chromatic Dispersion Parameters
    D = 17#Dispersion parameter in (ps/(nm x km))
    Clambda = 1550 / (10**9) #Central lambda in (m)
    L = 1000*(10**3)#Fibre length in (m)
    #NOTE: need FFT size of ≈ 4 * number of symbols dispersion is spread over (@2 sps) = 4 * D*∆lambda*L * (2*Rs)= 4 * D*(Clambda/f * ∆f)*L * (2*Rs) = 4 * D*(((Clambda**2)/c)*Rs)*L * (2*Rs)
    #NOTE: need NOverlap size of ≈ number of symbols dispersion is spread over (@2 sps) = 4 * D*∆lambda*L * (2*Rs)= 4 * D*(Clambda/f * ∆f)*L * (2*Rs) = 4 * D*(((Clambda**2)/c)*Rs)*L * (2*Rs)
    ideal_NOverlap = D*(1e-12/(1e-9*1e3)) * (Clambda**2/299792458)*Rs*L*2*Rs
    ideal_NFFT = 4*ideal_NOverlap
    NFFTpower = int(np.ceil(np.log2(ideal_NFFT))) 
    NFFT=2**NFFTpower #Adjusted to minimise complexity 
    NOverlap = int(ideal_NOverlap) #Given by minimum equaliser length N_CD : pg 113 CDOT graph?
    
    NFFT=NFFT*2 #temporary, since have issues with CD_compensation at higher Rs

    snr_begin = 9

    toggle_RRC = True #toggle RRC pulse shaping
    toggle_AWGNnoise = True
    toggle_phasenoise = True #toggle phase noise
    toggle_phasenoisecompensation = True #toggle phase noise compensation
    toggle_plotuncompensatedphase = False #toggle plotting constellation before phase compensation. Note this is before downsampling if using RRC pulseshaping.
    toggle_ploterrorindexes = False #toggle plotting error indexes on phase plot
    toggle_BPS = True #toggle blind phase searching algorithm: True is BPS, False is DD Phase compensation.
    toggle_DE = False #toggle Differential Encoding
    toggle_frequencyrecovery = False #toggle Frequency Recovery
    toggle_CD = True #Toggle Chromatic Dispersion
    toggle_NL = True
    toggle_CD_compensation = True #Toggle Chromatic Dispersion Compensation
    toggle_AIR = True
    toggle_adaptive_equalisation = True
    AIR_type = 'MI' #'MI' or 'GMI'
    
    if(toggle_RRC==False):
        sps=1               #overwrite sps if no RRC

    #Phase noise parameters
    #There is a max Linewidth * T = maxDvT
    #T = 1/(Rs*sps)
    Linewidth = 100*10**3 #Linewidth of laser in Hz
    maxDvT = Linewidth/(Rs*sps)
    laser_power = 0 #Total laser power in dBm

    print('Symbol Rate:         ', Rs/1e9, 'GBaud')
    print('Bit Rate:            ', Modbits*Rs/1e9, 'GBit/s')
    print('Modulation Format:   ', modulation_format)
    print('∆νT:                 ', maxDvT)
    print('Fibre Length:        ',L/1e3, 'km')
    print('Laser Linewidth:     ', Linewidth/1e3,'kHz')
    print('Laser Power:         ',laser_power, 'dBm')
    print('Polarisations:       ', NPol)
    print('No. of Symbols:      ', num_symbols)
    
    original_bits = f.generate_original_bits(num_symbols, Modbits, NPol) #NPol-dimensional array

    symbols = f.generate_symbols(original_bits, Modbits, NPol, toggle_DE) #NPol-dimensional array

    RRCimpulse , t1 = f.RRC(span, rolloff, sps)

    #Pulse shaping with RRC filter
    pulse_shaped_symbols = f.pulseshaping(symbols, sps, RRCimpulse, NPol, toggle = toggle_RRC) #if toggle False, this function just returns symbol
    print('RRC Parameters:')
    print('RRC Rolloff: ', rolloff)
    print('RRC span: ', span)

    snr_db = np.arange(snr_begin,snr_begin+10,1)
    #snr_db = np.array([10])

    fig1V, axs1V = plt.subplots(2, 2, figsize=(8, 8))   #Phase noise compensated constellation
    axs1V = axs1V.flatten()  # Flatten the array for easy indexing
    if(NPol==2):
        fig1H, axs1H = plt.subplots(2, 2, figsize=(8, 8))   #Phase noise compensated constellation
        axs1H = axs1H.flatten()  # Flatten the array for easy indexing

    if(toggle_plotuncompensatedphase==True): #Plots for uncompensated phase constellation
        fig3V, axs3V = plt.subplots(2, 2, figsize=(8, 8))  
        axs3V= axs3V.flatten()  # Flatten the array for easy indexing
        if(NPol==2):
            fig3H, axs3H = plt.subplots(2, 2, figsize=(8, 8))  
            axs3H= axs3H.flatten()  # Flatten the array for easy indexing

    fig4, axs4 = plt.subplots(2, 2, figsize=(8, 8))  #Phase noise plot
    axs4= axs4.flatten()  # Flatten the array for easy indexing

    if(toggle_adaptive_equalisation and NPol==2):
            fig5, axs5 = plt.subplots(2, 2, figsize=(8, 8))  #Adaptive equalisation magnitude plots
            axs5= axs5.flatten()  # Flatten the array for easy indexing

    BER = np.empty(len(snr_db))
    SER = np.empty(len(snr_db))
    AIR = np.empty(len(snr_db))
   
    #Phase_Noise_rx, theta = f.add_phase_noise(tx, num_symbols, sps, Rs, Linewidth, toggle=toggle_phasenoise) #Phase noise added at transmitting laser

    Elaser, theta = f.Laser(laser_power, Linewidth, sps, Rs, num_symbols, NPol, toggle_phasenoise) #Laser phase noise

    Laser_Eoutput = f.IQModulator(pulse_shaped_symbols, Elaser, Vpi, Bias, MaxExc, MinExc, NPol) #laser output E field with phase noise

    for i, snr_dbi in enumerate(snr_db):
        if(snr_dbi!=9):
            continue
        
        print(f'Processing SNR {snr_dbi}')
        Gaussian_noise_signal = f.add_noise(Laser_Eoutput, snr_dbi, sps, Modbits, NPol, toggle_AWGNnoise) 

        if(toggle_NL==False):
            
            CD_NL_signal = f.add_chromatic_dispersion(Gaussian_noise_signal, sps, Rs, D, Clambda, L, NPol, toggle_CD)
    
        else:   
            CD_NL_signal = f.SSFM(Gaussian_noise_signal, Rs, D, Clambda, L, 1, NPol)


        rx = CD_NL_signal #Skipping receiver front end for now

        #rx = f.mix_polarisation_signals(CD_NL_signal, 45)

        filtered_signal = f.matched_filter(rx, RRCimpulse, NPol, toggle=toggle_RRC) #if toggle is False, this function returns input
        
        ADC = f.downsample(filtered_signal, sps//2, NPol, toggle=toggle_RRC) #Simulate ADC downsampling to 2 sps

        #Chromatic Dispersion Compensation
        if(toggle_RRC==True):
            spsCD = 2
        else:
            spsCD=1

        CD_compensated_rx = f.CD_compensation(ADC, D, L, Clambda, Rs, NPol, spsCD, NFFT, NOverlap, toggle_CD_compensation)
        
        if(toggle_adaptive_equalisation == True and NPol == 2):
            NTaps = 4 
            mu = 5e-4 
            N1=2000
            N2=5000
            Ndiscard=12000
            if(Modbits==2):
                AE_Type='CMA'
            else:
                AE_Type='CMA+RDE'
            print('Adaptive Equalisation Parameters:')
            print('Update used:            ', AE_Type)
            print('Number of taps:         ', NTaps)
            print('μ:                      ', mu)
            print('2nd Filter starts at:   ', N1)
            print('CMA to RDE switch at:   ', N2)
            print('Samples discarded:      ', Ndiscard)
            adaptive_eq_rx = f.adaptive_equalisation(CD_compensated_rx ,2, AE_Type, NTaps, mu, True, N1, N2)
            downsampled_CD_compensated_rx = f.downsample(CD_compensated_rx, 2, NPol, True)
            downsampled_rx = np.concatenate([downsampled_CD_compensated_rx[:,:Ndiscard], adaptive_eq_rx[:, Ndiscard:]], axis=1) #Discard first NOut symbols of adaptive equalisation

            #downsampling done within adaptive equalisation

        else:
            adaptive_eq_rx = CD_compensated_rx
            downsampled_rx = f.downsample(adaptive_eq_rx, 2, NPol, toggle=toggle_RRC) #Downsampled from 2 sps to 1 sps

        frequency_recovered = f.frequency_recovery(downsampled_rx, Rs, NPol, toggle_frequencyrecovery)
        
        if(toggle_BPS==True):
            print('BPS parameters:')
            print('Test Angles:      ', B)
            print('Averaging number: ',N)
            Phase_Noise_compensated_rx, thetaHat = f.BPS(frequency_recovered, Modbits, N, B, NPol, toggle_phasenoisecompensation)
        else:
            #Note DD algorithm currently only set up for NPol==1
            #Note this currently uses SNR per bit, which should be changed to per symbol
            Phase_Noise_compensated_rx, thetaHat = dd.DD_phase_noise_compensation(downsampled_rx, sps, Rs, Linewidth, Modbits, snr_dbi, frac, toggle_phasenoisecompensation)

        if(toggle_DE==True):
            if(NPol==1):
                demod_bits = f.Differential_decode_symbols(Phase_Noise_compensated_rx, Modbits)
                #Plot erroneous symbols as the symbols that correspond to bit errors
                erroneous_bit_indexes = np.where(original_bits!=demod_bits)[0]
            if(NPol==2):
                demod_bits0 = f.Differential_decode_symbols(Phase_Noise_compensated_rx[0], Modbits)
                demod_bits1 = f.Differential_decode_symbols(Phase_Noise_compensated_rx[1], Modbits)
                demod_bits = np.array([demod_bits0, demod_bits1])

                erroneous_bit_indexesV = np.where(original_bits[0]!=demod_bits[0])[0] #Vertical Polarisation
                erroneous_bit_indexesH = np.where(original_bits[1]!=demod_bits[1])[0] #Horizontal Polarisation

        else:
            if(NPol==1):
                demod_symbols = f.max_likelihood_decision(Phase_Noise_compensated_rx, Modbits) #pass in Modbits which says 16QAM or 64QAM
                # SER only has meaning if Differential Encoding is NOT used 
                # Find erroneous symbol indexes
                erroneous_indexes = np.where(symbols != demod_symbols)[0]
                SER[i] = np.mean(symbols != demod_symbols)

                demod_bits = f.decode_symbols(demod_symbols, Modbits, NPol) #pass in Modbits which says 16QAM or 64QAM
            elif(NPol==2):
                demod_symbols0 = f.max_likelihood_decision(Phase_Noise_compensated_rx[0], Modbits) #pass in Modbits which says 16QAM or 64QAM
                demod_symbols1 = f.max_likelihood_decision(Phase_Noise_compensated_rx[1], Modbits)
                demod_symbols = np.array([demod_symbols0, demod_symbols1])
                # SER only has meaning if Differential Encoding is NOT used 
                # Find erroneous symbol indexes
                
                erroneous_indexesV = np.where(demod_symbols[0]!=symbols[0])[0] #Vertical Polarisation
                erroneous_indexesH = np.where(demod_symbols[1]!=symbols[1])[0] #Horizontal Polarisation

                SER[i] = (len(erroneous_indexesV)+len(erroneous_indexesH))/(NPol*num_symbols)

                demod_bits = f.decode_symbols(demod_symbols, Modbits, NPol) #pass in Modbits which says 16QAM or 64QAM

        if(NPol==1):
            BER[i] = np.mean(original_bits != demod_bits)
            if(toggle_AIR==True):
                if(AIR_type=='GMI'):
                    AIR[i] = p.AIR_SDBW(symbols, original_bits, Phase_Noise_compensated_rx, Modbits)
                elif(AIR_type=='MI'):
                    AIR[i] = p.AIR_SDSW(symbols, Phase_Noise_compensated_rx, Modbits)
        elif(NPol==2):
            #BER[i] = (np.sum(original_bits[0] != demod_bits[0])+np.sum(original_bits[1] != demod_bits[1]))/(NPol*num_symbols*Modbits)
            BER[i] = np.sum(original_bits[0] != demod_bits[0])/(num_symbols*Modbits)
            if(AIR_type=='GMI'):
                AIR[i] = (p.AIR_SDBW(symbols[0], original_bits[0], Phase_Noise_compensated_rx[0], Modbits) + p.AIR_SDBW(symbols[1], original_bits[1], Phase_Noise_compensated_rx[1], Modbits))/2
            elif(AIR_type=='MI'):
                AIR[i] = 0.5*(p.AIR_SDSW(symbols[0], Phase_Noise_compensated_rx[0], Modbits)+p.AIR_SDSW(symbols[1], Phase_Noise_compensated_rx[1], Modbits))

        # Plot downsampled received constellation, including different colour for erroneous results
        #Only plot some of the results:
        if(i%3==0):
            if(NPol==1):
                f.plot_constellation(axs1V[i//3], Phase_Noise_compensated_rx, title=f'Phase Noise Compensated Constellation \n SNR = {snr_dbi}dB', lim=plotsize)
                
                if(toggle_plotuncompensatedphase==True):
                    f.plot_constellation(axs3V[i//3], filtered_signal[0::sps], title=f'No Derotation at SNR= {snr_dbi}dB ', lim=plotsize)

            elif(NPol==2):
                f.plot_constellation(axs1V[i//3], Phase_Noise_compensated_rx[0], title=f'Phase Noise Compensated Constellation \n SNR = {snr_dbi}dB (V)', lim=plotsize)
                f.plot_constellation(axs1H[i//3], Phase_Noise_compensated_rx[1], title=f'Phase Noise Compensated Constellation \n SNR = {snr_dbi}dB (H)', lim=plotsize)
                
                if(toggle_plotuncompensatedphase==True):
                    f.plot_constellation(axs3V[i//3], filtered_signal[0][0::sps], title=f'No Derotation at SNR= {snr_dbi}dB (V)', lim=plotsize)
                    f.plot_constellation(axs3H[i//3], filtered_signal[0][0::sps], title=f'No Derotation at SNR= {snr_dbi}dB (H)', lim=plotsize)

            axs4[i//3].plot(1e9*np.arange(num_symbols)/(Rs*sps), theta[0::sps], label='Phase')
            #Note plotting theta before frequency recovery here
            axs4[i//3].set_title(f'Phase Noise at SNR = {snr_dbi}dB \n ∆νT = {maxDvT}')
            axs4[i//3].grid(True)
            axs4[i//3].set_xlabel('Time (ns)')
            axs4[i//3].set_ylabel('Phase (rad)')

            if(NPol==1):
                axs4[i//3].plot(1e9*np.arange(num_symbols)/(Rs*sps), thetaHat, color='red', label='Phase Estimate')
            elif(NPol==2):
                axs4[i//3].plot(1e9*np.arange(num_symbols)/(Rs*sps), thetaHat[:,0], color='red', label='Phase Estimate (V)')
                axs4[i//3].plot(1e9*np.arange(num_symbols)/(Rs*sps), thetaHat[:,1], color='orange', label='Phase Estimate (H)')
            
            if(toggle_DE==False):
                # Highlight erroneous symbols
                if(NPol==1):
                    axs1V[i//3].scatter(Phase_Noise_compensated_rx[erroneous_indexes].real, Phase_Noise_compensated_rx[erroneous_indexes].imag, color='red', label='Errors', alpha=0.5)
                    axs4[i//3].vlines(1e9*erroneous_indexes/(Rs*sps), ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes')
                if(NPol==2):
                    axs1V[i//3].scatter(Phase_Noise_compensated_rx[0][erroneous_indexesV].real, Phase_Noise_compensated_rx[0][erroneous_indexesV].imag, color='red', label='Errors', alpha=0.5)
                    axs1H[i//3].scatter(Phase_Noise_compensated_rx[1][erroneous_indexesH].real, Phase_Noise_compensated_rx[1][erroneous_indexesH].imag, color='red', label='Errors', alpha=0.5)
                    axs4[i//3].vlines(1e9*erroneous_indexesV/(Rs*sps), ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes (V)')
                    axs4[i//3].vlines(1e9*erroneous_indexesH/(Rs*sps), ymin=-0.05, ymax=0.05, colors='purple', alpha=0.2, label='Erroneous Indexes (H)')
            else:
                
                if(NPol==1):
                    axs1V[i//3].scatter(Phase_Noise_compensated_rx[erroneous_bit_indexes//Modbits].real, Phase_Noise_compensated_rx[erroneous_bit_indexes//Modbits].imag, color='red', label='Errors', alpha=0.5)
                    axs4[i//3].vlines(erroneous_bit_indexes//Modbits, ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes')
                if(NPol==2):
                    axs1V[i//3].scatter(Phase_Noise_compensated_rx[0][erroneous_bit_indexesV//Modbits].real, Phase_Noise_compensated_rx[0][erroneous_bit_indexesV//Modbits].imag, color='red', label='Errors', alpha=0.5)
                    axs1H[i//3].scatter(Phase_Noise_compensated_rx[1][erroneous_bit_indexesH//Modbits].real, Phase_Noise_compensated_rx[1][erroneous_bit_indexesH//Modbits].imag, color='red', label='Errors', alpha=0.5)
                    axs4[i//3].vlines(1e9*(erroneous_bit_indexesV//Modbits)/(Rs*sps), ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes (V)')
                    axs4[i//3].vlines(1e9*(erroneous_bit_indexesH//Modbits)/(Rs*sps), ymin=-0.05, ymax=0.05, colors='purple', alpha=0.2, label='Erroneous Indexes (H)')
                #Highlight rough location of symbol errors
            axs4[i//3].legend(loc='lower left')

            if(toggle_adaptive_equalisation == True and NPol == 2):
                axs5[i//3].plot(abs(adaptive_eq_rx[0]), linestyle='', marker='o', markersize='1', color='b', label='y1 mag.')
                axs5[i//3].set_ylabel('magnitude of symbols')
                axs5[i//3].plot(abs(adaptive_eq_rx[1]), linestyle='', marker='o', markersize='1', color='r', label='y2 mag.')
                axs5[i//3].set_ylim(0,3)
                axs5[i//3].vlines(N1, colors='purple', label='N1', ymin=0, ymax=3)
                axs5[i//3].vlines(N2, colors='green', label='N2', ymin=0, ymax=3)
                axs5[i//3].vlines(Ndiscard, colors='orange', label='Ndiscard', ymin=0, ymax=3)
                axs5[i//3].legend()

            autocorrV = np.real(ifft(np.conjugate(fft(symbols[0]))*fft(Phase_Noise_compensated_rx[0])))
            plt.figure()
            plt.plot(autocorrV)
            plt.title('V autocorrelation')
            print('Max V autocorrelation at index', np.argmax(autocorrV), 'or', np.argmax(autocorrV)-num_symbols)

            autocorrH = np.real(ifft(np.conjugate(fft(symbols[1]))*fft(Phase_Noise_compensated_rx[1])))
            plt.figure()
            plt.plot(autocorrH)
            plt.title('H autocorrelation')
            print('Max H autocorrelation at index', np.argmax(autocorrH), 'or', np.argmax(autocorrH)-num_symbols)
            
            plt.figure()
            plt.title('Phase of last 10 symbols, H Polarisation')
            plt.plot(np.angle(symbols[1,-10:]), label='Sent phase')
            plt.plot(np.angle(Phase_Noise_compensated_rx[1,-10:]), label='received phase')
            plt.legend()
            

            plt.figure()
            plt.title('Phase of last 10 symbols, V Polarisation')
            plt.plot(np.angle(symbols[0,-10:]), label='received phase')
            plt.plot(np.angle(Phase_Noise_compensated_rx[0,-10:]), label='received phase')
            plt.legend()
        
    plt.tight_layout()
    plt.show()
    if(toggle_DE==True):
        #SER has no meaning for Differential Encoding
        # Plot BER
        plt.plot()
        plt.semilogy(snr_db, BER, marker='o')
        plt.xlabel('SNR per bit/ dB')
        plt.ylabel('BER')
        plt.title('Bit Error Rate (BER)')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    else:
        fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))  
        axs2 = axs2.flatten()  # Flatten the array for easy indexing   

        # Plot BER
        axs2[1].plot()
        axs2[1].semilogy(snr_db, BER, marker='o')
        axs2[1].set_xlabel('SNR per symbol/ dB')
        axs2[1].set_ylabel('BER')
        axs2[1].set_title('Bit Error Rate (BER)')
        axs2[1].grid(True)   
        axs2[1].xaxis.set_major_locator(MaxNLocator(integer=True))     
        # Plot SER
        axs2[0].semilogy(snr_db, SER, marker='o')
        axs2[0].set_xlabel('SNR per symbol/ dB')
        axs2[0].set_ylabel('SER')
        axs2[0].set_title('Symbol Error Rate (SER)')
        axs2[0].grid(True)
        axs2[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    if(toggle_AIR==True):
        #Shannon limit 
        snr_dbLin = 10**(snr_db/10)
        shannon = np.log2(1+snr_dbLin)
        if(AIR_type == 'GMI'):
            AIR_theoretical = p.AIR_SDBW_theoretical(snr_db, Modbits)
            plt.figure()
            M = int(2**Modbits)
            plt.title(f"SD-BW AIRs with {M}-QAM")
            plt.xlabel("SNR (dB)")
            plt.ylabel("GMI (bits/symbols)")
            plt.plot(snr_db, AIR, marker='o', color='b', label='Emprirical AIR')
            plt.plot(snr_db, AIR_theoretical, marker='x', color='r', label='Theoretical AIR')
            plt.plot(snr_db, shannon, color='g', label='Shannon Limit')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc='lower left')
            plt.ylim(0, max(AIR_theoretical[-1],max(AIR))+0.5)
        elif(AIR_type=='MI'):
            AIR_theoretical = p. AIR_SDSW_theoretical(snr_db, Modbits)
            plt.figure()
            M = int(2**Modbits)
            plt.title(f"SD-SW AIRs with {M}-QAM")
            plt.xlabel("SNR (dB)")
            plt.ylabel("MI (bits/symbols)")
            plt.plot(snr_db, AIR, marker='o', color='b', label='Emprirical AIR')
            plt.plot(snr_db, AIR_theoretical, marker='x', color='r', label='Theoretical AIR')
            plt.plot(snr_db, shannon, color='g', label='Shannon Limit')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc='lower left')
            plt.ylim(0, max(AIR_theoretical[-1],max(AIR))+0.5)

    plt.tight_layout()
    plt.show()

    print("BER", BER)
    """
    #For Testing:
    # Create a figure for constellation plots
    fig0, axs0 = plt.subplots(3, 2, figsize=(15, 10))  
    axs0 = axs0.flatten()  # Flatten the array for easy indexing
    # Plot pre-shaped constellation
    f.plot_constellation(axs0[0], symbols, title=f'Pre-shaped Constellation', lim=2)
    # Plot pulse shaped constellation
    f.plot_constellation(axs0[1], tx, title=f'Transmitted Constellation', lim=2)    
    # Plot received constellation
    f.plot_constellation(axs0[2], rx, title=f'Received Constellation', lim=2)
    # Plot filtered received constellation
    f.plot_constellation(axs0[3], filtered_signal, title=f'Filtered Constellation', lim=2)   
    # Plot downsampled received constellation
    f.plot_constellation(axs0[4], downsampled_rx, title=f'Downsampled Constellation', lim=2)
    plt.tight_layout()
    plt.show()
    """

if __name__ == "__main__":
    main()