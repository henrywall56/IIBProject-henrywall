import numpy as np
import matplotlib.pyplot as plt
import functions as f
import DDPhaseRecoveryTesting as dd
import performance_evaluation as p
from matplotlib.ticker import MaxNLocator

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
    num_symbols = 2000 #Number of symbols in each polarisation
    
    Modbits = 2 #2 is QPSK, 4 is 16QAM, 6 is 64QAM

    NPol = 2 #Number of polarisations used

    #Phase noise parameters
    maxDvT = 1/(10**6) #There is a max Linewidth * T = maxDvT where T = 1/(sps*Rs)
    Linewidth = 10**5 #Linewidth of laser in Hz
    laser_power = 0 #Total laser power in dBm
    
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
    N=6 #Number of symbols to average over
    #"N = 6,..,10 will be a fairly good choice" - Pfau paper

    if(Modbits==2): #Values given by Pfau paper
        B=32 #Number of trial angles
        plotsize = 2
    elif(Modbits==4):
        B=32 #Number of trial angles
        plotsize = 2
    elif(Modbits==6):
        B=64 #Number of trial angles
        plotsize = 1.5

    #Overwriting for quick testing:
    #B=20

    #Chromatic Dispersion Parameters
    D = 17#Dispersion parameter in (ps/(nm x km))
    Clambda = 1550 / (10**9) #Central lambda in (m)
    L = 1000*(10**3)#Fibre length in (m)
    NFFT = 128 #Adjusted to minimise complexity
    NOverlap = 10 #Given by minimum equaliser length N_CD : pg 113 CDOT graph?

    snr_begin = 0

    toggle_RRC = True #toggle RRC pulse shaping
    toggle_AWGNnoise = True
    toggle_phasenoise = True #toggle phase noise
    toggle_phasenoisecompensation = True #toggle phase noise compensation
    toggle_plotuncompensatedphase = False #toggle plotting constellation before phase compensation. Note this is before downsampling if using RRC pulseshaping.
    toggle_ploterrorindexes = True #toggle plotting error indexes on phase plot
    toggle_BPS = True #toggle blind phase searching algorithm: True is BPS, False is DD Phase compensation.
    toggle_DE = False #toggle Differential Encoding
    toggle_frequencyrecovery = False #toggle Frequency Recovery
    toggle_CD = True #Toggle Chromatic Dispersion
    toggle_CD_compensation = True #Toggle Chromatic Dispersion Compensation
    toggle_AIR = True
    AIR_type = 'MI' #'MI' or 'GMI'
    

    if(toggle_RRC==False):
        sps=1               #overwrite sps if no RRC

    #There is a max Linewidth * T = maxDvT
    #T = 1/(Rs*sps)
    Rs = Linewidth/(sps*maxDvT) #Rs symbol rate symbols/second
    
    original_bits = f.generate_original_bits(num_symbols, Modbits, NPol) #NPol-dimensional array

    symbols = f.generate_symbols(original_bits, Modbits, NPol, toggle_DE) #NPol-dimensional array

    RRCimpulse , t1 = f.RRC(span, rolloff, sps)

    #Pulse shaping with RRC filter
    pulse_shaped_symbols = f.pulseshaping(symbols, sps, RRCimpulse, NPol, toggle = toggle_RRC) #if toggle False, this function just returns symbols

    snr_db = np.arange(snr_begin,snr_begin+20,2)

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

    BER = np.empty(len(snr_db))
    SER = np.empty(len(snr_db))
    AIR = np.empty(len(snr_db))
   
    #Phase_Noise_rx, theta = f.add_phase_noise(tx, num_symbols, sps, Rs, Linewidth, toggle=toggle_phasenoise) #Phase noise added at transmitting laser

    Elaser, theta = f.Laser(laser_power, Linewidth, sps, Rs, num_symbols, NPol, toggle_phasenoise) #Laser phase noise

    Laser_Eoutput = f.IQModulator(pulse_shaped_symbols, Elaser, Vpi, Bias, MaxExc, MinExc, NPol) #laser output E field with phase noise

    for i, snr_dbi in enumerate(snr_db):

        print(f'Processing SNR {snr_dbi}')
        Gaussian_noise_signal = f.add_noise(Laser_Eoutput, snr_dbi, sps, Modbits, NPol, toggle_AWGNnoise) 

        CD_signal = f.add_chromatic_dispersion(Gaussian_noise_signal, sps, Rs, D, Clambda, L, NPol, toggle_CD)

        rx = CD_signal #Skipping receiver front end for now

        filtered_signal = f.matched_filter(rx, RRCimpulse, NPol, toggle=toggle_RRC) #if toggle is False, this function returns input
        
        ADC = f.downsample(filtered_signal, sps//2, NPol, toggle=toggle_RRC) #Simulate ADC downsampling to 2 sps

        #Chromatic Dispersion Compensation
        if(toggle_RRC==True):
            spsCD = 2
        else:
            spsCD=1
        CD_compensated_rx = f.CD_compensation(ADC, D, L, Clambda, Rs, NPol, spsCD, NFFT, NOverlap, toggle_CD_compensation)

        #Adaptive Equalisation Here

        downsampled_rx = f.downsample(CD_compensated_rx, 2, NPol, toggle=toggle_RRC) #Downsampled from 2 sps to 1 sps
    
        frequency_recovered = f.frequency_recovery(downsampled_rx, Rs, NPol, toggle_frequencyrecovery)
        
        if(toggle_BPS==True):
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
            BER[i] = (np.sum(original_bits[0] != demod_bits[0])+np.sum(original_bits[1] != demod_bits[1]))/(NPol*num_symbols*Modbits)
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

            axs4[i//3].plot(np.arange(num_symbols), theta[0::sps], label='Phase')
            #Note plotting theta before frequency recovery here
            axs4[i//3].set_title(f'Phase Noise at SNR = {snr_dbi}dB \n Linewidth = {maxDvT}')
            axs4[i//3].grid(True)
            if(NPol==1):
                axs4[i//3].plot(np.arange(num_symbols), thetaHat, color='red', label='Phase Estimate')
            elif(NPol==2):
                axs4[i//3].plot(np.arange(num_symbols), thetaHat[:,0], color='red', label='Phase Estimate (V)')
                axs4[i//3].plot(np.arange(num_symbols), thetaHat[:,1], color='orange', label='Phase Estimate (H)')
            
            if(toggle_DE==False):
                # Highlight erroneous symbols
                if(NPol==1):
                    axs1V[i//3].scatter(Phase_Noise_compensated_rx[erroneous_indexes].real, Phase_Noise_compensated_rx[erroneous_indexes].imag, color='red', label='Errors', alpha=0.5)
                    axs4[i//3].vlines(erroneous_indexes, ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes')
                if(NPol==2):
                    axs1V[i//3].scatter(Phase_Noise_compensated_rx[0][erroneous_indexesV].real, Phase_Noise_compensated_rx[0][erroneous_indexesV].imag, color='red', label='Errors', alpha=0.5)
                    axs1H[i//3].scatter(Phase_Noise_compensated_rx[1][erroneous_indexesH].real, Phase_Noise_compensated_rx[1][erroneous_indexesH].imag, color='red', label='Errors', alpha=0.5)
                    axs4[i//3].vlines(erroneous_indexesV, ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes (V)')
                    axs4[i//3].vlines(erroneous_indexesH, ymin=-0.05, ymax=0.05, colors='purple', alpha=0.2, label='Erroneous Indexes (H)')
            else:
                
                if(NPol==1):
                    axs1V[i//3].scatter(Phase_Noise_compensated_rx[erroneous_bit_indexes//Modbits].real, Phase_Noise_compensated_rx[erroneous_bit_indexes//Modbits].imag, color='red', label='Errors', alpha=0.5)
                    axs4[i//3].vlines(erroneous_bit_indexes//Modbits, ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes')
                if(NPol==2):
                    axs1V[i//3].scatter(Phase_Noise_compensated_rx[0][erroneous_bit_indexesV//Modbits].real, Phase_Noise_compensated_rx[0][erroneous_bit_indexesV//Modbits].imag, color='red', label='Errors', alpha=0.5)
                    axs1H[i//3].scatter(Phase_Noise_compensated_rx[1][erroneous_bit_indexesH//Modbits].real, Phase_Noise_compensated_rx[1][erroneous_bit_indexesH//Modbits].imag, color='red', label='Errors', alpha=0.5)
                    axs4[i//3].vlines(erroneous_bit_indexesV//Modbits, ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes (V)')
                    axs4[i//3].vlines(erroneous_bit_indexesH//Modbits, ymin=-0.05, ymax=0.05, colors='purple', alpha=0.2, label='Erroneous Indexes (H)')
                #Highlight rough location of symbol errors
            axs4[i//3].legend(loc='lower left')

        
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