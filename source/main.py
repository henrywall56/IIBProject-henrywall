import numpy as np
import matplotlib.pyplot as plt
import functions as f
import DDPhaseRecoveryTesting as t

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
    num_symbols = 10000
    
    Modbits = 6 #2 is QPSK, 4 is 16QAM, 6 is 64QAM

    maxDvT = 1/(10**5) #There is a max Linewidth * T = maxDvT where T = 1/(sps*Rs)
    Linewidth = 10**5 #Linewidth of laser in Hz
    
    #Generate RRC filter impulse response
    #base 20, 16, 0.1
    span= 20 #Span of filter
    sps= 16 #Samples per symbol
    rolloff = 0.1 #Roll-off of RRC

    #DD Phase Recovery Parameters:
    frac = 0.05

    #BPS Phase Recovery Parameters:
    N=5 #Number of symbols to average over
    #"N = 6,..,10 will be a fairly good choice" - Pfau paper
    B=10 #Number of trial angles

    snr_begin = 8
    

    toggle_RRC = False #toggle RRC pulse shaping
    toggle_AWGNnoise = True
    toggle_phasenoise = True #toggle phase noise
    toggle_phasenoisecompensation = True #toggle phase noise compensation
    toggle_plotuncompensatedphase = True #toggle plotting constellation before phase compensation. Note this is before downsampling if using RRC pulseshaping.
    toggle_ploterrorindexes = True #toggle plotting error indexes on phase plot
    toggle_BPS = True #toggle blind phase searching algorithm: True is BPS, False is DD Phase compensation.
    toggle_DE = True #toggle Differential Encoding

    if(toggle_RRC==False):
        sps=1               #overwrite sps if no RRC

    #There is a max Linewidth * T = maxDvT
    #T = 1/(Rs*sps)
    Rs = Linewidth/(sps*maxDvT) #Rs symbol rate symbols/second
    
    original_bits = np.random.randint(0, 2, size=num_symbols * Modbits) 

    if(toggle_DE==False):
        # Generate symbols
        if(Modbits==2): #QPSK
            symbols = f.generate_QPSK_symbols(original_bits)
            plotsize = 2
            
        elif(Modbits==4): #16QAM
            symbols = f.generate_16qam_symbols(original_bits)
            plotsize = 2
            
        elif(Modbits==6): #64QAM
            symbols = f.generate_64qam_symbols(original_bits) 
            plotsize = 1.5
    else:
        # Generate symbols using differential encoding
        if(Modbits==2): #16QAM
            symbols = f.Differential_Encoding_qpsk(original_bits)
            plotsize = 2
            
        elif(Modbits==4): #16QAM
            symbols = f.Differential_Encoding_16qam (original_bits)
            plotsize = 2
            
        elif(Modbits==6): #64QAM
            symbols = f.Differential_Encoding_64qam(original_bits) 
            plotsize = 1.5

    RRCimpulse , t1 = f.RRC(span, rolloff, sps)

        #Pulse shaping with RRC filter
    tx = f.pulseshaping(symbols, sps, RRCimpulse, toggle = toggle_RRC) #if toggle False, this function just returns symbols

    snr_db = np.arange(snr_begin,snr_begin+10,1)

    fig1, axs1 = plt.subplots(2, 2, figsize=(8, 8))  
    axs1 = axs1.flatten()  # Flatten the array for easy indexing

    if(toggle_plotuncompensatedphase==True):
        fig3, axs3 = plt.subplots(2, 2, figsize=(8, 8))  
        axs3= axs3.flatten()  # Flatten the array for easy indexing

    fig4, axs4 = plt.subplots(2, 2, figsize=(8, 8))  
    axs4= axs4.flatten()  # Flatten the array for easy indexing

    BER = np.zeros(len(snr_db))
    SER = np.empty(len(snr_db))

    for i, snr_dbi in enumerate(snr_db):
        print(f'Processing SNR {snr_dbi}')
        Gaussian_noise_rx = f.add_noise(tx, snr_dbi, sps, Modbits, toggle_AWGNnoise)

        Phase_Noise_rx, theta = f.add_phase_noise(Gaussian_noise_rx, num_symbols, sps, Rs, Linewidth, toggle=toggle_phasenoise)
        
        filtered_signal = f.matched_filter(Phase_Noise_rx, RRCimpulse, toggle=toggle_RRC) #if toggle is False, this function returns input

        if(toggle_BPS==True):
            Phase_Noise_compensated, thetaHat = f.BPS(filtered_signal,Modbits,N,B, toggle_phasenoisecompensation)
        else:
            Phase_Noise_compensated, thetaHat = t.DD_phase_noise_compensation(filtered_signal, sps, Rs, Linewidth, Modbits, snr_dbi, frac, toggle_phasenoisecompensation)

        downsampled_rx = f.downsample(Phase_Noise_compensated, sps, toggle=toggle_RRC) #if toggle is False, this function returns input

        if(toggle_DE==True):
            demod_bits = f.Differential_decode_symbols(downsampled_rx, Modbits)
            #Plot erroneous symbols as the symbols that correspond to bit errors
            erroneous_bit_indexes = np.where(original_bits!=demod_bits)[0]
        else:
            demod_symbols = f.max_likelihood_decision(downsampled_rx, Modbits) #pass in Modbits which says 16QAM or 64QAM
            # SER only has meaning if Differential Encoding is NOT used 
            # Find erroneous symbol indexes
            erroneous_indexes = np.where(symbols != demod_symbols)[0]
            SER[i] = np.mean(symbols != demod_symbols)
        
            demod_bits = f.decode_symbols(demod_symbols, Modbits) #pass in Modbits which says 16QAM or 64QAM

        BER[i] = np.mean(original_bits != demod_bits)

        # Plot downsampled received constellation, including different colour for erroneous results
        #Only plot some of the results:
        if(i%3==0):

            f.plot_constellation(axs1[i//3], downsampled_rx, title=f'Downsampled Constellation \n SNR = {snr_dbi}dB', lim=plotsize)
            

            if(toggle_plotuncompensatedphase==True):
                f.plot_constellation(axs3[i//3], filtered_signal[1::sps], title=f'No Derotation at SNR= {snr_dbi}dB', lim=plotsize)
            
            axs4[i//3].plot(np.arange(num_symbols*sps), theta, label='Phase')
            axs4[i//3].set_title(f'Phase Noise at SNR = {snr_dbi}dB \n Linewidth = {maxDvT}')
            axs4[i//3].grid(True)

            axs4[i//3].plot(np.arange(num_symbols*sps), thetaHat, color='red', label='Phase Estimate')
            
            if(toggle_DE==False):
                if(toggle_ploterrorindexes==True):
                    axs4[i//3].vlines(erroneous_indexes*sps, ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes')
                # Highlight erroneous symbols
                axs1[i//3].scatter(downsampled_rx[erroneous_indexes].real, downsampled_rx[erroneous_indexes].imag, color='red', label='Errors', alpha=0.5)
            else:
                axs4[i//3].vlines(erroneous_bit_indexes*sps//Modbits, ymin=-0.05, ymax=0.05, colors='g', alpha=0.2, label='Erroneous Indexes')
                axs1[i//3].scatter(downsampled_rx[erroneous_bit_indexes//Modbits].real, downsampled_rx[erroneous_bit_indexes//Modbits].imag, color='red', label='Errors', alpha=0.5)
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

    else:
        fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))  
        axs2 = axs2.flatten()  # Flatten the array for easy indexing   

        # Plot BER
        axs2[1].plot()
        axs2[1].semilogy(snr_db, BER, marker='o')
        axs2[1].set_xlabel('SNR per bit/ dB')
        axs2[1].set_ylabel('BER')
        axs2[1].set_title('Bit Error Rate (BER)')
        axs2[1].grid(True)        
        # Plot SER
        axs2[0].semilogy(snr_db, SER, marker='o')
        axs2[0].set_xlabel('SNR per bit/ dB')
        axs2[0].set_ylabel('SER')
        axs2[0].set_title('Symbol Error Rate (SER)')
        axs2[0].grid(True)


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
