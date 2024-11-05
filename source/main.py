import numpy as np
import matplotlib.pyplot as plt
import functions as f
import testing as t
"""
TO DO:
    -probatilistic shaping
        -where does it come in (tx and rx) - papers been sent
            - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8640810&tag=1
            - https://www.nowpublishers.com/article/DownloadSummary/CIT-111
            - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7322261
            -CCDM
        -how to implement

    -toggle in code for RRC & matched filter or not (with sps set to 1)
    -FPGA?
"""
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
    
    Modbits = 2 #2 is QPSK, 4 is 16QAM, 6 is 64QAM

    maxDvT = 8/(10**6) #There is a max Linewidth * T = maxDvT where T = 1/(sps*Rs)
    Linewidth = 10**5 #Linewidth of laser in Hz
    
    #Generate RRC filter impulse response
    #base 20, 16, 0.1
    span= 20 #Span of filter
    sps= 16 #Samples per symbol
    rolloff = 0.1 #Roll-off of RRC

    #Wiener Filter length parameters:
    frac = 0.05

    toggle_RRC = False #true means on
    toggle_phasenoise = True #true means on
    toggle_phasenoisecompensation = True #true means on
    toggle_plotuncompensatedphase = True #true means on
    toggle_ploterrorindexes = True #true means on

    if(toggle_RRC==False):
        sps=1               #overwrite sps if no RRC

    #There is a max Linewidth * T = maxDvT
    #T = 1/(Rs*sps)
    Rs = Linewidth/(sps*maxDvT) #Rs symbol rate symbols/second
    
    original_bits = np.random.randint(0, 2, size=num_symbols * Modbits)  


    # Generate symbols
    if(Modbits==2): #16QAM
        symbols = f.generate_QPSK_symbols(original_bits)
        plotsize = 2
        snr_begin=5
    elif(Modbits==4): #16QAM
        symbols = f.generate_16qam_symbols(original_bits)
        plotsize = 2
        snr_begin=10
    elif(Modbits==6): #64QAM
        symbols = f.generate_64qam_symbols(original_bits) 
        plotsize = 1.5
        snr_begin=10
    
    

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

        Gaussian_noise_rx = f.add_noise(tx, snr_dbi, sps, Modbits)

        Phase_Noise_rx, theta = f.add_phase_noise(Gaussian_noise_rx, num_symbols, sps, Rs, Linewidth, toggle=toggle_phasenoise)
        
        filtered_signal = f.matched_filter(Phase_Noise_rx, RRCimpulse, toggle=toggle_RRC) #if toggle is False, this function returns input

        Phase_Noise_compensated, thetaHat = t.phase_noise_compensation(filtered_signal, sps, Rs, Linewidth, Modbits, snr_dbi, frac, toggle_phasenoisecompensation)

        downsampled_rx = f.downsample(Phase_Noise_compensated, sps, toggle=toggle_RRC) #if toggle is False, this function returns input

        demod_symbols = f.max_likelihood_decision(downsampled_rx, Modbits) #pass in Modbits which says 16QAM or 64QAM

        # Find erroneous symbol indexes
        erroneous_indexes = np.where(symbols != demod_symbols)[0]

        #print(erroneous_indexes) #plot where errors are

        SER[i] = np.mean(symbols != demod_symbols)
    
        demod_bits = f.decode_symbols(demod_symbols, Modbits) #pass in Modbits which says 16QAM or 64QAM

        BER[i] = np.mean(original_bits != demod_bits)

        # Plot downsampled received constellation, including different colour for erroneous results
        #Only plot some of the results:
        if(i%3==0):

            f.plot_constellation(axs1[i//3], downsampled_rx, title=f'Downsampled Constellation at SNR = {snr_dbi}dB', lim=plotsize)
                    # Highlight erroneous symbols
            axs1[i//3].scatter(downsampled_rx[erroneous_indexes].real, downsampled_rx[erroneous_indexes].imag, color='red', label='Errors', alpha=0.5)

            if(toggle_plotuncompensatedphase==True):
                f.plot_constellation(axs3[i//3], filtered_signal[1::sps], title=f'No Derotation at SNR= {snr_dbi}dB', lim=plotsize)
            
            axs4[i//3].plot(np.arange(num_symbols*sps), theta, label='Phase')
            axs4[i//3].set_title(f'Phase Noise at SNR = {snr_dbi}dB')
            axs4[i//3].grid(True)

            axs4[i//3].plot(np.arange(num_symbols*sps), thetaHat, color='red', label='Phase Estimate')
            

            if(toggle_ploterrorindexes==True):
                axs4[i//3].vlines(erroneous_indexes*sps, ymin=-0.01, ymax=0.01, colors='g', alpha=0.2, label='Erroneous Indexes')
            axs4[i//3].legend(loc='lower left')

    plt.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))  
    axs2 = axs2.flatten()  # Flatten the array for easy indexing
    # Plot SER
    axs2[0].semilogy(snr_db, SER, marker='o')
    axs2[0].set_xlabel('SNR per bit/ dB')
    axs2[0].set_ylabel('SER')
    axs2[0].set_title('Symbol Error Rate (SER)')
    axs2[0].grid(True)

    # Plot BER
    axs2[1].semilogy(snr_db, BER, marker='o')
    axs2[1].set_xlabel('SNR per bit/ dB')
    axs2[1].set_ylabel('BER')
    axs2[1].set_title('Bit Error Rate (BER)')
    axs2[1].grid(True)
    plt.tight_layout()
    plt.show()

    """
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
