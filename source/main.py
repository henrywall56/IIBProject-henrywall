import numpy as np
import matplotlib.pyplot as plt
import functions as f

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
    Modbits = 6 #4 if 16QAM, 6 is 64QAM


    #Generate RRC filter impulse response
    #base 20, 16, 0.1
    span= 20 #Span of filter
    sps= 16 #Samples per symbol
    rolloff = 0.1 #Roll-off of RRC

    toggle_RRC = False #true means on 

    if(toggle_RRC==False):
        sps=1               #overwrite sps if no RRC

    original_bits = np.random.randint(0, 2, size=num_symbols * Modbits)  

    # Generate symbols
    if(Modbits==4): #16QAM
        symbols = f.generate_16qam_symbols(original_bits)
        plotsize = 2
    elif(Modbits==6): #64QAM
        symbols = f.generate_64qam_symbols(original_bits) 
        plotsize = 1.5
    
    if(toggle_RRC):

        RRCimpulse , t1 = f.RRC(span, rolloff, sps)

        plt.plot(t1, RRCimpulse, '.')
        plt.title('RRC Filter Impulse Reponse')
        plt.show()

        #Pulse shaping with RRC filter
        tx = f.pulseshaping(symbols, sps, RRCimpulse)

    else:
        tx = symbols

    snr_begin = 10
    snr_db = np.arange(snr_begin,snr_begin+10,1)

    fig1, axs1 = plt.subplots(2, 2, figsize=(8, 8))  
    axs1 = axs1.flatten()  # Flatten the array for easy indexing

    BER = np.zeros(len(snr_db))
    SER = np.empty(len(snr_db))

    

    for i, snr_dbi in enumerate(snr_db):

        rx = f.add_noise(tx, snr_dbi, sps, Modbits)

        if(toggle_RRC):

            filtered_signal = f.matched_filter(rx, RRCimpulse)

            downsampled_rx = f.downsample(filtered_signal, sps)
        
        else:
            downsampled_rx = rx

        

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
