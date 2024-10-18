import numpy as np
import matplotlib.pyplot as plt
import functions as f


#TO DO:
#   64QAM : generation, demod_symbols, decode_bits
#   probatilistic shaping
#   why is there always a couple of symbol errors? 
def main():
    num_symbols = 10000
    Modbits = 6 #4 if 16QAM, 6 is 64QAM

    #Generate RRC filter impulse response
    span= 20 #Span of filter
    sps= 16 #Samples per symbol
    rolloff = 0.1 #Roll-off of RRC

    original_bits = np.random.randint(0, 2, size=num_symbols * Modbits)  

    # Generate symbols
    if(Modbits==4): #16QAM
        symbols = f.generate_16qam_symbols(original_bits)
        plotsize = 2
    elif(Modbits==6): #64QAM
        symbols = f.generate_64qam_symbols(original_bits) 
        plotsize = 1.5

    RRCimpulse , t1 = f.RRC(span, rolloff, sps)

    plt.plot(t1, RRCimpulse, '.')
    plt.title('RRC Filter Impulse Reponse')
    plt.show()

    #Pulse shaping with RRC filter
    tx = f.pulseshaping(symbols, sps, RRCimpulse)

    snr_db = np.arange(6,16,1)

    fig1, axs1 = plt.subplots(2, 2, figsize=(8, 8))  
    axs1 = axs1.flatten()  # Flatten the array for easy indexing

    BER = np.zeros(len(snr_db))
    SER = np.empty(len(snr_db))

    for i, snr_dbi in enumerate(snr_db):

        rx = f.add_noise(tx, snr_dbi, sps, Modbits)

        filtered_signal = f.matched_filter(rx, RRCimpulse)

        downsampled_rx = f.downsample(filtered_signal, sps)
        # Plot downsampled received constellation
        if(i%3==0):
            f.plot_constellation(axs1[i//3], downsampled_rx, title=f'Downsampled Constellation at SNR = {snr_dbi}dB', lim=plotsize)

        demod_symbols = f.max_likelihood_decision(downsampled_rx, Modbits) #pass in Modbits which says 16QAM or 64QAM

        SER[i] = np.mean(symbols != demod_symbols)
    
        demod_bits = f.decode_symbols(demod_symbols, Modbits) #pass in Modbits which says 16QAM or 64QAM

        BER[i] = np.mean(original_bits != demod_bits)
        
    plt.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))  
    axs2 = axs2.flatten()  # Flatten the array for easy indexing
    # Plot SER
    axs2[0].semilogy(snr_db, SER, marker='o')
    axs2[0].set_xlabel('SNR / dB')
    axs2[0].set_ylabel('SER')
    axs2[0].set_title('Symbol Error Rate (SER)')
    axs2[0].grid(True)

    # Plot BER
    axs2[1].semilogy(snr_db, BER, marker='o')
    axs2[1].set_xlabel('SNR / dB')
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


#TO DO:

#understand how the oversampling work properly
#how to do raised root cosine pulse shaping
