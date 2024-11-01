import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve #upfirdn for up/down sampling
import time

#References: Digital Coherent Optical Systems Textbook, Matlab Code, page 38

enable_benchmark = True  #True turns on benchmarking
def benchmark(enabled=enable_benchmark):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if enabled:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                print(f"Function '{func.__name__}' took {end_time - start_time:.6f} seconds to complete.")
                return result
            else:
                return func(*args, **kwargs)  # Call the function without timing
        return wrapper
    return decorator

@benchmark(enable_benchmark)
def generate_16qam_symbols(bits):
    # 16QAM is 4 bits/symbol, so create array of blocks of 4 bits
    # Mapping dictionary for 16QAM symbols
    symbol_mapping = {
        (0, 0, 0, 0): -3 + 3j,
        (0, 0, 0, 1): -1 + 3j,
        (0, 0, 1, 1):  1 + 3j,
        (0, 0, 1, 0):  3 + 3j,
        (0, 1, 1, 0):  3 + 1j,
        (0, 1, 1, 1):  1 + 1j,
        (0, 1, 0, 1): -1 + 1j,
        (0, 1, 0, 0): -3 + 1j,
        (1, 1, 0, 0): -3 - 1j,
        (1, 1, 0, 1): -1 - 1j,
        (1, 1, 1, 1):  1 - 1j,
        (1, 1, 1, 0):  3 - 1j,
        (1, 0, 1, 0):  3 - 3j,
        (1, 0, 1, 1):  1 - 3j,
        (1, 0, 0, 1): -1 - 3j,
        (1, 0, 0, 0): -3 - 3j,
    }
    # Convert bits to symbols using the dictionary
    symbols = np.array([
        symbol_mapping[tuple(bits[i:i + 4])] for i in range(0, len(bits), 4)
    ])
    return symbols/np.sqrt(10)

@benchmark(enable_benchmark)
def generate_64qam_symbols(bits):
    # 64QAM is 6 bits/symbol, so create array of blocks of 6 bits
    # Mapping dictionary for 64QAM symbols
    symbol_mapping = {
        (0, 0, 0, 0, 0, 0): -7 - 7j,
        (0, 0, 0, 0, 0, 1): -7 - 5j,
        (0, 0, 0, 0, 1, 1): -7 - 3j,
        (0, 0, 0, 0, 1, 0): -7 - 1j,
        (0, 0, 0, 1, 1, 0): -7 + 1j,
        (0, 0, 0, 1, 1, 1): -7 + 3j,
        (0, 0, 0, 1, 0, 1): -7 + 5j,
        (0, 0, 0, 1, 0, 0): -7 + 7j,
        (0, 0, 1, 0, 0, 0): -5 - 7j,
        (0, 0, 1, 0, 0, 1): -5 - 5j,
        (0, 0, 1, 0, 1, 1): -5 - 3j,
        (0, 0, 1, 0, 1, 0): -5 - 1j,
        (0, 0, 1, 1, 1, 0): -5 + 1j,
        (0, 0, 1, 1, 1, 1): -5 + 3j,
        (0, 0, 1, 1, 0, 1): -5 + 5j,
        (0, 0, 1, 1, 0, 0): -5 + 7j,
        (0, 1, 1, 0, 0, 0): -3 - 7j,
        (0, 1, 1, 0, 0, 1): -3 - 5j,
        (0, 1, 1, 0, 1, 1): -3 - 3j,
        (0, 1, 1, 0, 1, 0): -3 - 1j,
        (0, 1, 1, 1, 1, 0): -3 + 1j,
        (0, 1, 1, 1, 1, 1): -3 + 3j,
        (0, 1, 1, 1, 0, 1): -3 + 5j,
        (0, 1, 1, 1, 0, 0): -3 + 7j,
        (0, 1, 0, 0, 0, 0): -1 - 7j,
        (0, 1, 0, 0, 0, 1): -1 - 5j,
        (0, 1, 0, 0, 1, 1): -1 - 3j,
        (0, 1, 0, 0, 1, 0): -1 - 1j,
        (0, 1, 0, 1, 1, 0): -1 + 1j,
        (0, 1, 0, 1, 1, 1): -1 + 3j,
        (0, 1, 0, 1, 0, 1): -1 + 5j,
        (0, 1, 0, 1, 0, 0): -1 + 7j,
        (1, 1, 0, 0, 0, 0): 1 - 7j,
        (1, 1, 0, 0, 0, 1): 1 - 5j,
        (1, 1, 0, 0, 1, 1): 1 - 3j,
        (1, 1, 0, 0, 1, 0): 1 - 1j,
        (1, 1, 0, 1, 1, 0): 1 + 1j,
        (1, 1, 0, 1, 1, 1): 1 + 3j,
        (1, 1, 0, 1, 0, 1): 1 + 5j,
        (1, 1, 0, 1, 0, 0): 1 + 7j,
        (1, 1, 1, 0, 0, 0): 3 - 7j,
        (1, 1, 1, 0, 0, 1): 3 - 5j,
        (1, 1, 1, 0, 1, 1): 3 - 3j,
        (1, 1, 1, 0, 1, 0): 3 - 1j,
        (1, 1, 1, 1, 1, 0): 3 + 1j,
        (1, 1, 1, 1, 1, 1): 3 + 3j,
        (1, 1, 1, 1, 0, 1): 3 + 5j,
        (1, 1, 1, 1, 0, 0): 3 + 7j,
        (1, 0, 1, 0, 0, 0): 5 - 7j,
        (1, 0, 1, 0, 0, 1): 5 - 5j,
        (1, 0, 1, 0, 1, 1): 5 - 3j,
        (1, 0, 1, 0, 1, 0): 5 - 1j,
        (1, 0, 1, 1, 1, 0): 5 + 1j,
        (1, 0, 1, 1, 1, 1): 5 + 3j,
        (1, 0, 1, 1, 0, 1): 5 + 5j,
        (1, 0, 1, 1, 0, 0): 5 + 7j,
        (1, 0, 0, 0, 0, 0): 7 - 7j,
        (1, 0, 0, 0, 0, 1): 7 - 5j,
        (1, 0, 0, 0, 1, 1): 7 - 3j,
        (1, 0, 0, 0, 1, 0): 7 - 1j,
        (1, 0, 0, 1, 1, 0): 7 + 1j,
        (1, 0, 0, 1, 1, 1): 7 + 3j,
        (1, 0, 0, 1, 0, 1): 7 + 5j,
        (1, 0, 0, 1, 0, 0): 7 + 7j,
    }
    # Convert bits to symbols using the dictionary
    symbols = np.array([
        symbol_mapping[tuple(bits[i:i + 6])] for i in range(0, len(bits), 6)
    ])
    return symbols/np.sqrt(42)    

@benchmark(enable_benchmark)
def pulseshaping(symbols, sps, RRCimpulse, toggle):
    #performs pulse shaping of symbols with RRC filter. 
    #filter parameters are span and rolloff
    #sequence of symbols upsampled to sps samples per symbol then applied RRC filter
    if(toggle==True):
        #upsample the symbols by inserting (sps-1) zeros between each symbol
        upsampled = np.zeros(sps*len(symbols), dtype=complex)
        for i in range(0, sps*len(symbols), sps):
            upsampled[i] = symbols[i//sps]

        shaped = convolve(upsampled, RRCimpulse, mode='same')
        shaped = shaped/(np.sqrt(np.mean(abs(shaped)**2)))

        return shaped
    else:
        return symbols

@benchmark(enable_benchmark)
def add_noise(signal, snr_db, sps, Modbits): 
    #addition of circular Gaussian noise to transmitted signal
    #snr_db snr per bit in dB in transmitted signal
    #Modbits per symbol eg 16QAM or 64QAM etc.
    #sps samples per symbol

    snr = 10 ** (snr_db / 10) #dB to linear (10 since power)

    stdev= np.sqrt(np.mean(abs(signal)**2)*sps/(2*Modbits*snr))

    

    noise = stdev * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))


    return signal + noise 

@benchmark(enable_benchmark)
def matched_filter(signal, pulse_shape, toggle):
    #received_signal is original signals + noise
    #pulse shape eg raised-root-cosine, square etc.
    if(toggle==True):
        filtered = convolve(signal, pulse_shape, mode='same') 
        filtered = filtered/(np.sqrt(np.mean(abs(filtered)**2)))
        #should have peaks where received signal matches pulse shape, maximising SNR
        return filtered
    else:
        return signal

@benchmark(enable_benchmark)
def plot_constellation(ax, symbols, title, lim=2):
    ax.scatter(symbols.real, symbols.imag, color='blue', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('In-Phase')
    ax.set_ylabel('Quadrature')
    ax.set_xlim(-1*lim, lim)
    ax.set_ylim(-1*lim, lim)
    ax.grid(True)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)

@benchmark(enable_benchmark)
def downsample(signal, sps, toggle):
    if(toggle==True):
        #downsample a signal that has sps samples per symbol
        downsampled = signal[::sps]
        return downsampled
    else:
        return signal
    
@benchmark(enable_benchmark)
def RRC(span, rolloff, sps):
    #Generate root-raised cosine impulse response
    g = np.zeros(span*sps+1, dtype=float)
    k = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    i1 = np.where(k == 0)[0]
    i2 = np.where(np.abs(4 * rolloff * k) - 1 == 0)[0]
    i3 = np.arange(1, len(k) + 1)
    i3 = np.delete(i3, np.concatenate([i1, i2]))
    k = k[i3 - 1]
       
    #singularity in k=0  
    if len(i1) > 0:  
        g[i1] = 1 - rolloff + 4 * rolloff / np.pi

    #singularity in k = 1/(4*rolloff)
    if len(i2) > 0: 
        g[i2] = (rolloff / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi / (4 * rolloff)) + (1 - 2/np.pi) * np.cos(np.pi / (4 * rolloff)))

    #rest of the coefficients
    g[i3-1] = (np.sin(np.pi * k * (1 - rolloff)) + 4 * rolloff * k * np.cos(np.pi * k * (1 + rolloff))) / (np.pi * k * (1 - (4 * rolloff * k)**2))
    
    g = g / np.max(g)

    t = np.arange(-span * sps / 2, span * sps / 2 + 1)
    return g , t

@benchmark(enable_benchmark)
def max_likelihood_decision(rx_symbols, Modbits):
    #returns the closest symbols in the constellation to the inputed rx_symbols
    # Define the 16QAM constellation points
    if(Modbits==4):
        constellation = np.array([
            -3 + 3j, -3 + 1j, -3 - 1j, -3 - 3j,
            -1 + 3j, -1 + 1j, -1 - 1j, -1 - 3j,
            1 + 3j,  1 + 1j,  1 - 1j,  1 - 3j,
            3 + 3j,  3 + 1j,  3 - 1j,  3 - 3j
        ]) / np.sqrt(10) 

    elif(Modbits==6):
        constellation = np.array([
            -7 - 7j, -7 - 5j, -7 - 3j, -7 - 1j,
            -7 + 1j, -7 + 3j, -7 + 5j, -7 + 7j,
            -5 - 7j, -5 - 5j, -5 - 3j, -5 - 1j,
            -5 + 1j, -5 + 3j, -5 + 5j, -5 + 7j,
            -3 - 7j, -3 - 5j, -3 - 3j, -3 - 1j,
            -3 + 1j, -3 + 3j, -3 + 5j, -3 + 7j,
            -1 - 7j, -1 - 5j, -1 - 3j, -1 - 1j,
            -1 + 1j, -1 + 3j, -1 + 5j, -1 + 7j,
            1 - 7j,  1 - 5j,  1 - 3j,  1 - 1j,
            1 + 1j,  1 + 3j,  1 + 5j,  1 + 7j,
            3 - 7j,  3 - 5j,  3 - 3j,  3 - 1j,
            3 + 1j,  3 + 3j,  3 + 5j,  3 + 7j,
            5 - 7j,  5 - 5j,  5 - 3j,  5 - 1j,
            5 + 1j,  5 + 3j,  5 + 5j,  5 + 7j,
            7 - 7j,  7 - 5j,  7 - 3j,  7 - 1j,
            7 + 1j,  7 + 3j,  7 + 5j,  7 + 7j        
        ])/np.sqrt(42)
    

    ML_symbols = np.empty(len(rx_symbols), dtype=complex)
    print(len(ML_symbols))
    for i, rx_symbol in enumerate(rx_symbols):
        # Find the closest symbol (maximum likelihood detection)
        ML_symbols[i] = min(constellation, key=lambda s: np.abs(s - rx_symbol)) 

    # Return the detected symbols based on the index
    return ML_symbols

@benchmark(enable_benchmark)
def decode_symbols(symbols, Modbits):
    #turns symbols back to bitstream
    if(Modbits==4): #16QAM
        symbol_mapping = {
            (0, 0, 0, 0): -3 + 3j,
            (0, 0, 0, 1): -1 + 3j,
            (0, 0, 1, 1):  1 + 3j,
            (0, 0, 1, 0):  3 + 3j,
            (0, 1, 1, 0):  3 + 1j,
            (0, 1, 1, 1):  1 + 1j,
            (0, 1, 0, 1): -1 + 1j,
            (0, 1, 0, 0): -3 + 1j,
            (1, 1, 0, 0): -3 - 1j,
            (1, 1, 0, 1): -1 - 1j,
            (1, 1, 1, 1):  1 - 1j,
            (1, 1, 1, 0):  3 - 1j,
            (1, 0, 1, 0):  3 - 3j,
            (1, 0, 1, 1):  1 - 3j,
            (1, 0, 0, 1): -1 - 3j,
            (1, 0, 0, 0): -3 - 3j,
        }
        root =10

    elif(Modbits==6):
        symbol_mapping = {
        (0, 0, 0, 0, 0, 0): -7 - 7j,
        (0, 0, 0, 0, 0, 1): -7 - 5j,
        (0, 0, 0, 0, 1, 1): -7 - 3j,
        (0, 0, 0, 0, 1, 0): -7 - 1j,
        (0, 0, 0, 1, 1, 0): -7 + 1j,
        (0, 0, 0, 1, 1, 1): -7 + 3j,
        (0, 0, 0, 1, 0, 1): -7 + 5j,
        (0, 0, 0, 1, 0, 0): -7 + 7j,
        (0, 0, 1, 0, 0, 0): -5 - 7j,
        (0, 0, 1, 0, 0, 1): -5 - 5j,
        (0, 0, 1, 0, 1, 1): -5 - 3j,
        (0, 0, 1, 0, 1, 0): -5 - 1j,
        (0, 0, 1, 1, 1, 0): -5 + 1j,
        (0, 0, 1, 1, 1, 1): -5 + 3j,
        (0, 0, 1, 1, 0, 1): -5 + 5j,
        (0, 0, 1, 1, 0, 0): -5 + 7j,
        (0, 1, 1, 0, 0, 0): -3 - 7j,
        (0, 1, 1, 0, 0, 1): -3 - 5j,
        (0, 1, 1, 0, 1, 1): -3 - 3j,
        (0, 1, 1, 0, 1, 0): -3 - 1j,
        (0, 1, 1, 1, 1, 0): -3 + 1j,
        (0, 1, 1, 1, 1, 1): -3 + 3j,
        (0, 1, 1, 1, 0, 1): -3 + 5j,
        (0, 1, 1, 1, 0, 0): -3 + 7j,
        (0, 1, 0, 0, 0, 0): -1 - 7j,
        (0, 1, 0, 0, 0, 1): -1 - 5j,
        (0, 1, 0, 0, 1, 1): -1 - 3j,
        (0, 1, 0, 0, 1, 0): -1 - 1j,
        (0, 1, 0, 1, 1, 0): -1 + 1j,
        (0, 1, 0, 1, 1, 1): -1 + 3j,
        (0, 1, 0, 1, 0, 1): -1 + 5j,
        (0, 1, 0, 1, 0, 0): -1 + 7j,
        (1, 1, 0, 0, 0, 0): 1 - 7j,
        (1, 1, 0, 0, 0, 1): 1 - 5j,
        (1, 1, 0, 0, 1, 1): 1 - 3j,
        (1, 1, 0, 0, 1, 0): 1 - 1j,
        (1, 1, 0, 1, 1, 0): 1 + 1j,
        (1, 1, 0, 1, 1, 1): 1 + 3j,
        (1, 1, 0, 1, 0, 1): 1 + 5j,
        (1, 1, 0, 1, 0, 0): 1 + 7j,
        (1, 1, 1, 0, 0, 0): 3 - 7j,
        (1, 1, 1, 0, 0, 1): 3 - 5j,
        (1, 1, 1, 0, 1, 1): 3 - 3j,
        (1, 1, 1, 0, 1, 0): 3 - 1j,
        (1, 1, 1, 1, 1, 0): 3 + 1j,
        (1, 1, 1, 1, 1, 1): 3 + 3j,
        (1, 1, 1, 1, 0, 1): 3 + 5j,
        (1, 1, 1, 1, 0, 0): 3 + 7j,
        (1, 0, 1, 0, 0, 0): 5 - 7j,
        (1, 0, 1, 0, 0, 1): 5 - 5j,
        (1, 0, 1, 0, 1, 1): 5 - 3j,
        (1, 0, 1, 0, 1, 0): 5 - 1j,
        (1, 0, 1, 1, 1, 0): 5 + 1j,
        (1, 0, 1, 1, 1, 1): 5 + 3j,
        (1, 0, 1, 1, 0, 1): 5 + 5j,
        (1, 0, 1, 1, 0, 0): 5 + 7j,
        (1, 0, 0, 0, 0, 0): 7 - 7j,
        (1, 0, 0, 0, 0, 1): 7 - 5j,
        (1, 0, 0, 0, 1, 1): 7 - 3j,
        (1, 0, 0, 0, 1, 0): 7 - 1j,
        (1, 0, 0, 1, 1, 0): 7 + 1j,
        (1, 0, 0, 1, 1, 1): 7 + 3j,
        (1, 0, 0, 1, 0, 1): 7 + 5j,
        (1, 0, 0, 1, 0, 0): 7 + 7j,
    }
        root = 42

    reverse_mapping = {v: k for k, v in symbol_mapping.items()}
    bits = np.zeros(len(symbols) * Modbits, dtype=int)


    for i, symbol in enumerate(symbols):
        bits[i * Modbits:(i + 1) * Modbits] = reverse_mapping[int(symbol.real*np.sqrt(root))+int(symbol.imag*np.sqrt(root))*1j] #bit stream
    return bits

@benchmark(enable_benchmark)
def add_phase_noise(symbols, Nsymb, sps, Rs, Linewidth, toggle):
    #This function adds phase noise to the transmitted symbols, modelled by a Wiener Process.

    #symbols: symbols sent
    #sps: samples per symbol
    #Rs: symbol rate symbols/second
    #Nsymb: number of transmitted symbols
    #Linewidth: Laser linewidth in Hz 
    if(Linewidth != 0 and toggle==True):

        T = 1/(sps*Rs) #period between samples at the oversampled transmit signal

        #Calculating phase noise:
        Var = 2*np.pi*Linewidth*T           
        delta_theta = np.sqrt(Var)*np.random.randn(sps*Nsymb)
        theta = np.cumsum(delta_theta) #an array of phase shift vs time
        symbols *= np.exp(1j * theta)

        return symbols, theta
    else:
        return symbols, np.zeros(len(symbols))
    