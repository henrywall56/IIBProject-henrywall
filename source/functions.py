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
def generate_QPSK_symbols(bits):
    # QPSK is 2 bits/symbol, so create array of blocks of 4 bits
    # Mapping dictionary for 16QAM symbols
    symbol_mapping = {
        (0,0): -1 - 1j,
        (0,1): -1 + 1j,
        (1,0):  1 - 1j,
        (1,1):  1 + 1j,
    }
    # Convert bits to symbols using the dictionary
    symbols = np.array([
        symbol_mapping[tuple(bits[i:i + 2])] for i in range(0, len(bits), 2)
    ])
    return symbols/np.sqrt(2)

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
def add_noise(signal, snrb_db, sps, Modbits, toggle_AWGNnoise): 
    #addition of circular Gaussian noise to transmitted signal
    #snrb_db snr per bit in dB in transmitted signal
    #Modbits per symbol eg 16QAM or 64QAM etc.
    #sps samples per symbol
    if(toggle_AWGNnoise==True):
        snr = 10 ** (snrb_db / 10) #dB to linear (10 since power)

        stdev= np.sqrt(np.mean(abs(signal)**2)*sps/(2*Modbits*snr))

        

        noise = stdev * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))


        return signal + noise 
    else:
        return signal

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
        downsampled = signal[1::sps]
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

@benchmark(False)
def max_likelihood_decision(rx_symbols, Modbits):
    #returns the closest symbols in the constellation to the inputed rx_symbols
    if(Modbits==2):
        constellation = np.array([1+1j, -1+1j, 1-1j, -1-1j
                                  ])/np.sqrt(2)

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
    
    for i, rx_symbol in enumerate(rx_symbols):
        # Find the closest symbol (maximum likelihood detection)
        ML_symbols[i] = min(constellation, key=lambda s: np.abs(s - rx_symbol)) 

    # Return the detected symbols based on the index
    return ML_symbols

@benchmark(enable_benchmark)
def decode_symbols(symbols, Modbits):
    #turns symbols back to bitstream
    if(Modbits==2): #QPSK
        symbol_mapping = {
            (0,0): -1 - 1j,
            (0,1): -1 + 1j,
            (1,0):  1 - 1j,
            (1,1):  1 + 1j,
        }
        root = 2

    elif(Modbits==4): #16QAM
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
        root = 10

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
        rotated_symbols = symbols * np.exp(1j * theta)

        return rotated_symbols, theta
    else:
        return symbols, np.zeros(len(symbols))

def convmtx(vector, L):
    """
    Create a convolution matrix from the input vector with length L.
    
    Parameters:
    - vector: 1D array (input vector)
    - L: Length of the desired convolution matrix
    
    Returns:
    - Convolution matrix
    """
    # Get the length of the input vector
    N = len(vector)
    
    # Create a matrix with zero padding on the left and right
    result = np.zeros((L, N + L - 1),dtype=complex)
    
    for i in range(L):
        result[i, i:i+N] = vector  # Fill the matrix with shifted vector
    
    return result


@benchmark(enable_benchmark)
def BPS(z,Modbits,N,B, toggle_phasenoisecompensation):
    #Blind Phase Search Phase compensation
    #z is received signal to derotate
    #Modbits defines the QAM format
    #N is number of past and future symbols used in the BPS algorithm for phase noise estimation. 
    #Total number of symbols used in BPS = L = 2N+1
    #B is number of test rotations

    if(toggle_phasenoisecompensation==True):
        # Parameters
        p = np.pi / 2
        L = 2 * N + 1

        # Test carrier phase angles
        b = np.arange(-B//2, B//2)
        ThetaTest = p * b / B #B rotation angles

        #This should be 
        ThetaTestMatrix = np.tile(np.exp(-1j*ThetaTest),(L,1)) #L row x B column matrix, each row is phase angle vector

        zB_V = np.concatenate([np.zeros(L // 2, dtype=complex), z, np.zeros(L // 2, dtype=complex)])
        zB_V = convmtx(zB_V,L)
        
        zB_V = np.flipud(zB_V[:, L:-L+1])

        zBlocks = zB_V
        
        ThetaPU = np.zeros(zBlocks.shape[1]+1)
        ThetaPrev = 0.0
        #Phase noise estimates
        for i in range(zBlocks.shape[1]): #Over columns
            
            zrot = np.tile(zBlocks[:, i][:, np.newaxis],(1,B)) * ThetaTestMatrix #ith column repeated 
            zrot_decided = np.zeros((zrot.shape[0], zrot.shape[1]), dtype=complex)

            for j in range(zrot.shape[0]): #Decision of rotated symbols
                zrot_decided[j,:] = max_likelihood_decision(zrot[j,:], Modbits)

            #intermediate sum to be minimised
            m = np.sum(abs(zrot-zrot_decided)**2,0)
            
            #estimating phase noise as angle that minimises m
            im = np.argmin(m)

            Theta = np.reshape(ThetaTest[im], 1)
            
            ThetaPU[i] = Theta + np.floor(0.5-(Theta-ThetaPrev)/p)*p

            ThetaPrev = ThetaPU[i]
        ThetaPU[-1]=ThetaPrev
        v = z*np.exp(-1j*ThetaPU)

        return v, ThetaPU
        
    else:
        return z, np.zeros(z)

def invert(nparr):
    #Note input MUST be a numpy array
    #Same as ~ NOT in Matlab, for array of ones and zeros
    return 1-nparr

@benchmark(enable_benchmark)  
def Differential_Encoding_qpsk(bits):
    #Differential Encoding of QPSK
    Modbits = 2

    #Bits that define the quadrant:
    Qbits1 = np.array(bits[0:len(bits):Modbits], dtype=np.int64)
    Qbits2 = np.array(bits[1:len(bits):Modbits], dtype=np.int64)

     #Defining the quadrant:
    Quad = np.array(invert(Qbits1)&invert(Qbits2), dtype=np.complex128) + (invert(Qbits1)&Qbits2)*np.exp(1j*np.pi/2) + (Qbits1&invert(Qbits2))*np.exp(3j*np.pi/2) + (Qbits1&Qbits2)*np.exp(1j*np.pi)

    #Initial Quadrant:
    QuadPrev = 1

    #Modulation
    x = np.zeros(len(Quad), dtype=complex)
    
    for i in range(len(x)):
        x[i] = QuadPrev * Quad[i] * (1+1j)
        QuadPrev = QuadPrev * Quad[i]

    x = x/np.sqrt(2)

    return x

@benchmark(enable_benchmark)
def Differential_Encoding_16qam(bits):
    #Differential Encoding of 16-QAM
    Modbits = 4

    #Bits that define the quadrant:
    Qbits1 = np.array(bits[0:len(bits):Modbits], dtype=np.int64)
    Qbits2 = np.array(bits[1:len(bits):Modbits], dtype=np.int64)

    #Bits that define the symbols inside the quadrant:
    InQuadBits1 = np.array(bits[2:len(bits):Modbits], dtype=np.int64)
    InQuadBits2 = np.array(bits[3:len(bits):Modbits], dtype=np.int64)

    #Defining the quadrant:
    Quad = np.array(invert(Qbits1)&invert(Qbits2), dtype=np.complex128) + (invert(Qbits1)&Qbits2)*np.exp(1j*np.pi/2) + (Qbits1&invert(Qbits2))*np.exp(3j*np.pi/2) + (Qbits1&Qbits2)*np.exp(1j*np.pi)
    
    #Defining the symbol inside the quadrants:
    InQuadI = 2*InQuadBits1+ 1
    InQuadQ = 2*InQuadBits2 + 1
    InQuadSymbol = InQuadI + 1j*InQuadQ

    #Initial Quadrant
    QuadPrev = 1

    #Modulation
    x = np.zeros(len(InQuadBits2), dtype=complex)
    
    for i in range(len(x)):
        x[i] = QuadPrev * Quad[i] * InQuadSymbol[i]
        QuadPrev = QuadPrev * Quad[i]

    normalised_x = x/np.sqrt(10)

    return normalised_x

@benchmark(enable_benchmark)
def Differential_Encoding_64qam(bits):
    #Differential Encoding of 64-QAM
    Modbits = 6

    #Bits that define the quadrant:
    Qbits1 = np.array(bits[0:len(bits):Modbits], dtype=np.int64)
    Qbits2 = np.array(bits[1:len(bits):Modbits], dtype=np.int64)

    #Bits that define the symbols inside the quadrant:
    InQuadBits1 = np.array(bits[2:len(bits):Modbits], dtype=np.int64)
    InQuadBits2 = np.array(bits[3:len(bits):Modbits], dtype=np.int64)
    InQuadBits3 = np.array(bits[4:len(bits):Modbits], dtype=np.int64)
    InQuadBits4 = np.array(bits[5:len(bits):Modbits], dtype=np.int64)

    #Defining the quadrant:
    Quad = np.array(invert(Qbits1)&invert(Qbits2), dtype=np.complex128) + (invert(Qbits1)&Qbits2)*np.exp(1j*np.pi/2) + (Qbits1&invert(Qbits2))*np.exp(3j*np.pi/2) + (Qbits1&Qbits2)*np.exp(1j*np.pi)
    
    #Defining the symbol inside the quadrants:
    InQuadI = 7 - (InQuadBits1&invert(InQuadBits2))*2 - (InQuadBits1&InQuadBits2)*4 - (invert(InQuadBits1)&InQuadBits2)*6 
    InQuadQ = 7 - (InQuadBits3&invert(InQuadBits4))*2 - (InQuadBits3&InQuadBits4)*4 - (invert(InQuadBits3)&InQuadBits4)*6 
    InQuadSymbol = InQuadI + 1j*InQuadQ

    #Initial Quadrant
    QuadPrev = 1

    #Modulation
    x = np.zeros(len(InQuadBits2), dtype=complex)
    
    for i in range(len(x)):
        x[i] = QuadPrev * Quad[i] * InQuadSymbol[i]
        QuadPrev = QuadPrev * Quad[i]

    normalised_x = x/np.sqrt(42)

    return normalised_x


@benchmark(enable_benchmark)
def Differential_decode_symbols(symbols, Modbits):
    #Differential decoding of symbols to bits
    QPrev = 0
    Decided = np.zeros((len(symbols), Modbits), dtype=int)
    if(Modbits==2):
        #Decision Regions for In-Phase component
        R1 = np.real(symbols) >= 0
        R2 = np.real(symbols) < 0
        #Decision Regions for Quadrature component
        R3 = np.imag(symbols) >= 0
        R4 = np.imag(symbols) < 0
        

        #Defining the quadrant:
        Q = np.zeros(len(symbols), dtype=int)
        # Assign values based on conditions
        Q[R1 & R3] = 0  # Set Q to 0 where R1 and R3 are True
        Q[R2 & R3] = 1  # Set Q to 1 where R2 and R3 are True
        Q[R2 & R4] = 2  # Set Q to 2 where R2 and R4 are True
        Q[R1 & R4] = 3  # Set Q to 3 where R1 and R4 are True
        #Now Q contains the value of the Qudrant that each symbol is in

        for i in range(len(symbols)):
            QRx = Q[i] - QPrev
            #Bits defining the quadrant (and consequently the symbol)
            if QRx == 0:
                Decided[i, 0] = 0
                Decided[i, 1] = 0
            elif QRx in {1, -3}:
                Decided[i, 0] = 0
                Decided[i, 1] = 1
            elif QRx in {3, -1}:
                Decided[i, 0] = 1
                Decided[i, 1] = 0
            elif QRx in {2, -2}:
                Decided[i, 0] = 1
                Decided[i, 1] = 1
            
            QPrev = Q[i]
        
        Decided = np.array(Decided).flatten()

        return Decided

    elif(Modbits==4):
        #Decision Regions for In-Phase component
        R1 = np.real(symbols) >= 2/np.sqrt(10)
        R2 = np.real(symbols) >= 0
        R3 = np.real(symbols) < 0
        R4 = np.real(symbols) <= -2/np.sqrt(10)
        #Decision Regions for Quadrature component
        R5 = np.imag(symbols) >= 2/np.sqrt(10)
        R6 = np.imag(symbols) >= 0
        R7 = np.imag(symbols) < 0
        R8 = np.imag(symbols) <= -2/np.sqrt(10)  
        
        #Defining the quadrant
        Q = np.zeros(len(symbols), dtype=int)
        Q[R1 & R5 | R1&R6&~R5 | ~R1&R2&R5 | ~R1&R2&R6&~R5] = 0  
        Q[~R4&R3&R5 | ~R4&R3&R6&~R5 | R4&R5 | R4&R6&~R5] = 1
        Q[R4&R7&~R8 | R4&R8 | ~R4&R3&R7&~R8 | ~R4&R3&R8] = 2
        Q[~R1&R2&R7&~R8 | ~R1&R2&R8 | R1&R7*~R8 | R1&R8] = 3

        #Defining the symbol inside the quadrants:
        S = np.zeros(len(symbols), dtype=int)
        S[~R1&R2&~R5&R6 | ~R1&R2&~R8&R7 | ~R4&R3&~R5&R6 | ~R4&R3&~R8&R7] = 0
        S[~R1&R2&R5 | R4&R6&~R5 | ~R4&R3&R8 | R1&R7&~R8] = 1
        S[R1&R6&~R5 | R4&R7&~R8 | ~R4&R3&R5 | ~R1&R2&R8] = 2
        S[R1&R5 | R1&R8 | R4&R5 | R4&R8] = 3

        #Received binary sequence
        for i in range(len(symbols)):
            QRx = Q[i] - QPrev
            if QRx == 0:
                Decided[i,0] = 0
                Decided[i,1] = 0
            elif QRx in {1,-3}:
                Decided[i,0] = 0
                Decided[i,1] = 1
            elif QRx in {3,-1}:
                Decided[i,0] = 1
                Decided[i,1] = 0
            elif QRx in {2,-2}:
                Decided[i,0] = 1
                Decided[i,1] = 1
        
            QPrev = Q[i]

            if S[i]==0:
                Decided[i,2] = 0
                Decided[i,3] = 0
            elif S[i] == 1:
                Decided[i,2] = 0
                Decided[i,3] = 1
            elif S[i] == 2:
                Decided[i,2] = 1
                Decided[i,3] = 0
            elif S[i] == 3:
                Decided[i,2] = 1
                Decided[i,3] = 1
        
        Decided = np.array(Decided).flatten()

        return Decided
    
    
    elif(Modbits==6):
        #Decision Regions for In-Phase component
        R9 = np.real(symbols) >= 0
        R10 = np.real(symbols) >= 2/np.sqrt(42)
        R11 = np.real(symbols) >= 4/np.sqrt(42)
        R12 = np.real(symbols) >= 6/np.sqrt(42)
        R13 = np.real(symbols) < 0
        R14 = np.real(symbols) <= -2/np.sqrt(42)
        R15 = np.real(symbols) <= -4/np.sqrt(42)
        R16 = np.real(symbols) <= -6/np.sqrt(42)
        #Decision Regions for Quadrature component
        R1 = np.imag(symbols) >= 0
        R2 = np.imag(symbols) >= 2/np.sqrt(42)
        R3 = np.imag(symbols) >= 4/np.sqrt(42)
        R4 = np.imag(symbols) >= 6/np.sqrt(42)
        R5 = np.imag(symbols) < 0
        R6 = np.imag(symbols) <= -2/np.sqrt(42)
        R7 = np.imag(symbols) <= -4/np.sqrt(42)
        R8 = np.imag(symbols) <= -6/np.sqrt(42)
        
        #Defining the quadrant (each SOP relates to 16 symbols in quadrant)
        Q = np.zeros(len(symbols), dtype=int)
        Q[R1&R9] = 0  
        Q[R1&R13] = 1
        Q[R5&R13] = 2
        Q[R5&R9] = 3

        #Defining the symbol inside the quadrants:
        S = np.zeros(len(symbols), dtype=int)
        S[R4&R12 | R4&R16 | R8&R16 | R8&R12] = 0
        S[R1&~R2&R12 | R4&R13&~R14 | R5&~R6&R16 | R9&~R10&R8] = 1
        S[R3&~R4&R12 | R15&~R16&R4 | R7&~R8&R16 | R11&~R12&R8] = 2
        S[R2&~R3&R12 | R14&~R15&R4 | R6&~R7&R16 | R10&~R11&R8] = 3
        S[R9&~R10&R4 | R1&~R2&R16 | R13&~R14&R8 | R5&~R6&R12] = 4
        S[R1&~R2&R9&~R10 | R13&~R14&R1&~R2 | R5&~R6&R13&~R14 | R9&~R10&R5&~R6] = 5
        S[R3&~R4&R9&~R10 | R15&~R16&R1&~R2 | R7&~R8&R13&~R14 | R5&~R6&R11&~R12] = 6
        S[R2&~R3&R9&~R10 | R1&~R2&R14&~R15 | R6&~R7&R13&~R14 | R10&~R11&R5&~R6] = 7 
        S[R4&R11&~R12 | R16&R3&~R4 | R8&R15&~R16 | R7&~R8&R12] = 8
        S[R1&~R2&R11&~R12 | R3&~R4&R13&~R14 | R5&~R6&R15&~R16 | R7&~R8&R9&~R10] = 9
        S[R3&~R4&R11&~R12 | R15&~R16&R3&~R4 | R7&~R8&R15&~R16 | R7&~R8&R11&~R12 ] = 10
        S[R2&~R3&R11&~R12 | R14&~R15&R3&~R4 | R15&~R16&R6&~R7 | R10&~R11&R7&~R8] = 11
        S[R4&R10&~R11 | R16&R2&~R3 | R8&R14&~R15 | R6&~R7&R12] = 12
        S[R1&~R2&R10&~R11 | R2&~R3&R13&~R14 | R5&~R6&R14&~R15 | R9&~R10&R6&~R7] = 13
        S[R3&~R4&R10&~R11 | R15&~R16&R2&~R3 | R7&~R8&R14&~R15 | R11&~R12&R6&~R7] = 14
        S[R2&~R3&R10&~R11 | R2&~R3&R14&~R15 | R6&~R7&R14&~R15 | R6&~R7&R10&~R11] = 15

        #Received binary sequence
        for i in range(len(symbols)):
            QRx = Q[i] - QPrev
            if QRx == 0:
                Decided[i,0] = 0
                Decided[i,1] = 0
            elif QRx in {1,-3}:
                Decided[i,0] = 0
                Decided[i,1] = 1
            elif QRx in {3,-1}:
                Decided[i,0] = 1
                Decided[i,1] = 0
            elif QRx in {2,-2}:
                Decided[i,0] = 1
                Decided[i,1] = 1
        
            QPrev = Q[i]

            if S[i]==0:
                Decided[i,2] = 0
                Decided[i,3] = 0
                Decided[i,4] = 0
                Decided[i,5] = 0
            if S[i]==1:
                Decided[i,2] = 0
                Decided[i,3] = 0
                Decided[i,4] = 0
                Decided[i,5] = 1
            if S[i]==2:
                Decided[i,2] = 0
                Decided[i,3] = 0
                Decided[i,4] = 1
                Decided[i,5] = 0
            if S[i]==3:
                Decided[i,2] = 0
                Decided[i,3] = 0
                Decided[i,4] = 1
                Decided[i,5] = 1
            if S[i]==4:
                Decided[i,2] = 0
                Decided[i,3] = 1
                Decided[i,4] = 0
                Decided[i,5] = 0
            if S[i]==5:
                Decided[i,2] = 0
                Decided[i,3] = 1
                Decided[i,4] = 0
                Decided[i,5] = 1
            if S[i]==6:
                Decided[i,2] = 0
                Decided[i,3] = 1
                Decided[i,4] = 1
                Decided[i,5] = 0
            if S[i]==7:
                Decided[i,2] = 0
                Decided[i,3] = 1
                Decided[i,4] = 1
                Decided[i,5] = 1
            if S[i]==8:
                Decided[i,2] = 1
                Decided[i,3] = 0
                Decided[i,4] = 0
                Decided[i,5] = 0
            if S[i]==9:
                Decided[i,2] = 1
                Decided[i,3] = 0
                Decided[i,4] = 0
                Decided[i,5] = 1
            if S[i]==10:
                Decided[i,2] = 1
                Decided[i,3] = 0
                Decided[i,4] = 1
                Decided[i,5] = 0
            if S[i]==11:
                Decided[i,2] = 1
                Decided[i,3] = 0
                Decided[i,4] = 1
                Decided[i,5] = 1
            if S[i]==12:
                Decided[i,2] = 1
                Decided[i,3] = 1
                Decided[i,4] = 0
                Decided[i,5] = 0
            if S[i]==13:
                Decided[i,2] = 1
                Decided[i,3] = 1
                Decided[i,4] = 0
                Decided[i,5] = 1
            if S[i]==14:
                Decided[i,2] = 1
                Decided[i,3] = 1
                Decided[i,4] = 1
                Decided[i,5] = 0
            if S[i]==15:
                Decided[i,2] = 1
                Decided[i,3] = 1
                Decided[i,4] = 1
                Decided[i,5] = 1

        Decided = np.array(Decided).flatten()
        return Decided
    
@benchmark(enable_benchmark)
def frequency_recovery(y, Rs, toggle_frequencyrecovery):
    #y is input signal. y must be obtained with one sample per symbol
    # Rs is symbol rate in symbols/second
    if(toggle_frequencyrecovery):
        f = np.arange(-1/2 + 1/len(y), 1/2, 1/len(y))*Rs
        Ts = 1/Rs

        signal_power_4 = y** 4

        # Compute the FFT and take the absolute value
        spectrum = np.fft.fft(signal_power_4)

        # Shift the zero frequency component to the center
        SignalSpectrum = np.fft.fftshift(np.abs(spectrum))

        max_index = np.argmax(SignalSpectrum)  # Find the index of the maximum value
        Delta_f = (1/4) * f[max_index]  # Calculate the frequency offset

        k = np.arange(len(y))
        z = y*np.exp(-1j*2*np.pi*Delta_f*Ts*k)

        return z, Delta_f
    else:
        return y, 0