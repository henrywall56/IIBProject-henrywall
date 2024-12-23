import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve #upfirdn for up/down sampling
import time
from scipy.fft import fft, ifft, fftshift, ifftshift

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
def generate_original_bits(num_symbols, Modbits, NPol):
    #If NPol == 1: generate 1D array of bits
    #If NPol == 2: generate 2D array of bits
    bits = np.random.randint(0, 2, size=num_symbols * Modbits*NPol)

    if(NPol==2):
        # Reshape it into a 2D array
        bits = bits.reshape(2,num_symbols*Modbits)
    return bits

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
def generate_symbols(original_bits, Modbits, NPol, toggle_DE):
    #Wrapper for symbols generation
    #NPol is number of polarisations used

    if(NPol==1):
        if(toggle_DE==False):
            # Generate symbols
            if(Modbits==2): #QPSK
                symbols = generate_QPSK_symbols(original_bits)
        
            elif(Modbits==4): #16QAM
                symbols = generate_16qam_symbols(original_bits)
                
            elif(Modbits==6): #64QAM
                symbols = generate_64qam_symbols(original_bits) 
                
        else:
            # Generate symbols using differential encoding
            if(Modbits==2): #16QAM
                symbols = Differential_Encoding_qpsk(original_bits)
                
                
            elif(Modbits==4): #16QAM
                symbols = Differential_Encoding_16qam (original_bits)
                
                
            elif(Modbits==6): #64QAM
                symbols = Differential_Encoding_64qam(original_bits) 

    elif(NPol==2):
        if(toggle_DE==False):
            # Generate symbols
            if(Modbits==2): #QPSK
                symbols0 = generate_QPSK_symbols(original_bits[0,:])
                symbols1 = generate_QPSK_symbols(original_bits[1,:])
        
            elif(Modbits==4): #16QAM
                symbols0 = generate_16qam_symbols(original_bits[0,:])
                symbols1 = generate_16qam_symbols(original_bits[1,:])

            elif(Modbits==6): #64QAM
                symbols0 = generate_64qam_symbols(original_bits[0,:]) 
                symbols1 = generate_64qam_symbols(original_bits[1,:])
        else:
            # Generate symbols using differential encoding
            if(Modbits==2): #16QAM
                symbols0 = Differential_Encoding_qpsk(original_bits[0,:])
                symbols1 = Differential_Encoding_qpsk(original_bits[1,:])
                
            elif(Modbits==4): #16QAM
                symbols0 = Differential_Encoding_16qam (original_bits[0,:])
                symbols1 = Differential_Encoding_16qam (original_bits[1,:])
                
            elif(Modbits==6): #64QAM
                symbols0 = Differential_Encoding_64qam(original_bits[0,:]) 
                symbols1 = Differential_Encoding_64qam(original_bits[1,:]) 
        symbols = np.array([symbols0, symbols1])

    return symbols


@benchmark(enable_benchmark)
def pulseshaping(symbols, sps, RRCimpulse, NPol, toggle):
    #performs pulse shaping of symbols with RRC filter. 
    #filter parameters are span and rolloff
    #sequence of symbols upsampled to sps samples per symbol then applied RRC filter
    #Applies pulse shaping two NPol polarisations
    if(toggle==True):
        if(NPol==1):
        
            #upsample the symbols by inserting (sps-1) zeros between each symbol
            upsampled = np.zeros(sps*len(symbols), dtype=complex)
            for i in range(0, sps*len(symbols), sps):
                upsampled[i] = symbols[i//sps]

            shaped = convolve(upsampled, RRCimpulse, mode='same')
            shaped = shaped/(np.sqrt(np.mean(abs(shaped)**2)))

            return shaped
        elif(NPol==2):
            #upsample the symbols by inserting (sps-1) zeros between each symbol
            upsampled0 = np.zeros(sps*len(symbols[0]), dtype=complex)
            upsampled1 = np.zeros(sps*len(symbols[1]), dtype=complex)
            for i in range(0, sps*len(symbols[0]), sps):
                upsampled0[i] = symbols[0][i//sps]
                upsampled1[i] = symbols[1][i//sps]

            shaped0 = convolve(upsampled0, RRCimpulse, mode='same')
            shaped1 = convolve(upsampled1, RRCimpulse, mode='same')
            shaped0 = shaped0/(np.sqrt(np.mean(abs(shaped0)**2)))
            shaped1 = shaped1/(np.sqrt(np.mean(abs(shaped1)**2)))

            return np.array([shaped0, shaped1])

    else:
        return symbols

@benchmark(enable_benchmark)
def add_noise(signal, snrb_db, sps, Modbits, NPol, toggle_AWGNnoise): 
    #addition of circular Gaussian noise to transmitted signal
    #snrb_db snr per symbol in dB in transmitted signal
    #Modbits per symbol eg 16QAM or 64QAM etc.
    #sps samples per symbol
    if(toggle_AWGNnoise==True):
        if(NPol==1):
            snr = 10 ** (snrb_db / 10) #dB to linear (10 since power)

            stdev= np.sqrt(np.mean(abs(signal)**2)*sps/(2*snr))
            noise = stdev * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

            return signal + noise 
        elif(NPol==2):
            snr = 10 ** (snrb_db / 10) #dB to linear (10 since power)

            stdev0= np.sqrt(np.mean(abs(signal[0])**2)*sps/(2*snr))
            stdev1= np.sqrt(np.mean(abs(signal[1])**2)*sps/(2*snr))
            noise0 = stdev0 * (np.random.randn(len(signal[0])) + 1j * np.random.randn(len(signal[0])))
            noise1 = stdev1 * (np.random.randn(len(signal[1])) + 1j * np.random.randn(len(signal[1])))

            return np.array([signal[0]+noise0, signal[1]+noise1], dtype=complex)

    else:
        return signal

@benchmark(enable_benchmark)
def matched_filter(signal, pulse_shape, NPol, toggle):
    #received_signal is original signals + noise
    #pulse shape eg raised-root-cosine, square etc.
    if(toggle==True):
        if(NPol==1):
            filtered = convolve(signal, pulse_shape, mode='same') 
            filtered = filtered/(np.sqrt(np.mean(abs(filtered)**2)))
            #should have peaks where received signal matches pulse shape, maximising SNR
            return filtered
        elif(NPol==2):
            filtered0 = convolve(signal[0], pulse_shape, mode='same') 
            filtered0 = filtered0/(np.sqrt(np.mean(abs(filtered0)**2)))
            filtered1= convolve(signal[1], pulse_shape, mode='same') 
            filtered1 = filtered1/(np.sqrt(np.mean(abs(filtered1)**2)))
            return np.array([filtered0, filtered1], dtype=complex)

    else:
        return signal/(np.sqrt(np.mean(abs(signal)**2)))

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
def downsample(signal, sps, NPol, toggle):
    if(toggle==True):
        if(NPol==1):
            #downsample a signal that has sps samples per symbol
            downsampled = signal[0::sps]
            return downsampled
        elif(NPol==2):
            downsampled0 = signal[0][0::sps]
            downsampled1 = signal[1][0::sps]
            return np.array([downsampled0, downsampled1], dtype=complex)
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
def decode_symbols(symbols, Modbits, NPol):
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

    if(NPol==1):
        bits = np.zeros(len(symbols) * Modbits, dtype=int)
        for i, symbol in enumerate(symbols):
            bits[i * Modbits:(i + 1) * Modbits] = reverse_mapping[int(symbol.real*np.sqrt(root))+int(symbol.imag*np.sqrt(root))*1j] #bit stream
        return bits
    elif(NPol==2):
        bits = np.zeros((NPol, len(symbols[0]) * Modbits), dtype=int)
        
        for i, symbol in enumerate(symbols[0]):
            bits[0][i * Modbits:(i + 1) * Modbits] = reverse_mapping[int(symbol.real*np.sqrt(root))+int(symbol.imag*np.sqrt(root))*1j] #bit stream
        for i, symbol in enumerate(symbols[1]):
            bits[1][i * Modbits:(i + 1) * Modbits] = reverse_mapping[int(symbol.real*np.sqrt(root))+int(symbol.imag*np.sqrt(root))*1j] #bit stream
        return bits


@benchmark(enable_benchmark)
def add_phase_noise(symbols, Nsymb, sps, Rs, Linewidth, toggle):
    #This function adds phase noise to the transmitted symbols, modelled by a Wiener Process.
    #This approach only used when not considering physical process of electrical field of laser
    #This function is no longer used

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

@benchmark(enable_benchmark)
def Laser(laser_power, Linewidth, sps, Rs, num_symbols, NPol, toggle_phasenoise):
    #Simulates a laser as a continuous wave optical source.
    #Phase noise is inserted into the electrical field E
    #laser_power: total laser power in dBm
    #sps: samples per signal in oversampled signal
    #Rs: symbol rate in symbols/second
    #num_symbols: number of symbols transmitted in each polarisation
    #Linedwith: of laser in Hz
    #NPol: number of polarisations

    laser_power_linear = 10**(-3) * (10**(laser_power/10))
    if NPol == 1:
        E = np.ones(sps*num_symbols, dtype=complex)*np.sqrt(laser_power_linear)
    elif NPol == 2:
        E = np.array([np.ones(sps*num_symbols, dtype=complex)*np.sqrt(laser_power_linear/2), np.ones(sps*num_symbols, dtype=complex)*np.sqrt(laser_power_linear/2)])
    
    if(toggle_phasenoise==True):
        T = 1/(sps*Rs) #Period between samples at the oversampled transmit signal

        #Calculating phase noise:
        Var = 2*np.pi*Linewidth*T           
        delta_theta = np.sqrt(Var)*np.random.randn(sps*num_symbols)
        theta = np.cumsum(delta_theta) #an array of phase shift vs time
        if(NPol==1):
            Erot = E*np.exp(1j*theta)
        elif(NPol==2):
            Erot = np.array([E[0]*np.exp(1j*theta), E[1]*np.exp(1j*theta)], dtype=complex)
        
        return Erot, theta
    
    else:
        return E, np.zeros(sps*num_symbols)

@benchmark(enable_benchmark)
def IQModulator(symbols, Einput, Vpi, Bias, MaxExc, MinExc, NPol):
    #Simulates an in-phase and quadrature modulator (IQM). The electrical field 'Einput' of the optical signal is
    #modulated according to the signal "symbols", generating Eoutput. 
    #the Mach-Zehnder modulators (MZMs) that compose the IQM are considered identical.
    #symbols: NPol dimensional array of symbols
    #Einput: NPol dimensional optical carrier symbol 
    #MZM parameters:
    #Vpi: MZM Vpi
    #Bias: Bias Voltage
    #MaxExc: Upper limit for the excursion of the modulation signal. The modulation signal is scaled so it fits in the the excursion
    #MinExc: Lower limit for the excursion of the modulation signal
    #Produces an NPol dimensional IQM output signal

    #In-Phase and Quadrature components of electrical signal:
    if(NPol==1):
        mI = np.real(symbols)
        mI = mI/max(abs(mI))
        mQ = np.imag(symbols)
        mQ = mQ/max(abs(mQ))

        #Setting signal excursion
        mI = mI*(MaxExc-MinExc)/2
        mQ = mQ*(MaxExc-MinExc)/2

        #Obtaining signals after considering bias:
        vI = mI + Bias
        vQ = mQ + Bias
        
        #Phase modulation in the I and Q branches:
        PhiI = np.pi*vI/Vpi
        PhiQ = np.pi*vQ/Vpi

        #IQM output signal:
        EOutput = (0.5*np.cos(0.5*PhiI) + 0.5j*np.cos(0.5*PhiQ))*Einput

        return EOutput

    elif(NPol==2):
        mI0 = np.real(symbols[0])
        mI0 = mI0/max(abs(mI0))
        mI1 = np.real(symbols[1])
        mI1 = mI1/max(abs(mI1))

        mQ0 = np.imag(symbols[0])
        mQ0 = mQ0/max(abs(mQ0))
        mQ1 = np.imag(symbols[1])
        mQ1 = mQ1/max(abs(mQ1))

        #Setting signal excursion
        mI0 = mI0*(MaxExc-MinExc)/2
        mQ0 = mQ0*(MaxExc-MinExc)/2
        mI1 = mI1*(MaxExc-MinExc)/2
        mQ1 = mQ1*(MaxExc-MinExc)/2

        #Obtaining signals after considering bias:
        vI0 = mI0 + Bias
        vQ0 = mQ0 + Bias
        vI1 = mI1 + Bias
        vQ1 = mQ1 + Bias
        
        #Phase modulation in the I and Q branches:
        PhiI0 = np.pi*vI0/Vpi
        PhiQ0 = np.pi*vQ0/Vpi
        PhiI1 = np.pi*vI1/Vpi
        PhiQ1 = np.pi*vQ1/Vpi

        #IQM output signal:
        EOutput0 = (0.5*np.cos(0.5*PhiI0) + 0.5j*np.cos(0.5*PhiQ0))*Einput[0]
        EOutput1 = (0.5*np.cos(0.5*PhiI1) + 0.5j*np.cos(0.5*PhiQ1))*Einput[1]

        return np.array([EOutput0, EOutput1], dtype=complex)


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
def BPS(z, Modbits, N, B, NPol, toggle_phasenoisecompensation):
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

        
        ThetaTestMatrix = np.tile(np.exp(-1j*ThetaTest),(L,1)) #L row x B column matrix, each row is phase angle vector

        if(NPol==1):
            zB_V = np.concatenate([np.zeros(L // 2, dtype=complex), z, np.zeros(L // 2, dtype=complex)])
            zB_V = convmtx(zB_V,L)
            zB_V = np.flipud(zB_V[:, L:-L+1])

            zBlocks = zB_V

        elif(NPol==2):
            ThetaTestMatrix = np.stack((ThetaTestMatrix, ThetaTestMatrix) , axis=2)
            zB_V = np.concatenate([np.zeros(L // 2, dtype=complex), z[0], np.zeros(L // 2, dtype=complex)])
            zB_V = convmtx(zB_V,L)
            zB_V = np.flipud(zB_V[:, L:-L+1])

            zB_H = np.concatenate([np.zeros(L // 2, dtype=complex), z[1], np.zeros(L // 2, dtype=complex)])
            zB_H = convmtx(zB_H,L)
            zB_H = np.flipud(zB_H[:, L:-L+1])

            zBlocks = np.stack((zB_V, zB_H), axis=2)

        if(NPol==1):
            ThetaPU = np.zeros(zBlocks.shape[1]+1)
            ThetaPrev = 0.0

        elif(NPol==2):
            ThetaPU = np.zeros((zBlocks.shape[1]+1, 2))
            ThetaPrev = np.zeros((1, NPol))

        if(NPol==1):
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

                Theta = ThetaTest[im]
                
                ThetaPU[i] = Theta + np.floor(0.5-(Theta-ThetaPrev)/p)*p

                ThetaPrev = ThetaPU[i]
            ThetaPU[-1]=ThetaPrev
            v = z*np.exp(-1j*ThetaPU)
            return v, ThetaPU

        elif(NPol==2):
            #Phase noise estimates
            for i in range(zBlocks.shape[1]): #Over columns
                
                zrot = np.repeat(zBlocks[:,i,:][:, np.newaxis, :], B, axis=1) * ThetaTestMatrix #ith column repeated 
                zrot_decided = np.zeros((zrot.shape[0], zrot.shape[1], zrot.shape[2]), dtype=complex)

                for j in range(zrot.shape[0]): #Decision of rotated symbols
                    for pol in range(zrot.shape[2]):
                        zrot_decided[j,:,pol] = max_likelihood_decision(zrot[j,:,pol], Modbits)

                #intermediate sum to be minimised
                m = np.sum(abs(zrot-zrot_decided)**2,axis=0)
                
                #estimating phase noise as angle that minimises m
                im = np.argmin(m, axis=0)

                Theta = ThetaTest[im].reshape(1,2)
                
                ThetaPU[i, :] = Theta + np.floor(0.5-(Theta-ThetaPrev)/p)*p

                ThetaPrev = ThetaPU[i,:]
            ThetaPU[-1]=ThetaPrev

            v0 = z[0]*np.exp(-1j*ThetaPU[:,0]) #Matlab deals with 2 x column vectors, so swap to 2 x row vectors
            v1 = z[1]*np.exp(-1j*ThetaPU[:,1])
            return np.array([v0,v1]), ThetaPU
        
    else:
        if(NPol==1):
            return z, np.zeros(len(z))
        elif(NPol==2):
            return z, np.zeros((len(z[0]),NPol))

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
def frequency_recovery(y, Rs, NPol, toggle_frequencyrecovery):
    #y is input signal. y must be obtained with one sample per symbol
    # Rs is symbol rate in symbols/second
    if(toggle_frequencyrecovery==True):
        if(NPol==1):
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
            
            print(f'Delta_f from frequency recovery: {Delta_f}Hz')
            return z
        elif(NPol==2):
            f = np.arange(-1/2 + 1/len(y[0]), 1/2, 1/len(y[0]))*Rs
            Ts = 1/Rs

            signal_power_4 = y[0]** 4

            # Compute the FFT and take the absolute value
            spectrum = np.fft.fft(signal_power_4)

            # Shift the zero frequency component to the center
            SignalSpectrum = np.fft.fftshift(np.abs(spectrum))

            max_index = np.argmax(SignalSpectrum)  # Find the index of the maximum value
            Delta_f = (1/4) * f[max_index]  # Calculate the frequency offset

            k = np.arange(len(y[0]))
            z0 = y[0]*np.exp(-1j*2*np.pi*Delta_f*Ts*k)
            z1 = y[1]*np.exp(-1j*2*np.pi*Delta_f*Ts*k)
            
            print(f'Delta_f from frequency recovery: {Delta_f}Hz')
            return np.array([z0,z1], dtype=complex)
    else:
        return y
    
@benchmark(enable_benchmark)
def add_chromatic_dispersion(symbols, sps, Rs, D, Clambda, L, NPol, toggle):
    #symbols: to be transmitted
    #sps: samples per symbol
    #Rs: symbol rate in (symbols/s)
    #D: Dispersion parameter in (ps/(nm x km))
    #CLambda: Central lambda in (m)
    #L: Fibre length in (m)
    #NPol: Number of polarisations

    if(toggle==False):
        return symbols
    else:

        c = 299792458 #speed of light

        #Dispersion
        D = D*(1/10**12)/((1/10**9)*1000) #scale to m

        #Frequency vector:
        if(NPol==1):
            w = 2 * np.pi * (np.arange(-1/2, 1/2, 1/len(symbols))) * sps * Rs
        elif(NPol==2):
            w = 2 * np.pi * (np.arange(-1/2, 1/2, 1/len(symbols[0]))) * sps * Rs

        #Calculating the CD frequency response:
        G = np.exp(1j * ((D*Clambda**2)/(4*np.pi*c))*L*(w**2)) #Quadratic phase shift

        if(NPol==1):
            output = ifft(ifftshift(G * fftshift(fft(symbols))))
            return output
        elif(NPol==2):
            output0 = ifft(ifftshift(G * fftshift(fft(symbols[0]))))
            output1 = ifft(ifftshift(G * fftshift(fft(symbols[1]))))
            return np.array([output0, output1], dtype=complex)

        #fft shift w first then don't have to do double shift - fft wants it to look like this - better way to define

@benchmark(enable_benchmark)
def CD_compensation(input, D1, L, CLambda, Rs, NPol, sps, NFFT, NOverlap, toggle_CD):
    #input: input signal
    #D: Dispersion parameter (ps\(ns*km))
    #L: Fiber length (m)
    #CLambda: Central wavelength (m)
    #Rs: Symbol rate 
    #NPol: Number of polarisations used
    #sps: number of samples per symbol in input signal "in"
    #NFFT: FFT size
    #NOverlap: Over lap sized. if odd, forced to nearest even number > Noveralp
    if(toggle_CD==False):
        return input
    
    c = 299792458
    D = D1/10**6

    #index for coeffificent calculation and Nyquist frequency
    n = np.arange(-NFFT//2, NFFT//2, 1)
    
    fN = sps*Rs/2

    if(NPol==1):
        #Calculating CD frequency response
        HCD = np.exp((-1j*np.pi*(CLambda**2)*D*L/c)*(n*2*fN/NFFT)**2)
        
        #NOverlap made even:
        NOverlap = NOverlap + NOverlap%2

        #Extending input signal so blocks are properly formed
        AuxLen = len(input)/(NFFT-NOverlap)

        if(AuxLen != np.ceil(AuxLen)):
            NExtra = np.ceil(AuxLen)*(NFFT-NOverlap) - len(input)
            input = np.concatenate([input[int(-NExtra//2):], input, input[:int(NExtra//2)]], dtype=complex)
            
        else:
            NExtra = NOverlap
            input = np.concatenate([input[-NExtra//2:], input, input[:NExtra//2]],dtype=complex)

        #Blocks
        Blocks = input.reshape( NFFT-NOverlap, len(input)//(NFFT-NOverlap),order='F') #order to get same reshape as matlab

        output = np.zeros(Blocks.shape, dtype=complex)
        overlap = np.zeros(NOverlap, dtype=complex)

        #Compensating for chromatic dispersion
        for i in range(len(Blocks[1])):
            #input block with overlap:
            InB = np.concatenate([overlap,Blocks[:,i]], dtype=complex)
            #FFT of input block
            InBFreq = fftshift(fft(InB))
            #Filtering in freq. domain
            OutFDEFreq = InBFreq * HCD
            #IFFT of block after filtering
            OutFDE = ifft(ifftshift(OutFDEFreq))
            #Overlap
            overlap = InB[-NOverlap:]
            #Output block
            OutB  = OutFDE[NOverlap//2:-NOverlap//2]
            #Assigning samples to output signal:
            output[:,i] = OutB
 
        outputT = output.T
        output = outputT.reshape(-1)
 
        #Quantity of samples to discard
        DInit = int((NExtra+NOverlap)//2)
        DFin = int((NExtra-NOverlap)//2)
        
        output = output[DInit:-DFin] 

        return output

    elif(NPol==2):
        output0 = CD_compensation(input[0], D1, L, CLambda, Rs, 1, sps, NFFT, NOverlap, toggle_CD)
        output1 = CD_compensation(input[1], D1, L, CLambda, Rs, 1, sps, NFFT, NOverlap, toggle_CD)

        return np.array([output0,output1])

