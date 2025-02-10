import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
import distribution_matcher as DM
#Probabilistic Amplitude Shaping Architecture

def nCr(n, r):
    if r > n:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
    
def PCS_encoder(C, bits, k, blocks, Modbits):
    bitsI = bits[::2] #in-phase bits
    bitsQ = bits[1::2] #quadrature bits
    N = np.sum(C)
    amplitudesI = DM.DM_encode(C, bitsI, k, blocks).reshape((blocks,N))
    amplitudesQ = DM.DM_encode(C, bitsQ, k, blocks).reshape((blocks,N))
    XI = PCS_signs(amplitudesI, Modbits).flatten()
    XQ = PCS_signs(amplitudesQ, Modbits).flatten()
    X = XI + 1j*XQ
    return X

def PCS_signs(A, Modbits):
    #A: Input matrix of amplitudes, with block length N
    #Modbits: Modulation format
    #NPol: Number of polarisations

    if(Modbits==4): #16-QAM (2**m ASK constellation, m=2, so 2**(m-1)=2 amplitudes)
        #G = [I|P] systemartic generator matrix
        #P is a ((m-1)N),(N) = (N,N) matrix
        #A[i] is length-N vector
        #b_A[i] is a length N(m-1) = N vector
        #b_S[i] = b_A[i]*P is a length-N vector of signs represented by bits
        #P = [[1,0,0,0,1],[0,1,0,1,0],[1,0,1,0,1],[1,0,1,0,1],[1,0,0,1,0]] #Large enough P gives close to uniform sign distribution
        
        m=2
        n_P = A.shape[1]*m
        k_P = A.shape[1]*(m-1)
        
        prob=0.03875
        P = np.random.choice([0, 1], size=(k_P, n_P-k_P), p=[1-prob, prob]) #to be replaced

        b_A = np.where(A == 1, 0, 1)#Replace ampltidues by a binary label
        b_S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Bits representing sign blocks
        S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Sign blocks
        X = np.empty((A.shape[0],A.shape[1]), dtype=int)
        for i in range(b_A.shape[0]): #For each block
            b_S[i] = np.dot(b_A[i],P)
            S[i] = np.where(b_S[i]==0, -1, 1)
            X[i] = A[i]*S[i]
        
        return X

    elif(Modbits==6): #64-QAM (2**m ASK constellation, m=3, so 2**(m-1)=4 amplitudes)
       
        prob=0.018411 
        m=3
        n_P = A.shape[1]*m
        k_P = A.shape[1]*(m-1)
        
        P = np.random.choice([0, 1], size=(k_P, n_P-k_P), p=[1-prob, prob]) #to be replaced

        b_A = []
        for row in range(A.shape[0]):
            b_A_row = []
            for i in A[row]:
                if(i==1):
                    b_A_row.append(1)
                    b_A_row.append(0)
                if(i==3):
                    b_A_row.append(1)
                    b_A_row.append(1)
                if(i==5):
                    b_A_row.append(0)
                    b_A_row.append(1)
                if(i==7):
                    b_A_row.append(0)
                    b_A_row.append(0)
            b_A.append(b_A_row)
        
        b_A = np.array(b_A)
        b_S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Bits representing sign blocks
        S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Sign blocks
        X = np.empty((A.shape[0],A.shape[1]), dtype=int)
        for i in range(b_A.shape[0]): #For each block
            b_S[i] = np.dot(b_A[i],P)

            b_S[i] = np.random.choice([0,1], size=A.shape[1], p=[0.5,0.5]) #Overwriting for testing with actual uniform distributed signs

            S[i] = np.where(b_S[i]==0, -1, 1)
            X[i] = A[i]*S[i]
        
        return X

Modbits = 6
blocks = 1000

if(Modbits==4):
    signal_points = [1,3]
    λ = 0.06
    N_target = 50
    const = 0
    for i in signal_points:
        const += np.exp(-λ*np.abs(i)**2)
    C = [int(N_target*np.exp(-λ)/const),int(N_target*np.exp(-λ*9)/const),0,0]
    N = np.sum(C)
    k = int(np.floor(math.log2(nCr(N,C[1]))))
    bits = np.random.randint(0, 2, size= k*blocks*2) #In-Phase and Quadrature

if(Modbits==6):
    signal_points = [1,3,5,7]
    λ = 0.04
    N_target = 30
    const = 0
    for i in signal_points:
        const += np.exp(-λ*np.abs(i)**2)
    C = [int(N_target*np.exp(-λ)/const),int(N_target*np.exp(-λ*9)/const),int(N_target*np.exp(-λ*25)/const),int(N_target*np.exp(-λ*49)/const)]
    k=int(np.floor(math.log2(math.factorial(C[0]+C[1]+C[2]+C[3])/(math.factorial(C[0])*math.factorial(C[1])*math.factorial(C[2])*math.factorial(C[3])))))
    N = np.sum(C)
    bits = np.random.randint(0, 2, size= k*blocks*2) #In-Phase and Quadrature

X = PCS_encoder(C, bits, k, blocks, Modbits)

XI_abs = np.abs(np.real(X))
XQ_abs = np.abs(np.imag(X))

bits_decodedI = DM.DM_decode(XI_abs, C, k, blocks)
bits_decodedQ = DM.DM_decode(XQ_abs, C, k, blocks)

bits_decoded = np.array([bit for pair in zip(bits_decodedI, bits_decodedQ) for bit in pair])

symbol_counts = Counter(X)
symbols = list(symbol_counts.keys())


print('k:', k)
print('N:', np.sum(C))
print('λ:', λ)
print('C :', C)
print('Blocks:',blocks)
print('Rate:', 2*k/N, 'bits/symbol')



print(bits, 'Source Bits')
print(X,'CCDM Symbols')
print(bits_decoded, 'Decoded Bits')

if(np.array_equal(bits_decoded,bits)):
    print('No Errors')
else:
    print('Errors')
print(np.where(bits!=bits_decoded))

probabilities = np.array(list(symbol_counts.values()))/len(X)

# Convert complex numbers to real and imaginary parts for plotting
real_parts = np.array([c.real for c in symbols])
imag_parts = np.array([c.imag for c in symbols])

# Normalize frequencies for color mapping (0 to 1 scale)
norm = (probabilities - min(probabilities)) / (max(probabilities) - min(probabilities) + 1e-9)  # Avoid division by zero
colors = cm.coolwarm(norm)

# Create a 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Define bar width
bar_width = 0.4

# Plot bars with color-coding
for x, y, h, color in zip(real_parts, imag_parts, probabilities, colors):
    ax.bar3d(x, y, 0, bar_width, bar_width, h, color=color, alpha=0.9)

# Labels and title
ax.set_title('Probabilistically Shaped Symbols')
ax.set_xlabel('In-Phase')
ax.set_ylabel('Quadrature')
ax.set_zlabel('Probability of Symbol')

# Show the plot
plt.show()

