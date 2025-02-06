import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
#Attempt 1 at PAS

def DM_rescale(bits, k, C, Modbits, NPol):
    #Distribution Matcher With Interval Rescaling
    #bits: Input bit sequence. Length must be an multiple of k.
    #Modbits: Modulation format (2**(Modbits)-QAM)
    #NPol: Number of polarisations
    #k: number of bits in input blocks
    #N: number of symbols in output blocks
    #C: Composition of output blocks, eg number of each amplitude in output blocks.

    N = np.sum(C)

    if(Modbits==4): #16-QAM
        v = bits.reshape((len(bits)//k, k)) #Make the blocks of input bits, length k
        x = np.empty((v.shape[0], N), dtype=int) #Initialise the blocks of output symbols, length n
        for i in range(v.shape[0]):
            w = 1 #Initialise initial interval width to 1
            I = 0
            n0 = C[0]
            n1 = C[1]
            for j in range(k):
                I += v[i][j]*(1/2**(j+1)) #Initialise initial I to d(v)
            for h in range(N):
                if(I>=n0/(N-h)):
                    x[i][h] = 3
                    I = I - n0/(N-h)
                    w = w*n1/(N-h)
                    I = I/w #rescale interval
                    w=1 #interval width back to 1
                    n1 = n1 -1 
                else:
                    x[i][h] = 1
                    w = w*n0/(N-h)
                    I=I/w #rescale interval
                    w=1 #interval width back to 1
                    n0 = n0 - 1
        return x #Unsigned
    
def PCS_encoder(bits, k, C, Modbits, NPol):
    bitsI = bits[::2] #in-phase bits
    bitsQ = bits[1::2] #quadrature bits
    amplitudesI = DM_rescale(bitsI, k, C, 4, 1)
    amplitudesQ = DM_rescale(bitsQ, k, C, 4, 1)
    XI = PCS_signs(amplitudesI, 4, 1).flatten()
    XQ = PCS_signs(amplitudesQ, 4, 1).flatten()
    X = XI + 1j*XQ
    return X


def DM(bits, k, C, Modbits, NPol):
    #Distribution Matcher
    #bits: Input bit sequence. Length must be an multiple of k.
    #Modbits: Modulation format (2**(Modbits)-QAM)
    #NPol: Number of polarisations
    #k: number of bits in input blocks
    #N: number of symbols in output blocks
    #C: Composition of output blocks, eg number of each amplitude in output blocks.

    N = np.sum(C)

    if(Modbits==4): #16-QAM
        v = bits.reshape((len(bits)//k, k)) #Make the blocks of input bits, length k
        x = np.empty((v.shape[0], N), dtype=int) #Initialise the blocks of output symbols, length n
        for i in range(v.shape[0]):
            w = 1 #Initialise initial interval width to 1
            I = 0
            n0 = C[0]
            n1 = C[1]
            for j in range(k):
                I += v[i][j]*(1/2**(j+1)) #Initialise initial I to d(v)
            for h in range(N):
                if(I>=w*n0/(N-h)):
                    x[i][h] = 3
                    I = I - w*n0/(N-h)
                    w = w*n1/(N-h)
                    n1 = n1 -1 
                else:
                    x[i][h] = 1
                    w = w*n0/(N-h)
                    n0 = n0 - 1
                print(w)
        
        return x #Unsigned

def PCS_signs(A, Modbits, NPol):
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
        prob=0.01156
        
        P = np.random.choice([0, 1], size=(A.shape[1],A.shape[1] ), p=[1-prob, prob]) #to be replaced

        b_A = np.where(A == 1, 0, 1)#Replace ampltidues by a binary label
        b_S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Bits representing sign blocks
        S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Sign blocks
        X = np.empty((A.shape[0],A.shape[1]), dtype=int)
        for i in range(b_A.shape[0]): #For each block
            b_S[i] = np.dot(b_A[i],P)
            S[i] = np.where(b_S[i]==0, -1, 1)
            X[i] = A[i]*S[i]
        
        return X
    
def PCS_encoder_norescale(bits, k, C, Modbits, NPol):
    bitsI = bits[::2] #in-phase bits
    bitsQ = bits[1::2] #quadrature bits
    amplitudesI = DM(bitsI, k, C, 4, 1)
    amplitudesQ = DM(bitsQ, k, C, 4, 1)
    XI = PCS_signs(amplitudesI, 4, 1).flatten()
    XQ = PCS_signs(amplitudesQ, 4, 1).flatten()
    X = XI + 1j*XQ
    return X


C = [100,60]
N=np.sum(C) #symbol block length
k=10000 #bit block length
l = k*2**7
bits = np.random.randint(0, 2, size= l)

X = PCS_encoder(bits, k, C, 4, 1)

symbol_counts = Counter(X)
symbols = list(symbol_counts.keys())
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
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
ax.set_zlabel('Probability')

# Show the plot
plt.show()

# C = [100,60]
# N=np.sum(C) #symbol block length
# k=10000 #bit block length
# b = 2**1 #number of blocks
# l = k*b #number of bits
# bits = np.random.randint(0, 2, size= l)
# A = DM_rescale(bits, k, C, 4, 1).flatten()

# count3 = np.count_nonzero(A == 3)
# count1 = np.count_nonzero(A == 1)
# labels = ['1', '3']
# values = [count1/len(A), count3/len(A)]

# # Create bar chart
# plt.bar(labels, values, color=['blue', 'green'])
# plt.xlabel("Number")
# plt.ylabel("Frequency")
# plt.title("Frequency of 1 and 3 in Vector A")

# # Show the plot
# plt.show()