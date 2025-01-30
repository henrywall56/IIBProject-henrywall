import numpy as np
import matplotlib.pyplot as plt

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
        P = [[1,0,0,0,1],[0,1,0,1,0],[1,0,1,0,1],[1,0,1,0,1],[1,0,0,1,0]] #Large enough P gives close to uniform sign distribution

        b_A = np.where(A == 1, 0, 1)#Replace ampltidues by a binary label
        b_S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Bits representing sign blocks
        S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Sign blocks
        X = np.empty((A.shape[0],A.shape[1]), dtype=int)
        for i in range(b_A.shape[0]): #For each block
            b_S[i] = np.dot(b_A[i],P)
            S[i] = np.where(b_S[i]==0, -1, 1)
            X[i] = A[i]*S[i]
        
        return X


C = [4,1]
k=3
l = k*2**15
bits = np.random.randint(0, 2, size= l)

amplitudes = DM(bits, k, C, 4, 1)

X = PCS_signs(amplitudes, 4, 1)

count_3 = np.count_nonzero(X==3)
count_1 = np.count_nonzero(X==1)
count_n3 = np.count_nonzero(X==-3)
count_n1 = np.count_nonzero(X==-1)


counts = np.array([count_n3/(l*N/k),count_n1/(l*N/k),count_1/(l*N/k), count_3/(l*N/k)])

plt.bar(['-3','-1','1','3'], counts)
plt.show()