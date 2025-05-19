import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
from scipy.special import expit
from scipy.special import logsumexp
#expit function computes 1/(1+np.exp(-llr)) without overflow issues
#logsumexp calculates log of sum of exponents of operands

testing=False
if(testing==False):
    import PAS.DMencode as DMencode
    import PAS.DMdecode as DMdecode
    from PAS.ldpc_jossy.py import ldpc
    import sys
    import os
    source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(source_dir)
    import functions as f

else:
    import DMencode as DMencode
    import DMdecode as DMdecode
    from ldpc_jossy.py import ldpc
    import sys
    import os
    source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(source_dir)
    import functions as f
#Probabilistic Amplitude Shaping Architecture

def PAS_truncate(processed_rx, source_symbols, original_bits, block_len, NPol,k):
    if(NPol==1):
        N = len(source_symbols)
        trun_len = N%block_len
        num_blocks = N//block_len

        processed_rx = processed_rx[:-trun_len]
        source_symbols = source_symbols[:-trun_len]
        original_bits = original_bits[:num_blocks*k*2] 

        # processed_rx = processed_rx[trun_len:]
        # source_symbols = source_symbols[trun_len:]
        # original_bits = original_bits[num_blocks*k*2:] 
        
        return processed_rx, source_symbols, original_bits
    else:
        processed_rx0, source_symbols0, original_bits0 = PAS_truncate(processed_rx[0], source_symbols[0], original_bits[0], block_len, 1, k)
        processed_rx1, source_symbols1, original_bits1 = PAS_truncate(processed_rx[1], source_symbols[1], original_bits[1], block_len, 1, k)
        return  np.array([processed_rx0,processed_rx1]), np.array([source_symbols0,source_symbols1]),np.array([original_bits0,original_bits1])


def nCr(n, r):
    if r > n:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

def PAS_encoder(C, bits, k, blocks, Modbits, LDPC_encoder):
    bitsI = bits[::2] #in-phase bits
    bitsQ = bits[1::2] #quadrature bits
    N = np.sum(C)
    amplitudesI = DMencode.DMencode(C, bitsI, k, blocks).reshape((blocks,N))
    amplitudesQ = DMencode.DMencode(C, bitsQ, k, blocks).reshape((blocks,N))
    XI = PAS_signs(amplitudesI, Modbits, LDPC_encoder).flatten()
    XQ = PAS_signs(amplitudesQ, Modbits, LDPC_encoder).flatten()
    X = XI + 1j*XQ
    return X

def PAS_signs(A, Modbits, LDPC_encoder):
    #A: Input matrix of amplitudes, with block length N
    #Modbits: Modulation format
    #NPol: Number of polarisations

    if(Modbits==4): #16-QAM (2**m ASK constellation, m=2, so 2**(m-1)=2 amplitudes)
        #G = [I|P] systematic generator matrix
        #P is a ((m-1)N),(N) = (N,N) matrix
        #A[i] is length-N vector
        #b_A[i] is a length N(m-1) = N vector
        #b_S[i] = b_A[i]*P is a length-N vector of signs represented by bits
        
        m=2
        n_P = A.shape[1]*m
        k_P = A.shape[1]*(m-1)

        b_A = np.where(A == 1, 0, 1)#Replace ampltidues by a binary label
        b_S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Bits representing sign blocks
        S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Sign blocks
        X = np.empty((A.shape[0],A.shape[1]), dtype=int)
        for i in range(b_A.shape[0]): #For each block
            b_S[i] = LDPC_encoder.encode(b_A[i])[LDPC_encoder.K:]
            S[i] = np.where(b_S[i]==0, 1, -1)
            X[i] = A[i]*S[i]
        
        return X

    elif(Modbits==6): #64-QAM (2**m ASK constellation, m=3, so 2**(m-1)=4 amplitudes)
        m=3
        n_P = A.shape[1]*m
        k_P = A.shape[1]*(m-1)

        b_A = []
        for row in range(A.shape[0]):
            b_A_row = []
            for i in A[row]:
                if(i==1):
                    b_A_row.append(1)
                    b_A_row.append(1)
                if(i==3):
                    b_A_row.append(1)
                    b_A_row.append(0)
                if(i==5):
                    b_A_row.append(0)
                    b_A_row.append(0)
                if(i==7):
                    b_A_row.append(0)
                    b_A_row.append(1)
            b_A.append(b_A_row)
        
        b_A = np.array(b_A)
        b_S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Bits representing sign blocks
        S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Sign blocks
        X = np.empty((A.shape[0],A.shape[1]), dtype=int)
        for i in range(b_A.shape[0]): #For each block
            #b_S[i] = np.dot(b_A[i],P)
            b_S[i] = LDPC_encoder.encode(b_A[i])[LDPC_encoder.K:]

            #b_S[i] = np.random.choice([0,1], size=A.shape[1], p=[0.5,0.5]) #Overwriting for testing with actual uniform distributed signs

            S[i] = np.where(b_S[i]==0, 1, -1)
            X[i] = A[i]*S[i]
        
        return X
    elif(Modbits==8): #256-QAM (2**m ASK constellation, m=4, so 2**(m-1)=8 amplitudes)
        m=4
        n_P = A.shape[1]*m
        k_P = A.shape[1]*(m-1)

        b_A = []
        for row in range(A.shape[0]):
            b_A_row = []
            for i in A[row]:
                if(i==1):
                    b_A_row.append(1)
                    b_A_row.append(0)
                    b_A_row.append(0)
                if(i==3):
                    b_A_row.append(1)
                    b_A_row.append(0)
                    b_A_row.append(1)
                if(i==5):
                    b_A_row.append(1)
                    b_A_row.append(1)
                    b_A_row.append(1)
                if(i==7):
                    b_A_row.append(1)
                    b_A_row.append(1)
                    b_A_row.append(0)
                if(i==9):
                    b_A_row.append(0)
                    b_A_row.append(1)
                    b_A_row.append(0)
                if(i==11):
                    b_A_row.append(0)
                    b_A_row.append(1)
                    b_A_row.append(1)
                if(i==13):
                    b_A_row.append(0)
                    b_A_row.append(0)
                    b_A_row.append(1)
                if(i==15):
                    b_A_row.append(0)
                    b_A_row.append(0)
                    b_A_row.append(0)

            b_A.append(b_A_row)
        
        b_A = np.array(b_A)
        b_S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Bits representing sign blocks
        S = np.empty((A.shape[0],A.shape[1]), dtype=int) #Sign blocks
        X = np.empty((A.shape[0],A.shape[1]), dtype=int)
        for i in range(b_A.shape[0]): #For each block
            #b_S[i] = np.dot(b_A[i],P)

            b_S[i] = LDPC_encoder.encode(b_A[i])[LDPC_encoder.K:]

            #b_S[i] = np.random.choice([0,1], size=A.shape[1], p=[0.5,0.5]) #Overwriting for testing with actual uniform distributed signs

            S[i] = np.where(b_S[i]==0, 1, -1)
            X[i] = A[i]*S[i]
        
        return X
    
def LLR_approx(Y, Modbits, λ, sigma, blocks, norm):
    print('Using Approximate LLRs')
    N = len(Y)
    Y = Y.reshape((blocks,Y.shape[0]//blocks))
    epsilon = 1e-20 #to protect when finding entropy

    if(Modbits==4):
        X_0_0 = np.array([1,3])/np.sqrt(norm) #Set of symbols that have 0 at the 0th bit level
        X_0_1 = np.array([-3,-1])/np.sqrt(norm) #Set of symbols that have 1 at the 0th bit level
        X_1_0 = np.array([-1,1])/np.sqrt(norm) #Set of symbols that have 0 at the 1th bit level
        X_1_1 = np.array([-3,3])/np.sqrt(norm) #Set of symbols that have 1 at the 1th bit level
        LLR = []
        H_0 = 0 #Entropy for performance metrics H(B_0|Y)
        H_1 = 0 #Entropy for performance metrics H(B_1|Y)

        for i in range(blocks):
            LLR_ampli = []
            LLR_signsi = []
            for y in Y[i]:
                x0_0 = X_0_0[np.argmax(np.exp(-1*((y-X_0_0)**2/(2*sigma**2))-λ*X_0_0**2))]
                x0_1 = X_0_1[np.argmax(np.exp(-1*((y-X_0_1)**2/(2*sigma**2))-λ*X_0_1**2))]
                llr = ((x0_0-x0_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x0_0**2-x0_1**2)
                LLR_signsi.append(llr)

                P_B0_0_Y = expit(llr) #P(B_0 = 0 | Y=y)
                P_B0_1_Y = 1 - P_B0_0_Y #P(B_0 = 1 | Y=y)
                P_B0_0_Y = np.clip(P_B0_0_Y, epsilon, 1 - epsilon)
                P_B0_1_Y = np.clip(P_B0_1_Y, epsilon, 1 - epsilon)
                H_0 += -1*P_B0_0_Y*np.log2(P_B0_0_Y)
                H_0 += -1*P_B0_1_Y*np.log2(P_B0_1_Y)

                x1_0 = X_1_0[np.argmax(np.exp(-1*((y-X_1_0)**2/(2*sigma**2))-λ*X_1_0**2))]
                x1_1 = X_1_1[np.argmax(np.exp(-1*((y-X_1_1)**2/(2*sigma**2))-λ*X_1_1**2))]
                llr = ((x1_0-x1_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x1_0**2-x1_1**2)
                LLR_ampli.append(llr)

                P_B1_0_Y = expit(llr) #P(B_1 = 0 | Y=y)
                P_B1_1_Y = 1 - P_B1_0_Y #P(B_1 = 1 | Y=y)
                P_B1_0_Y = np.clip(P_B1_0_Y, epsilon, 1 - epsilon)
                P_B1_1_Y = np.clip(P_B1_1_Y, epsilon, 1 - epsilon)
                H_1 += -1*P_B1_0_Y*np.log2(P_B1_0_Y)
                H_1 += -1*P_B1_1_Y*np.log2(P_B1_1_Y)

            LLRi = []
            for llr in LLR_ampli:
                LLRi.append(llr)
            for llr in LLR_signsi:
                LLRi.append(llr)
            LLR.append(LLRi)

        H_0 = H_0/N #Empirical average 
        H_1 = H_1/N
        H = np.array([H_0,H_1])
        
    elif(Modbits==6):
        
        X_0_0 = np.array([1,3,5,7])/np.sqrt(norm) #Set of symbols that have a 0 at the 0th bit level
        X_0_1 = np.array([-7,-5,-3,-1])/np.sqrt(norm) #Set of symbols that have a 1 at the 0th bit level
        X_1_0 = np.array([-7,-5,5,7])/np.sqrt(norm) #Set of symbols that have a 0 at the 1th bit level
        X_1_1 = np.array([-3,-1,1,3])/np.sqrt(norm) #Set of symbols that have a 1 at the 1th bit level
        X_2_0 = np.array([-5,-3,3,5])/np.sqrt(norm) #Set of symbols that have a 0 at the 2th bit level
        X_2_1= np.array([-7,-1,1,7])/np.sqrt(norm) #Set of symbols that have a 1 at the 2th bit level
        LLR=[]

        H_0 = 0 #Entropy for performance metrics H(B_0|Y)
        H_1 = 0 #Entropy for performance metrics H(B_1|Y)
        H_2 = 0 #Entropy for performance metrics H(B_2|Y)

        for i in range(blocks):
            LLR_ampli = []
            LLR_signsi = []
            for y in Y[i]:
                x0_0 = X_0_0[np.argmax(np.exp(-1*((y-X_0_0)**2/(2*sigma**2))-λ*X_0_0**2))]
                x0_1 = X_0_1[np.argmax(np.exp(-1*((y-X_0_1)**2/(2*sigma**2))-λ*X_0_1**2))]
                llr = ((x0_0-x0_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x0_0**2-x0_1**2)
                LLR_signsi.append(llr)

                P_B0_0_Y = expit(llr) #P(B_0 = 0 | Y=y)
                P_B0_1_Y = 1 - P_B0_0_Y #P(B_0 = 1 | Y=y)
                P_B0_0_Y = np.clip(P_B0_0_Y, epsilon, 1 - epsilon)
                P_B0_1_Y = np.clip(P_B0_1_Y, epsilon, 1 - epsilon)
                H_0 += -1*P_B0_0_Y*np.log2(P_B0_0_Y)
                H_0 += -1*P_B0_1_Y*np.log2(P_B0_1_Y)
                

                x1_0 = X_1_0[np.argmax(np.exp(-1*((y-X_1_0)**2/(2*sigma**2))-λ*X_1_0**2))]
                x1_1 = X_1_1[np.argmax(np.exp(-1*((y-X_1_1)**2/(2*sigma**2))-λ*X_1_1**2))]
                llr = ((x1_0-x1_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x1_0**2-x1_1**2)
                LLR_ampli.append(llr)

                P_B1_0_Y = expit(llr) #P(B_1 = 0 | Y=y)
                P_B1_1_Y = 1 - P_B1_0_Y #P(B_1 = 1 | Y=y)
                P_B1_0_Y = np.clip(P_B1_0_Y, epsilon, 1 - epsilon)
                P_B1_1_Y = np.clip(P_B1_1_Y, epsilon, 1 - epsilon)
                H_1 += -1*P_B1_0_Y*np.log2(P_B1_0_Y)
                H_1 += -1*P_B1_1_Y*np.log2(P_B1_1_Y)

                x2_0 = X_2_0[np.argmax(np.exp(-1*((y-X_2_0)**2/(2*sigma**2))-λ*X_2_0**2))]
                x2_1 = X_2_1[np.argmax(np.exp(-1*((y-X_2_1)**2/(2*sigma**2))-λ*X_2_1**2))]
                llr = ((x2_0-x2_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x2_0**2-x2_1**2)
                LLR_ampli.append(llr)

                P_B2_0_Y = expit(llr) #P(B_2 = 0 | Y=y)
                P_B2_1_Y = 1 - P_B2_0_Y #P(B_2 = 1 | Y=y)
                P_B2_0_Y = np.clip(P_B2_0_Y, epsilon, 1 - epsilon)
                P_B2_1_Y = np.clip(P_B2_1_Y, epsilon, 1 - epsilon)
                H_2 += -1*P_B2_0_Y*np.log2(P_B2_0_Y)
                H_2 += -1*P_B2_1_Y*np.log2(P_B2_1_Y)

            LLRi = []
            for llr in LLR_ampli:
                LLRi.append(llr)
            for llr in LLR_signsi:
                LLRi.append(llr)
            LLR.append(LLRi)
        
        H_0 = H_0/N #Empirical average 
        H_1 = H_1/N
        H_2 = H_2/N
        H = np.array([H_0,H_1,H_2])
        
    elif(Modbits==8):
        
        X_0_0 = np.array([1,3,5,7,9,11,13,15])/np.sqrt(norm) #Set of symbols that have a 0 at the 0th bit level
        X_0_1 = np.array([-15,-13,-11,-9,-7,-5,-3,-1])/np.sqrt(norm) #Set of symbols that have a 1 at the 0th bit level
        X_1_0 = np.array([-15,-13,-11,-9,9,11,13,15])/np.sqrt(norm) #Set of symbols that have a 0 at the 1th bit level
        X_1_1 = np.array([-7,-5,-3,-1,1,3,5,7])/np.sqrt(norm) #Set of symbols that have a 1 at the 1th bit level
        X_2_0 = np.array([-15,-13,-3,-1,1,3,13,15])/np.sqrt(norm) #Set of symbols that have a 0 at the 2th bit level
        X_2_1 = np.array([-11,-9,-7,-5,5,7,9,11])/np.sqrt(norm) #Set of symbols that have a 1 at the 2th bit level
        X_3_0 = np.array([-15,-9,-7,-1,1,7,9,15])/np.sqrt(norm) #Set of symbols that have a 0 at the 3th bit level
        X_3_1 = np.array([-13,-11,-5,-3,3,5,11,13])/np.sqrt(norm) #Set of symbols that have a 1 at the 3th bit level
        LLR=[]

        H_0 = 0 #Entropy for performance metrics H(B_0|Y)
        H_1 = 0 #Entropy for performance metrics H(B_1|Y)
        H_2 = 0 #Entropy for performance metrics H(B_2|Y)
        H_3 = 0 #Entropy for performance metrics H(B_3|Y)

        for i in range(blocks):
            LLR_ampli = []
            LLR_signsi = []
            for y in Y[i]:
                x0_0 = X_0_0[np.argmax(np.exp(-1*((y-X_0_0)**2/(2*sigma**2))-λ*X_0_0**2))]
                x0_1 = X_0_1[np.argmax(np.exp(-1*((y-X_0_1)**2/(2*sigma**2))-λ*X_0_1**2))]
                llr = ((x0_0-x0_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x0_0**2-x0_1**2)
                LLR_signsi.append(llr)

                P_B0_0_Y = expit(llr) #P(B_0 = 0 | Y=y)
                P_B0_1_Y = 1 - P_B0_0_Y #P(B_0 = 1 | Y=y)
                P_B0_0_Y = np.clip(P_B0_0_Y, epsilon, 1 - epsilon)
                P_B0_1_Y = np.clip(P_B0_1_Y, epsilon, 1 - epsilon)
                H_0 += -1*P_B0_0_Y*np.log2(P_B0_0_Y)
                H_0 += -1*P_B0_1_Y*np.log2(P_B0_1_Y)

                x1_0 = X_1_0[np.argmax(np.exp(-1*((y-X_1_0)**2/(2*sigma**2))-λ*X_1_0**2))]
                x1_1 = X_1_1[np.argmax(np.exp(-1*((y-X_1_1)**2/(2*sigma**2))-λ*X_1_1**2))]
                llr = ((x1_0-x1_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x1_0**2-x1_1**2)
                LLR_ampli.append(llr)

                P_B1_0_Y = expit(llr) #P(B_1 = 0 | Y=y)
                P_B1_1_Y = 1 - P_B1_0_Y #P(B_1 = 1 | Y=y)
                P_B1_0_Y = np.clip(P_B1_0_Y, epsilon, 1 - epsilon)
                P_B1_1_Y = np.clip(P_B1_1_Y, epsilon, 1 - epsilon)
                H_1 += -1*P_B1_0_Y*np.log2(P_B1_0_Y)
                H_1 += -1*P_B1_1_Y*np.log2(P_B1_1_Y)

                x2_0 = X_2_0[np.argmax(np.exp(-1*((y-X_2_0)**2/(2*sigma**2))-λ*X_2_0**2))]
                x2_1 = X_2_1[np.argmax(np.exp(-1*((y-X_2_1)**2/(2*sigma**2))-λ*X_2_1**2))]
                llr = ((x2_0-x2_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x2_0**2-x2_1**2)
                LLR_ampli.append(llr)

                P_B2_0_Y = expit(llr) #P(B_2 = 0 | Y=y)
                P_B2_1_Y = 1 - P_B2_0_Y #P(B_2 = 1 | Y=y)
                P_B2_0_Y = np.clip(P_B2_0_Y, epsilon, 1 - epsilon)
                P_B2_1_Y = np.clip(P_B2_1_Y, epsilon, 1 - epsilon)
                H_2 += -1*P_B2_0_Y*np.log2(P_B2_0_Y)
                H_2 += -1*P_B2_1_Y*np.log2(P_B2_1_Y)

                x3_0 = X_3_0[np.argmax(np.exp(-1*((y-X_3_0)**2/(2*sigma**2))-λ*X_3_0**2))]
                x3_1 = X_3_1[np.argmax(np.exp(-1*((y-X_3_1)**2/(2*sigma**2))-λ*X_3_1**2))]
                llr = ((x3_0-x3_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x3_0**2-x3_1**2)
                LLR_ampli.append(llr)

                P_B3_0_Y = expit(llr) #P(B_2 = 0 | Y=y)
                P_B3_1_Y = 1 - P_B3_0_Y #P(B_2 = 1 | Y=y)
                P_B3_0_Y = np.clip(P_B3_0_Y, epsilon, 1 - epsilon)
                P_B3_1_Y = np.clip(P_B3_1_Y, epsilon, 1 - epsilon)
                H_3 += -1*P_B3_0_Y*np.log2(P_B3_0_Y)
                H_3 += -1*P_B3_1_Y*np.log2(P_B3_1_Y)

            LLRi = []
            for llr in LLR_ampli:
                LLRi.append(llr)
            for llr in LLR_signsi:
                LLRi.append(llr)
            LLR.append(LLRi)

        H_0 = H_0/N #Empirical average 
        H_1 = H_1/N
        H_2 = H_2/N
        H_3 = H_3/N
        H = np.array([H_0,H_1,H_2,H_3])
    
    return np.array(LLR).flatten(), H

def LLR_full(Y, Modbits, λ, sigma, blocks, norm):
    print('Using Full LLRs')
    N = len(Y)
    Y = Y.reshape((blocks,Y.shape[0]//blocks))
    epsilon = 1e-20 #to protect when finding entropy

    if(Modbits==4):
        X_0_0 = np.array([1,3])/np.sqrt(norm) #Set of symbols that have 0 at the 0th bit level
        X_0_1 = np.array([-3,-1])/np.sqrt(norm) #Set of symbols that have 1 at the 0th bit level
        X_1_0 = np.array([-1,1])/np.sqrt(norm) #Set of symbols that have 0 at the 1th bit level
        X_1_1 = np.array([-3,3])/np.sqrt(norm) #Set of symbols that have 1 at the 1th bit level
        LLR = []
        H_0 = 0 #Entropy for performance metrics H(B_0|Y)
        H_1 = 0 #Entropy for performance metrics H(B_1|Y)

        for i in range(blocks):
            LLR_ampli = []
            LLR_signsi = []
            for y in Y[i]:
                num = logsumexp(-((y - X_0_0)**2) / (2 * sigma**2) - λ * X_0_0**2)
                den = logsumexp(-((y - X_0_1)**2) / (2 * sigma**2) - λ * X_0_1**2)
                llr = num - den
                LLR_signsi.append(llr)

                P_B0_0_Y = expit(llr) #P(B_0 = 0 | Y=y)
                P_B0_1_Y = 1 - P_B0_0_Y #P(B_0 = 1 | Y=y)
                P_B0_0_Y = np.clip(P_B0_0_Y, epsilon, 1 - epsilon)
                P_B0_1_Y = np.clip(P_B0_1_Y, epsilon, 1 - epsilon)
                H_0 += -1*P_B0_0_Y*np.log2(P_B0_0_Y)
                H_0 += -1*P_B0_1_Y*np.log2(P_B0_1_Y)


                num = logsumexp(-((y - X_1_0)**2) / (2 * sigma**2) - λ * X_1_0**2)
                den = logsumexp(-((y - X_1_1)**2) / (2 * sigma**2) - λ * X_1_1**2)
                llr = num - den
                LLR_ampli.append(llr)

                P_B1_0_Y = expit(llr) #P(B_1 = 0 | Y=y)
                P_B1_1_Y = 1 - P_B1_0_Y #P(B_1 = 1 | Y=y)
                P_B1_0_Y = np.clip(P_B1_0_Y, epsilon, 1 - epsilon)
                P_B1_1_Y = np.clip(P_B1_1_Y, epsilon, 1 - epsilon)
                H_1 += -1*P_B1_0_Y*np.log2(P_B1_0_Y)
                H_1 += -1*P_B1_1_Y*np.log2(P_B1_1_Y)

            LLRi = []
            for llr in LLR_ampli:
                LLRi.append(llr)
            for llr in LLR_signsi:
                LLRi.append(llr)
            LLR.append(LLRi)

        H_0 = H_0/N #Empirical average 
        H_1 = H_1/N
        H = np.array([H_0,H_1])
        
    elif(Modbits==6):
        
        X_0_0 = np.array([1,3,5,7])/np.sqrt(norm) #Set of symbols that have a 0 at the 0th bit level
        X_0_1 = np.array([-7,-5,-3,-1])/np.sqrt(norm) #Set of symbols that have a 1 at the 0th bit level
        X_1_0 = np.array([-7,-5,5,7])/np.sqrt(norm) #Set of symbols that have a 0 at the 1th bit level
        X_1_1 = np.array([-3,-1,1,3])/np.sqrt(norm) #Set of symbols that have a 1 at the 1th bit level
        X_2_0 = np.array([-5,-3,3,5])/np.sqrt(norm) #Set of symbols that have a 0 at the 2th bit level
        X_2_1= np.array([-7,-1,1,7])/np.sqrt(norm) #Set of symbols that have a 1 at the 2th bit level
        LLR=[]

        H_0 = 0 #Entropy for performance metrics H(B_0|Y)
        H_1 = 0 #Entropy for performance metrics H(B_1|Y)
        H_2 = 0 #Entropy for performance metrics H(B_2|Y)

        for i in range(blocks):
            LLR_ampli = []
            LLR_signsi = []
            for y in Y[i]:
                x0_0 = X_0_0[np.argmax(np.exp(-1*((y-X_0_0)**2/(2*sigma**2))-λ*X_0_0**2))]
                x0_1 = X_0_1[np.argmax(np.exp(-1*((y-X_0_1)**2/(2*sigma**2))-λ*X_0_1**2))]
                llr = ((x0_0-x0_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x0_0**2-x0_1**2)
                LLR_signsi.append(llr)

                P_B0_0_Y = expit(llr) #P(B_0 = 0 | Y=y)
                P_B0_1_Y = 1 - P_B0_0_Y #P(B_0 = 1 | Y=y)
                P_B0_0_Y = np.clip(P_B0_0_Y, epsilon, 1 - epsilon)
                P_B0_1_Y = np.clip(P_B0_1_Y, epsilon, 1 - epsilon)
                H_0 += -1*P_B0_0_Y*np.log2(P_B0_0_Y)
                H_0 += -1*P_B0_1_Y*np.log2(P_B0_1_Y)
                

                x1_0 = X_1_0[np.argmax(np.exp(-1*((y-X_1_0)**2/(2*sigma**2))-λ*X_1_0**2))]
                x1_1 = X_1_1[np.argmax(np.exp(-1*((y-X_1_1)**2/(2*sigma**2))-λ*X_1_1**2))]
                llr = ((x1_0-x1_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x1_0**2-x1_1**2)
                LLR_ampli.append(llr)

                P_B1_0_Y = expit(llr) #P(B_1 = 0 | Y=y)
                P_B1_1_Y = 1 - P_B1_0_Y #P(B_1 = 1 | Y=y)
                P_B1_0_Y = np.clip(P_B1_0_Y, epsilon, 1 - epsilon)
                P_B1_1_Y = np.clip(P_B1_1_Y, epsilon, 1 - epsilon)
                H_1 += -1*P_B1_0_Y*np.log2(P_B1_0_Y)
                H_1 += -1*P_B1_1_Y*np.log2(P_B1_1_Y)

                x2_0 = X_2_0[np.argmax(np.exp(-1*((y-X_2_0)**2/(2*sigma**2))-λ*X_2_0**2))]
                x2_1 = X_2_1[np.argmax(np.exp(-1*((y-X_2_1)**2/(2*sigma**2))-λ*X_2_1**2))]
                llr = ((x2_0-x2_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x2_0**2-x2_1**2)
                LLR_ampli.append(llr)

                P_B2_0_Y = expit(llr) #P(B_2 = 0 | Y=y)
                P_B2_1_Y = 1 - P_B2_0_Y #P(B_2 = 1 | Y=y)
                P_B2_0_Y = np.clip(P_B2_0_Y, epsilon, 1 - epsilon)
                P_B2_1_Y = np.clip(P_B2_1_Y, epsilon, 1 - epsilon)
                H_2 += -1*P_B2_0_Y*np.log2(P_B2_0_Y)
                H_2 += -1*P_B2_1_Y*np.log2(P_B2_1_Y)

            LLRi = []
            for llr in LLR_ampli:
                LLRi.append(llr)
            for llr in LLR_signsi:
                LLRi.append(llr)
            LLR.append(LLRi)
        
        H_0 = H_0/N #Empirical average 
        H_1 = H_1/N
        H_2 = H_2/N
        H = np.array([H_0,H_1,H_2])
        
    elif(Modbits==8):
        
        X_0_0 = np.array([1,3,5,7,9,11,13,15])/np.sqrt(norm) #Set of symbols that have a 0 at the 0th bit level
        X_0_1 = np.array([-15,-13,-11,-9,-7,-5,-3,-1])/np.sqrt(norm) #Set of symbols that have a 1 at the 0th bit level
        X_1_0 = np.array([-15,-13,-11,-9,9,11,13,15])/np.sqrt(norm) #Set of symbols that have a 0 at the 1th bit level
        X_1_1 = np.array([-7,-5,-3,-1,1,3,5,7])/np.sqrt(norm) #Set of symbols that have a 1 at the 1th bit level
        X_2_0 = np.array([-15,-13,-3,-1,1,3,13,15])/np.sqrt(norm) #Set of symbols that have a 0 at the 2th bit level
        X_2_1 = np.array([-11,-9,-7,-5,5,7,9,11])/np.sqrt(norm) #Set of symbols that have a 1 at the 2th bit level
        X_3_0 = np.array([-15,-9,-7,-1,1,7,9,15])/np.sqrt(norm) #Set of symbols that have a 0 at the 3th bit level
        X_3_1 = np.array([-13,-11,-5,-3,3,5,11,13])/np.sqrt(norm) #Set of symbols that have a 1 at the 3th bit level
        LLR=[]

        H_0 = 0 #Entropy for performance metrics H(B_0|Y)
        H_1 = 0 #Entropy for performance metrics H(B_1|Y)
        H_2 = 0 #Entropy for performance metrics H(B_2|Y)
        H_3 = 0 #Entropy for performance metrics H(B_3|Y)

        for i in range(blocks):
            LLR_ampli = []
            LLR_signsi = []
            for y in Y[i]:
                x0_0 = X_0_0[np.argmax(np.exp(-1*((y-X_0_0)**2/(2*sigma**2))-λ*X_0_0**2))]
                x0_1 = X_0_1[np.argmax(np.exp(-1*((y-X_0_1)**2/(2*sigma**2))-λ*X_0_1**2))]
                llr = ((x0_0-x0_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x0_0**2-x0_1**2)
                LLR_signsi.append(llr)

                P_B0_0_Y = expit(llr) #P(B_0 = 0 | Y=y)
                P_B0_1_Y = 1 - P_B0_0_Y #P(B_0 = 1 | Y=y)
                P_B0_0_Y = np.clip(P_B0_0_Y, epsilon, 1 - epsilon)
                P_B0_1_Y = np.clip(P_B0_1_Y, epsilon, 1 - epsilon)
                H_0 += -1*P_B0_0_Y*np.log2(P_B0_0_Y)
                H_0 += -1*P_B0_1_Y*np.log2(P_B0_1_Y)

                x1_0 = X_1_0[np.argmax(np.exp(-1*((y-X_1_0)**2/(2*sigma**2))-λ*X_1_0**2))]
                x1_1 = X_1_1[np.argmax(np.exp(-1*((y-X_1_1)**2/(2*sigma**2))-λ*X_1_1**2))]
                llr = ((x1_0-x1_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x1_0**2-x1_1**2)
                LLR_ampli.append(llr)

                P_B1_0_Y = expit(llr) #P(B_1 = 0 | Y=y)
                P_B1_1_Y = 1 - P_B1_0_Y #P(B_1 = 1 | Y=y)
                P_B1_0_Y = np.clip(P_B1_0_Y, epsilon, 1 - epsilon)
                P_B1_1_Y = np.clip(P_B1_1_Y, epsilon, 1 - epsilon)
                H_1 += -1*P_B1_0_Y*np.log2(P_B1_0_Y)
                H_1 += -1*P_B1_1_Y*np.log2(P_B1_1_Y)

                x2_0 = X_2_0[np.argmax(np.exp(-1*((y-X_2_0)**2/(2*sigma**2))-λ*X_2_0**2))]
                x2_1 = X_2_1[np.argmax(np.exp(-1*((y-X_2_1)**2/(2*sigma**2))-λ*X_2_1**2))]
                llr = ((x2_0-x2_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x2_0**2-x2_1**2)
                LLR_ampli.append(llr)

                P_B2_0_Y = expit(llr) #P(B_2 = 0 | Y=y)
                P_B2_1_Y = 1 - P_B2_0_Y #P(B_2 = 1 | Y=y)
                P_B2_0_Y = np.clip(P_B2_0_Y, epsilon, 1 - epsilon)
                P_B2_1_Y = np.clip(P_B2_1_Y, epsilon, 1 - epsilon)
                H_2 += -1*P_B2_0_Y*np.log2(P_B2_0_Y)
                H_2 += -1*P_B2_1_Y*np.log2(P_B2_1_Y)

                x3_0 = X_3_0[np.argmax(np.exp(-1*((y-X_3_0)**2/(2*sigma**2))-λ*X_3_0**2))]
                x3_1 = X_3_1[np.argmax(np.exp(-1*((y-X_3_1)**2/(2*sigma**2))-λ*X_3_1**2))]
                llr = ((x3_0-x3_1)/(sigma**2))*y - (1/(2*sigma**2)+ λ)*(x3_0**2-x3_1**2)
                LLR_ampli.append(llr)

                P_B3_0_Y = expit(llr) #P(B_2 = 0 | Y=y)
                P_B3_1_Y = 1 - P_B3_0_Y #P(B_2 = 1 | Y=y)
                P_B3_0_Y = np.clip(P_B3_0_Y, epsilon, 1 - epsilon)
                P_B3_1_Y = np.clip(P_B3_1_Y, epsilon, 1 - epsilon)
                H_3 += -1*P_B3_0_Y*np.log2(P_B3_0_Y)
                H_3 += -1*P_B3_1_Y*np.log2(P_B3_1_Y)

            LLRi = []
            for llr in LLR_ampli:
                LLRi.append(llr)
            for llr in LLR_signsi:
                LLRi.append(llr)
            LLR.append(LLRi)

        H_0 = H_0/N #Empirical average 
        H_1 = H_1/N
        H_2 = H_2/N
        H_3 = H_3/N
        H = np.array([H_0,H_1,H_2,H_3])
    
    return np.array(LLR).flatten(), H

def LDPC_decision(llr):
    b = [0 if j > 0 else 1 for j in llr]
    return b

def BRGC_btos(bits, Modbits):
    if(Modbits==4):
        btos_map = {(1,1):-3,
                    (1,0):-1,
                    (0,0):1,
                    (0,1):3
                    }

    elif(Modbits==6):
        btos_map = {(1,0,1):-7,
               (1,0,0):-5,
               (1,1,0):-3,
               (1,1,1):-1,
               (0,1,1):1,
               (0,1,0):3,
               (0,0,0):5,
               (0,0,1):7
        }
    
    elif(Modbits==8):
        btos_map = {(1,0,0,0): -15,
                    (1,0,0,1): -13,
                    (1,0,1,1): -11,
                    (1,0,1,0): -9,
                    (1,1,1,0): -7,
                    (1,1,1,1): -5,
                    (1,1,0,1): -3,
                    (1,1,0,0): -1,
                    (0,1,0,0): 1,
                    (0,1,0,1): 3,
                    (0,1,1,1): 5,
                    (0,1,1,0): 7,
                    (0,0,1,0): 9,
                    (0,0,1,1): 11,
                    (0,0,0,1): 13,
                    (0,0,0,0): 15
                    }
    
    symbols = np.array([
        btos_map[tuple(bits[i:i + Modbits//2])] for i in range(0, len(bits), Modbits//2)
    ])

    return symbols

def regroup_LLRs(LLR, Modbits):
    #regroup LLRs from [LLR(A1_1),...,LLR(A1_(m-1)),......,LLR(Anc_1),...,LLR(Anc_(m-1)), LLR(S1),...LLR(S_nc)] (same as codeword)
    #eg regroup from codeword format back to symbols identified by consecutive LLRs
    #eg put sign LLRs at the start if each set of amplitude LLRs
    if Modbits==4:
        LLR_regrouped = []
        m=2
        nc = LLR.shape[1]//m
        for i in range(LLR.shape[0]):
            sign_LLRs = LLR[i][nc*(m-1):] #LLRs representing sign bits
            amplitude_LLRs = LLR[i][:nc*(m-1)]
            j=0
            LLR_regroupedi = []
            while(j<nc*(m-1)):
                LLR_regroupedi.append(sign_LLRs[j//(m-1)])
                LLR_regroupedi.append(amplitude_LLRs[j])
                j=j+m-1
            for llr in LLR_regroupedi:
                LLR_regrouped.append(llr)
        return np.array(LLR_regrouped)
    
    elif Modbits==6:
        LLR_regrouped = []
        m=3
        nc = LLR.shape[1]//m
        for i in range(LLR.shape[0]):
            sign_LLRs = LLR[i][nc*(m-1):] #LLRs representing sign bits
            amplitude_LLRs = LLR[i][:nc*(m-1)]
            j=0
            LLR_regroupedi = []
            while(j<(nc*(m-1))):
                LLR_regroupedi.append(sign_LLRs[j//(m-1)])
                LLR_regroupedi.append(amplitude_LLRs[j])
                LLR_regroupedi.append(amplitude_LLRs[j+1])
                j=j+m-1
            for llr in LLR_regroupedi:
                LLR_regrouped.append(llr)
        return np.array(LLR_regrouped)

    elif Modbits==8:
        LLR_regrouped = []
        m=4
        nc = LLR.shape[1]//m
        for i in range(LLR.shape[0]):
            sign_LLRs = LLR[i][nc*(m-1):] #LLRs representing sign bits
            amplitude_LLRs = LLR[i][:nc*(m-1)]
            j=0
            LLR_regroupedi = []
            while(j<(nc*(m-1))):
                LLR_regroupedi.append(sign_LLRs[j//(m-1)])
                LLR_regroupedi.append(amplitude_LLRs[j])
                LLR_regroupedi.append(amplitude_LLRs[j+1])
                LLR_regroupedi.append(amplitude_LLRs[j+2])
                j=j+m-1
            for llr in LLR_regroupedi:
                LLR_regrouped.append(llr)
        return np.array(LLR_regrouped)

def PAS_parameters(Modbits,λ):
    if(Modbits==2):
        return 1,1,1,1
    elif(Modbits==4):
        signal_points = [1,3]
        if(λ==0.0 or λ==0.01 or λ==0.05 or λ==0.0625 or λ==0.08 or λ==0.1 or λ==0.15 or λ==0.025 or λ==0.075):
            N_target = 901
            z = 75
        else:
            raise ValueError("Need valid λ")
        
        const = 0
        for i in signal_points:
            const += np.exp(-λ*np.abs(i)**2)
        C = [int(N_target*np.exp(-λ)/const),int(N_target*np.exp(-λ*9)/const),0,0,0,0,0,0]
        N = np.sum(C)
        k = int(np.floor(math.log2(nCr(N,C[1]))))
        LDPC_encoder = ldpc.code(standard = '802.16', rate = '1/2', z=z, ptype='A')
        #LDPC_encoder.K should be (m-1)N = N
        # print("KKKKK",LDPC_encoder.K)
        # print("NNNNN",N)

    elif(Modbits==6):
        signal_points = [1,3,5,7]
        if(λ==0.05):
            N_target = 402
            z=50
        elif(λ==0.1):
            N_target = 401
            z=50
        elif(λ==0.025 or λ==0.01 or λ==0.0375 or λ==0.03125 or λ==0.034375 or λ==0.075):
            N_target = 417
            z=52

        else:
            raise ValueError("Need valid λ")
        
        const = 0
        for i in signal_points:
            const += np.exp(-λ*np.abs(i)**2)
        C = [int(N_target*np.exp(-λ)/const),int(N_target*np.exp(-λ*9)/const),int(N_target*np.exp(-λ*25)/const),int(N_target*np.exp(-λ*49)/const),0,0,0,0]
        C[0] += 1 #to get N to 400
        k=int(np.floor(math.log2(math.factorial(C[0]+C[1]+C[2]+C[3])/(math.factorial(C[0])*math.factorial(C[1])*math.factorial(C[2])*math.factorial(C[3])))))
        N = np.sum(C)
        LDPC_encoder = ldpc.code(standard = '802.16', rate = '2/3', z=z, ptype='A')
        #LDPC encoder rate is (m-1)/m = 2/3
        #LDPC_encoder.K should be (m-1)N = 2N
        print(LDPC_encoder.K,'K')
        print(2*N,'2N')

    elif(Modbits==8):
        signal_points = [1,3,5,7,9,11,13,15]
        N_target = 369
        const = 0
        #*********************** REMOVE AFTER TESTING ***********************
        for i in signal_points:
            const += np.exp(-λ*np.abs(i)**2)
        C = [int(N_target*np.exp(-λ)/const),int(N_target*np.exp(-λ*9)/const),int(N_target*np.exp(-λ*25)/const),int(N_target*np.exp(-λ*49)/const), int(N_target*np.exp(-λ*81)/const), int(N_target*np.exp(-λ*121)/const), int(N_target*np.exp(-λ*169)/const), int(N_target*np.exp(-λ*225)/const)]
        
        k=int(np.floor(math.log2(math.factorial(C[0]+C[1]+C[2]+C[3]+C[4]+C[5]+C[6]+C[7])/(math.factorial(C[0])*math.factorial(C[1])*math.factorial(C[2])*math.factorial(C[3])*math.factorial(C[4])*math.factorial(C[5])*math.factorial(C[6])*math.factorial(C[7])))))
        N = np.sum(C)
        LDPC_encoder = ldpc.code(standard = '802.16', rate = '3/4', z=61, ptype='A')
        #LDPC encoder rate is (m-1)/m = 3/4
        #LDPC_encoder.K should be (m-1)N = 3N

    return k, N, C, LDPC_encoder


def PAS_decoder(Y, Modbits, λ, sigma, blocks, LDPC_encoder, k, C, norm):
    
    YI = np.real(Y) #Received in-phase symbols
    YQ = np.imag(Y) #Received quadrature symbols
    
    #LLr full use full form of LLR
    #LLR approx uses linear approximation to find LLR (fine at high SNRs)
    LLRI, HI = LLR_full(YI, Modbits, λ, sigma, blocks, norm) #Calculate LLRs for each bit level of each symbol
    LLRQ, HQ = LLR_full(YQ, Modbits, λ, sigma, blocks, norm)
    #LLRs organised as [LLR(A1_1),...,LLR(A1_(m-1)),......,LLR(Anc_1),...,LLR(Anc_(m-1)), LLR(S1),...LLR(S_nc)] (same as codeword)
    LLRI = LLRI.reshape((blocks,LLRI.shape[0]//blocks))
    LLRQ = LLRQ.reshape((blocks,LLRQ.shape[0]//blocks))
    LLR_LDPCI=np.empty(LLRI.shape)
    LLR_LDPCQ=np.empty(LLRQ.shape)


    for i in range(blocks):
        LLR_LDPCI[i],itI = LDPC_encoder.decode(LLRI[i]) #LDPC decoding to final LLRs
        LLR_LDPCQ[i],itQ = LDPC_encoder.decode(LLRQ[i])

    #organise LLRs as [LLR(S1),LLR(A1_1),...,LLR(A1_(m-1)),......,LLR(Snc),LLR(Anc_1),...,LLR(Anc_(m-1))] so consecutive LLRs represent the same symbol
    LLRI_decoded_grouped = regroup_LLRs(LLR_LDPCI, Modbits)
    LLRQ_decoded_grouped = regroup_LLRs(LLR_LDPCQ, Modbits)

    bits_LDPCI = LDPC_decision(LLRI_decoded_grouped.flatten()) #Turn LLRs to bits
    bits_LDPCQ = LDPC_decision(LLRQ_decoded_grouped.flatten())

    YI_decoded = BRGC_btos(bits_LDPCI, Modbits) #Use BRGC mapping to map bits to symbols
    YQ_decoded = BRGC_btos(bits_LDPCQ, Modbits)

    Y_decoded = YI_decoded + 1j*YQ_decoded


    bits_decodedI = DMdecode.DMdecode(np.abs(YI_decoded), C, k, blocks) #Decode the distribution matching to the information bits
    bits_decodedQ = DMdecode.DMdecode(np.abs(YQ_decoded), C, k, blocks)

    bits_decoded = np.array([bit for pair in zip(bits_decodedI, bits_decodedQ) for bit in pair])

    return Y_decoded, bits_decoded, HI, HQ

def PAS_barplot(X):
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
    bar_width = 0.1

    # Plot bars with color-coding
    for x, y, h, color in zip(real_parts, imag_parts, probabilities, colors):
        ax.bar3d(x, y, 0, bar_width, bar_width, h, color=color, alpha=0.9)

    # Labels and title
    ax.set_title('Probabilistically Shaped Symbols')
    ax.set_xlabel('In-Phase')
    ax.set_ylabel('Quadrature')
    ax.set_zlabel('Probability of Symbol')


if(testing==True):

    Modbits = 4
    blocks = 100
    np.random.seed(1)
    λ = 0.05

    k, N, C, LDPC_encoder = PAS_parameters(Modbits,λ)

    bits = np.random.randint(0, 2, size= k*blocks*2) #In-Phase and Quadrature
    X = PAS_encoder(C, bits, k, blocks, Modbits, LDPC_encoder)
    norm = np.sum(np.abs(X)**2)/len(X)
    X=X/np.sqrt(norm)

    #Channel: Receive symbols Y
    snr_db = 13
    sps = 1

    Y,sigma = f.add_noise(X, snr_db, sps, Modbits, 1, True)

    print(sigma,'sigma')

    Y_decoded, bits_decoded = PAS_decoder(Y, Modbits, λ, sigma, blocks, LDPC_encoder, k, C, norm)   


    print('k:', k)
    print('N:', np.sum(C))
    print('λ:', λ)
    print('C :', C)
    print('Blocks:',blocks)
    print('Rate:', 2*k/N, 'bits/symbol')

    # print(bits, 'Source Bits')
    # print(X,'CCDM Symbols')
    # print(bits_decoded, 'Decoded Bits')

    if(np.array_equal(bits_decoded,bits)):
        print('No Bit Errors')
    else:
        print('Bit Errors')
    print(list(np.where(bits!=bits_decoded)))

    print('BER:', np.sum(bits!=bits_decoded)/bits.size)

    PAS_barplot(X)

    fig1, axs1 = plt.subplots(1, 1, figsize=(8, 8))
    f.plot_constellation(axs1, Y, 'Received Symbols', lim=Modbits*1.5)

    plt.show()

