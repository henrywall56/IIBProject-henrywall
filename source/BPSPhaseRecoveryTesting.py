import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import functions as f

#Follow the algorithm not the Matlab
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



def BPS(z,Modbits,N,B, toggle_phasenoisecompensation):
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
                zrot_decided[j,:] = f.max_likelihood_decision(zrot[j,:], Modbits)

            
            #intermediate sum to be minimised
            m = np.sum(abs(zrot-zrot_decided)**2,0)
            
            #estimating phase noise as angle that minimises m
            im = np.argmin(m)

            Theta = np.reshape(ThetaTest[im], 1)
            
            ThetaPU[i] = Theta + np.floor(0.5-(Theta-ThetaPrev)/p)*p

            ThetaPrev = ThetaPU[i]

        v = z*np.exp(-1j*ThetaPU)

        return v, ThetaPU
        
    else:
        return z, np.zeros(z)


