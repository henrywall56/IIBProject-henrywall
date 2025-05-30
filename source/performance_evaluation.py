import numpy as np
import functions as f
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import parameters as p
from matplotlib.ticker import MaxNLocator


def LLR(y,x,bmap,sigma2, Modbits):
    #Not for use with differential encoding
    #Calculates the LLR's for circularly symmetric Gaussian channel.
    #y: input signal normaliased to unit power
    #x: reference constellation normalised to unit power
    #bmap: Binary label of each symbol of the reference constellation
    #sigma2: estimate of the variance of the circularly symmetric Gaussian channel. sigma2 is variance per dimension.
    #Output L: estimated LLRs.
    L = np.zeros((len(y),Modbits))
    
    for k in range(Modbits):
        #Sets with bit "b = {0,1}" at position "k"
        xSet_b1 = x[bmap[:,k]==1]
        xSet_b0 = x[bmap[:,k]==0]

        #LLR estimation
        num = np.zeros(len(y))
        den = np.zeros(len(y))
        for i in range((Modbits**2)//2):
            num = num + np.exp(-1*abs(y-xSet_b1[i])**2/(2*sigma2))

            den = den + np.exp(-1*abs(y-xSet_b0[i])**2/(2*sigma2))

        L[:,k] = np.log(num/den)
    return L

def AIR_SDBW(x,b,y,Modbits):
    #Not for use with differential encoding
    #Estmiates the AIRs for CM schemes with soft decision bitwise encoders.
    #Assumes a circularly symmetric Gaussian channel
    #Noise is complex Gaussian random variable with total variance Tsigma2 = 2*sigma2
    #AIRs estimated using LLRs.
    #x: symbols transmitted in one polarisation, normalised to unit power
    #b: bits transmitted in one polarisation, normalised to unit power
    #y: sequence of symbols received in one polarisation.
    #Modbits: modulation format used.

    if(Modbits==2):
        
        bmap = np.array([[1,1],[0,1],[1,0],[0,0]])
        s = np.array([1+1j, -1+1j, 1-1j, -1-1j])/np.sqrt(2)
        
    elif(Modbits==4):

        bmap = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], 
                    [0, 1, 1, 0],[0, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 0],
                    [1, 1, 0, 0],[1, 1, 0, 1],[1, 1, 1, 1],[1, 1, 1, 0], 
                    [1, 0, 1, 0],[1, 0, 1, 1],[1, 0, 0, 1], [1, 0, 0, 0]])

        s = np.array([
            -3 + 3j,-1 + 3j,1 + 3j,3 + 3j,
            3 + 1j,1 + 1j,-1 + 1j,-3 + 1j,
            -3 - 1j,-1 - 1j,1 - 1j,3 - 1j,
            3 - 3j,1 - 3j,-1 - 3j,-3 - 3j
        ])/np.sqrt(10)

    elif(Modbits==6):
        
        bmap = np.array([
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 1], [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0]
        ])

        s = np.array([
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

    elif(Modbits==8):


        s = np.array([
            -15+15j, -13+15j, -11+15j, -9+15j, -7+15j, -5+15j, -3+15j, -1+15j, 1+15j, 3+15j, 5+15j, 7+15j, 9+15j, 11+15j, 13+15j, 15+15j,
            -15+13j, -13+13j, -11+13j, -9+13j, -7+13j, -5+13j, -3+13j, -1+13j, 1+13j, 3+13j, 5+13j, 7+13j, 9+13j, 11+13j, 13+13j, 15+13j,
            -15+11j, -13+11j, -11+11j, -9+11j, -7+11j, -5+11j, -3+11j, -1+11j, 1+11j, 3+11j, 5+11j, 7+11j, 9+11j, 11+11j, 13+11j, 15+11j,
            -15+9j, -13+9j, -11+9j, -9+9j, -7+9j, -5+9j, -3+9j, -1+9j, 1+9j, 3+9j, 5+9j, 7+9j, 9+9j, 11+9j, 13+9j, 15+9j,
            -15+7j, -13+7j, -11+7j, -9+7j, -7+7j, -5+7j, -3+7j, -1+7j, 1+7j, 3+7j, 5+7j, 7+7j, 9+7j, 11+7j, 13+7j, 15+7j,
            -15+5j, -13+5j, -11+5j, -9+5j, -7+5j, -5+5j, -3+5j, -1+5j, 1+5j, 3+5j, 5+5j, 7+5j, 9+5j, 11+5j, 13+5j, 15+5j,
            -15+3j, -13+3j, -11+3j, -9+3j, -7+3j, -5+3j, -3+3j, -1+3j, 1+3j, 3+3j, 5+3j, 7+3j, 9+3j, 11+3j, 13+3j, 15+3j,
            -15+1j, -13+1j, -11+1j, -9+1j, -7+1j, -5+1j, -3+1j, -1+1j, 1+1j, 3+1j, 5+1j, 7+1j, 9+1j, 11+1j, 13+1j, 15+1j,
            -15-1j, -13-1j, -11-1j, -9-1j, -7-1j, -5-1j, -3-1j, -1-1j, 1-1j, 3-1j, 5-1j, 7-1j, 9-1j, 11-1j, 13-1j, 15-1j,
            -15-3j, -13-3j, -11-3j, -9-3j, -7-3j, -5-3j, -3-3j, -1-3j, 1-3j, 3-3j, 5-3j, 7-3j, 9-3j, 11-3j, 13-3j, 15-3j,
            -15-5j, -13-5j, -11-5j, -9-5j, -7-5j, -5-5j, -3-5j, -1-5j, 1-5j, 3-5j, 5-5j, 7-5j, 9-5j, 11-5j, 13-5j, 15-5j,
            -15-7j, -13-7j, -11-7j, -9-7j, -7-7j, -5-7j, -3-7j, -1-7j, 1-7j, 3-7j, 5-7j, 7-7j, 9-7j, 11-7j, 13-7j, 15-7j,
            -15-9j, -13-9j, -11-9j, -9-9j, -7-9j, -5-9j, -3-9j, -1-9j, 1-9j, 3-9j, 5-9j, 7-9j, 9-9j, 11-9j, 13-9j, 15-9j,
            -15-11j, -13-11j, -11-11j, -9-11j, -7-11j, -5-11j, -3-11j, -1-11j, 1-11j, 3-11j, 5-11j, 7-11j, 9-11j, 11-11j, 13-11j, 15-11j,
            -15-13j, -13-13j, -11-13j, -9-13j, -7-13j, -5-13j, -3-13j, -1-13j, 1-13j, 3-13j, 5-13j, 7-13j, 9-13j, 11-13j, 13-13j, 15-13j,
            -15-15j, -13-15j, -11-15j, -9-15j, -7-15j, -5-15j, -3-15j, -1-15j, 1-15j, 3-15j, 5-15j, 7-15j, 9-15j, 11-15j, 13-15j, 15-15j

        ])/np.sqrt(170)


    #estimating the variance of the complex Gaussian distribution
    h = np.vdot(x,y)/(np.linalg.norm(x)**2)
    #h = np.sum(np.conjugate(x)*y)/np.sum(abs(x)**2)
    
    sigma2 = np.mean(abs(y-h*x)**2)/2
    
    #Calculating LLRs
    L = LLR(y,h*s, bmap, sigma2, Modbits)
    
    b = b.reshape(len(b)//Modbits, Modbits)

    def objective_function(a, b, L):
        return np.sum(np.mean(np.log2(1 + np.exp(a * ((-1)**b) * L))))

    AIRaux = minimize_scalar(objective_function, bounds=(0,2), args=(b,L), method='bounded')
    AIR = Modbits - AIRaux.fun


    return AIR

def AIR_SDBW_theoretical(SNRdB, Modbits):
    #Not for differential encoding
    #Calculates approximations of the theoretical AIRs for SD-BW encoders, under a circularly symmetric Gaussian channel.
    #Approximation given by a 10-point Gauss-Hermite Quadrature.
    #Gray labelling assumed
    if(Modbits==2):
        M=4
        bmap = np.array([[1,1],[0,1],[1,0],[0,0]])
        s = np.array([1+1j, -1+1j, 1-1j, -1-1j
                                  ])/np.sqrt(2)
        
    elif(Modbits==4):
        M=16
        bmap = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], 
                    [0, 1, 1, 0],  [0, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 0],
                    [1, 1, 0, 0],[1, 1, 0, 1],[1, 1, 1, 1],[1, 1, 1, 0], 
                    [1, 0, 1, 0],[1, 0, 1, 1],[1, 0, 0, 1], [1, 0, 0, 0]])

        s = np.array([
            -3 + 3j,-1 + 3j,1 + 3j,3 + 3j,
            3 + 1j,1 + 1j,-1 + 1j,-3 + 1j,
            -3 - 1j,-1 - 1j,1 - 1j,3 - 1j,
            3 - 3j,1 - 3j,-1 - 3j,-3 - 3j,
        ])/np.sqrt(10)

    elif(Modbits==6):
        M=64
        bmap = np.array([
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 1], [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0]
        ])

        s = np.array([
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

    elif(Modbits==8):
        M=256

        s = np.array([
            -15+15j, -13+15j, -11+15j, -9+15j, -7+15j, -5+15j, -3+15j, -1+15j, 1+15j, 3+15j, 5+15j, 7+15j, 9+15j, 11+15j, 13+15j, 15+15j,
            -15+13j, -13+13j, -11+13j, -9+13j, -7+13j, -5+13j, -3+13j, -1+13j, 1+13j, 3+13j, 5+13j, 7+13j, 9+13j, 11+13j, 13+13j, 15+13j,
            -15+11j, -13+11j, -11+11j, -9+11j, -7+11j, -5+11j, -3+11j, -1+11j, 1+11j, 3+11j, 5+11j, 7+11j, 9+11j, 11+11j, 13+11j, 15+11j,
            -15+9j, -13+9j, -11+9j, -9+9j, -7+9j, -5+9j, -3+9j, -1+9j, 1+9j, 3+9j, 5+9j, 7+9j, 9+9j, 11+9j, 13+9j, 15+9j,
            -15+7j, -13+7j, -11+7j, -9+7j, -7+7j, -5+7j, -3+7j, -1+7j, 1+7j, 3+7j, 5+7j, 7+7j, 9+7j, 11+7j, 13+7j, 15+7j,
            -15+5j, -13+5j, -11+5j, -9+5j, -7+5j, -5+5j, -3+5j, -1+5j, 1+5j, 3+5j, 5+5j, 7+5j, 9+5j, 11+5j, 13+5j, 15+5j,
            -15+3j, -13+3j, -11+3j, -9+3j, -7+3j, -5+3j, -3+3j, -1+3j, 1+3j, 3+3j, 5+3j, 7+3j, 9+3j, 11+3j, 13+3j, 15+3j,
            -15+1j, -13+1j, -11+1j, -9+1j, -7+1j, -5+1j, -3+1j, -1+1j, 1+1j, 3+1j, 5+1j, 7+1j, 9+1j, 11+1j, 13+1j, 15+1j,
            -15-1j, -13-1j, -11-1j, -9-1j, -7-1j, -5-1j, -3-1j, -1-1j, 1-1j, 3-1j, 5-1j, 7-1j, 9-1j, 11-1j, 13-1j, 15-1j,
            -15-3j, -13-3j, -11-3j, -9-3j, -7-3j, -5-3j, -3-3j, -1-3j, 1-3j, 3-3j, 5-3j, 7-3j, 9-3j, 11-3j, 13-3j, 15-3j,
            -15-5j, -13-5j, -11-5j, -9-5j, -7-5j, -5-5j, -3-5j, -1-5j, 1-5j, 3-5j, 5-5j, 7-5j, 9-5j, 11-5j, 13-5j, 15-5j,
            -15-7j, -13-7j, -11-7j, -9-7j, -7-7j, -5-7j, -3-7j, -1-7j, 1-7j, 3-7j, 5-7j, 7-7j, 9-7j, 11-7j, 13-7j, 15-7j,
            -15-9j, -13-9j, -11-9j, -9-9j, -7-9j, -5-9j, -3-9j, -1-9j, 1-9j, 3-9j, 5-9j, 7-9j, 9-9j, 11-9j, 13-9j, 15-9j,
            -15-11j, -13-11j, -11-11j, -9-11j, -7-11j, -5-11j, -3-11j, -1-11j, 1-11j, 3-11j, 5-11j, 7-11j, 9-11j, 11-11j, 13-11j, 15-11j,
            -15-13j, -13-13j, -11-13j, -9-13j, -7-13j, -5-13j, -3-13j, -1-13j, 1-13j, 3-13j, 5-13j, 7-13j, 9-13j, 11-13j, 13-13j, 15-13j,
            -15-15j, -13-15j, -11-15j, -9-15j, -7-15j, -5-15j, -3-15j, -1-15j, 1-15j, 3-15j, 5-15j, 7-15j, 9-15j, 11-15j, 13-15j, 15-15j

        ])/np.sqrt(170)
    
    #Constants of the 10-point Gauss Hermite Quadrature:
    Zeta = np.array([-3.43615911883773760, -2.53273167423278980, -1.75668364929988177,
                     -1.03661082978951365, -0.34290132722370461, 0.34290132722370461,
                     1.03661082978951365, 1.75668364929988177, 2.53273167423278980,
                     3.43615911883773760])
    Gamma = np.array([0.76404328552326206/10**5, 0.13436457467812327/10**2,
                      0.33874394455481063/10, 0.24013861108231469,
                      0.61086263373532580, 0.61086263373532580, 
                      0.24013861108231469, 0.33874394455481063/10,
                      0.13436457467812327/10**2, 0.76404328552326206/10**5])
    
    #Calculating the standard deviation
    SNR = 10**(SNRdB/10)
    Stdev = np.sqrt(np.mean(abs(s)**2)/(2*SNR))

    #GMI calculation (Guass-Hermite quadrature)
    AIR = np.zeros(len(SNRdB))
    for n in range(len(SNRdB)):
        Sum = 0
        for k in range(Modbits):
            for b in {0,1}:
                xSet = s[bmap[:, Modbits-k-1]==b]
                for i in range(M//2):
                    for l_1 in range(len(Zeta)):
                        for l_2 in range(len(Zeta)):
                            #Numerator
                            num = np.sum(np.exp(-(abs(xSet[i]-s)**2 + 2*np.sqrt(2)*Stdev[n]*np.real((Zeta[l_1]+1j*Zeta[l_2])*(xSet[i]-s)))/(2*Stdev[n]**2)))
                            #Denominator
                            den = np.sum(np.exp(-(abs(xSet[i]-xSet)**2 + 2*np.sqrt(2)*Stdev[n]*np.real((Zeta[l_1]+1j*Zeta[l_2])*(xSet[i]-xSet)))/(2*Stdev[n]**2)))
                            
                            Sum = Sum + Gamma[l_1]*Gamma[l_2]*np.log2(num/den)
        AIR[n] = Modbits - Sum/(M*np.pi)


    return AIR

def AIR_SDSW_theoretical(SNRdB, Modbits):
    #Not for differential encoding
    #Calculates approximations of the theoretical AIRs for SD-SW encoders, under a circularly symmetric Gaussian channel.
    #Approximation given by a 10-point Gauss-Hermite Quadrature.
    #Gray labelling assumed
    if(Modbits==2):
        M=4
        bmap = np.array([[1,1],[0,1],[1,0],[0,0]])
        s = np.array([1+1j, -1+1j, 1-1j, -1-1j
                                  ])/np.sqrt(2)
        
    elif(Modbits==4):
        M=16
        bmap = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], 
                    [0, 1, 1, 0],  [0, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 0],
                    [1, 1, 0, 0],[1, 1, 0, 1],[1, 1, 1, 1],[1, 1, 1, 0], 
                    [1, 0, 1, 0],[1, 0, 1, 1],[1, 0, 0, 1], [1, 0, 0, 0]])

        s = np.array([
            -3 + 3j,-1 + 3j,1 + 3j,3 + 3j,
            3 + 1j,1 + 1j,-1 + 1j,-3 + 1j,
            -3 - 1j,-1 - 1j,1 - 1j,3 - 1j,
            3 - 3j,1 - 3j,-1 - 3j,-3 - 3j,
        ])/np.sqrt(10)

    elif(Modbits==6):
        M=64
        bmap = np.array([
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 1], [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0]
        ])

        s = np.array([
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
    
    elif(Modbits==8):
        M = 256

        #don't need bmap

        s = np.array([
            -15+15j, -13+15j, -11+15j, -9+15j, -7+15j, -5+15j, -3+15j, -1+15j, 1+15j, 3+15j, 5+15j, 7+15j, 9+15j, 11+15j, 13+15j, 15+15j,
            -15+13j, -13+13j, -11+13j, -9+13j, -7+13j, -5+13j, -3+13j, -1+13j, 1+13j, 3+13j, 5+13j, 7+13j, 9+13j, 11+13j, 13+13j, 15+13j,
            -15+11j, -13+11j, -11+11j, -9+11j, -7+11j, -5+11j, -3+11j, -1+11j, 1+11j, 3+11j, 5+11j, 7+11j, 9+11j, 11+11j, 13+11j, 15+11j,
            -15+9j, -13+9j, -11+9j, -9+9j, -7+9j, -5+9j, -3+9j, -1+9j, 1+9j, 3+9j, 5+9j, 7+9j, 9+9j, 11+9j, 13+9j, 15+9j,
            -15+7j, -13+7j, -11+7j, -9+7j, -7+7j, -5+7j, -3+7j, -1+7j, 1+7j, 3+7j, 5+7j, 7+7j, 9+7j, 11+7j, 13+7j, 15+7j,
            -15+5j, -13+5j, -11+5j, -9+5j, -7+5j, -5+5j, -3+5j, -1+5j, 1+5j, 3+5j, 5+5j, 7+5j, 9+5j, 11+5j, 13+5j, 15+5j,
            -15+3j, -13+3j, -11+3j, -9+3j, -7+3j, -5+3j, -3+3j, -1+3j, 1+3j, 3+3j, 5+3j, 7+3j, 9+3j, 11+3j, 13+3j, 15+3j,
            -15+1j, -13+1j, -11+1j, -9+1j, -7+1j, -5+1j, -3+1j, -1+1j, 1+1j, 3+1j, 5+1j, 7+1j, 9+1j, 11+1j, 13+1j, 15+1j,
            -15-1j, -13-1j, -11-1j, -9-1j, -7-1j, -5-1j, -3-1j, -1-1j, 1-1j, 3-1j, 5-1j, 7-1j, 9-1j, 11-1j, 13-1j, 15-1j,
            -15-3j, -13-3j, -11-3j, -9-3j, -7-3j, -5-3j, -3-3j, -1-3j, 1-3j, 3-3j, 5-3j, 7-3j, 9-3j, 11-3j, 13-3j, 15-3j,
            -15-5j, -13-5j, -11-5j, -9-5j, -7-5j, -5-5j, -3-5j, -1-5j, 1-5j, 3-5j, 5-5j, 7-5j, 9-5j, 11-5j, 13-5j, 15-5j,
            -15-7j, -13-7j, -11-7j, -9-7j, -7-7j, -5-7j, -3-7j, -1-7j, 1-7j, 3-7j, 5-7j, 7-7j, 9-7j, 11-7j, 13-7j, 15-7j,
            -15-9j, -13-9j, -11-9j, -9-9j, -7-9j, -5-9j, -3-9j, -1-9j, 1-9j, 3-9j, 5-9j, 7-9j, 9-9j, 11-9j, 13-9j, 15-9j,
            -15-11j, -13-11j, -11-11j, -9-11j, -7-11j, -5-11j, -3-11j, -1-11j, 1-11j, 3-11j, 5-11j, 7-11j, 9-11j, 11-11j, 13-11j, 15-11j,
            -15-13j, -13-13j, -11-13j, -9-13j, -7-13j, -5-13j, -3-13j, -1-13j, 1-13j, 3-13j, 5-13j, 7-13j, 9-13j, 11-13j, 13-13j, 15-13j,
            -15-15j, -13-15j, -11-15j, -9-15j, -7-15j, -5-15j, -3-15j, -1-15j, 1-15j, 3-15j, 5-15j, 7-15j, 9-15j, 11-15j, 13-15j, 15-15j

        ])/np.sqrt(170)
    
    #Constants of the 10-point Gauss Hermite Quadrature:
    Zeta = np.array([-3.43615911883773760, -2.53273167423278980, -1.75668364929988177,
                     -1.03661082978951365, -0.34290132722370461, 0.34290132722370461,
                     1.03661082978951365, 1.75668364929988177, 2.53273167423278980,
                     3.43615911883773760])
    Gamma = np.array([0.76404328552326206/10**5, 0.13436457467812327/10**2,
                      0.33874394455481063/10, 0.24013861108231469,
                      0.61086263373532580, 0.61086263373532580, 
                      0.24013861108231469, 0.33874394455481063/10,
                      0.13436457467812327/10**2, 0.76404328552326206/10**5])
    
    #Calculating the standard deviation
    SNR = 10**(SNRdB/10)
    Stdev = np.sqrt(np.mean(abs(s)**2)/(2*SNR))

    #GMI calculation (Guass-Hermite quadrature)
    AIR = np.zeros(len(SNRdB))
    for n in range(len(SNRdB)):
        Sum = 0
        for i in range(M):
            for l_1 in range(len(Zeta)):
                for l_2 in range(len(Zeta)):
                    #Numerator
                    num = np.sum(np.exp(-(abs(s[i]-s)**2 + 2*np.sqrt(2)*Stdev[n]*np.real((Zeta[l_1]+1j*Zeta[l_2])*(s[i]-s)))/(2*Stdev[n]**2)))
                    
                    Sum = Sum + Gamma[l_1]*Gamma[l_2]*np.log2(num)
        AIR[n] = Modbits - Sum/(M*np.pi)
    
    return AIR

def AIR_SDSW(x,y,Modbits):
    #Not for use with differential encoding
    #Estmiates the AIRs for CM schemes with soft decision symbol-wise encoders.
    #Estimates Mutual Information "MI"
    #Assumes a circularly symmetric Gaussian channel
    #Noise is complex Gaussian random variable with total variance Tsigma2 = 2*sigma2
    #x: symbols transmitted in one polarisation, normalised to unit power
    #y: sequence of symbols received in one polarisation.
    #Modbits: modulation format used.

    if(Modbits==2):
        M=4
        s = np.array([1+1j, -1+1j, 1-1j, -1-1j
                                  ])/np.sqrt(2)
        
    elif(Modbits==4):
        M=16
        s = np.array([
            -3 + 3j,-1 + 3j,1 + 3j,3 + 3j,
            3 + 1j,1 + 1j,-1 + 1j,-3 + 1j,
            -3 - 1j,-1 - 1j,1 - 1j,3 - 1j,
            3 - 3j,1 - 3j,-1 - 3j,-3 - 3j
        ])/np.sqrt(10)

    elif(Modbits==6):
        M=64

        s = np.array([
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

    elif(Modbits==8):
        M=256

        s = np.array([
            -15+15j, -13+15j, -11+15j, -9+15j, -7+15j, -5+15j, -3+15j, -1+15j, 1+15j, 3+15j, 5+15j, 7+15j, 9+15j, 11+15j, 13+15j, 15+15j,
            -15+13j, -13+13j, -11+13j, -9+13j, -7+13j, -5+13j, -3+13j, -1+13j, 1+13j, 3+13j, 5+13j, 7+13j, 9+13j, 11+13j, 13+13j, 15+13j,
            -15+11j, -13+11j, -11+11j, -9+11j, -7+11j, -5+11j, -3+11j, -1+11j, 1+11j, 3+11j, 5+11j, 7+11j, 9+11j, 11+11j, 13+11j, 15+11j,
            -15+9j, -13+9j, -11+9j, -9+9j, -7+9j, -5+9j, -3+9j, -1+9j, 1+9j, 3+9j, 5+9j, 7+9j, 9+9j, 11+9j, 13+9j, 15+9j,
            -15+7j, -13+7j, -11+7j, -9+7j, -7+7j, -5+7j, -3+7j, -1+7j, 1+7j, 3+7j, 5+7j, 7+7j, 9+7j, 11+7j, 13+7j, 15+7j,
            -15+5j, -13+5j, -11+5j, -9+5j, -7+5j, -5+5j, -3+5j, -1+5j, 1+5j, 3+5j, 5+5j, 7+5j, 9+5j, 11+5j, 13+5j, 15+5j,
            -15+3j, -13+3j, -11+3j, -9+3j, -7+3j, -5+3j, -3+3j, -1+3j, 1+3j, 3+3j, 5+3j, 7+3j, 9+3j, 11+3j, 13+3j, 15+3j,
            -15+1j, -13+1j, -11+1j, -9+1j, -7+1j, -5+1j, -3+1j, -1+1j, 1+1j, 3+1j, 5+1j, 7+1j, 9+1j, 11+1j, 13+1j, 15+1j,
            -15-1j, -13-1j, -11-1j, -9-1j, -7-1j, -5-1j, -3-1j, -1-1j, 1-1j, 3-1j, 5-1j, 7-1j, 9-1j, 11-1j, 13-1j, 15-1j,
            -15-3j, -13-3j, -11-3j, -9-3j, -7-3j, -5-3j, -3-3j, -1-3j, 1-3j, 3-3j, 5-3j, 7-3j, 9-3j, 11-3j, 13-3j, 15-3j,
            -15-5j, -13-5j, -11-5j, -9-5j, -7-5j, -5-5j, -3-5j, -1-5j, 1-5j, 3-5j, 5-5j, 7-5j, 9-5j, 11-5j, 13-5j, 15-5j,
            -15-7j, -13-7j, -11-7j, -9-7j, -7-7j, -5-7j, -3-7j, -1-7j, 1-7j, 3-7j, 5-7j, 7-7j, 9-7j, 11-7j, 13-7j, 15-7j,
            -15-9j, -13-9j, -11-9j, -9-9j, -7-9j, -5-9j, -3-9j, -1-9j, 1-9j, 3-9j, 5-9j, 7-9j, 9-9j, 11-9j, 13-9j, 15-9j,
            -15-11j, -13-11j, -11-11j, -9-11j, -7-11j, -5-11j, -3-11j, -1-11j, 1-11j, 3-11j, 5-11j, 7-11j, 9-11j, 11-11j, 13-11j, 15-11j,
            -15-13j, -13-13j, -11-13j, -9-13j, -7-13j, -5-13j, -3-13j, -1-13j, 1-13j, 3-13j, 5-13j, 7-13j, 9-13j, 11-13j, 13-13j, 15-13j,
            -15-15j, -13-15j, -11-15j, -9-15j, -7-15j, -5-15j, -3-15j, -1-15j, 1-15j, 3-15j, 5-15j, 7-15j, 9-15j, 11-15j, 13-15j, 15-15j

        ])/np.sqrt(170)
    
    #estimating the variance of the complex Gaussian distribution
    h = np.vdot(x,y)/(np.linalg.norm(x)**2)
    #h = np.sum(np.conjugate(x)*y)/np.sum(abs(x)**2)
    
    sigma2 = np.mean(abs(y-h*x)**2)/2

    #Estimation of the conditional pdf qY|X(y|x)
    qY_X = (1/(2*np.pi*sigma2))*np.exp(-1*(abs(y-h*x)**2)/(2*sigma2))

    #Estimaiton of the conditional pdf qY|X(y|s), where 's' represents constellation symbols

    sT=s.reshape(len(s),1)

    qY_S = np.sum((1/(2*np.pi*sigma2))*np.exp(-1*abs(np.tile(y, (len(s), 1))-h*np.tile(sT, (1, len(y))))**2/(2*sigma2)),0)

    #Estimating AIR (MI)
    AIR = np.sum(np.log2(qY_X/(qY_S/M)))/len(y)
    
    return AIR

def performance_metrics(original_bits, demodulated_bits, source_symbols, processed_symbols, H):
    processed_symbols0 = processed_symbols[0]/np.sqrt(np.mean(np.abs(processed_symbols[0])**2))
    processed_symbols1 = processed_symbols[1]/np.sqrt(np.mean(np.abs(processed_symbols[1])**2))
    processed_symbols = np.array([processed_symbols0,processed_symbols1])
    
    if(p.Mod_param.Modbits==2):
        source_symbols = source_symbols/np.sqrt(2)
    elif(p.Mod_param.Modbits==4):
        source_symbols = source_symbols/np.sqrt(10)
    elif(p.Mod_param.Modbits==6):
        source_symbols = source_symbols/np.sqrt(42)
    elif(p.Mod_param.Modbits==8):
        source_symbols = source_symbols/np.sqrt(170)

    if(p.Mod_param.NPol==1):
        BER = np.mean(original_bits != demodulated_bits)
        if(p.toggle.toggle_AIR==True):
            if(p.toggle.toggle_PAS==False):
                if(p.toggle.AIR_type=='GMI'):
                    AIR = AIR_SDBW(source_symbols, original_bits, processed_symbols, p.Mod_param.Modbits)
                elif(p.toggle.AIR_type=='MI'):
                    AIR = AIR_SDSW(source_symbols, processed_symbols, p.Mod_param.Modbits)
            else:
                AIR = 0 #undefined yet
        else:
            AIR = 0
            
    elif(p.Mod_param.NPol==2):
        BER = (np.sum(original_bits[0] != demodulated_bits[0])+np.sum(original_bits[1] != demodulated_bits[1]))/(2*original_bits.shape[1])
        if(p.toggle.toggle_AIR==True):
            if(p.toggle.toggle_PAS==False):
                if(p.toggle.AIR_type=='GMI'):
                    AIR = 0.5*(AIR_SDBW(source_symbols[0], original_bits[0],processed_symbols[0], p.Mod_param.Modbits) + AIR_SDBW(source_symbols[1], original_bits[1], processed_symbols[1], p.Mod_param.Modbits))
                elif(p.toggle.AIR_type=='MI'):
                    print('AIR0:',AIR_SDSW(source_symbols[0], processed_symbols[0], p.Mod_param.Modbits),'bits/symbols') #TESTING
                    print('AIR1:',AIR_SDSW(source_symbols[1], processed_symbols[1], p.Mod_param.Modbits),'bits/symbol') #TESTING
                    AIR = 0.5*(AIR_SDSW(source_symbols[0], processed_symbols[0], p.Mod_param.Modbits)+AIR_SDSW(source_symbols[1], processed_symbols[1], p.Mod_param.Modbits))
                    
            else:
            #     HI0 = H[0][0]
            #     HQ0 = H[0][1]
            #     HI1 = H[1][0]
            #     HQ1 = H[1][1]
            #     H0 = (HI0+HQ0)/2 #Average of in-phase and quadrature for Pol 0.
            #     H1 = (HI1+HQ1)/2 #Average of in-phase and quadrature for Pol 1.

            #     m = len(H0)
                
            #     NGMI0 = 0
            #     for i in range(m):
            #         NGMI0 += (H0[i])/m
            #     NGMI0 = 1-NGMI0

            #     NGMI1 = 0
            #     for i in range(m):
            #         NGMI1 += (H1[i])/m
            #     NGMI1 = 1-NGMI1

            #     print("NGMI Pol 0:", NGMI0,'bits/bit')
            #     print("NGMI Pol 1:", NGMI1,'bits/bit')


            #     Cpas = p.PAS_param.C
            #     sumC = np.sum(Cpas)*2 #negative values too, when Cpas only has positive amplitudes
            #     P = np.zeros(len(Cpas))
            #     H=0
            #     for i in range(len(Cpas)):
            #         P[i] = Cpas[i]/sumC    
            #         if(P[i]!=0):
            #             H += -1*P[i]*np.log2(P[i])

            #     H*=2 #Positive and negative signs

            #     AIR0 = 2*NGMI0*H
            #     AIR1 = 2*NGMI1*H

            #     print('AIR0:',AIR0,'bits/symbol') 
            #     print('AIR1:', AIR1,'bits/symbol')
            #     AIR = (AIR0+AIR1)/2

                # HI0 = H[0][0]
                # HQ0 = H[0][1]
                # HI1 = H[1][0]
                # HQ1 = H[1][1]
                # m = len(HI0)

                # Cpas = p.PAS_param.C
                # sumC = np.sum(Cpas)*2 #negative values too, when Cpas only has positive amplitudes
                # P = np.zeros(len(Cpas))
                # H=0
                # for i in range(len(Cpas)):
                #     P[i] = Cpas[i]/sumC    
                #     if(P[i]!=0):
                #         H += -1*P[i]*np.log2(P[i])
                # H *= 2 #For negative values too
                
                # GMI_0_I = 0
                # for i in range(m):
                #     GMI_0_I += HI0[i]
                # GMI_0_I = H - GMI_0_I
                
                # GMI_0_Q = 0
                # for i in range(m):
                #     GMI_0_Q += HQ0[i]
                # GMI_0_Q = H - GMI_0_Q

                # GMI_1_I = 0
                # for i in range(m):
                #     GMI_1_I += HI1[i]
                # GMI_1_I = H - GMI_1_I

                # GMI_1_Q = 0
                # for i in range(m):
                #     GMI_1_Q += HQ1[i]
                # GMI_1_Q = H - GMI_1_Q

                # GMI_0 = GMI_0_I + GMI_0_Q
                # GMI_1 = GMI_1_I + GMI_1_Q

                # print('AIR0:',GMI_0,'bits/symbol') 
                # print('AIR1:', GMI_1,'bits/symbol')
                # AIR = (GMI_0+GMI_1)/2

                HI0 = H[0][0]
                HQ0 = H[0][1]
                HI1 = H[1][0]
                HQ1 = H[1][1]
                m = len(HI0)

                Cpas = p.PAS_param.C
                sumC = np.sum(Cpas) #Only positive amplitudes
                P = np.zeros(len(Cpas))
                H=0
                for i in range(len(Cpas)):
                    P[i] = Cpas[i]/sumC    
                    if(P[i]!=0):
                        H += -1*P[i]*np.log2(P[i])
                
                
                GMI_0_I = 0
                for i in range(1,m): #Only amplitude bits are shaped
                    GMI_0_I += HI0[i]
                GMI_0_I = H - GMI_0_I
                GMI_0_I = GMI_0_I + 1 - HI0[0] #GMI of sign bit
                
                GMI_0_Q = 0
                for i in range(1,m):
                    GMI_0_Q += HQ0[i]
                GMI_0_Q = H - GMI_0_Q
                GMI_0_Q = GMI_0_Q + 1 - HQ0[0] 

                GMI_1_I = 0
                for i in range(1,m):
                    GMI_1_I += HI1[i]
                GMI_1_I = H - GMI_1_I
                GMI_1_I = GMI_1_I + 1 - HI1[0]

                GMI_1_Q = 0
                for i in range(1,m):
                    GMI_1_Q += HQ1[i]
                GMI_1_Q = H - GMI_1_Q
                GMI_1_Q = GMI_1_Q + 1 - HQ1[0]

                GMI_0 = GMI_0_I + GMI_0_Q
                GMI_1 = GMI_1_I + GMI_1_Q

                print('AIR0:',GMI_0,'bits/symbol') 
                print('AIR1:', GMI_1,'bits/symbol')
                AIR = (GMI_0+GMI_1)/2


        else:
            AIR = 0


    if(p.toggle.toggle_AIR==True):
        #Shannon limit 
        snr_db = p.fibre_param.snr_db
        dbrange = 21
        snr_dbarr = np.arange(snr_db-dbrange//2, snr_db+dbrange//2,1)
        snr_dbLinarr = 10**(snr_dbarr/10)
        shannon = np.log2(1+snr_dbLinarr)
        if(p.toggle.AIR_type == 'GMI'):
            AIR_theoretical = AIR_SDBW_theoretical(snr_dbarr, p.Mod_param.Modbits)
            plt.figure()
            M = int(2**p.Mod_param.Modbits)
            plt.title(f"SD-BW AIRs with {M}-QAM")
            plt.xlabel("SNR (dB)")
            plt.ylabel("GMI (bits/symbols)")
            plt.plot(p.fibre_param.snr_db, AIR, marker='o', color='b', label='Emprirical AIR')
            plt.plot(snr_dbarr, AIR_theoretical, marker='x', color='r', label='Theoretical AIR')
            plt.plot(snr_dbarr, shannon, color='g', label='Shannon Limit')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc='lower left')
            plt.ylim(0, max(AIR_theoretical[-1],AIR)+0.5)
        elif(p.toggle.AIR_type=='MI'):
            AIR_theoretical = AIR_SDSW_theoretical(snr_dbarr, p.Mod_param.Modbits)
            plt.figure()
            M = int(2**p.Mod_param.Modbits)
            plt.title(f"SD-SW AIRs with {M}-QAM")
            plt.xlabel("SNR (dB)")
            plt.ylabel("MI (bits/symbols)")
            plt.plot(p.fibre_param.snr_db, AIR, marker='o', color='b', label='Emprirical AIR')
            plt.plot(snr_dbarr, AIR_theoretical, marker='x', color='r', label='Theoretical AIR')
            plt.plot(snr_dbarr, shannon, color='g', label='Shannon Limit')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc='lower left')
            plt.ylim(min(min(AIR_theoretical)-2, AIR), max(AIR_theoretical[-1],AIR)+2)

    return BER, AIR, AIR_theoretical
