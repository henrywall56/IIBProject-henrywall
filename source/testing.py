import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import functions as f

def soft_phase_estimator(y, Modbits):
    #SOFT-DECISION PHASE-ESTIMATOR:
    #transmitted symbols x
    #received symbols y
    #theta is actual phase noise

    #have estimate of phase noise thetaHat

    #De-rotare received symbols by estimate of noise
    xHat_predecision = y*np.exp(-1j*thetaHat) #Compensate for phase estimate
    
    #Pass de-rotated through a decision device
    xHat = f.max_likelihood_decision(xHat_predecision, Modbits) 

    #At high SNR, xHat = x with high probability

    #psi = theta + n'   
    #n' is angular projection of Gaussian noise in direction orthogonal to x*exp(j*theta)
    psi = np.angle(y) - np.angle(xHat)

    psi_unwrapped = np.unwrap([psi])

    return psi_unwrapped #To be passed through a Wiener Filter

def hard_phase_estimator(psi, r, alpha, L, delta, wplot_toggle):
    #HARD DECISION PHASE ESTIMATOR:
    #psi is the phase noise corrupted by Gaussian noise n'
    #Pass psi through a Wiener filter W(z)
    #Output is the MMSE estimate thetaHat of the actual carrier phase theta
    #alpha and r are parameters of Wiener Filter
    #L is length of filter
    #delta is delay of filter
    #wplot_toggle = True to display filter plot

    n  = np.linspace(-delta, L-delta, L+1) #make L odd, so symmetrical about 0

    w = alpha**(abs(n)) * alpha*r / (1-alpha**2)

    if(wplot_toggle==True):
        plt.plot(n,w)
        plt.show()  
    
    return convolve(psi, w, mode="same") #return best estimate of carrier phase
    

def generate_Wiener_parameters(sps, Rs, Linewidth, snrb_db, Modbits, frac):
    #rx: Received signal
    #sps: Samples per symbol
    #Rs: symbol rate symbols/second
    #Linewidth: Laser linewidth in Hz 
    #Modbits: 4 for 16-QAM, 6 for 64-QAM
    #snrb_db: SNR per bit in dB
    #frac: Neglect Wiener coefficients if they are less than a fraction frac (0<=frac<=1) than largest coefficient w_0

    #var_p = phase noise variance
    #var_n = half_constellation_penalty / SNR per symbol
    #SNR per symbol = SNR per bit * bits per symbol

    T = 1/(sps*Rs)
    var_p = 2*np.pi*Linewidth*T #phase noise variance
    snrb = 10 ** (snrb_db / 10) #snr per bit (Linear)
    snrsymb = Modbits * snrb #snr per symbol
    if(Modbits == 4): #16-QAM
        constellation_penalty = 1.889
        half_constellation_penalty = constellation_penalty/2
    elif(Modbits == 6): #64-QAM
        constellation_penalty = 2.685
        half_constellation_penalty = constellation_penalty/2
    var_n = half_constellation_penalty / snrsymb
    r = var_p / var_n 
    alpha = (1 + 0.5*r) - np.sqrt((1+0.5*r)**2 - 1)
    #neglect coefficients that are less than a fraction f og value of largest coefficient
    L=np.ceil(2*np.log2(1/frac)/np.log2(1/alpha))
    L=int(L)
    if(L%2)==0:
        L+=1    

    return r, alpha, L

def phase_noise_compensation(rx, sps, Rs, Linewidth, Modbits, snrb_db, frac, wplot_toggle=False, thetaplot_toggle=False):
    #rx: Received signal
    #sps: Samples per symbol
    #Rs: symbol rate symbols/second
    #Linewidth: Laser linewidth in Hz 
    #Modbits: 4 for 16-QAM, 6 for 64-QAM
    #snrb_db: SNR per bit in dB
    #frac: Neglect Wiener coefficients if they are less than a fraction frac (0<=frac<=1) than largest coefficient w_0
    #wplot_toggle: True means plot the Wiener filter coefficients
    #thetaplot_toggle: True means plot phase noise estimate

    #Wiener filter parameters:
    r, alpha, L = generate_Wiener_parameters(sps, Rs, Linewidth, snrb_db, Modbits, frac)

    SD_delta = 0 #delta for soft decision phase estimator

    psi_unwrapped = soft_phase_estimator(rx, Modbits) #Returns first estimate of phase noise, corrupted by Gaussian noise

    #delta: delay of FIR filter.

    HD_delta = np.floor(L-0.5) #delta for hard decision Wiener filter

    thetaHat = hard_phase_estimator(psi_unwrapped, r, alpha, L, HD_delta, wplot_toggle) #Best estimate of phase noise

    if(thetaplot_toggle==True):
        plt.plot(np.arange(len(thetaHat)), thetaHat)
        plt.show()

    rx *= np.exp(-1j*thetaHat) #de-rotate by phase noise estimate

    return rx

#TODO:
    #need to have thetahat estimate for soft decision part
    #check if implementing correctly, eg look at offset Delta in paper

hard_phase_estimator(psi=[1,2,4,4,5,6,7], sps=1, Rs=10000000, Linewidth=1000, Modbits=4, snrb_db=10, frac=0.001, delta=2 ,wplot_toggle=True)

""""
def phase_unwrap(psi):
    # psi is carrier phase prior to the unwrapper
    
    for k in range(len(psi) - 1):
        p = np.floor(0.5 + (psi[k] - psi[k + 1]) / (2 * np.pi))
        psi[k + 1] = psi[k+1] + p * (2 * np.pi) #note psi[k] doesn't rely on psi[k+1] like paper suggests
    return psi

#or use np.unwrap
"""""