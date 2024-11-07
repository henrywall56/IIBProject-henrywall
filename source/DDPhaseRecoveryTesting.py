import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import functions as f


#Decision Directed Phase Recovery Testing
def soft_phase_estimator(y, sps, Rs, Linewidth, snrb_db, Modbits, frac):
    #SOFT-DECISION PHASE-ESTIMATOR:
    #transmitted symbols x
    #received symbols y
    #theta is actual phase noise
    T = 1/(Rs*sps)
    
    r,alpha,L = generate_Wiener_parameters(sps, Rs, Linewidth, snrb_db, Modbits, frac)
    delta = 0   
    L = L//2 #Do we use half value for soft phase estimator?
    w = generate_Wiener_coefficients(r, alpha, L, delta)
    
    print('SD Wiener', 'r=', r, 'L=', L, 'a=', alpha, 'snr_db=', snrb_db)

    num_samples = len(y)
    psi_unwrapped = np.zeros(num_samples)  # Output array
    theta_estimate = 0.0  # Initial theta[k+1] estimate

    psi = np.zeros(num_samples)  # Intermediate phase differences

    for k in range(num_samples):

        xHatk_predecision = y[k] * np.exp(-1j * theta_estimate)
        xHatk = f.max_likelihood_decision(np.array([xHatk_predecision]), Modbits)[0] #Pass through decision function
    
        psi[k] = np.angle(y[k]) - np.angle(xHatk)
        
        # Unwrapping logic
        if k >=3:
            #p = np.floor(0.5 + (psi_unwrapped[k-1] - psi[k]) / (2 * np.pi))
            
            p = np.floor(0.5 + (((1/3)*(psi_unwrapped[k-1]+psi_unwrapped[k-2]+psi_unwrapped[k-3])-psi[k])/(2*np.pi)))

            psi_unwrapped[k] = psi[k-1] + p * 2 * np.pi
        
        else:
            psi_unwrapped[k] = psi[k]
       

        # Update theta estimate using the Wiener filter after enough samples
        if k >= L-1:  # Ensure we have enough samples for the filter
            #Only need one element of convolution:
            theta_estimate = sum(w[l] * psi_unwrapped[k - l] for l in range(L))
       

    """""
    for x,y in enumerate(w):
        plt.plot([x, x],[0,y], color='blue', linestyle='-', linewidth=2)
    plt.title("SD Wiener")
    plt.show()
    """""
    return psi_unwrapped


def generate_Wiener_coefficients(r, alpha, L, delta):
    n  = np.linspace(-delta, L-delta-1, L) #make L odd, so symmetrical about 0
    w = alpha**(abs(n)) * alpha*r / (1-alpha**2)
    return w

    

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

    if(Modbits == 2): #QPSK
        constellation_penalty = 1
        half_constellation_penalty = constellation_penalty/2
    elif(Modbits == 4): #16-QAM
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

def DD_phase_noise_compensation(rx, sps, Rs, Linewidth, Modbits, snrb_db, frac, toggle_phasenoisecompensation):
    #rx: Received signal
    #sps: Samples per symbol
    #Rs: symbol rate symbols/second
    #Linewidth: Laser linewidth in Hz 
    #Modbits: 4 for 16-QAM, 6 for 64-QAM
    #snrb_db: SNR per bit in dB
    #frac: Neglect Wiener coefficients if they are less than a fraction frac (0<=frac<=1) than largest coefficient w_0
    #wplot_toggle: True means plot the Wiener filter coefficients
    #thetaplot_toggle: True means plot phase noise estimate

    if(toggle_phasenoisecompensation==False):
        
        return rx, np.zeros(len(rx)) 
    
    else:

        #Wiener filter parameters:
        r, alpha, L = generate_Wiener_parameters(sps, Rs, Linewidth, snrb_db, Modbits, frac)

        
        #delta: delay of FIR filter.

        psi_unwrapped = soft_phase_estimator(rx, sps, Rs, Linewidth, snrb_db, Modbits, frac) #Returns first estimate of phase noise, corrupted by Gaussian noise

        #HARD DECISION PHASE ESTIMATOR:

        HD_delta = np.floor((L-1)/2) #delta for hard decision Wiener filter
        
        print('HD Wiener', 'r=', r, 'L=', L, 'a=', alpha, 'snr_db=', snrb_db)
        wiener_HD = generate_Wiener_coefficients(r, alpha, L, HD_delta)

        """""
        for x,y in enumerate(wiener_HD):
            plt.plot([x, x],[0,y], color='blue', linestyle='-', linewidth=2)
        plt.title("HD Wiener")
        plt.show()
        """""

        #convolve psi_unwrapped with wiener_HD to get thetaHat

        thetaHat = convolve(psi_unwrapped, wiener_HD, mode="same")

        derotated = rx*np.exp(-1j*thetaHat) #de-rotate by phase noise estimate
       
        return derotated, thetaHat 

