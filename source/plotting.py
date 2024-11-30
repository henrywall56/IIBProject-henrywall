import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

AIR_16QAM = [0.98751639, 1.36158661, 1.73166976, 2.19000541, 2.68635947, 3.17475941
 ,3.57694062 ,3.84631423, 3.97098125 ,3.99649874]
BER_16QAM = [3.72375e-01 ,3.24450e-01,2.74775e-01 ,2.17625e-01 ,1.54550e-01 ,9.33750e-02
 ,4.71500e-02 ,1.60250e-02, 3.00000e-03 ,1.50000e-04]

AIR_4QAM = [0.97362657, 1.28424035 ,1.59528509 ,1.8156295 , 1.95241441 ,1.99384761,
 1.99950612 ,2.     ,    2. ,        2.    ,    ]
BER_4QAM = [2.6295e-01 ,1.8550e-01, 1.0410e-01, 4.5900e-02, 1.1400e-02 ,1.4000e-03
 ,5.0000e-05 ,0.0000e+00, 0.0000e+00, 0.0000e+00]

AIR_64QAM = [0.97976809 ,1.35370904 ,1.76928409, 2.21949269 ,2.73303164 ,3.22787312,
 3.79344528 ,4.33137532 ,4.88527216 ,5.34361232]
BER_64QAM = [0.4193,     0.38643333, 0.34751667, 0.30016667 ,0.25326667 ,0.2053,
 0.15955    ,0.11418333,0.07191667 ,0.03988333]

snr_db = np.arange(0,20,2)
snr_dbLin = 10**(snr_db/10)
shannon = np.log2(1+snr_dbLin)

plt.figure()
plt.title(f"Pre-FEC BER vs SNR Plot")
plt.xlabel("SNR (dB)")
plt.ylabel("BER ")
plt.semilogy(snr_db, BER_4QAM, color='b', label='QPSK')
plt.semilogy(snr_db, BER_16QAM, color='purple', label='16 QAM')
plt.semilogy(snr_db, BER_64QAM, color='orange', label='64 QAM')
plt.legend(loc='lower left')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()




"""""
        plt.figure(figsize=(8,8))
        plt.plot(np.arange(num_symbols), theta[0::sps], label='Phase')
        plt.plot(np.arange(num_symbols), thetaHat, color='red', label='Phase Estimate with BPS')
        plt.title('Laser Phase Noise vs Time plot')
        plt.xlabel('Time / Arbitrary Units')
        plt.ylabel('Phase / rad')
        plt.grid(True)
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.legend(loc='lower left')
        plt.show()
    
    
        plt.figure(figsize=(8,8))
        plt.plot(np.arange(num_symbols), theta[0::sps], label='Phase')
        plt.title('Laser Phase Noise vs Time plot')
        plt.xlabel('Time / Arbitrary Units')
        plt.ylabel('Phase / rad')
        plt.grid(True)
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.legend(loc='lower left')
        plt.show()

"""
""""
        if(snr_dbi==14):
            plt.figure(figsize=(8,8))
            plt.scatter(Phase_Noise_compensated_rx.real, Phase_Noise_compensated_rx.imag, alpha=0.5, color='blue', label='Correct Symbols')
            plt.scatter(Phase_Noise_compensated_rx[erroneous_indexes].real, Phase_Noise_compensated_rx[erroneous_indexes].imag, color='red', label='Errors', alpha=0.5)
            plt.title(f'Recovered Constellation at SNR={snr_dbi}')
            plt.xlabel('In-Phase')
            plt.ylabel('Quadrature')
            num=1.5
            plt.xlim(-1*num, num)
            plt.ylim(-1*num, num)
            plt.grid(True)
            plt.axhline(0, color='black', lw=0.5)
            plt.axvline(0, color='black', lw=0.5)
            plt.legend(loc='lower left')
            plt.show()
"""""