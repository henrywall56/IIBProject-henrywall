import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Original data
QPSK = [1.26991272e-02 ,6.14547729e-03, 2.45666504e-03 ,7.62939453e-04,
 2.09808350e-04, 1.52587891e-05 ,3.81469727e-06, 0.00000000e+00,
 0.00000000e+00]

QPSKDE = [2.53295898e-02 ,1.17492676e-02, 4.96673584e-03 ,1.49536133e-03,
 4.11987305e-04, 7.62939453e-05, 1.52587891e-05, 7.62939453e-06,
 0.00000000e+00]

snr_db = np.arange(7, 16, 1)

# Remove zeros (logarithms of zero are undefined)
epsilon = 0  # Small value to avoid issues with zeros
QPSK = np.array(QPSK) + epsilon
QPSKDE = np.array(QPSKDE) + epsilon

# Interpolation in the log domain
snr_smooth = np.linspace(snr_db.min(), snr_db.max(), 300)
qpsk_log = np.log10(QPSK)
qpskde_log = np.log10(QPSKDE)

qpsk_smooth_log = np.interp(snr_smooth, snr_db, qpsk_log)
qpskde_smooth_log = np.interp(snr_smooth, snr_db, qpskde_log)

qpsk_smooth = 10 ** qpsk_smooth_log
qpskde_smooth = 10 ** qpskde_smooth_log

# Plotting
plt.figure(figsize=(8, 6))
plt.semilogy(snr_smooth, qpskde_smooth, color='red', label='Differential Encoding')
plt.semilogy(snr_smooth, qpsk_smooth, color='blue', label='Gray Mapping')
plt.xlabel('SNR per bit (dB)')
plt.ylabel('BER')
plt.title('Bit Error Rate (BER) vs SNR per B it (dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()
