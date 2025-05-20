import numpy as np
import matplotlib.pyplot as plt
import parameters as p
import functions as f
import matplotlib as mpl

plt.rcParams['font.size'] = 22  # Change the font size
plt.rcParams['font.family'] = 'Times New Roman' 
mpl.rcParams['mathtext.fontset'] = 'stix'

plot_ngvariation = False
if(plot_ngvariation==True):
    wavelength = np.arange(1200,1400,1)
    ng = 1.4565 + 3*(57.086/wavelength)**2 + (wavelength/17436)**2
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength,ng, color='black')
    plt.xlabel('Wavelength/ nm')
    plt.ylabel('Group Refractive Index')
    plt.show()

plot_RRC = True
if(plot_RRC==True):
    rolloffs = [0.0, 0.25, 0.5, 1.0]
    span = 10
    sps = 8

    plt.figure(figsize=(6, 6))

    # Time domain
    
    for alpha in rolloffs:
        h,t = f.RRC(span, alpha, sps)
        plt.plot(t, h, label=f'α={alpha}')
    # plt.title('RRC Impulse Response (Time Domain)')
    plt.xlabel('Time [symbols]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Frequency domain
    plt.figure(figsize=(10,6))
    for alpha in rolloffs:
        h,t = f.RRC(span, alpha, sps)
        offset = 350
        H = np.fft.fftshift(np.fft.fft(h, 1024))
        H /= max(H)
        H = H[offset:len(H)-offset]
        fre = np.linspace(-1, 1, len(H))
        plt.plot(fre, np.abs(H), label=rf'$\alpha^{{\mathrm{{RRC}}}} = {alpha}$')
    # plt.title('RRC Frequency Response (Magnitude)')
    plt.xlabel('Normalised Frequency $fT_s$')
    plt.ylabel(r'$|H_{\mathrm{RRC}}( f )|$')
    plt.grid(False)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()