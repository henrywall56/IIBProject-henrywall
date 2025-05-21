import numpy as np
import matplotlib.pyplot as plt
import parameters as p
import functions as f
import matplotlib as mpl
import performance_evaluation as pe

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

plot_RRC = False
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

plot_MI=False
if(plot_MI==True):
    dbrange = 32
    snr_db = 16  # Center SNR value, adjust as needed
    snr_dbarr = np.arange(snr_db - dbrange//2, snr_db + dbrange//2 + 1, 1)
    snr_lin = 10**(snr_dbarr / 10)

    # Shannon capacity curve
    shannon = np.log2(1 + snr_lin)

    # Modulation formats
    mod_formats = {
        'QPSK': 2,
        '16-QAM': 4,
        '64-QAM': 6,
        '256-QAM': 8
    }

    plt.figure(figsize=(8, 5))
    plt.plot(snr_dbarr, shannon, 'k-', label='Shannon Limit')

    # Plot MI curves for each format
    for label, modbits in mod_formats.items():
        MI = pe.AIR_SDSW_theoretical(snr_dbarr, modbits)
        plt.plot(snr_dbarr, MI, label=f'{label}')
        asymptote = modbits  # log2(M) = modbits
        idx = -1  # rightmost point
        x_pos = snr_dbarr[idx] - 4.1
        y_pos = MI[idx] + 0.2  # offset above the curve
        plt.text(x_pos, y_pos, f'{label}', fontsize=18)
    x_pos = snr_dbarr[-1] - 17  # adjust as needed
    y_pos = shannon[-1] - 5
    plt.text(x_pos, y_pos, 'Shannon Limit', fontsize=18, color='black', rotation=31)

    plt.xlabel('SNR (dB)')
    plt.ylabel('MI (bits/symbol)')
    plt.grid(False)
    # plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


plot_GMI=False
if(plot_GMI==True):
    dbrange = 32
    snr_db = 16  # Center SNR value, adjust as needed
    snr_dbarr = np.arange(snr_db - dbrange//2, snr_db + dbrange//2 + 1, 1)
    snr_lin = 10**(snr_dbarr / 10)

    # Shannon capacity curve
    shannon = np.log2(1 + snr_lin)

    # Modulation formats
    mod_formats = {
        'QPSK': 2,
        '16-QAM': 4,
        '64-QAM': 6,
        '256-QAM': 8
    }

    plt.figure(figsize=(8, 5))
    

    # Plot MI curves for each format
    for label, modbits in mod_formats.items():
        if(modbits!=8):
            MI = pe.AIR_SDBW_theoretical(snr_dbarr, modbits)
        else:
            MI = pe.AIR_SDSW_theoretical(snr_dbarr, modbits)
        plt.plot(snr_dbarr, MI, label=f'{label}')
        asymptote = modbits  # log2(M) = modbits
        idx = -1  # rightmost point
        x_pos = snr_dbarr[idx] - 4.1
        y_pos = MI[idx] + 0.2  # offset above the curve
        plt.text(x_pos, y_pos, f'{label}', fontsize=18)
    x_pos = snr_dbarr[-1] - 17  # adjust as needed
    y_pos = shannon[-1] - 5
    plt.text(x_pos, y_pos, 'Shannon Limit', fontsize=18, color='black', rotation=31)
    plt.plot(snr_dbarr, shannon, 'k-', label='Shannon Limit')
    plt.xlabel('SNR (dB)')
    plt.ylabel('GMI (bits/symbol)')
    plt.grid(False)
    # plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


plot_CD_filter=False
if(plot_CD_filter==True):
    N_FFT = 1024
    D = 17 *1e-12*1e9*1e-3  # ps/nm/km to s/m
    c = 3e8  # speed of light [m/s]
    lam = 1550e-9  # wavelength [m]
    Ts = 1 / (100e9)  # 2 samples per symbol at 16 Gbaud
    fs = 1 / Ts
    delta_f = fs / N_FFT

    # Frequency index
    n = np.arange(-N_FFT // 2, N_FFT // 2)
    f = n * delta_f

    # --- Magnitude for 100 km only ---
    L_100 = 100e3  # 100 km in meters
    phi_100 = -(np.pi * lam**2 * D * L_100 / c) * f**2
    H_CD_100 = np.exp(1j * phi_100)
    magnitude_100 = np.abs(H_CD_100)

    plt.figure(figsize=(6,4))
    plt.plot(n, magnitude_100, color='black')
    plt.ylim((0,2)) 
    
    plt.xlabel('FFT index $n$')
    plt.ylabel(r'$|H_{\mathrm{CD}}[n]|$')
    plt.tight_layout()

    # --- Unwrapped Phase for 100 km and 200 km ---
    L_200 = 200e3  # 200 km in meters
    phi_200 = -(np.pi * lam**2 * D * L_200 / c) * f**2
    H_CD_200 = np.exp(1j * phi_200)

    unwrapped_phase_100 = np.unwrap(np.angle(H_CD_100))
    unwrapped_phase_200 = np.unwrap(np.angle(H_CD_200))
    unwrapped_phase_100 -= unwrapped_phase_100[N_FFT // 2]
    unwrapped_phase_200 -= unwrapped_phase_200[N_FFT // 2]

    plt.figure(figsize=(6, 4))
    plt.plot(n, unwrapped_phase_100, color='black', label='100 km')
    plt.plot(n, unwrapped_phase_200, 'k--', label='200 km')  # dashed black
    plt.xlabel('FFT index $n$')
    plt.ylabel(r'$\angle H_{\mathrm{CD}}[n]$ / rad')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_CD_complexity = False
if(plot_CD_complexity==True):
    D = 17  # s/m (converted from ps/nm/km)
    c = 3e8  # speed of light [m/s]
    lam = 1550e-9  # wavelength [m]
    Ts = 1 / (100e9)  # sampling period (100 GSa/s = 2 samples per 50 Gbaud)
    fs = 1 / Ts

    # Fibre lengths
    L_values = [100, 200]  # in meters
    labels = ['100 km', '200 km']
    colors = ['black', 'gray']

    # Assume a spectral width (in nm)
    delta_lambda = 0.4  # nm eg 50GBaud

    # Plot setup
    plt.figure(figsize=(6, 4))

    for L, label, color in zip(L_values, labels, colors):
        delta_T = abs(D) * L * delta_lambda * 1e-12  # seconds
        N_CD = int(np.ceil(delta_T / Ts))  # samples
        N_FFTs = [2**i for i in range(8, 15) if 2**i > N_CD]
        N_cm = [(N * np.log2(N) + N) / (N - N_CD + 1) for N in N_FFTs]
        plt.plot(N_FFTs, N_cm, label=label, color=color)
        plt.xscale('log', base=2)


    plt.xlabel(r'$N_{\mathrm{FFT}}$')
    plt.ylabel(r'$N_{\mathrm{cm}}$')
    plt.xticks(N_FFTs, [r'$2^{{{}}}$'.format(int(np.log2(n))) for n in N_FFTs])
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_shaping_gain_16qam=True
if(plot_shaping_gain_16qam==True):

    SNR = np.array([29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8])
    lam16 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0.01,0.01,0.025,0.03125,0.05,0.0625,0.075,0.08,0.1]
    AIR16 = [4,4,4,4,4,4,4,4,4,4,4,4, 3.989,3.971,3.932,3.866,3.768,3.635,3.467,3.272,3.051,2.817]
    GMI16 = pe.AIR_SDBW_theoretical(SNR, 4)

    
    lam64 = [0,0,0,0,0,0,0,0.0025,0.0075,0.01325,0.014,0.015,0.01625,0.025, 0.03125,0.05]
    AIR64 = [6,6,6,6,6,5.994,5.982,5.955,5.901,5.809,5.698,5.549,5.353,5.118,4.886,3.999]
    SNR64 = np.array([29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,12])
    GMI64 = pe.AIR_SDBW_theoretical(SNR64,6)

    lam256 = [0.01,0.015,0.015]
    AIR256 = [6.743,6.214,5.599]
    SNR256 = np.array([21,19,17])
    #MI
    GMI256 = pe.AIR_SDSW_theoretical(SNR256,8)
    plt.figure()
    plt.plot(SNR, GMI16, color='red')
    plt.plot(SNR, AIR16, color='blue')
    SNR_lin =  10**(SNR / 10)
    shannon = np.log2(1 + SNR_lin)
    plt.plot(SNR, shannon, 'black')
    plt.xlabel('SNR / dB')
    plt.ylabel('GMI / bits/symbol')

    plt.figure()
    plt.plot(SNR64, GMI64, color='green')
    plt.plot(SNR64, AIR64, color='orange')
    SNR_lin64 =  10**(SNR64 / 10)
    shannon64 = np.log2(1 + SNR_lin64)
    plt.plot(SNR64, shannon64, 'black')
    plt.xlabel('SNR / dB')
    plt.ylabel('GMI / bits/symbol')

    plt.figure()
    plt.plot(SNR256, GMI256, color='red')
    plt.plot(SNR256, AIR256, color='blue')
    SNR_lin =  10**(SNR256 / 10)
    shannon = np.log2(1 + SNR_lin)
    plt.plot(SNR256, shannon, 'black')
    plt.xlabel('SNR / dB')
    plt.ylabel('GMI / bits/symbol')

    plt.figure()
    plt.plot(SNR, GMI16, color='red')
    plt.plot(SNR, AIR16, color='blue')
    plt.plot(SNR64, GMI64, color='green')
    plt.plot(SNR64, AIR64, color='orange')
    SNRfull = np.arange(8,23,1)
    SNR_linfull =  10**(SNRfull / 10)
    shannonfull = np.log2(1 + SNR_linfull)
    plt.plot(SNRfull, shannonfull, 'black')
    plt.xlabel('SNR / dB')
    plt.ylabel('GMI / bits/symbol')

    plt.figure()
    plt.plot(SNR, lam16,color='blue')
    plt.plot(SNR64,lam64,color='orange')
    plt.xlabel('SNR / dB')
    plt.ylabel(r'$\lambda$')

    plt.show()
    #Repeat this for 64QAM and 256QAM, eg optimise lamda for different SNRs, and plot AIR with uniform vs PCS
