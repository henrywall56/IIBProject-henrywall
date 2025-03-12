function plot_spectrum(y_Tx,Fs)
Npoints = length(y_Tx);
n = (1:length(y_Tx))';
FFT_Ex_1 = fftshift(fft(y_Tx));
FFT_Ex = abs(FFT_Ex_1)./(length(y_Tx));
Frek = (Fs*(-(Npoints)/2:((Npoints/2)-1)))/Npoints;
% figure;
plot(Frek./1e9, 10*log10(FFT_Ex.^2));
title('Spectrum of the signal');
xlabel('Frequency, GHz');ylabel('Power, dB');