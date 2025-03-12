function [Q_time phase_mis]=procedureRIGHT2(Q_time,Length_R_misN) 

%coefficients for [angle amplitude]

[angle amplitude] = procedureRIGHT1;
angle(1)=Length_R_misN;

%parameters
samplingrates = 50 ;% 50 -- [GHz] 

%load data here
% Q_time=load('z:data-1002.dat');

Q_freq=fft(Q_time.');

ang_freq_posi=zeros(1,length(Q_freq)/2);
amp_freq_posi=zeros(1,length(Q_freq)/2);

I2=1:length(Q_freq)/2;

for J1=1:length(angle)-1
        ang_freq_posi=ang_freq_posi+angle(J1)*(samplingrates/length(Q_time)*I2).^(length(angle)-J1);
end

% for J1=1:length(amplitude)
%         amp_freq_posi=amp_freq_posi+amplitude(J1)*(samplingrates/length(Q_time)*I2).^(length(amplitude)-J1);
% end

Q_freq_II=Q_freq(1:length(Q_freq)/2).*(exp(j*(ang_freq_posi)).*amplitude);
Q_freq_III=Q_freq_II(2:length(Q_freq_II));
Q_freq_II=[real(Q_freq_II(1)) Q_freq_III (Q_freq_II(length(Q_freq)/2)) fliplr(conj(Q_freq_III))];

Q_time=ifft(Q_freq_II);
Q_time=real(Q_time);
phase_mis=angle(end);
Q_time = Q_time.';
% close all;
    
end