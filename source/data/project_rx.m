%% Load the waveforms
close all;
clc;
% tic;
measurement_index = 1;
X_I = Signal_X_I(:,measurement_index);
X_Q = Signal_X_Q(:,measurement_index);
Y_I = Signal_Y_I(:,measurement_index);
Y_Q = Signal_Y_Q(:,measurement_index);

% Normalise 
[Rx,Ry] = NormaliseSignal(X_I,X_Q,Y_I,Y_Q);
plot_spectrum(Rx,256e9);
plot_spectrum(Ry,256e9);

scatterplot(Rx)
scatterplot(Ry)

%% Rx DeSkew
Rx_SkewParam.skew_x = 0;
Rx_SkewParam.skew_xIQ = 0.2;
Rx_SkewParam.skew_y = 0;
Rx_SkewParam.skew_yIQ = -0.04; % -0;04 0.06
[Rx_XI,Rx_XQ,Rx_YI,Rx_YQ] = RxDeskew(Rx,Ry,Rx_SkewParam);

%% GSO
[Rx_XI, Rx_XQ] = func_hybridIQcomp(Rx_XI, Rx_XQ);
[Rx_YI, Rx_YQ] = func_hybridIQcomp(Rx_YI, Rx_YQ);

Rx = Rx_XI + 1i*Rx_XQ;
Ry = Rx_YI + 1i*Rx_YQ;

scatterplot(Rx)
scatterplot(Ry)

%% Resample to 2 sps
NormX = mean(abs(pattern_x));
NormY = mean(abs(pattern_y));

X_2sps = resample(Rx,100e9,256e9);
X_2sps = X_2sps.*(NormX/mean(abs(X_2sps)));

Y_2sps = resample(Ry,100e9,256e9);
Y_2sps = Y_2sps.*(NormY/mean(abs(Y_2sps)));

% RRC Match Filter

X_2sps_rrc = upfirdn(X_2sps,rctFilt,1,1);
X_2sps_rrc = X_2sps_rrc(fltDelay:end-fltDelay-1);

Y_2sps_rrc = upfirdn(Y_2sps,rctFilt,1,1);
Y_2sps_rrc = Y_2sps_rrc(fltDelay:end-fltDelay-1);

scatterplot(X_2sps_rrc);
scatterplot(Y_2sps_rrc);
%% Timing Sync
[X_payload,X_header,X_pos] = Rx_Syncronisation(X_2sps_rrc,Sync_PN_X);
[Y_payload,Y_header,Y_pos] = Rx_Syncronisation(Y_2sps_rrc,Sync_PN_Y);

% save('16QAM_2Pol_1657_rx.mat', 'X_payload', 'Y_payload');
save('16QAM_1Pol_1615_rx.mat', 'X_payload');