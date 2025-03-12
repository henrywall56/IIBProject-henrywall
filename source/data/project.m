% clear;close all;clc;
% warning off;
% addpath SOA\Matlab\
% addpath SOA\Matlab\lib
% addpath SOA\Matlab\Traces\
%%
Mqam = 4;
Nsym = 131072;
Nbit = Nsym*log2(Mqam);
Sps = 2;
pre_len = 2;
sync_len = 512;
Nhead = sync_len/2;% *2 synchead length 
Nhead_rep=2;
TS_len = 1024;
frame_len = (Nhead*Nhead_rep + TS_len*1 + Nsym)*Sps;

%% Symbols
pattern_x = source(1,:)';
pattern_y = source(2,:)';
%% Set AWG Parameters 
AWG = 100e9;
QAM_BaudRate = 50e9;

%% Set RRC Filter Parameters
beta = 0.1;
span = 100;
rctFilt = rcosdesign(beta,span,Sps,'sqrt');
%% Set Frame Headers [512 Sync QPSK Symbols + 1024 FO Training Symbols + Payload]
% ------------------------ X POL -----------------------%
% ------------------------FO header -----------------------%
NormAmp = mean(abs(pattern_x));
TS = pattern_x(1)*ones(1*TS_len,1);
TS = TS*NormAmp/mean(abs(TS));

% ------------------------Sync header --------------------%
% [Nbits, PRBS, shift] -- > 
[Sync_header,Sync_PN_X] = SyncHeaderGenerator((sync_len/2)*log2(4),9,33);

% -----------------------Generate Frame ------------------%
FrameX = [0.9*NormAmp*Sync_header;0.80*TS;pattern_x];

fff = linspace(-1,1,length(FrameX));
figure;plot(fff,10*log10(fftshift(abs(fft(FrameX)))))

%% ------------------------ Y POL -----------------------%
% ------------------------FO header -----------------------%
NormAmp = mean(abs(pattern_y));
TS = pattern_y(1)*ones(1*TS_len,1);
TS = TS*NormAmp/mean(abs(TS));

% ------------------------Sync header --------------------%
% [Nbits, PRBS, shift] -- > 
[Sync_header,Sync_PN_Y] = SyncHeaderGenerator((sync_len/2)*log2(4),9,33);

% -----------------------Generate Frame ------------------%
FrameY = [0.9*NormAmp*Sync_header;0.80*TS;pattern_y];

fff = linspace(-1,1,length(FrameY));
figure;plot(fff,10*log10(fftshift(abs(fft(FrameY)))))

%% Apply Pre-emphasis filtering and RRC Pulse Shaping
load("CH4_pred.mat");
f1 = zeros(125,1);
filt_r = pred(1:125);
maxi = max([max(filt_r),abs(min(filt_r))]);
filt_r = filt_r./maxi;
f_vector = linspace(0,1,125); 
f1(1:125) = filt_r(1:125);

d = fdesign.arbmagnphase('n,f,h',128,f_vector,f1); % N=100, f and h set to defaults.
D = design(d,'freqsamp','SystemObject',true);
% freqz(D);

Pre_emphasis_FrameX = D(FrameX);
Pre_emphasis_FrameY = D(FrameY);
% Pre_emphasis_FrameX = FrameX;
% Pre_emphasis_FrameY = FrameY;

SignalX = upfirdn(Pre_emphasis_FrameX,rctFilt,Sps);
SignalY = upfirdn(Pre_emphasis_FrameY,rctFilt,Sps);

SignalX_nopre = upfirdn(FrameX,rctFilt,Sps);
SignalY_nopre = upfirdn(FrameY,rctFilt,Sps);

% Correct for propagation delay by removing filter transients
fltDelay = span;
SignalX = SignalX(fltDelay+1:end-fltDelay+1);
SignalY = SignalY(fltDelay+1:end-fltDelay+1);
SignalX_nopre = SignalX_nopre(fltDelay+1:end-fltDelay+1);
SignalY_nopre = SignalY_nopre(fltDelay+1:end-fltDelay+1);

%% Tx Skew 
% X.I =0; X.Q = -4.3; Y_I = 9.8; Y_Q = 15;
Channel_1 = real(SignalX)/max(abs(real(SignalX)));
Channel_2 = imag(SignalX)/max(abs(imag(SignalX)));
Channel_3 = real(SignalY)/max(abs(real(SignalY)));
Channel_4 = imag(SignalY)/max(abs(imag(SignalY)));

% X Pol.
skew_ix = 0; skew_qx = -4.3;
skew_iy = 9.8; skew_qy = 15;

ParamSkewX = struct('TauIV',skew_ix*1e-12,'TauQV',skew_qx*1e-12,'TauIH',skew_iy*1e-12,'TauQH',skew_qy*1e-12);
Signal_deskew = InsertSkew([Channel_1,Channel_2,Channel_3,Channel_4],Sps,50e9,2,ParamSkewX);

% Y Pol.
% ParamSkewY = struct('TauIV',skew_i*1e-12,'TauQV',skew_q*1e-12);
% SignalY_deskew = InsertSkew([Channel_3,Channel_4],Sps,50e9,1,ParamSkewY);

Channel_1 = Signal_deskew(:,1);
Channel_2 = Signal_deskew(:,2);
Channel_3 = Signal_deskew(:,3);
Channel_4 = Signal_deskew(:,4);

scatterplot(Signal_deskew(1:2:end,1:2))
scatterplot(Signal_deskew(1:2:end,3:4))
%% AWG Length Check : Check the signal length mod 128 == 0; otherwise append 0s
Channel_1(end+1:floor(size(Channel_1,1)/128+1)*128) = 0;
Channel_2(end+1:floor(size(Channel_2,1)/128+1)*128) = 0;
Channel_3(end+1:floor(size(Channel_3,1)/128+1)*128) = 0;
Channel_4(end+1:floor(size(Channel_4,1)/128+1)*128) = 0;

function Sync_bit_header = BinaryHeaders(Nbit,prbs,shift)
    switch prbs
        case 7
            g = [7 6 0];
        case 9
            g = [9 5 0];
        case 11
            g = [11 9 0];
        case 15
            g = [15 14 0];
        case 20
            g = [20 3 0];
        case 23
            g = [23 18 0];
        case 31
            g = [31 28 0];
        otherwise
            error('Polynom not defined for given pattern length')
    end

    % h = comm.PNSequence('Polynomial',g,'SamplesPerFrame', Nbit+shift,'InitialConditions', 1);
    % data=step(h);
    % data=data(shift+1:end);    
    % stream = data;
    % h = seqgen.pn('GenPoly',  g);
    h = commsrc.pn('GenPoly',  g);
    set(h, 'NumBitsOut',Nbit+shift);
    % h = comm.PNSequence('Polynomial',g,'InitialStates', 1);
    % set(h, 'NumBitsOut',Nbit+shift);
    data=generate(h);
    data=data(shift+1:end);    
    Sync_bit_header = data;
end

function [SyncHead,Sync_PN_saved] = SyncHeaderGenerator(Nbit,prbs,shift)
    % -------------------- Generate Binary Headers ---------------------
    Sync_bit_header = BinaryHeaders(Nbit,prbs,shift);
    
    % -------------------- Convert to Symbol Headers --------------------
    bin = reshape(Sync_bit_header,log2(4),[]).';
    Dec = bi2de(bin,'left-msb');
    Sync_qpsk_header = qammod(Dec,4,'gray');
    
    Sync_qpsk_header = repmat(Sync_qpsk_header,2,1); % Two sync headers

    Sync_PN_sequence = BinaryHeaders(length(Sync_qpsk_header),9,1000)*2-1;

    Sync_PN_saved = Sync_PN_sequence(1:end/2).*Sync_PN_sequence(1+end/2:end);

    SyncHead = Sync_qpsk_header.*Sync_PN_sequence;
end

%% 
% 
% X = X_I + 1i*X_Q;
% scatterplot(X);
% Y = Y_I + 1i*Y_Q;
% scatterplot(Y);