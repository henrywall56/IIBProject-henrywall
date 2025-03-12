function [Rx,Ry] = NormaliseSignal(XI,XQ,YI,YQ)

    % X Pol.
    Rx_data_I = XI - mean(XI);
    Rx_data_I = Rx_data_I/(max(abs(Rx_data_I)));

    Rx_data_Q = XQ - mean(XQ);
    Rx_data_Q = Rx_data_Q/(max(abs(Rx_data_Q)));

    Rx = Rx_data_I + 1i*Rx_data_Q;

    % Y Pol.
    Ry_data_I = YI - mean(YI);
    Ry_data_I = Ry_data_I/(max(abs(Ry_data_I)));

    Ry_data_Q = YQ - mean(YQ);
    Ry_data_Q = Ry_data_Q/(max(abs(Ry_data_Q)));

    Ry = Ry_data_I + 1i*Ry_data_Q;
end