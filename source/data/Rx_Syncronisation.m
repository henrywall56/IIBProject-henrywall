function [sync_signal,sync_header,start_position] = Rx_Syncronisation(signal,header)
    N_us = 2;
    Nhead = 256;
    Nhead_repeated = 2;
    frame_len_ue_down = 265216; % 68608

    kkk = Nhead_repeated*Nhead*N_us;

    header = reshape(repmat(header,1,N_us)',N_us*numel(header),1); % Upsample to 2 sps
    
    header_copy = header;

    correlations = zeros(length(signal)-2*Nhead,1);
    
    for m = 1:frame_len_ue_down*1
        if(m==1)
            reg_sch = conj(signal(1:kkk/2)).*(signal(1+kkk/2:kkk)); %conj(signal(1:kkk/2)).*(signal(1+kkk/2:kkk));
        else
            reg_sch_update = conj(signal((m-1)+kkk/2)).*(signal(kkk+m-1)); %conj(signal((m-1)+kkk/2)).*(signal(kkk+m-1));
            reg_sch = [reg_sch(2:end);reg_sch_update];
        end

        P_av_test = [signal(m:m+kkk-1)];
        P_av_test_abs = (sum(abs(P_av_test).^2)/2)^2;

        correlations(m) = 64*abs(sum(reg_sch.*header_copy))^2/P_av_test_abs;
    end

    [~,d] = max(abs(correlations));
    % figure; plot(abs(correlations));

    start_position = d;
    location = start_position+kkk;
    
    sync_header = signal(start_position:start_position+kkk-1);
    sync_signal = signal(location:location+frame_len_ue_down-kkk-1);

end