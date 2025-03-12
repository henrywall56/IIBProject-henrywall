function [XI,XQ_deskew,YI,YQ_deskew] = RxDeskew(X,Y,param)
    XI = real(X);
    XQ = imag(X);
    YI = real(Y);
    YQ = imag(Y);
    
    [XQ_deskew,~] = procedureRIGHT2(XQ,param.skew_xIQ);
    [YQ_deskew,~] = procedureRIGHT2(YQ,param.skew_yIQ);
end
