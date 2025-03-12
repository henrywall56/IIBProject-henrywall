function [Out] = InsertSkew(In,SpS,Rs,NPol,ParamSkew)
    iIV = In(:,1) ; 
    iQV = In(:,2); 
    if NPol == 2 
        iIH = In(:,3) ; 
        iQH = In(:,4); 
    end
    % Calculatingthe interpolation factor given the timing skew 
    % TauIV/H and TauQV/H, for in-phase and quadrature components of V pol. and H pol. orientations, respectively: 
    Ts = 1/(SpS*Rs) ; 
    Skew = [ParamSkew.TauIV/Ts ParamSkew.TauQV/Ts]; 
    if NPol== 2 
        Skew = [Skew ParamSkew.TauIH/Ts ParamSkew.TauQH/Ts]; 
    end
    % Using the min skew as reference: 
    Skew = Skew-min(Skew); 
    % Inserting skew in the samples: 
    Len = length(iIV); 
    iIV = interp1(0:Len-1,iIV,Skew(1):Len-1,'spline','extrap').'; 
    iQV = interp1(0:Len-1,iQV,Skew(2):Len-1,'spline','extrap').'; 
    if NPol== 2 
        iIH= interp1(0:Len-1,iIH,Skew(3):Len-1,'spline','extrap').'; 
        iQH= interp1(0:Len-1,iQH,Skew(4):Len-1,'spline','extrap').';
        % Output signals with the samelength: 
        MinLength= min([length(iIV) length(iQV) length(iIH) length(iQH)]); 
        Out(:,1) = iIV(1:MinLength) ; 
        Out(:,2) = iQV(1:MinLength); 
        Out(:,3) = iIH(1:MinLength) ; 
        Out(:,4) = iQH(1:MinLength); 
    else
        % Output signals with the same length: 
        MinLength= min([length(iIV) length(iQV)]); 
        Out(:,1) = iIV(1:MinLength) ;
        Out(:,2) = iQV(1:MinLength); 
    end
end
