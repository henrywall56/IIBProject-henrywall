function [RxInew, RxQnew] = func_hybridIQcomp(RxI, RxQ)

    Pt = mean(RxI.^2+RxQ.^2);
    Pi = mean(RxI.^2);
    Pq = mean(RxQ.^2);

    rho = mean(RxI.*RxQ);
    
    RxInew = RxI/sqrt(Pi);
    
    RxQ = RxQ - rho.*RxI/Pi;
    RxQnew = RxQ/sqrt(Pq);
    
    RxInew = sqrt(Pt/2)*RxInew;
    RxQnew = sqrt(Pt/2)*RxQnew;

end