import numpy as np
from intervaltree import IntervalTree
#My implementation of distribution matcher
    
def DM_prototype(C, v, k):
    # Distribution Matcher using arithmetic coding

    # C = {n1, n2} #composition (for 16-QAM, number of 1's and 3's in each block)
    # k = 100 #number of bits in input block

    # v #input bits
    # x #output symbols
    N = np.sum(C)
    x = np.zeros(N,dtype=int)
    #intialise
    nA = C[0]
    nB = C[1]
    wi = 1 #input interval width
    bi = 0 #input interval base
    h=0 #Number of symbols sent to output
    bc = 0 #code interval base
    wc = 1 #code interval width

    for i in range(k):
        if(v[i] == 0):
            bi = bi
            wi = wi/2
        else:
            bi = bi + wi/2
            wi = wi/2

        m = nA/(N-h)

        #Code interval is [bc,m) and [m,bc+wc)
       
        if(bi+wi < m): #input interval identifies output A interval
            x[h] = 1
            wc = m - bc #upper is m
            bc = bc #lower
            bi = (bi-bc)/(wc)
            bc = 0
            h=h+1 #one more output symbol sent to buffer (one less left in bag)
            wi = wi / (wc)
            wc = 1
            nA = nA - 1 #one less symbol A left in bag

        elif(bi > m): #input interval identifies output B interval
            x[h] = 3
            wc = bc+wc-m #upper is as before
            bc = m #lower
            bi = (bi-bc)/(wc)
            bc = 0
            h=h+1 #one more output symbol sent to buffer (one less left in bag)
            wi = wi / (wc)
            wc = 1
            nB = nB - 1 #one less symbol B left in bag

        # else:  #Check for lower level candidates and rescale
        #need to adapt to use bc, wc
        #     u = m + (nA/(N-h))*(1-m)
        #     l = m - (nB/(N-h))*(m-0)
         
        #     if ((0<bi) and (bi+wi<u)):  #if (bi,bi+wi) in [0, u)
        #         #scale to [0,u)
        #         bi = bi/u
        #         wi = wi/u

        #     elif((l<bi) and (bi+wi<1)): #if (bi,bi+wi) in [l,1)
        #         #scale to [l,1)
        #         bi = (bi-l)/(1-l)
        #         wi = wi/(1-l)

        #     elif((l<bi) and (bi+wi<u)): #if (bi,bi+wi) in [l,u)
        #         #scale to [l,u)
        #         bi = (bi-l)/(u-l)
        #         wi = wi/(u-l)


    print(x)

    #Finalisation step so that the codeword identifies the source interval [bi, bi+wi)
    m = nA/(N-h) #seperating point of code interval: [0,m) is symbol A, [m,1) is symbol B
    #lower of code interval is 0
    #upper of code interval is 1
    #bi is base of source interval
    #wi is width of source interval
    #nA of symbol A left
    #nB of symbol B left
    #N-h symbols left to add to output
    if(m==0 or m==1):
        return x
    
    code_intervals = IntervalTree()
    code_intervals.addi(0, m, [int(1)])
    code_intervals.addi(m, 1, [int(3)])
    H=h #number of symbols added to x by this point
    #sorted(code_intervals.overlap(bi, bi+wi)) returns sorted list of intervals that overal with source interval [bi, bi+wi)

    while(h<N):
        new_intervals = []
        old_intervals = []
        for interval in code_intervals: #refine intervals
            nAtemp = nA
            nBtemp = nB
            htemp = h
            for d in interval.data: 
                if(d==1):
                    nAtemp = nAtemp - 1
                    htemp = htemp + 1
                elif(d==3):
                    nBtemp = nBtemp - 1
                    htemp = htemp +1

            if(nAtemp>0):
                data = interval.data.copy()
                data.append(int(1))
                new_intervals.append((interval.begin, interval.begin+(nAtemp/(N-htemp))*(interval.end-interval.begin), data))
            if(nBtemp>0):
                data = interval.data.copy()
                data.append(int(3))
                new_intervals.append((interval.begin+(nAtemp/(N-htemp))*(interval.end-interval.begin), interval.end, data))

            old_intervals.append(interval)
        
        for iv_begin, iv_end, iv_data in new_intervals:
            code_intervals.addi(iv_begin, iv_end,iv_data)
    
        for iv in old_intervals:
            code_intervals.discard(iv)
        
        #find intervals that overlap with source interval
        overlapping_intervals = code_intervals.overlap(bi, bi+wi)
            
        epsilon_overlap = []
        epsilon = 0
        for interval in overlapping_intervals:
            # Calculate the overlap size (intersection length)
            overlap_start = max(interval.begin, bi)
            overlap_end = min(interval.end, bi+wi)
            overlap_size = overlap_end - overlap_start
            
            # If the overlap size is less than epsilon, add to discard list
            if overlap_size < epsilon:
                epsilon_overlap.append(interval)
        # Discard intervals after the loop finishes
        for interval in epsilon_overlap:
            overlapping_intervals.discard(interval)

        l = min(iv.begin for iv in overlapping_intervals)
        u = max(iv.end for iv in overlapping_intervals)

        scaled_intervals = []

        for iv in overlapping_intervals: #scale intervals
            iv_begin = (iv.begin-l)/(u-l)
            iv_end = (iv.end-l)/(u-l)
            # iv_begin = iv.begin
            # iv_end = iv.end
            scaled_intervals.append((iv_begin,iv_end,iv.data))

        code_intervals.clear()

        for iv_begin, iv_end, iv_data in scaled_intervals:
            code_intervals.addi(iv_begin, iv_end,iv_data)
        
        #code_intervals contains only the overlapping intervals scaled from 0 to 1
        
        bi = (bi-l)/(u-l)
        wi = wi/(u-l)

        e=0

        within_intervals = [iv for iv in overlapping_intervals if iv.begin >= bi-e and iv.end <= bi+wi+e]

        if(within_intervals):
            largest_interval = max(within_intervals, key=lambda iv: iv.end-iv.begin)
            finalised_seq = largest_interval.data
            print(np.count_nonzero(x),'pre-finalisation')
            print(len(finalised_seq),'finalised')
            print(np.count_nonzero(x)+len(finalised_seq),'total')
            print(N,'N')
            x[-(N-H):-(N-H)+len(finalised_seq)] = finalised_seq

            # nAfin = np.count_nonzero(x==1)
            # nBfin = np.count_nonzero(x==3)
    
            # if(nAfin-C[0]!=0):
            #     #C[0]-nAfin symbol A's left
            #     x[-(C[0]-nAfin):] = np.full(C[0]-nAfin,1)
            # elif(nBfin-C[1]!=0):
            #     #C[1]-nBfin symbol A's left
            #     x[-(C[1]-nBfin):] = np.full(C[1]-nBfin,3)
           
            break
        

    return x


k=7

#k = floor(log2(T^n_{P_A}))

bits = np.random.randint(0, 2, size= k)

C = [5,5]

x = DM_prototype(C,bits,k)
print(x,'x')

count1 = np.count_nonzero(x==1)
count3 = np.count_nonzero(x==3)
print(count1)
print(count3)