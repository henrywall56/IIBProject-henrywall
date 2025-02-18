import numpy as np
from intervaltree import IntervalTree, Interval
import math
import mpmath
#My implementation of distribution matcher for 16-QAM only - No longer in use
    
def nCr(n, r):
    if r > n:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

def DM_encode_prototype(C, v, k):
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
    p=0 #precision parameter
    wi = 2**p #input interval width
    bi = 0 #input interval base
    h=0 #Number of symbols sent to output
    bc = 0 #code interval base
    wc = 2**p #code interval width

    for i in range(k):
        if(v[i] == 0):
            bi = bi
            wi = wi/2
        else:
            bi = bi + wi/2
            wi = wi/2

        m = (nA/(N-h))*(wc)+bc

        #Code interval is [bc,m) and [m,bc+wc)
       
        if(bi+wi < m and bi > bc): #input interval identifies output A interval
            x[h] = 1
            wc = m - bc #upper is m
            bc = bc #lower
            bi = (bi-bc)/((wc)/2**p)
            bc = 0
            h=h+1 #one more output symbol sent to buffer (one less left in bag)
            wi = wi / ((wc)/2**p)
            wc = 2**p
            nA = nA - 1 #one less symbol A left in bag


        elif(bi > m and bi+wi<bc+wc): #input interval identifies output B interval
            x[h] = 3
            wc = bc+wc-m #upper is as before
            bc = m #lower
            bi = (bi-bc)/((wc)/2**p)
            bc = 0
            h=h+1 #one more output symbol sent to buffer (one less left in bag)
            wi = wi / ((wc)/2**p)
            wc = 2**p
            nB = nB - 1 #one less symbol B left in bag

        else:  #Check for lower level candidates and rescale
            u1 = bc+wc
            l1 = bc

            u2 = max(m + (nA/(N-h))*(u1-m),1)
            l2 = min(m - (nB/(N-h))*(m-l1),0)

            if ((l1<=bi) and (bi+wi<=u2)): 
                bi = (bi-l1)/((u2-l1)/2**p)
                wi = wi/((u2-l1)/2**p)
                bc = (bc-l1)/((u2-l1)/2**p)
                wc = wc/((u2-l1)/2**p)

            elif((bi>=l2) and (bi+wi<=u1)): 
                bi = (bi-l2)/((u1-l2)/2**p)
                wi = wi/((u1-l2)/2**p)
                bc = (bc-l2)/((u1-l2)/2**p)
                wc = wc/((u1-l2)/2**p)
            
            elif((bi>=l2) and (bi+wi<=u2)):
                bi = (bi-l2)/((u2-l2)/2**p)
                wi = wi/((u2-l2)/2**p)
                bc = (bc-l2)/((u2-l2)/2**p)
                wc = wc/((u2-l2)/2**p)

    #Finalisation step so that the codeword identifies the source interval [bi, bi+wi)
    m = (nA/(N-h))*2**p #seperating point of code interval: [0,m) is symbol A, [m,1) is symbol B
    #lower of code interval is 0
    #upper of code interval is 1
    #bi is base of source interval
    #wi is width of source interval
    #nA of symbol A left
    #nB of symbol B left
    #N-h symbols left to add to output
    
    code_intervals = IntervalTree()
    code_intervals.addi(0, m, [int(1)])
    code_intervals.addi(m, 2**p, [int(3)])

    while(1):
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
                    htemp = htemp + 1

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
            iv_begin = (iv.begin-l)/((u-l)/2**p)
            iv_end = (iv.end-l)/((u-l)/2**p)
            # iv_begin = iv.begin
            # iv_end = iv.end
            scaled_intervals.append((iv_begin,iv_end,iv.data))

        code_intervals.clear()

        for iv_begin, iv_end, iv_data in scaled_intervals:
            code_intervals.addi(iv_begin, iv_end,iv_data)
        
        #code_intervals contains only the overlapping intervals scaled from 0 to 1
        
        bi = (bi-l)/((u-l)/2**p)
        wi = wi/((u-l)/2**p)

        e=0

        lower_border_inside = [iv for iv in code_intervals if iv.begin >= bi-e and iv.begin <= bi+wi+e]

        lower_border_sorted = sorted(lower_border_inside, key=lambda iv: iv.begin)

        for iv in lower_border_sorted:

            if((iv.data.count(1)==nA)):
                finalised_seq = iv.data
                finalised_seq.extend([3]*(nB-iv.data.count(3)))
                next_symbol_index = np.where(x==0)[0][0]
                x[next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                  
                return x
            elif((iv.data.count(3)==nB)):
                finalised_seq = iv.data
                finalised_seq.extend([1]*(nA-iv.data.count(1)))
                next_symbol_index = np.where(x==0)[0][0]
                x[next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                  
                return x


def DM_decode_prototype(codeword, C, k):
    #Distributin Matcher Decoding Function
    #C: the number of each symbol type in the codeword
    #k: the number of information bits in each block

    bits = []
    nA = C[0]
    nB = C[1]
    N = nA + nB

    p=0 #precision parameter

    S_intervals = IntervalTree() #Source intervals

    m = 2**p*nA/N
    if(codeword[0]==1):
        C_interval = Interval(0,m)
        C_countA = 1 #number of code symbols A scanned
        C_countB = 0 #number of code symbols B scanned
        K_interval = Interval(0,m)
        K_countA = 1 #number of code symbols A scanned in K interval
        K_countB = 0 #number of code symbols B scanned in K interval
    else:
        C_interval = Interval(m,2**p) #add interval [m,1) to code intervals
        C_countA = 0 #number of code symbols A scanned
        C_countB = 1 #number of code symbols B scanned
        K_interval = Interval(m,2**p)
        K_countA = 0 #number of code symbols A scanned in K interval
        K_countB = 1 #number of code symbols B scanned in K interval

    S_intervals.addi(0,0.5*2**p,0) #bit 0 is interval (0,0.5)
    S_intervals.addi(0.5*2**p,2**p,1) #bit 1 is interval (0.5,1)
    S_count = 0 #number of output bits identified

    bit_identified = False

    while(S_count<k):
        # print(C_interval.end-C_interval.begin, 'code width decode')
        # print(K_interval.end-K_interval.begin, 'K width decode')
        # print(next(iter(S_intervals)).end-next(iter(S_intervals)).begin, 'source 0 width decode')

        if((C_countA+C_countB == N) or (C_countA==nA) or (C_countB==nB)): #If read whole codeword
            bit_interval = next(iter(S_intervals.at(C_interval.begin))) #Source interval that the lower border of the code interval is in. next(iter(.)) extracts interval from the set produced by .at()
            bits.append(bit_interval.data)
            bit_identified = True
            S_count = S_count + 1
            
        
        else:
            S_containing_C = [si for si in S_intervals if si.begin <= C_interval.begin and C_interval.end <= si.end] #Source intervals that are identified by a code interval
            if(S_containing_C): #if any code intervals are within source intervals
                bit_interval = S_containing_C[0]
                bits.append(bit_interval.data)
                bit_identified = True
                S_count = S_count + 1

        if(S_count==k):
            return bits
        
        if(bit_identified): #If refined source interval
            bit_identified=False
            if(K_interval.begin <= bit_interval.begin and bit_interval.end <= K_interval.end): #If source interval within K interval
                u = K_interval.end
                l = K_interval.begin
                S_intervals.clear()

                s_begin = (bit_interval.begin-l)/((u-l)/2**p) #scale source interval
                s_end = (bit_interval.end-l)/((u-l)/2**p)

                S_intervals.addi(s_begin, s_begin+0.5*(s_end-s_begin),0)
                S_intervals.addi(s_begin+0.5*(s_end-s_begin),s_end,1)

                next_symbol = codeword[K_countA+K_countB] #Read next symbol
                if(next_symbol==1):
                    mK = (nA-K_countA)/(N-K_countA-K_countB)
                    K_countA = K_countA+1
                    K_interval = Interval(0,mK*2**p)
                else:
                    mK = (nA-K_countA)/(N-K_countA-K_countB)
                    K_countB = K_countB+1
                    K_interval = Interval(mK*2**p,2**p)
                
                C_countA = K_countA #Reset code interval to state of K interval
                C_countB = K_countB
                C_interval = Interval(K_interval.begin, K_interval.end)

                #K_interval is rescaled to [0,1), and then refined
            else:
                S_intervals.clear()
                s_begin = bit_interval.begin
                s_end = bit_interval.end
                S_intervals.addi(s_begin, s_begin+0.5*(s_end-s_begin),0)
                S_intervals.addi(s_begin+0.5*(s_end-s_begin),s_end,1)

        
        else:
            next_symbol = codeword[C_countA+C_countB] #Read next symbol
            if(next_symbol==1):
                mC = (nA-C_countA)/(N-C_countA-C_countB)
                C_countA = C_countA+1
                C_interval = Interval(C_interval.begin, C_interval.begin+ mC*(C_interval.end-C_interval.begin))
                
            else:
                mC = (nA-C_countA)/(N-C_countA-C_countB)
                C_countB = C_countB+1
                C_interval = Interval(C_interval.begin+ mC*(C_interval.end-C_interval.begin), C_interval.end)
    
    return bits
                
C = [45,25]
N = np.sum(C)
k=int(np.floor(math.log2(nCr(N,C[1]))))
bits = np.random.randint(0, 2, size= k)

x = DM_encode_prototype(C,bits,k)

count1 = np.count_nonzero(x==1)
count3 = np.count_nonzero(x==3)

print('k:', k)
print('N:', np.sum(C))
print('Rate:', k/N)
print('C input:', C)
print('N output:', count1+count3)
print('C output:', [count1,count3])

bits_decoded = DM_decode_prototype(x, C, k)
print(list(bits), 'Source Bits')
print(list(x),'CCDM Symbols')
print(bits_decoded, 'Decoded Bits')
                
if(np.array_equal(bits_decoded,bits)):
    print('No Errors')
else:
    print('Errors')

print(np.where(bits!=bits_decoded))

        

