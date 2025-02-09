import numpy as np
from intervaltree import IntervalTree, Interval
import math
import mpmath
#My implementation of distribution matcher

mpmath.mp.dps = 70
    
def nCr(n, r):
    if r > n:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

def DM_encode_64QAM(C, v, k):
    # Distribution Matcher using arithmetic coding

    # C = {nA, nB, nC, nD} #composition (for 64-QAM, number of 1's, 3's, 5's and 7's in each block)
    # k = number of bits in input block

    # v #input bits
    # x #output symbols
    N = np.sum(C)
    x = np.zeros(N,dtype=int)
    #intialise
    nA = C[0]
    nB = C[1]
    nC = C[2]
    nD = C[3]

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

        m1 = (nA/(N-h))*(wc)+bc
        m2 = m1 + (nB/(N-h))*(wc)
        m3 = m2 + (nC/(N-h))*(wc)

        #Code interval is [bc,m) and [m,bc+wc)
       
        if(bi+wi < m1 and bi > bc): #input interval identifies output A interval
            x[h] = 1
            wc = m1 - bc #upper is m1
            bc = bc #lower
            bi = (bi-bc)/(wc)
            bc = 0
            h=h+1 #one more output symbol sent to buffer (one less left in bag)
            wi = wi / (wc)
            wc = 1
            nA = nA - 1 #one less symbol A left in bag


        elif(bi > m1 and bi+wi<m2): #input interval identifies output B interval
            x[h] = 3
            wc = m2-m1 #upper is m2
            bc = m1 #lower
            bi = (bi-bc)/(wc)
            bc = 0
            h=h+1 #one more output symbol sent to buffer (one less left in bag)
            wi = wi / (wc)
            wc = 1
            nB = nB - 1 #one less symbol B left in bag

        elif(bi > m2 and bi+wi<m3): #input interval identifies output B interval
            x[h] = 5
            wc = m3-m2 #upper is m3
            bc = m2 #lower
            bi = (bi-bc)/(wc)
            bc = 0
            h=h+1 #one more output symbol sent to buffer (one less left in bag)
            wi = wi / (wc)
            wc = 1
            nC = nC- 1 #one less symbol B left in bag

        elif(bi > m3 and bi+wi<bc+wc): #input interval identifies output B interval
            x[h] = 7
            wc = bc+wc-m3 #upper is as before
            bc = m3 #lower
            bi = (bi-bc)/(wc)
            bc = 0
            h=h+1 #one more output symbol sent to buffer (one less left in bag)
            wi = wi / (wc)
            wc = 1
            nD= nD - 1 #one less symbol B left in bag

    #Finalisation step so that the codeword identifies the source interval [bi, bi+wi)
    m1 = (nA/(N-h))*(wc)+bc
    m2 = m1 + (nB/(N-h))*(wc)
    m3 = m2 + (nC/(N-h))*(wc)
    #lower of code interval is 0
    #upper of code interval is 1
    #bi is base of source interval
    #wi is width of source interval
    #nA of symbol A left
    #nB of symbol B left
    #N-h symbols left to add to output
    
    code_intervals = IntervalTree()
    if(nA!=0):
        code_intervals.addi(bc, m1, [int(1)])
    if(nB!=0):
        code_intervals.addi(m1, m2, [int(3)])
    if(nC!=0):
        code_intervals.addi(m2, m3, [int(5)])
    if(nD!=0):
        code_intervals.addi(m3, bc+wc, [int(7)])
    i=0
    while(1):
        new_intervals = []
        old_intervals = []
 
        for interval in code_intervals: #refine intervals
            nAtemp = nA
            nBtemp = nB
            nCtemp = nC
            nDtemp = nD
            htemp = h
            for d in interval.data: 
                if(d==1):
                    nAtemp = nAtemp - 1
                    htemp = htemp + 1
                elif(d==3):
                    nBtemp = nBtemp - 1
                    htemp = htemp +1
                elif(d==5):
                    nCtemp = nCtemp - 1
                    htemp = htemp +1
                elif(d==7):
                    nDtemp = nDtemp - 1
                    htemp = htemp +1

            if(nAtemp>0):
                data = interval.data.copy()
                data.append(int(1))
                m1 = interval.begin+(nAtemp/(N-htemp))*(interval.end-interval.begin)
                new_intervals.append((interval.begin, m1, data))
            if(nBtemp>0):
                data = interval.data.copy()
                data.append(int(3))
                m1 = interval.begin+(nAtemp/(N-htemp))*(interval.end-interval.begin)
                m2 = m1 + (nBtemp/(N-htemp))*(interval.end-interval.begin)
                new_intervals.append((m1, m2, data))
            if(nCtemp>0):
                data = interval.data.copy()
                data.append(int(5))
                m1 = interval.begin+(nAtemp/(N-htemp))*(interval.end-interval.begin)
                m2 = m1 + (nBtemp/(N-htemp))*(interval.end-interval.begin)
                m3 = m2 + (nCtemp/(N-htemp))*(interval.end-interval.begin)
                new_intervals.append((m2, m3, data))
            if(nDtemp>0):
                data = interval.data.copy()
                data.append(int(7))
                m1 = interval.begin+(nAtemp/(N-htemp))*(interval.end-interval.begin)
                m2 = m1 + (nBtemp/(N-htemp))*(interval.end-interval.begin)
                m3 = m2 + (nCtemp/(N-htemp))*(interval.end-interval.begin)
                new_intervals.append((m3, interval.end, data))

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

        lower_border_inside = [iv for iv in code_intervals if iv.begin >= bi-e and iv.begin <= bi+wi+e]

        lower_border_sorted = sorted(lower_border_inside, key=lambda iv: iv.begin)

        for iv in lower_border_sorted:

            if(iv.data.count(1)==nA and iv.data.count(3)==nB and iv.data.count(5)==nC):
                finalised_seq = iv.data
                finalised_seq.extend([7]*(nD-iv.data.count(7)))
                next_symbol_index = np.where(x==0)[0][0]
                x[next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                return x

            elif(iv.data.count(1)==nA and iv.data.count(3)==nB and iv.data.count(7)==nD):
                finalised_seq = iv.data
                finalised_seq.extend([5]*(nC-iv.data.count(5)))
                next_symbol_index = np.where(x==0)[0][0]
                x[next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                return x

            elif(iv.data.count(1)==nA and iv.data.count(5)==nC and iv.data.count(7)==nD):
                finalised_seq = iv.data
                finalised_seq.extend([3]*(nB-iv.data.count(3)))
                next_symbol_index = np.where(x==0)[0][0]
                x[next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                return x

            elif(iv.data.count(3)==nB and iv.data.count(5)==nC and iv.data.count(7)==nD):
                finalised_seq = iv.data
                finalised_seq.extend([1]*(nA-iv.data.count(1)))
                next_symbol_index = np.where(x==0)[0][0]
                x[next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                return x

    return x

def DM_decode_64QAM(codeword, C, k):
    #Distributin Matcher Decoding Function
    #C: the number of each symbol type in the codeword
    #k: the number of information bits in each block

    bits = []
    nA = C[0]
    nB = C[1]
    nC = C[2]
    nD = C[3]

    N = nA + nB +nC + nD

    S_intervals = IntervalTree() #Source intervals

    m1 = (nA/(N))
    m2 = m1 + (nB/(N))
    m3 = m2 + (nC/(N))

    if(codeword[0]==1):
        C_interval = Interval(0,m1)
        C_countA = 1 #number of code symbols A scanned
        C_countB = 0 #number of code symbols B scanned
        C_countC = 0
        C_countD = 0
        K_interval = Interval(0,m1)
        K_countA = 1 #number of code symbols A scanned in K interval
        K_countB = 0 #number of code symbols B scanned in K interval
        K_countC = 0
        K_countD = 0
    elif(codeword[0]==3):
        C_interval = Interval(m1,m2) #add interval [m,1) to code intervals
        C_countA = 0 #number of code symbols A scanned
        C_countB = 1 #number of code symbols B scanned
        C_countC = 0
        C_countD = 0
        K_interval = Interval(m1,m2)
        K_countA = 0 #number of code symbols A scanned in K interval
        K_countB = 1 #number of code symbols B scanned in K interval
        K_countC = 0
        K_countD = 0
    elif(codeword[0]==5):
        C_interval = Interval(m2,m3) #add interval [m,1) to code intervals
        C_countA = 0 #number of code symbols A scanned
        C_countB = 0 #number of code symbols B scanned
        C_countC = 1
        C_countD = 0
        K_interval = Interval(m2,m3)
        K_countA = 0 #number of code symbols A scanned in K interval
        K_countB = 0 #number of code symbols B scanned in K interval
        K_countC = 1
        K_countD = 0
    elif(codeword[0]==7):
        C_interval = Interval(m3,1) #add interval [m,1) to code intervals
        C_countA = 0 #number of code symbols A scanned
        C_countB = 0 #number of code symbols B scanned
        C_countC = 0
        C_countD = 1
        K_interval = Interval(m3,1)
        K_countA = 0 #number of code symbols A scanned in K interval
        K_countB = 0 #number of code symbols B scanned in K interval
        K_countC = 0
        K_countD = 1

    S_intervals.addi(0,0.5,0) #bit 0 is interval (0,0.5)
    S_intervals.addi(0.5,1,1) #bit 1 is interval (0.5,1)
    S_count = 0 #number of output bits identified

    bit_identified = False

    while(S_count<k):
        # print(C_interval.end-C_interval.begin, 'code width decode')
        # print(K_interval.end-K_interval.begin, 'K width decode')
        # print(next(iter(S_intervals)).end-next(iter(S_intervals)).begin, 'source 0 width decode')

        if((C_countA+C_countB+C_countC+C_countD == N) or (C_countA==nA and C_countB==nB and C_countC==nC) or (C_countA==nA and C_countB==nB and C_countD==nD) or (C_countA==nA and C_countC==nC and C_countD==nD) or (C_countB==nB and C_countC==nC and C_countD==nD)): #If read whole codeword
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

                s_begin = (bit_interval.begin-l)/(u-l) #scale source interval
                s_end = (bit_interval.end-l)/(u-l)

                S_intervals.addi(s_begin, s_begin+0.5*(s_end-s_begin),0)
                S_intervals.addi(s_begin+0.5*(s_end-s_begin),s_end,1)

                next_symbol = codeword[K_countA+K_countB+K_countC+K_countD] #Read next symbol
                if(next_symbol==1):
                    mK1 = (nA-K_countA)/(N-K_countA-K_countB-K_countC-K_countD)
                    K_countA = K_countA+1
                    K_interval = Interval(0,mK1)
                elif(next_symbol==3):
                    mK1 = (nA-K_countA)/(N-K_countA-K_countB-K_countC-K_countD)
                    mK2 = mK1 + (nB-K_countB)/(N-K_countA-K_countB-K_countC-K_countD)
                    K_countB = K_countB+1
                    K_interval = Interval(mK1,mK2)
                elif(next_symbol==5):
                    mK1 = (nA-K_countA)/(N-K_countA-K_countB-K_countC-K_countD)
                    mK2 = mK1 + (nB-K_countB)/(N-K_countA-K_countB-K_countC-K_countD)
                    mK3 = mK2 + (nC-K_countC)/(N-K_countA-K_countB-K_countC-K_countD)
                    K_countC = K_countC+1
                    K_interval = Interval(mK2,mK3)
                elif(next_symbol==7):
                    mK1 = (nA-K_countA)/(N-K_countA-K_countB-K_countC-K_countD)
                    mK2 = mK1 + (nB-K_countB)/(N-K_countA-K_countB-K_countC-K_countD)
                    mK3 = mK2 + (nC-K_countC)/(N-K_countA-K_countB-K_countC-K_countD)
                    K_countD = K_countD+1
                    K_interval = Interval(mK3,1)
                
                C_countA = K_countA #Reset code interval to state of K interval
                C_countB = K_countB
                C_countC = K_countC
                C_countD = K_countD
                C_interval = Interval(K_interval.begin, K_interval.end)

                #K_interval is rescaled to [0,1), and then refined
            else:
                S_intervals.clear()
                s_begin = bit_interval.begin
                s_end = bit_interval.end
                S_intervals.addi(s_begin, s_begin+0.5*(s_end-s_begin),0)
                S_intervals.addi(s_begin+0.5*(s_end-s_begin),s_end,1)

        
        else:
            next_symbol = codeword[C_countA+C_countB+C_countC+C_countD] #Read next symbol
            if(next_symbol==1):
                mC1 = C_interval.begin + ((nA-C_countA)/(N-C_countA-C_countB-C_countC-C_countD))*(C_interval.end-C_interval.begin)
                C_countA = C_countA+1
                C_interval = Interval(C_interval.begin, mC1)
                
            elif(next_symbol==3):
                mC1 = C_interval.begin + ((nA-C_countA)/(N-C_countA-C_countB-C_countC-C_countD))*(C_interval.end-C_interval.begin)
                mC2 = mC1 + ((nB-C_countB)/(N-C_countA-C_countB-C_countC-C_countD))*(C_interval.end-C_interval.begin)
                C_countB = C_countB+1
                C_interval = Interval(mC1,mC2)

            elif(next_symbol==5):
                mC1 = C_interval.begin + ((nA-C_countA)/(N-C_countA-C_countB-C_countC-C_countD))*(C_interval.end-C_interval.begin)
                mC2 = mC1 + ((nB-C_countB)/(N-C_countA-C_countB-C_countC-C_countD))*(C_interval.end-C_interval.begin)
                mC3 = mC2 + ((nC-C_countC)/(N-C_countA-C_countB-C_countC-C_countD))*(C_interval.end-C_interval.begin)
                C_countC = C_countC+1
                C_interval = Interval(mC2,mC3)

            elif(next_symbol==7):
                mC1 = C_interval.begin + ((nA-C_countA)/(N-C_countA-C_countB-C_countC-C_countD))*(C_interval.end-C_interval.begin)
                mC2 = mC1 + ((nB-C_countB)/(N-C_countA-C_countB-C_countC-C_countD))*(C_interval.end-C_interval.begin)
                mC3 = mC2 + ((nC-C_countC)/(N-C_countA-C_countB-C_countC-C_countD))*(C_interval.end-C_interval.begin)
                C_countD = C_countD+1
                C_interval = Interval(mC3,C_interval.end)
    
    return bits



C = [13,10,5,4]
N = np.sum(C)
k=50
bits = np.random.randint(0, 2, size= k)


x = DM_encode_64QAM(C,bits,k)
count1 = np.count_nonzero(x==1)
count3 = np.count_nonzero(x==3)
count5 = np.count_nonzero(x==5)
count7 = np.count_nonzero(x==7)

print('k:', k)
print('N:', np.sum(C))
print('Rate:', k/N, 'bits/symbol')
print('C input:', C)
print('N output:', count1+count3+count5+count7)
print('C output:', [count1,count3,count5,count7])

bits_decoded = DM_decode_64QAM(x, C, k)
print(list(bits), 'Source Bits')
print(list(x),'CCDM Symbols')
print(bits_decoded, 'Decoded Bits')
                
if(np.array_equal(bits_decoded,bits)):
    print('No Errors')
else:
    print('Errors')

print(np.where(bits!=bits_decoded))