import numpy as np
from intervaltree import IntervalTree, Interval
import math
import matplotlib.pyplot as plt
#My implementation of distribution matcher
    
def nCr(n, r):
    if r > n:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

def DM_encode(C, v, k,blocks):
    # Distribution Matcher using arithmetic coding

    # C = {nA, nB, nC, nD} #composition (for 64-QAM, number of 1's, 3's, 5's and 7's in each block)
    # k = number of bits in input block

    # v #input bits
    # x #output symbols
    v = v.reshape((blocks,k))
    N = np.sum(C)
    x = np.zeros((blocks,N),dtype=int)
    for row in range(blocks):
        
        #intialise
        nA = C[0]
        nB = C[1]
        nC = C[2]
        nD = C[3]
        
        ui = 1.0 #input interval upper
        bi = 0.0 #input interval base
        h=0 #Number of symbols sent to output
        bc = 0.0 #code interval base
        uc = 1.0 #code interval upper

        for i in range(k):
            if(v[row][i] == 0):
                bi = bi
                ui = bi + (ui-bi)*0.5

            else:
                bi = bi + (ui-bi)/2
                ui = ui

            m1 = (nA/(N-h))*(uc-bc)+bc
            m2 = ((nA+nB)/(N-h))*(uc-bc) + bc
            m3 = ((nA+nB+nC)/(N-h))*(uc-bc) + bc

            #Code interval is [bc,m) and [m,bc+wc)
        
            if(bi > bc and ui < m1): #input interval identifies output A interval
                x[row][h] = 1
                uc = m1 #upper is m1
                bc = bc #lower
                bi = (bi-bc)/(uc-bc)
                ui = (ui-bc)/(uc-bc)
                if(ui>1):
                    ui=1.0
                bc = 0.0
                h=h+1 #one more output symbol sent to buffer (one less left in bag)
                uc = 1.0
                nA = nA - 1 #one less symbol A left in bag


            elif(bi > m1 and ui<m2): #input interval identifies output B interval
                x[row][h] = 3
                uc = m2 #upper is m2
                bc = m1 #lower
                bi = (bi-bc)/(uc-bc)
                ui = (ui-bc)/(uc-bc)
                if(ui>1):
                    ui=1.0
                bc = 0.0
                h=h+1 #one more output symbol sent to buffer (one less left in bag)
                uc = 1.0
                nB = nB - 1 #one less symbol B left in bag

            elif(bi > m2 and ui<m3): #input interval identifies output B interval
                x[row][h] = 5
                uc = m3 #upper is m3
                bc = m2 #lower
                bi = (bi-bc)/(uc-bc)
                ui = (ui-bc)/(uc-bc)
                if(ui>1):
                    ui=1.0
                bc = 0.0
                h=h+1 #one more output symbol sent to buffer (one less left in bag)
                uc = 1.0
                nC = nC- 1 #one less symbol B left in bag

            elif(bi > m3 and ui<uc): #input interval identifies output B interval
                x[row][h] = 7
                #upper is as before
                bc = m3 #lower
                bi = (bi-bc)/(uc-bc)
                ui = (ui-bc)/(uc-bc)
                if(ui>1):
                    ui=1.0
                bc = 0.0
                h=h+1 #one more output symbol sent to buffer (one less left in bag)
                uc = 1.0
                nD= nD - 1 #one less symbol B left in bag

            # else:  #Check for lower level candidates and rescale ONLY FOR 16-QAM SO FAR
                
            #     u1 = uc
            #     l1 = bc
            #     u2 = m1 + (nA/(N-h))*(u1-m1)
            #     l2 = m1 - (nB/(N-h))*(m1-l1)

            #     if((bi>=l2) and (ui<=u2)):
            #         print('else1')
            #         bi = (bi-l2)/(u2-l2)
            #         ui = (ui-l2)/(u2-l2)
            #         bc = (bc-l2)/(u2-l2)
            #         uc = (uc-l2)/(u2-l2)

            #     elif ((bi>=l1) and (ui<=u2)): 
            #         print('else2')
            #         bi = (bi-l1)/(u2-l1)
            #         ui = (ui-l1)/(u2-l1)
            #         bc = (bc-l1)/(u2-l1)
            #         uc = (uc-l1)/(u2-l1)

            #     elif((bi>=l2) and (ui<=u1)): 
            #         print('else3')
            #         bi = (bi-l2)/(u1-l2)
            #         ui = (ui-l2)/(u1-l2)
            #         bc = (bc-l2)/(u1-l2)
            #         uc = (uc-l2)/(u1-l2)

            #     if(ui>1):
            #         ui=1.0
            #     if(uc>1):
            #         print(uc)
            #         uc=1.0
                


        #Finalisation step so that the codeword identifies the source interval [bi, bi+wi)
        m1 = (nA/(N-h))*(uc-bc)+bc
        m2 = ((nA+nB)/(N-h))*(uc-bc)+bc
        m3 = ((nA+nB+nC)/(N-h))*(uc-bc)+bc
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
            code_intervals.addi(m3, uc, [int(7)])
        
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
                    m2 = interval.begin+((nAtemp+nBtemp)/(N-htemp))*(interval.end-interval.begin)
                    new_intervals.append((m1, m2, data))
                if(nCtemp>0):
                    data = interval.data.copy()
                    data.append(int(5))
                    m2 = interval.begin+((nAtemp+nBtemp)/(N-htemp))*(interval.end-interval.begin)
                    m3 = interval.begin+((nAtemp+nBtemp+nCtemp)/(N-htemp))*(interval.end-interval.begin)
                    new_intervals.append((m2, m3, data))
                if(nDtemp>0):
                    data = interval.data.copy()
                    data.append(int(7))
                    m3 = interval.begin+((nAtemp+nBtemp+nCtemp)/(N-htemp))*(interval.end-interval.begin)
                    new_intervals.append((m3, interval.end, data))

                old_intervals.append(interval)

            for iv_begin, iv_end, iv_data in new_intervals:
                code_intervals.addi(iv_begin, iv_end,iv_data)
        
            for iv in old_intervals:
                code_intervals.discard(iv)

            #find intervals that overlap with source interval
            overlapping_intervals = code_intervals.overlap(bi, ui)
                
            epsilon_overlap = []
            epsilon = 0
            for interval in overlapping_intervals:
                # Calculate the overlap size (intersection length)
                overlap_start = max(interval.begin, bi)
                overlap_end = min(interval.end, ui)
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
            ui = (ui-l)/(u-l)

            e=0

            lower_border_inside = [iv for iv in code_intervals if iv.begin >= bi-e and iv.begin <= ui+e]

            lower_border_sorted = sorted(lower_border_inside, key=lambda iv: iv.begin)

            row_done = False
            for iv in lower_border_sorted:
                
                if(iv.data.count(1)==nA and iv.data.count(3)==nB and iv.data.count(5)==nC):
                    finalised_seq = iv.data
                    finalised_seq.extend([7]*(nD-iv.data.count(7)))
                    next_symbol_index = np.where(x[row]==0)[0][0]
                    x[row][next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                    row_done = True
                    break

                elif(iv.data.count(1)==nA and iv.data.count(3)==nB and iv.data.count(7)==nD):
                    finalised_seq = iv.data
                    finalised_seq.extend([5]*(nC-iv.data.count(5)))
                    next_symbol_index = np.where(x[row]==0)[0][0]
                    x[row][next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                    row_done = True
                    break

                elif(iv.data.count(1)==nA and iv.data.count(5)==nC and iv.data.count(7)==nD):
                    finalised_seq = iv.data
                    finalised_seq.extend([3]*(nB-iv.data.count(3)))
                    next_symbol_index = np.where(x[row]==0)[0][0]
                    x[row][next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                    row_done = True
                    break

                elif(iv.data.count(3)==nB and iv.data.count(5)==nC and iv.data.count(7)==nD):
                    finalised_seq = iv.data
                    finalised_seq.extend([1]*(nA-iv.data.count(1)))
                    next_symbol_index = np.where(x[row]==0)[0][0]
                    x[row][next_symbol_index:next_symbol_index+len(finalised_seq)] = finalised_seq
                    row_done = True
                    break
            
            if(row_done==True):
                break #go to next row

    return x.flatten()

def DM_decode(codeword, C, k, blocks):
    #Distributin Matcher Decoding Function
    #C: the number of each symbol type in the codeword
    #k: the number of information bits in each block
    bits = []
    nA = C[0]
    nB = C[1]
    nC = C[2]
    nD = C[3]

    N = nA + nB +nC + nD

    codeword = codeword.reshape((blocks,N))

    for row in range(blocks):
        bit_row = []
        S_intervals = IntervalTree() #Source intervals

        m1 = (nA/(N))
        m2 = m1 + (nB/(N))
        m3 = m2 + (nC/(N))

        if(codeword[row][0]==1):
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
        elif(codeword[row][0]==3):
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
        elif(codeword[row][0]==5):
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
        elif(codeword[row][0]==7):
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
                bit_row.append(bit_interval.data)
                bit_identified = True
                S_count = S_count + 1
                
            else:
                S_containing_C = [si for si in S_intervals if si.begin <= C_interval.begin and C_interval.end <= si.end] #Source intervals that are identified by a code interval
                if(S_containing_C): #if any code intervals are within source intervals
                    bit_interval = S_containing_C[0]
                    bit_row.append(bit_interval.data)
                    bit_identified = True
                    S_count = S_count + 1

            if(S_count==k):
                bits.append(bit_row)
                break
            
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

                    next_symbol = codeword[row][K_countA+K_countB+K_countC+K_countD] #Read next symbol
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
                next_symbol = codeword[row][C_countA+C_countB+C_countC+C_countD] #Read next symbol
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
        
    return np.array(bits).flatten()

plot=True

if(plot==True):
    Modbits = 4
    blocks = 1
    np.random.seed(1)
    
    if(Modbits==4):
        signal_points = [1,3]
        λ = 0.06
        # N_target = 50
        N_target = 100
        A = 0
        for i in signal_points:
            A += np.exp(-λ*np.abs(i)**2)
        C = [int(N_target*np.exp(-λ)/A),int(N_target*np.exp(-λ*9)/A),0,0]
        N = np.sum(C)
        k = int(np.floor(math.log2(nCr(N,C[1]))))
        bits = np.random.randint(0, 2, size= k*blocks)

    if(Modbits==6):
        signal_points = [1,3,5,7]
        # λ = 0.04
        # N_target = 32
        λ = 0.05
        N_target = 100

        A = 0
        for i in signal_points:
            A += np.exp(-λ*np.abs(i)**2)
        C = [int(N_target*np.exp(-λ)/A),int(N_target*np.exp(-λ*9)/A),int(N_target*np.exp(-λ*25)/A),int(N_target*np.exp(-λ*49)/A)]
        k=int(np.floor(math.log2(math.factorial(C[0]+C[1]+C[2]+C[3])/(math.factorial(C[0])*math.factorial(C[1])*math.factorial(C[2])*math.factorial(C[3])))))
        N = np.sum(C)
        bits = np.random.randint(0, 2, size= k*blocks)


    x = DM_encode(C,bits,k,blocks)

    count1 = np.count_nonzero(x==1)
    count3 = np.count_nonzero(x==3)
    count5 = np.count_nonzero(x==5)
    count7 = np.count_nonzero(x==7)

    print('k:', k)
    print('N:', np.sum(C))
    print('Blocks:',blocks)
    print('Rate:', k/N, 'bits/symbol')
    print('C input:', C)
    print('N output average:', (count1+count3+count5+count7)/blocks)
    print('C output average:', [count1/blocks,count3/blocks,count5/blocks,count7/blocks])

    bits_decoded = DM_decode(x, C, k, blocks)
    print(bits, 'Source Bits')
    print(list(x),'CCDM Symbols')
    print(bits_decoded, 'Decoded Bits')
                    
    if(np.array_equal(bits_decoded,bits)):
        print('No Errors')
    else:
        print('Errors')

    print(np.where(bits!=bits_decoded))


    labels = [1,3,5,7]
    values = [count1/(N*blocks*2), count3/(N*blocks*2), count5/(N*blocks*2), count7/(N*blocks*2) ]
    #eg the probability of each symbol divded by the width it covers, eg 2 ((0,2),(2,4),(4,6),(6,8))

    # Create bar chart
    plt.bar(labels, values, color=['red', 'orange', 'green', 'blue'])
    plt.xlabel("Symbol")
    plt.ylabel(f"Maxwell-Boltzmann PMF, λ={λ}")
    plt.title("Symbol Probability Mass Function in Codeword")
    plt.plot()


    xaxis = np.linspace(0, 8, 400)  # Define range of x

    MB_dist = np.exp(-λ * np.abs(xaxis)**2)*2*np.sqrt(λ/np.pi)
    plt.plot(xaxis, MB_dist)

    # plt.show()
