import numpy as np
from intervaltree import IntervalTree, Interval
import math
import matplotlib.pyplot as plt

#My implementation of distribution matcher
    
def nCr(n, r):
    if r > n:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

def DMencode(C, v, k,blocks):
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
        m1 = (nA/(N-h))*(uc-bc)+bc
        m2 = ((nA+nB)/(N-h))*(uc-bc) + bc
        m3 = ((nA+nB+nC)/(N-h))*(uc-bc) + bc

        if(nC==0 and nD==0):
            m2 = 1.0
            m3 = 1.0
        
        code_intervals = IntervalTree()
        if(nA!=0):
            code_intervals.addi(bc, m1, int(1))
        if(nB!=0):
            code_intervals.addi(m1, m2, int(3))
        if(nC!=0):
            code_intervals.addi(m2, m3, int(5))
        if(nD!=0):
            code_intervals.addi(m3, uc, int(7))

        source_interval = Interval(0,1)

        for i in range(k):
            if(v[row][i] == 0): #Update source intervals
                ui = source_interval.begin + 0.5*(source_interval.end-source_interval.begin)
                source_interval = Interval(source_interval.begin,ui)

            else:
                bi = source_interval.begin + 0.5*(source_interval.end-source_interval.begin)
                source_interval = Interval(bi,source_interval.end)
            

            #Check for output and rescale
            symbol_identified=False
            for iv in code_intervals:
                if(source_interval.begin >= iv.begin and source_interval.end <= iv.end):
                    symbol_identified = True
                    correct_code_interval = Interval(iv.begin,iv.end,iv.data)
                    break 
            
            while(symbol_identified==True):
                #rescale source interval
                bi = (source_interval.begin - correct_code_interval.begin)/(correct_code_interval.end-correct_code_interval.begin)
                ui = (source_interval.end - correct_code_interval.begin)/(correct_code_interval.end-correct_code_interval.begin)
                source_interval = Interval(bi,ui)
                if(ui>1):
                    ui=1

                #Update code intervals
                latest_symbol = correct_code_interval.data

                x[row][h] = latest_symbol

                h=h+1 #Number of symbols sent to output
                if(latest_symbol==1):
                    nA=nA-1
                elif(latest_symbol==3):
                    nB=nB-1
                elif(latest_symbol==5):
                    nC=nC-1
                elif(latest_symbol==7): 
                    nD=nD-1
                bc = 0.0 #code interval base
                uc = 1.0 #code interval upper
                m1 = (nA/(N-h))*(uc-bc)+bc
                m2 = ((nA+nB)/(N-h))*(uc-bc) + bc
                m3 = ((nA+nB+nC)/(N-h))*(uc-bc) + bc

                if(nC==0 and nD==0):
                    m2 = 1.0
                    m3 = 1.0
                
                # symbols = correct_code_interval.data.copy()
                # symbols.append(int(latest_symbol))
                code_intervals.clear()
                if(nA!=0):
                    code_intervals.addi(bc, m1, int(1))
                if(nB!=0):
                    code_intervals.addi(m1, m2, int(3))
                if(nC!=0):
                    code_intervals.addi(m2, m3, int(5))
                if(nD!=0):
                    code_intervals.addi(m3, uc, int(7))   

                symbol_identified = False

                for iv in code_intervals:
                    if((source_interval.begin >= iv.begin) and (source_interval.end <= iv.end)):
                        symbol_identified = True
                        correct_code_interval = Interval(iv.begin,iv.end,iv.data)
                        break 

        #Finalisation step so that the codeword identifies the source interval [bi, bi+wi)
        m1 = (nA/(N-h))*(uc-bc)+bc
        m2 = ((nA+nB)/(N-h))*(uc-bc)+bc
        m3 = ((nA+nB+nC)/(N-h))*(uc-bc)+bc

        if(nC==0 and nD==0):
            m2 = 1.0
            m3 = 1.0

        code_intervals.clear()
        code_intervals = IntervalTree()
        if(nA!=0):
            code_intervals.addi(bc, m1, int(1))
        if(nB!=0):
            code_intervals.addi(m1, m2, int(3))
        if(nC!=0):
            code_intervals.addi(m2, m3, int(5))
        if(nD!=0):
            code_intervals.addi(m3, uc, int(7))
        
        finalised = False
        for iv in code_intervals:
            if(source_interval.begin <= iv.begin and iv.begin <= source_interval.end): #Lower border of code candidate is in source interval
                correct_code_interval = Interval(iv.begin,iv.end,iv.data)
                finalised = True
                break

        if(finalised==False): #Output sequence doesn't need finalising
            continue

        latest_symbol = correct_code_interval.data
        x[row][h] = latest_symbol

        h=h+1 #Number of symbols sent to output
        if(latest_symbol==1):
            nA=nA-1
        elif(latest_symbol==3):
            nB=nB-1
        elif(latest_symbol==5):
            nC=nC-1
        elif(latest_symbol==7): 
            nD=nD-1

        for i in range(nA):
            x[row][h] = int(1)
            h = h+1
        for i in range(nB):
            x[row][h] = int(3)
            h = h+1
        for i in range(nC):
            x[row][h] = int(5)
            h = h+1
        for i in range(nD):
            x[row][h] = int(7)
            h = h+1

    return x.flatten()  



plot=False

if(plot==True):
    import DMdecode as DMdecode
    Modbits = 6
    blocks = 1
    # np.random.seed(1)
    
    if(Modbits==4):
        signal_points = [1,3]
        λ = 0.06
        # N_target = 50
        N_target = 800
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
        N_target = 600

        A = 0
        for i in signal_points:
            A += np.exp(-λ*np.abs(i)**2)
        C = [int(N_target*np.exp(-λ)/A),int(N_target*np.exp(-λ*9)/A),int(N_target*np.exp(-λ*25)/A),int(N_target*np.exp(-λ*49)/A)]
        k=int(np.floor(math.log2(math.factorial(C[0]+C[1]+C[2]+C[3])/(math.factorial(C[0])*math.factorial(C[1])*math.factorial(C[2])*math.factorial(C[3])))))
        N = np.sum(C)
        bits = np.random.randint(0, 2, size= k*blocks)

    x = DMencode(C,bits,k,blocks)

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

    print(bits)
    bits_decoded = DMdecode.DMdecode(x, C, k, blocks)
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

    #[1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]