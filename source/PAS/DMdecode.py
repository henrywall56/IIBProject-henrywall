import numpy as np
from intervaltree import IntervalTree, Interval
import math
import matplotlib.pyplot as plt

def DMdecode(codeword, C, k, blocks):
    #Distributin Matcher Decoding Function
    #C: the number of each symbol type in the codeword
    #k: the number of information bits in each block
    bits = []
    nA = C[0]
    nB = C[1]
    nC = C[2]
    nD = C[3]
    nE = C[4]
    nF = C[5]
    nG = C[6]
    nH = C[7]

    symbol_counts = [nA,nB,nC,nD,nE,nF,nG,nH]

    N = nA + nB +nC + nD + nE + nF + nG + nH

    codeword = codeword.reshape((blocks,N))

    for row in range(blocks):
        bit_row = []
        symbol_counts = [nA,nB,nC,nD,nE,nF,nG,nH]
        m1 = (symbol_counts[0]/(N))
        m2 = ((symbol_counts[0]+symbol_counts[1])/(N))
        m3 = ((symbol_counts[0]+symbol_counts[1]+symbol_counts[2])/(N))
        m4 = ((symbol_counts[0]+symbol_counts[1]+symbol_counts[2]+symbol_counts[3])/(N))
        m5 = ((symbol_counts[0]+symbol_counts[1]+symbol_counts[2]+symbol_counts[3]+symbol_counts[4])/(N))
        m6 = ((symbol_counts[0]+symbol_counts[1]+symbol_counts[2]+symbol_counts[3]+symbol_counts[4]+symbol_counts[5])/(N))
        m7 = ((symbol_counts[0]+symbol_counts[1]+symbol_counts[2]+symbol_counts[3]+symbol_counts[4]+symbol_counts[5]+symbol_counts[6])/(N))

        ########## INITIALISE ##########
        C_intervals = IntervalTree()    
        if(nA!=0):
            C_intervals.addi(0.0,m1,int(1)) 
        if(nB!=0):
            C_intervals.addi(m1,m2,int(3))
        if(nC!=0):
            C_intervals.addi(m2,m3,int(5))
        if(nD!=0):
            C_intervals.addi(m3,m4,int(7))
        if(nE!=0):
            C_intervals.addi(m4,m5,int(9))
        if(nF!=0):
            C_intervals.addi(m5,m6,int(11))
        if(nG!=0):
            C_intervals.addi(m6,m7,int(13))
        if(nH!=0):
            C_intervals.addi(m7,1.0,int(15))

        S_interval = Interval(0,1.0)   #intialise source interval

        symbol_index = 0   #current symbol
        row_done = False

        while(symbol_index != N):
            ######### READ CODE SYMBOL #########
            unprocessed_symbol = codeword[row][symbol_index] 

            for iv in C_intervals: #Choose interval corresponding to symbol
                if(unprocessed_symbol == iv.data):
                    code_interval = Interval(iv.begin,iv.end)
                    break

            symbol_counts_preview = symbol_counts.copy() #Used later in preview stage

            if(unprocessed_symbol == 1):    #Keep preview up to date with current state at this stage
                symbol_counts_preview[0] -= 1 #nA 
            elif(unprocessed_symbol == 3):
                symbol_counts_preview[1] -= 1 #nB
            elif(unprocessed_symbol == 5):
                symbol_counts_preview[2] -= 1 #nC
            elif(unprocessed_symbol == 7):
                symbol_counts_preview[3] -= 1 #nD
            elif(unprocessed_symbol == 9):
                symbol_counts_preview[4] -= 1 #nE
            elif(unprocessed_symbol == 11):
                symbol_counts_preview[5] -= 1 #nF
            elif(unprocessed_symbol == 13):
                symbol_counts_preview[6] -= 1 #nG
            elif(unprocessed_symbol == 15):
                symbol_counts_preview[7] -= 1 #nH

            preview_symbol_index = symbol_index #Refine code interval until it identifies enough source symbols to imitate the behaviour of the decoder.
            preview_symbol_index +=1            #If the encoder performs a scaling operation, reset the code interval to the state of the encoder and restart.
            
            scaled = False 
            
            while(scaled==False):
            ######## IDENTIFY SOURCE SYMBOLS ########
                s = S_interval.begin + (S_interval.end-S_interval.begin)*0.5 #source interval midpoint
                while((code_interval.begin >= s) or (code_interval.end < s)): #code interval does not straddle border. If it is on one side, it will stay there.
                    if(code_interval.begin >= s):
                        bit_row.append(1)
                        S_interval = Interval(s, S_interval.end)
                    elif(code_interval.end < s):
                        bit_row.append(0)
                        S_interval = Interval(S_interval.begin, s)

                    if(len(bit_row)==k):
                        row_done = True
                        break
                
                    s = S_interval.begin + (S_interval.end-S_interval.begin)*0.5

                    S_interval_check = Interval(S_interval.begin,S_interval.end) #used to check if source interval has been rescaled
                    symbol_counts_temp = symbol_counts.copy()

                    ##########   MIMICKING ENCODER OPERATION  ######### To check if there is a rescaling
                    symbol_identified=False
                    for iv in C_intervals:
                        if(S_interval_check.begin >= iv.begin and S_interval_check.end <= iv.end):
                            symbol_identified = True
                            correct_code_interval = Interval(iv.begin,iv.end,iv.data)
                            break 
                    number_identified = 0
                    while(symbol_identified==True):
                        number_identified += 1 #number of symbols identified by this encoder. This is the number that the symbol_index will need to increase by to match state of the encoder.
                        #rescale source interval
                        bi = (S_interval_check.begin - correct_code_interval.begin)/(correct_code_interval.end-correct_code_interval.begin)
                        ui = (S_interval_check.end - correct_code_interval.begin)/(correct_code_interval.end-correct_code_interval.begin)
                        S_interval_check = Interval(bi,ui)
                        if(ui>1):
                            ui=1

                        #Update code intervals
                        latest_symbol = correct_code_interval.data

                        if(latest_symbol==1):
                            symbol_counts_temp[0] -=1
                        elif(latest_symbol==3):
                            symbol_counts_temp[1] -=1
                        elif(latest_symbol==5):
                            symbol_counts_temp[2] -=1
                        elif(latest_symbol==7): 
                            symbol_counts_temp[3] -=1
                        elif(latest_symbol==9): 
                            symbol_counts_temp[4] -=1
                        elif(latest_symbol==11): 
                            symbol_counts_temp[5] -=1
                        elif(latest_symbol==13): 
                            symbol_counts_temp[6] -=1
                        elif(latest_symbol==15): 
                            symbol_counts_temp[7] -=1

                        bc = 0.0 #code interval base
                        uc = 1.0 #code interval upper
                        symbols_left = symbol_counts_temp[0] + symbol_counts_temp[1] + symbol_counts_temp[2] + symbol_counts_temp[3] + symbol_counts_temp[4] + symbol_counts_temp[5] + symbol_counts_temp[6] + symbol_counts_temp[7]
                        m1 = (symbol_counts_temp[0]/(symbols_left))*(uc-bc)+bc
                        m2 = ((symbol_counts_temp[0]+symbol_counts_temp[1])/(symbols_left))*(uc-bc) + bc
                        m3 = ((symbol_counts_temp[0]+symbol_counts_temp[1]+symbol_counts_temp[2])/(symbols_left))*(uc-bc) + bc
                        m4 = ((symbol_counts_temp[0]+symbol_counts_temp[1]+symbol_counts_temp[2]+symbol_counts_temp[3])/(symbols_left))*(uc-bc) + bc
                        m5 = ((symbol_counts_temp[0]+symbol_counts_temp[1]+symbol_counts_temp[2]+symbol_counts_temp[3]+symbol_counts_temp[4])/(symbols_left))*(uc-bc) + bc
                        m6 = ((symbol_counts_temp[0]+symbol_counts_temp[1]+symbol_counts_temp[2]+symbol_counts_temp[3]+symbol_counts_temp[4]+symbol_counts_temp[5])/(symbols_left))*(uc-bc) + bc
                        m7 = ((symbol_counts_temp[0]+symbol_counts_temp[1]+symbol_counts_temp[2]+symbol_counts_temp[3]+symbol_counts_temp[4]+symbol_counts_temp[5]+symbol_counts_temp[6])/(symbols_left))*(uc-bc) + bc
                        
                        C_intervals.clear()
                        if(symbol_counts_temp[0]!=0):
                            C_intervals.addi(bc, m1, int(1))
                        if(symbol_counts_temp[1]!=0):
                            C_intervals.addi(m1, m2, int(3))
                        if(symbol_counts_temp[2]!=0):
                            C_intervals.addi(m2, m3, int(5))
                        if(symbol_counts_temp[3]!=0):
                            C_intervals.addi(m3, m4, int(7)) 
                        if(symbol_counts_temp[4]!=0):
                            C_intervals.addi(m4, m5, int(9))  
                        if(symbol_counts_temp[5]!=0):
                            C_intervals.addi(m5, m6, int(11))  
                        if(symbol_counts_temp[6]!=0):
                            C_intervals.addi(m6, m7, int(13))  
                        if(symbol_counts_temp[7]!=0):
                            C_intervals.addi(m7, uc, int(15))    

                        symbol_identified = False

                        for iv in C_intervals:
                            if((S_interval_check.begin >= iv.begin) and (S_interval_check.end <= iv.end)):
                                symbol_identified = True
                                correct_code_interval = Interval(iv.begin,iv.end,iv.data)
                                break 
                #######################################################
                    ###### CHECK IF SCALING PERFORMED ###### If scaling if performed, update the symbol counter to the state of the encoder
                    if((S_interval_check.begin != S_interval.begin) or (S_interval_check.end != S_interval.end)):
                        symbol_index += number_identified
                        symbol_counts = symbol_counts_temp.copy()
                        S_interval = Interval(S_interval_check.begin, S_interval_check.end)
                        scaled = True
                        break
                
                if(row_done==True):
                    break
                ######### No Scaling Performed: PREVIEW CODE SYMBOL #########
                        
                if(preview_symbol_index == N):
                    code_interval = Interval(code_interval.begin, code_interval.begin + 0.01*(code_interval.end - code_interval.begin)) #"magic"

                else: #Preview next symbol
                    preview_symbol = codeword[row][preview_symbol_index]
                    symbol_shortlist = []
                    symbol_shortlist.append(preview_symbol)

                    symbols_left = symbol_counts_preview[0] + symbol_counts_preview[1] + symbol_counts_preview[2] + symbol_counts_preview[3] + symbol_counts_preview[4] + symbol_counts_preview[5] + symbol_counts_preview[6] + symbol_counts_preview[7]
                    if(preview_symbol==1):
                        l=0
                        u = symbol_counts_preview[0]/symbols_left
                    elif(preview_symbol==3):
                        l = symbol_counts_preview[0]/symbols_left
                        u = (symbol_counts_preview[0]+symbol_counts_preview[1])/symbols_left
                    elif(preview_symbol==5):
                        l = (symbol_counts_preview[0]+symbol_counts_preview[1])/symbols_left
                        u = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2])/symbols_left
                    elif(preview_symbol==7):
                        l = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2])/symbols_left
                        u = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2]+symbol_counts_preview[3])/symbols_left
                    elif(preview_symbol==9):
                        l = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2]+symbol_counts_preview[3])/symbols_left
                        u = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2]+symbol_counts_preview[3]+symbol_counts_preview[4])/symbols_left
                    elif(preview_symbol==11):
                        l = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2]+symbol_counts_preview[3]+symbol_counts_preview[4])/symbols_left
                        u = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2]+symbol_counts_preview[3]+symbol_counts_preview[4]+symbol_counts_preview[5])/symbols_left
                    elif(preview_symbol==13):
                        l = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2]+symbol_counts_preview[3]+symbol_counts_preview[4]+symbol_counts_preview[5])/symbols_left
                        u = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2]+symbol_counts_preview[3]+symbol_counts_preview[4]+symbol_counts_preview[5]+symbol_counts_preview[6])/symbols_left
                    elif(preview_symbol==15):
                        l = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2]+symbol_counts_preview[3]+symbol_counts_preview[4]+symbol_counts_preview[5]+symbol_counts_preview[6])/symbols_left
                        # u = (symbol_counts_preview[0]+symbol_counts_preview[1]+symbol_counts_preview[2]+symbol_counts_preview[3]+symbol_counts_preview[4]+symbol_counts_preview[5]+symbol_counts_preview[6]+symbol_counts_preview[7])/symbols_left
                        u=1.0

                    buffer_l = code_interval.begin + (code_interval.end - code_interval.begin)*l
                    buffer_u = code_interval.begin + (code_interval.end - code_interval.begin)*u
                    code_interval = Interval(buffer_l,buffer_u)

                    if(preview_symbol==1):
                        symbol_counts_preview[0] -=1
                    elif(preview_symbol==3):
                        symbol_counts_preview[1] -=1
                    elif(preview_symbol==5):
                        symbol_counts_preview[2] -=1
                    elif(preview_symbol==7): 
                        symbol_counts_preview[3] -=1
                    elif(preview_symbol==9): 
                        symbol_counts_preview[4] -=1
                    elif(preview_symbol==11): 
                        symbol_counts_preview[5] -=1
                    elif(preview_symbol==13): 
                        symbol_counts_preview[6] -=1
                    elif(preview_symbol==15): 
                        symbol_counts_preview[7] -=1
                    
                    preview_symbol_index += 1

                    # Go back to identifying source symbols, as still in "no scaling performed" loop.
                

            if(row_done==True):
                break

        bits.append(bit_row)
    return np.array(bits).flatten()
 