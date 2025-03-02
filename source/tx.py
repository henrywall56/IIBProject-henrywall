import numpy as np
import matplotlib.pyplot as plt
import functions as f
import PAS.PAS_architecture as pas
import parameters as p

def tx(original_bits):
    #num_power: num_symbols = 2**num_power
    #mod_param is a Modulation_param class type
    #RRC_param is a RRC_param class type
    #fibre_param is a fibre_param class type
    #t is a toggle class type

    NPol = p.Mod_param.NPol
    Rs = p.Mod_param.Rs
    Modbits = p.Mod_param.Modbits
    
    #Generate RRC filter impulse response
    span = p.RRC_param.span #Span of filter
    rolloff = p.RRC_param.rolloff #Roll-off of RRC

    #BPS Phase Recovery Parameters:
    N=20 #Number of symbols to average over
    #"N = 6,..,10 will be a fairly good choice" - Pfau paper

    if(Modbits==2): #Values given by Pfau paper
        B=64 #Number of trial angles
        modulation_format='QPSK'
    elif(Modbits==4):
        B=32 #Number of trial angles
        modulation_format='16-QAM'
    elif(Modbits==6):
        B=64 #Number of trial angles
        modulation_format='64-QAM'

    L = p.fibre_param.L

    if(p.toggle.toggle_RRC==False):
        p.RRC_param.sps=1               #overwrite sps if no RRC    


    print('Modulation Format:        ', modulation_format)
    print('Fibre Length:             ', L/1e3, 'km')
    print('Polarisations:            ', NPol)
    print('Wavelength λ:             ', round(p.fibre_param.Clambda/1e-9), 'nm')
    print('Dispersion Parameter D:   ', p.fibre_param.D, 'ps/(nm.km)')
    print('--------------------------------------')
    

    if(p.toggle.toggle_PAS==False):

        symbols = f.generate_symbols(original_bits, Modbits, NPol, p.toggle.toggle_DE) #NPol-dimensional array
        print('--------------------------------------')
        print('No. of Symbols:                ', p.Mod_param.num_symbols)
        print('Symbol Rate:         ', Rs/1e9, 'GBaud')
        print('Bit Rate:            ', Modbits*Rs/1e9, 'GBit/s')
        print('--------------------------------------')

    else:
        
        λpas = p.PAS_param.λ
        kpas, Npas, Cpas, LDPC_encoderpas = p.PAS_param.k, p.PAS_param.N, p.PAS_param.C, p.PAS_param.LDPC_encoder
        symbols = pas.PAS_encoder(Cpas, original_bits, kpas, p.PAS_param.blocks, Modbits, LDPC_encoderpas)
        pas.PAS_barplot(symbols)
        PAS_normalisation = np.sum(abs(symbols)**2)/len(symbols)
        p.PAS_param.PAS_normalisation = PAS_normalisation
        symbols = symbols/np.sqrt(PAS_normalisation)
        if(NPol==1):
            p.Mod_param.num_symbols = len(symbols)
        else:
            p.Mod_param.num_symbols = symbols.shape[1]
        
        print('PAS Parameters:')
        print('Info Bits per Block k-N:       ', kpas)
        print('Symbols per Block N:           ', Npas)
        print('Number of Blocks:              ', p.PAS_param.blocks)
        print('Maxwell-Boltzmann Parameter λ: ', λpas)
        print('Shaping Rate:                  ', 2*kpas/Npas, 'bits/symbol') 
        print('--------------------------------------')
        print('No. of Symbols:      ', p.Mod_param.num_symbols)
        print('Symbol Rate:         ', Rs/1e9, 'GBaud')
        print('Bit Rate:            ', 2*((kpas)/Npas)*Rs/1e9, 'GBit/s')
        print('--------------------------------------')

    
    #Pulse shaping with RRC filter
    pulse_shaped_symbols = f.pulseshaping(symbols, p.RRC_param.sps, p.RRC_param.RRCimpulse, NPol, toggle = p.toggle.toggle_RRC) #if toggle False, this function just returns symbol
    print('--------------------------------------')
    print('RRC Parameters:')
    print('RRC Rolloff: ', rolloff)
    print('RRC span: ', span)
    print('--------------------------------------')
    
    return pulse_shaped_symbols

