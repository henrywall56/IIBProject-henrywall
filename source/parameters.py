import functions as f
import numpy as np
import PAS.PAS_architecture as pas

class RRC_paramX:
    def __init__(self, span, sps ,rolloff):
        self.span = span
        self.sps = sps
        self.rolloff = rolloff
        self.RRCimpulse , t1 = f.RRC(span, rolloff, sps)

class fibre_paramX:
    def __init__(self, L, D, Clambda, snrb_db):
        self.L = L #in m
        self.D = D #in ps/(nm.km)
        self.Clambda = Clambda #in m
        self.snrb_db = snrb_db

class IQ_Mod_paramX:
    def __init__(self, Vpi):
        self.Vpi = Vpi
        self.Bias = -1*Vpi
        self.MinExc = -1.5*Vpi
        self.MaxExc = -0.5*Vpi

class laser_paramX:
    def __init__(self, Linewidth, maxDvT, laser_power):
        self.Linewidth=Linewidth #Hz
        self.maxDvT=maxDvT
        self.laser_power=laser_power

class toggleX:
    def __init__(self, toggle_RRC, toggle_AWGNnoise, toggle_phasenoise, toggle_phasenoisecompensation, toggle_plotuncompensatedphase, toggle_ploterrorindexes, toggle_BPS, toggle_DE, toggle_frequencyrecovery, toggle_CD, toggle_NL, toggle_CD_compensation, toggle_AIR, toggle_adaptive_equalisation, toggle_PAS, AIR_type):
        self.toggle_RRC = toggle_RRC
        self.toggle_AWGNnoise = toggle_AWGNnoise
        self.toggle_phasenoise = toggle_phasenoise
        self.toggle_phasenoisecompensation = toggle_phasenoisecompensation
        self.toggle_plotuncompensatedphase = toggle_plotuncompensatedphase
        self.toggle_ploterrorindexes = toggle_ploterrorindexes
        self.toggle_BPS = toggle_BPS
        self.toggle_DE = toggle_DE
        self.toggle_frequencyrecovery = toggle_frequencyrecovery
        self.toggle_CD = toggle_CD
        self.toggle_NL = toggle_NL
        self.toggle_CD_compensation = toggle_CD_compensation
        self.toggle_AIR = toggle_AIR
        self.toggle_adaptive_equalisation = toggle_adaptive_equalisation
        self.toggle_PAS = toggle_PAS
        self.AIR_type = AIR_type

class Modulation_paramX:
    def __init__(self, Modbits, Rs, NPol, num_power):
        self.Modbits = Modbits
        self.Rs = Rs
        self.NPol = NPol
        self.num_symbols = 2**num_power


class CD_paramX:
    def __init__(self, D, Clambda, Rs, L):
        #NOTE: need FFT size of ≈ 4 * number of symbols dispersion is spread over (@2 sps) = 4 * D*∆lambda*L * (2*Rs)= 4 * D*(Clambda/f * ∆f)*L * (2*Rs) = 4 * D*(((Clambda**2)/c)*Rs)*L * (2*Rs)
        #NOTE: need NOverlap size of ≈ number of symbols dispersion is spread over (@2 sps) = 4 * D*∆lambda*L * (2*Rs)= 4 * D*(Clambda/f * ∆f)*L * (2*Rs) = 4 * D*(((Clambda**2)/c)*Rs)*L * (2*Rs)
        ideal_NOverlap = D*(1e-12/(1e-9*1e3)) * (Clambda**2/299792458)*Rs*L*2*Rs
        ideal_NFFT = 4*ideal_NOverlap
        NFFTpower = int(np.ceil(np.log2(ideal_NFFT))) 
        self.NFFT=2**NFFTpower #Adjusted to minimise complexity 
        self.NOverlap = int(ideal_NOverlap) #Given by minimum equaliser length N_CD : pg 113 CDOT graph?

class AE_paramX:
    def __init__(self, NTaps, mu ,N1, N2, Ndiscard):
        self.NTaps = NTaps
        self.mu = mu
        self.N1 = N1
        self.N2 = N2
        self.Ndisard = Ndiscard

class BPS_paramX:
    def __init__(self, B, N):
        self.B = B
        self.N = N

class PAS_paramX:
    def __init__(self, Modbits, λ):
        self.k, self.N, self.C, self.LDPC_encoder = pas.PAS_parameters(Modbits, λ)
        self.λ = λ
        self.blocks = Mod_param.num_symbols//self.k
        self.sigma = 0
        self.PAS_normalisation = 0

############################# SETUP PARAMETERS #############################

Mod_param = Modulation_paramX(
        Modbits = 6,
        Rs = 100e9,
        NPol = 1,
        num_power = 16
)

RRC_param = RRC_paramX(
        span=20, 
        sps=16, 
        rolloff=0.1
)

fibre_param = fibre_paramX(
        L=1000*1e3, 
        D=17, 
        Clambda=1550/1e9,
        snrb_db = 21
)

IQ_Mod_param = IQ_Mod_paramX(
        Vpi=1.05
)

Linewidth = 100*10**3


toggle = toggleX(
        toggle_RRC = True,
        toggle_AWGNnoise = True,
        toggle_phasenoise = False,
        toggle_phasenoisecompensation = False,
        toggle_plotuncompensatedphase = False,
        toggle_ploterrorindexes = False,
        toggle_BPS = True,
        toggle_DE = False,
        toggle_frequencyrecovery = False,
        toggle_CD = False,
        toggle_NL = False,
        toggle_CD_compensation = False,
        toggle_AIR = False,
        toggle_adaptive_equalisation = False,
        toggle_PAS = True,
        AIR_type = 'MI'
)

if(toggle.toggle_RRC == True):
    sps = RRC_param.sps
else:
    sps = 1

maxDvT = Linewidth/(Mod_param.Rs*sps)

laser_param = laser_paramX(
        Linewidth = Linewidth,
        maxDvT = maxDvT, 
        laser_power = 0 #dBm
)


CD_param = CD_paramX(fibre_param.D, fibre_param.Clambda, Mod_param.Rs, fibre_param.L)

AE_param = AE_paramX(
        NTaps = 4,
        mu = 5e-4,
        N1 = 2000,
        N2 = 5000,
        Ndiscard = 12000
)

BPS_param = BPS_paramX(
        B = 64,
        N = 20
)

PAS_param = PAS_paramX(
        Mod_param.Modbits,
        λ = 0.05
)