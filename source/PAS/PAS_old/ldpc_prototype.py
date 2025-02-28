from ldpc_jossy.py import ldpc
import numpy as np


#standard: string : Specifies the IEEE standard used, 802.11n or 802.16

#rate: string : Specifies the code rate, 1/2, 2/3, 3/4 or 5/6

#z: int : Optional parameter (not needed for for 802.16, required for 802.11n)
    # Specifies the protograph expansion factor, freely chooseable >= 3 for
    # IEEE 802.16, restricted to (27, 54, 81) for IEEE 802.11n 

#ptype: character : Optional parameter.
    # Either A or B for 802.16 rates 2/3 and 3/4 where two options are
    # specified in the standard. Parameter unused for all other codes.
encoder = ldpc.code(standard = '802.16', rate = '1/2', z=50, ptype='A')
P = encoder.pcmat()

print('k:', encoder.K)
print('N:', encoder.N)
print('Rate:', encoder.K/encoder.N)


bits = np.random.randint(0, 2, size= encoder.K)
x = encoder.encode(bits)

y = 10*(.5-x)


app,it = encoder.decode(y)

corr = [0 if j > 0 else 1 for j in app]
print(it)

print(np.where(x!=corr))



