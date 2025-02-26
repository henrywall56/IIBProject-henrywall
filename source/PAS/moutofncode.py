import numpy as np
import matplotlib.pyplot as plt
import math
#m out of n coding, need to adapt to work with non-binary code alphabet

def nCr(n, r):
    if r > n:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

def moutofn_encoder(m, N, u, k, w, blocks):
    #m: Number of "High" symbols in each codeword
    #N: Length of each codeword
    #u: Input bits
    #k: Length of each input bit block
    #w: precision parameter
    u = u.reshape((blocks, k))
    v = np.empty((u.shape[0], N), dtype=int)
    for row in range (u.shape[0]):
        i=0
        j=0
        x=0
        y=1
        
        n=N
        n0 = N-m

        while(1):
            if(y<2**w):
                if(j<k):
                    j=j+1
                    x = 2*x + u[row][j-1]
                else:
                    x = 2*x
                y = 2*y
            
            else:
                z = np.floor((2*y*n0+n)/(2*n))
                n = n-1
                i = i+1
                if(x<z):
                    v[row][i-1]=1
                    y=z
                    n0=n0-1
                else:
                    v[row][i-1] = 3
                    x = x-z
                    y=y-z
                if(i>=N):
                    break

    return v.flatten()

def moutofn_decoder(v, m, N, k , w):
    #v symbols
    #m out of N code
    #k is bit block length
    #w precision parameter
    v =v.reshape((len(v)//N,N))
    u = np.empty((v.shape[0], k), dtype=int)
    for row in range(v.shape[0]):
        i=0
        j=-2
        x=0
        y=2**w
        L=0
        rl = 0
        n = N
        n0 = N-m
        nb=0

        while(1):
            if(L>k): #end sequence
                if(x>0):
                    nb=nb+1
                    rb=0
                else:
                    rb=1
                j=j+1
                u[row][j-1]=nb
                while(rl>0):
                    j=j+1
                    u[row][j-1]=rb
                    rl=rl-1
                break #end
            else:
                z = np.floor((2*y*n0+n)/(2*n))
                n=n-1
                i=i+1
                if(v[row][i-1]==1):
                    y=z
                    n0=n0-1
                else:
                    x=x+z
                    y=y-z
                while(y<2**w):
                    if(L>k):
                        break #enter end sequence
                    else:
                        y=2*y
                        L=L+1
                        e=np.floor(x/(2**w))
                        x=2*(x%(2**w))
                        if(e==1):
                            rl=rl+1
                        else:
                            if(e>1):
                                nb=nb+1
                                rb=0
                                e=e-2
                            else:
                                rb=1
                            j=j+1
                            u[row][j-1]=nb
                            nb=e
                            while(rl>0):
                                j=j+1
                                u[row][j-1]=rb
                                rl=rl-1
                        
    return u.flatten()

C = [500,300]


N=np.sum(C)
m=C[1]
print(m)
print(N)

w=8
blocks=1
k=int(np.floor(math.log2(nCr(N,m)))) #There are nCr(N,m) codewords, so have maximum log2(number of codewords) input bits.

np.random.seed(1)
bits = np.random.randint(0, 2, size= k*blocks)
bits = np.zeros(k)



A = moutofn_encoder(m,N,bits,k, w, blocks)
print('k: ',k)
print('m: ',m)
print('N: ',N)
print('blocks: ',blocks)
print('Total number of bits: ',k*blocks)
print('Rate: ', k/N, 'bits/symbol')
print(bits)
print(A)
u=moutofn_decoder(A,m,N,k,w)
print(u)
if(np.array_equal(u,bits)):
    print('No Errors')
else:
    print('Errors')

count3 = np.count_nonzero(A == 3)
count1 = np.count_nonzero(A == 1)
labels = ['1', '3']
values = [count1/len(A), count3/len(A)]

# Create bar chart
plt.bar(labels, values, color=['blue', 'green'])
plt.xlabel("Number")
plt.ylabel("Frequency")
plt.title("Frequency of 1 and 3 in Vector A")

# Show the plot
plt.show()

