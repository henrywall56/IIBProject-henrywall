import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_color_constellation(wavein,axs:plt.Axes):
    N = 4
    Nbeside = 20  # Neighboring points to consider
    indexdata = np.zeros((len(wavein), N), dtype=int)
    angledata = np.linspace(0, np.pi/2, N)
    
    for m in range(N):
        data = (np.real(np.exp(1j * angledata[m])) * np.real(wavein) +
                np.imag(np.exp(1j * angledata[m])) * np.imag(wavein))
        indexdata[:, m] = np.argsort(data)
    
    dis = np.zeros((len(indexdata), N))
    
    for kk in range(N):
        for k in range(len(indexdata)):
            indexbeg = max(k - Nbeside, 0)
            indexend = min(indexbeg + 2 * Nbeside + 1, len(wavein))
            indexbeg = max(indexend - 2 * Nbeside, 0)
            
            datatemp = wavein[indexdata[indexbeg:indexend, kk]] - wavein[indexdata[k, kk]]
            sorted_distances = np.sort(np.abs(datatemp))
            dis[indexdata[k, kk], kk] = sorted_distances[1]
    
    disdata = np.min(dis, axis=1)
    disdata = (disdata / np.max(np.abs(disdata))) ** 0.1
    
    Nslice = 256
    counts, centers = np.histogram(disdata, bins=Nslice)
    cpdf = np.cumsum(counts) / np.sum(counts)
    indexcolor = np.ceil(cpdf * Nslice).astype(int)
    
    halfwide = (centers[1] - centers[0]) / 2
    factor = 0.65
    istart = round(Nslice / factor * (1 - factor) / 2)
    cmap = cm.jet(np.linspace(0, 1, round(Nslice / factor)))[istart:istart + Nslice]
    cmap = np.flipud(cmap)
    

    axs.axis([-5, 5, -5, 5])
    
    for k in range(Nslice - 1, -1, -1):
        mask = (disdata >= centers[k] - halfwide) & (disdata <= centers[k] + halfwide)
        axs.scatter(np.real(wavein[mask]), np.imag(wavein[mask]), c=[cmap[indexcolor[k] - 1]], s=1, marker='.')
    
    axs.grid(alpha=0.2)
    axs.set_xlabel('In-Phase')
    axs.set_ylabel('Quadrature')