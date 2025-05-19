import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 22  # Change the font size
plt.rcParams['font.family'] = 'Times New Roman' 

plot_ngvariation = True
if(plot_ngvariation==True):
    wavelength = np.arange(1200,1400,1)
    ng = 1.4565 + 3*(57.086/wavelength)**2 + (wavelength/17436)**2
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength,ng, color='black')
    plt.xlabel('Wavelength/ nm')
    plt.ylabel('Group Refractive Index')
    plt.show()