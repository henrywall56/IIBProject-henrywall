import numpy as np
import matplotlib.pyplot as plt
import functions as f
import tx
import channel
import rx
import parameters as p

if(p.toggle.toggle_PAS==False):
    original_bits = f.generate_original_bits(p.Mod_param.num_symbols, p.Mod_param.Modbits, p.Mod_param.NPol) #NPol-dimensional array
else:
    original_bits = np.random.randint(0, 2, size= p.PAS_param.k*p.PAS_param.blocks*2*p.Mod_param.NPol)

pulse_shaped_symbols= tx.tx(original_bits)

channel_output = channel.channel(pulse_shaped_symbols)

demodulated_bits, demodulated_symbols, processed_symbols = rx.rx(channel_output) 


fig, axs = plt.subplots(1, 1, figsize=(8, 8))
f.plot_constellation(axs, pulse_shaped_symbols[::p.RRC_param.sps], title='pre-channel', lim=2)
fig2, axs2 = plt.subplots(1, 1, figsize=(8, 8))
f.plot_constellation(axs2, processed_symbols, title='processed', lim=5)
plt.show()
