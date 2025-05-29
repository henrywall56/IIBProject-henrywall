## IIBProject-henrywall

# High Capacity Digital Coherent Transceivers

Optical digital coherent transceivers have the capability of achieving 1 Tbit/s links in the core networks. This project will focus on the state-of-the-art digital signal processing used in a coherent transceiver to achieve this capacities. Algorithms including those required for synchronisation and equalisation (both linear and nonlinear) will be researched, as well as modulation techniques such as probabilistic shaping. The project will begin by simulating the expected performance using python before moving to experimentally validating the performance using the experimental testbed which includes 45 GHz arbitrary waveform generators and 70 GHz real time oscilloscope and over 1000 km of fibre.  I will gain hands on experience of working with ultra-high capacity communication systems, including metrology and the design the DSP associated with modern optical transceivers.

# File Structure

Source:
- All Python files developed in this project.
- PAS contains the probabilistic amplitude shaping architecture
  - ldpc_jossy contains Jossy Sayir's C implementation of LDPC code, all tests using       PAS must be run from this folder.
- PAS_old contains all testing for PAS.
- All other files just in source.

Data:
- All experimental data not uploaded to github due to file size.
- Matlab code used during experiments.
