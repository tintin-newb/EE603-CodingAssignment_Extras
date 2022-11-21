import matplotlib.pyplot as plt
import numpy as np
from math import pi

def sine_sequence(w0, A=1, phi=0, L= 100):
  locs = np.arange(-L, L+1)                             #locations of samples
  signal = np.zeros(locs.shape)                            #initialize array to store signal
  for i in range(len(locs)):
    signal[i] = A*np.sin(w0*locs[i] + phi)

  return signal, locs

#utility function to plot a simple line plot
def simple_line_plot(loc, y, title="Dummy", xlabel="Time", ylabel="Amplitude"):
  plt.figure()
  plt.plot(loc, y)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()


# w1 = 0.3*pi
# w2 = 0.75*pi

# #we compute the signals with default values of other input parameters
# signal1, locs1 = sine_sequence(w1)
# signal2, locs2 = sine_sequence(w2)

# simple_line_plot(locs1, signal1, "Question 5.a.i", "n", "Amplitude")
# simple_line_plot(locs2, signal2, "Question 5.a.ii", "n", "Amplitude")

#calculate the signals individually and add them
w1 = 0.1*pi
phi1 = 0.75*pi
signal_1, locs = sine_sequence(w1, phi = phi1)

w2 = 0.8*pi
phi2 = 0.2*pi + pi/2                  #since this function is cosine we just add a pi/2 offset to the sine function
A2 = -3
signal_2, __ = sine_sequence(w2, A2, phi2)          #no need to get sample location hence dummy variable '__'

w3 = 1.3*pi
phi3 = pi/2                           #as done previously
signal_3, __ = sine_sequence(w3, phi=phi3)

final_signal = signal_1 + signal_2 + signal_3

simple_line_plot(locs, final_signal, "Question 5.b", 'n', "Amplitude")


