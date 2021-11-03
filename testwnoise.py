from pylab import *
from filter import lowpass,bandpass
import numpy
import matplotlib.pyplot as plt
from pylab import fft, array, convolve, shape, zeros, size
from specfn import MorletSpec

plt.rcParams['font.size'] = 14

#
def timefreq(x,fs=200):
    """
    TIMEFREQ
    This function takes the time series and the sampling rate and calculates the
    total number of points, the maximum frequency, the minimum (or change in)
    frequency, and the vector of frequency points F.    
    Version: 2011may04
    """
    from numpy import size, shape, arange, append    
    maxfreq=float(fs)/2.0 # Maximum frequency
    minfreq=float(fs)/float(size(x,0)) # Minimum and delta frequency -- simply the inverse of the length of the recording in seconds
    F=arange(minfreq,maxfreq+minfreq,minfreq) # Create frequencies evenly spaced from 0:minfreq:maxfreq
    F=append(0,F) # Add zero-frequency component    
    return F


ion()

wlength01       = 100;          # waveform length in milliseconds
fs = sampling_rate   = 100000;       # sampling rate    
frqband = [5000.0, 40000.0];           # frequency range of noise in Hz
###### creating 100 ms long "white noise"
dt              = 1.0/fs;  
nsamples        = round(wlength01*(sampling_rate/1000));
hzperpoint      = (1.0/(nsamples*dt)); # resolutiuon of FFT, Hz
lowfrequency    = frqband[0]
highfrequency   = frqband[1]
lowcut          = round(lowfrequency/hzperpoint)+1;
highcut         = round(highfrequency/hzperpoint)+1;
numcomponents   = (highcut-lowcut)+1;

num_samples = nsamples
samples = numpy.random.standard_normal(size=num_samples)

tvec = numpy.linspace(0,wlength01,nsamples)

useFFT = False


#
F=timefreq(samples,fs) # frequencies
P = abs(fft(samples))**2 # FFT power
P=P[0:size(F,0)] # Trim so they're always the same size    
kernel=array([0.25,0.5,0.25]) # Convolution kernel
smooth = 100
PSmoothed = P
for q in range(smooth): PSmoothed=convolve(PSmoothed,kernel,'same') # Convolve FFT with kernel  
prng = (0, max(max(P),max(PSmoothed)))

#
def calcDrawFFT ():
  #
  subplot(2,3,1)
  title('Signal')
  plot(tvec,samples); xlabel('Time (ms)'); ylabel('Amplitude')
  subplot(2,3,2)
  plot(F,P); xlabel('Frequency (Hz)'); title('Signal FFT Power')
  plot(F,ones(len(P))*mean(P),'r--')
  ylim(prng)
  subplot(2,3,3)
  plot(F,PSmoothed); xlabel('Frequency (Hz)'); title('Signal FFT Power (smoothed)')
  plot(F,ones(len(PSmoothed))*mean(PSmoothed),'r--')
  ylim(prng)
  # now do bandpass
  samplesB = bandpass(samples, frqband[0], frqband[1], fs, zerophase=True)
  #
  P = abs(fft(samplesB))**2 # FFT power
  P=P[0:size(F,0)] # Trim so they're always the same size    
  kernel=array([0.25,0.5,0.25]) # Convolution kernel
  smooth = 100
  PSmoothed = P
  for q in range(smooth): PSmoothed=convolve(PSmoothed,kernel,'same') # Convolve FFT with kernel  
  prng = (0, max(max(P),max(PSmoothed)))
  #
  subplot(2,3,4)
  title('Bandpassed Signal (' + str(frqband[0]) + '-' + str(frqband[1]) + ' Hz)')
  plot(tvec,samples); xlabel('Time (ms)'); ylabel('Amplitude')
  subplot(2,3,5)
  plot(F,P); xlabel('Frequency (Hz)'); title('Bandpassed Signal FFT Power')
  plot(F,ones(len(P))*mean(P),'r--')
  plot([frqband[0],frqband[0]],[prng[0],prng[1]],'k--')
  plot([frqband[1],frqband[1]],[prng[0],prng[1]],'k--')
  ylim(prng)
  subplot(2,3,6)
  plot(F,PSmoothed); xlabel('Frequency (Hz)'); title('Bandpassed Signal FFT Power (smoothed)')
  plot(F,ones(len(PSmoothed))*mean(PSmoothed),'r--')
  plot([frqband[0],frqband[0]],[prng[0],prng[1]],'k--')
  plot([frqband[1],frqband[1]],[prng[0],prng[1]],'k--')
  ylim(prng)

#
def calcDrawWavelets (ms=None):
  dt = tvec[1] - tvec[0]
  tstop = tvec[-1]
  prm = {'f_max_spec':frqband[1]+frqband[1]/10.0,'dt':dt,'tstop':tstop}
  if ms is None: ms = MorletSpec(tvec,samples,None,None,prm)
  wpsd = np.mean(ms.TFR,axis=1)
  subplot(1,3,1)
  title('Signal')
  plot(tvec,samples); xlabel('Time (ms)'); ylabel('Amplitude')
  subplot(1,3,2)
  plot(ms.f,wpsd); xlabel('Frequency (Hz)'); title('Morlet Wavelet Time-Averaged Power')
  subplot(1,3,3)
  title('Morlet Wavelet Spectrogram')
  ylabel('Frequency (Hz)'); xlabel('Time (ms)')
  imshow(ms.TFR, extent=[tvec[0], tvec[-1], ms.f[-1], ms.f[0]], aspect='auto', origin='upper',cmap=plt.get_cmap('jet'))
  return ms


if useFFT:
  print('calculating FFT...')
  calcDrawFFT()
else:
  print('calculating wavelets...')
  ms = calcDrawWavelets()


