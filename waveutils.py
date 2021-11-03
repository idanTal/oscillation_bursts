import wave, struct
from scipy import io
from pylab import *
import numpy
import math
from psd import *
from filter import lowpass, envelope
from filt import hilb
import soundfile as sf
import numpy as np
import os

def readwav (fn):
  dat, sampr = sf.read(fn)
  return dat, sampr

def writewav (dat, fn, sampr):
  sf.write(fn, dat, sampr)

#
def modulate (dat, fcarrier, sampr, Ac = 1.0):
  aout = []; carr = []
  for i in xrange(len(dat)):
    carrier_sample = math.cos(fcarrier * (i / double(sampr)) * math.pi * 2)
    signal_am = signal_amsc = dat[i] * carrier_sample
    signal_am += carrier_sample
    signal_am /= 2
    aout.append(signal_am)
    carr.append(carrier_sample)
  return aout, carr

#
def demodulate (dat): # , fcarrier, sampr):
  dout = []
  for n in xrange(len(dat)):
    dout.append(abs(dat[n]))
  return dout

lfile = ['rfire.wav', 'pcr.wav', 'kars.wav', 'pierre.wav']

#
def extract1chan (fn):
  print(fn)
  dat,sampr = sf.read(fn)
  if len(dat.shape) > 1:
    return dat[:,0], sampr
  else:
    return dat, sampr

#
def saveall1chan (lfile):
  for fn in lfile:
    p = os.path.join('data',fn)
    dat,sampr = extract1chan(p)
    writewav(dat,p.split('.wav')[0]+'_0.wav',sampr)

fn = 'data/rfire_0.wav'
# f = sf.SoundFile(fn)
# len(f), f.channels, f.samplerate # (746339, 2, 44100)

ion()
# plot(dat,'b'); plot(datlow,'r') # compare original with low-pass filtered signal

#
def testmoddemod (fn, lowf=8000.0, fcarrier=20000.0):
  dat, sampr = readwav(fn)
  datlow = lowpass(dat, lowf, df=sampr, zerophase=True)

  fout = fn.split('.wav')[0] + '_low.wav'
  writewav(datlow,os.path.join('data',fout),sampr) # low-pass filtered version

  # amod = modulate(dat,fcarrier,sampr)
  amod, carr = modulate(datlow,fcarrier,sampr, Ac = 1.0)
  fout = fout.split('.wav')[0] + '_am.wav'
  writewav(amod,os.path.join('data',fout),sampr)

  demod = demodulate(amod)
  demod = lowpass(demod, lowf/4, df=sampr, zerophase=True) # lowpass(dat, lowf, df=sampr, zerophase=True)
  #demod = lowpass(demod, lowf, df=sampr, zerophase=True) # lowpass(dat, lowf, df=sampr, zerophase=True)
  demod = 2.0 * (demod - np.mean(demod))
  # flow

  #subplot(2,1,1); plot(carr)
  #subplot(2,1,2);
  ns = 5000
  plot(datlow[0:ns],'b'); # plot(amod[0:ns],'g'); 
  plot(demod[0:ns],'r') # compare original with demodulated

  # fout
  writewav(demod,'data/demod.wav',sampr)  

# https://epxx.co/artigos/ammodulation.html

def test ():
  # modulated = wave.open("amsc.wav", "r")
  modulated = wave.open(fn,'r')

  demod_amsc_ok = wave.open("demod_amsc_ok.wav", "w")
  demod_amsc_nok = wave.open("demod_amsc_nok.wav", "w")
  demod_amsc_nok2 = wave.open("demod_amsc_nok2.wav", "w")

  cfreq = 96000.0 # 2*96000.0 # 44100.0

  for f in [demod_amsc_ok, demod_amsc_nok, demod_amsc_nok2]:
      f.setnchannels(1)
      f.setsampwidth(2)
      f.setframerate(int(cfreq))

  cutoff = 25.0 # 12000.0 # 6000.0 # 3000.0

  for n in range(0, modulated.getnframes()):
      signal = struct.unpack('h', modulated.readframes(1))[0] / 32768.0
      carrier = math.cos(cutoff * (n / cfreq) * math.pi * 2)
      #carrier_phased = math.sin(3000.0 * (n / cfreq) * math.pi * 2)
      #carrier_freq = math.cos(3100.0 * (n / cfreq) * math.pi * 2)

      base = signal * carrier
      #base_nok = signal * carrier_phased
      #base_nok2 = signal * carrier_freq

      demod_amsc_ok.writeframes(struct.pack('h', base * 32767))
      #demod_amsc_nok.writeframes(struct.pack('h', base_nok * 32767))
      #demod_amsc_nok2.writeframes(struct.pack('h', base_nok2 * 32767))

"""
f = demod_am = wave.open("demod_am.wav", "w")
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(96000)

ns = 2 * 32768.0

modulated = wave.open(fn,'r')
for n in range(0, modulated.getnframes()):
    signal = struct.unpack('h', modulated.readframes(1))[0] / ns
    signal = abs(signal)
    demod_am.writeframes(struct.pack('h', signal * int(ns-1)))




"""

if __name__ == '__main__':
  dat, sampr = readwav(fn)
  lowf = 8000.0 # 1500.0
  datlow = lowpass(dat, lowf, df=sampr, zerophase=True)
  writewav(dat,'data/rfire0.wav',sampr) # single channel from original
  writewav(datlow,'data/rfire0low.wav',sampr) # single channel from original
  fcarrier = 20000.0 # 80000.0 # 44000.0
  # amod = modulate(dat,fcarrier,sampr)
  amod, carr = modulate(datlow,fcarrier,sampr, Ac = 1.0)
  writewav(amod,'data/rfire0lowam.wav',sampr)
  demod = demodulate(amod)
  demod = lowpass(demod, lowf/4, df=sampr, zerophase=True) # lowpass(dat, lowf, df=sampr, zerophase=True)
  #demod = lowpass(demod, lowf, df=sampr, zerophase=True) # lowpass(dat, lowf, df=sampr, zerophase=True)
  demod = 2.0 * (demod - np.mean(demod))
  # flow
  #subplot(2,1,1); plot(carr)
  #subplot(2,1,2);
  ns = 5000
  plot(datlow[0:ns],'b'); # plot(amod[0:ns],'g'); 
  plot(demod[0:ns],'r') # compare original with demodulated
  writewav(demod,'data/demod.wav',sampr)  

