from nhpdat import *
from csd import *
import numpy as np
from waveutils import *
from load import noiseampLFP,noiseampCSD
from morlet import MorletSpec
from scipy.stats import pearsonr
from scipy.signal import hilbert

# separating functions for speech analysis here

def envelope (x): return np.abs(hilbert(x))

#
def getdsig (LFP,CSD,chan,trigtimes,trigIDs,dwavedur,sampr,minf=-1,maxf=-1,noiseampLFP=noiseampLFP,noiseampCSD=noiseampCSD,aftertrig=False,aftertrigdur=1.0,othersig=None):
  # returns the LFP,CSD signals from specific channel (chan) during each stimulus in a dictionary indexed by stimulus ID (trigger ID) as dsig
  # each element in dsig is a list
  # when minf,maxf are specified filters the signals
  # 
  highampskipLFP = highampskipCSD = 0
  dsig = {}
  for whichtrigid in range(min(trigIDs),max(trigIDs)+1,1): dsig[whichtrigid] = []
  for i,tstart in enumerate(trigtimes):
    sidx,eidx = tstart, tstart + dwavedur[trigIDs[i]][1]
    if aftertrig and aftertrigdur > 0.0: # use period after stimulus?
      sidx = eidx
      eidx = min(LFP.shape[0]-1, sidx + int(aftertrigdur * sampr))
    x = LFP[sidx:eidx,chan]
    if max(abs(x)) > noiseampLFP: # skip high amplitude LFP
      highampskipLFP+=1
      continue # get rid of high amplitude noise
    x = CSD[chan,sidx:eidx]
    if max(abs(x)) > noiseampCSD: # skip high amplitude CSD
      highampskipCSD += 1
      continue
    if minf > -1 and maxf > -1: 
      if minf == maxf:
        dsig[trigIDs[i]].append(lowpass(x,minf,sampr,zerophase=True))
      else:
        dsig[trigIDs[i]].append(bandpass(x,minf,maxf,sampr,zerophase=True))
    elif othersig is None:
      dsig[trigIDs[i]].append(x)
    else:
      dsig[trigIDs[i]].append(othersig[sidx:eidx]) # can use to store other signal, e.g. amplitude envelope of gamma
  print('high amp skip LFP',highampskipLFP,'high amp skip CSD',highampskipCSD)
  return dsig

#
def getdsigavg (trigIDs,dsig):
  dsigavg = {}
  for i in range(min(trigIDs),max(trigIDs)+1,1):
    npdat = np.array(dsig[i])
    dsigavg[i] = mean(npdat,axis=0)
  return dsigavg

#
def getdmspec (ddt,sampr,minf=0.5,maxf=9.0,stepf=0.25):
  # returns the CSD wavelet spectrograms during each stimulus in a dictionary indexed by stimulus ID (trigger ID) as dsig
  # each element in dsig is a list
  # minf,maxf,stepf are the bandwidths/steps for the wavelet spectrogram
  dms = {}
  for stim in ddt.keys():
    print('stim:',stim)
    dms[stim] = []
    for sig in ddt[stim]:
      dms[stim].append(MorletSpec(sig,sampr,freqmin=minf,freqmax=maxf,freqstep=stepf))
  return dms

def getdmspecfull (ms,trigtimes,trigIDs,dwavedur):
  dms = {}
  for whichtrigid in range(min(trigIDs),max(trigIDs)+1,1): dms[whichtrigid] = []
  for i,tstart in enumerate(trigtimes):
    sidx,eidx = tstart, min(tstart + dwavedur[trigIDs[i]][1], ms.shape[1])
    x = ms[:,sidx:eidx]
    dms[trigIDs[i]].append(x)
  return dms
  

"""
#
dcgram = {}
for k in dwave.keys():
  sound = Sound(dwave[k][0][:,0], samplerate=samprds*Hz)
  dcgram[int(k.split('_')[0])] = getcgram(sound,maxf=(samprds/2.0)*Hz)

cf = dcgram[1][0]

#
dcgramtheta = {}; dcgramlowf = {}
for i in range(1,17,1):
  cres = sum(dcgram[i][2].T,axis=0)
  dcgramtheta[i] = bandpass(cres,3,9,sampr,zerophase=True)
  dcgramlowf[i] = lowpass(cres,max(20.0,sampr/2.0-500),sampr,zerophase=True)
"""

"""
#
def getddtwtrials (dsig,lstim=[1,6,12,15]):
  ddtwtrials = {}
  for idx in range(len(lstim)):
    i = lstim[idx]
    for jdx in range(idx,len(lstim),1):
      j = lstim[jdx]
      ddtwtrials[(i,j)] = []
      print(i,j)
      for sdx1,sig1 in enumerate(dsig[i]):
        nsig1 = normarr(sig1)
        for sdx2,sig2 in enumerate(dsig[j]):
          if i == j and sdx2 <= sdx1: continue
          nsig2 = normarr(sig2)
          d,p = fastdtw(nsig1,nsig2)
          ddtwtrials[(i,j)].append(d)
  return ddtwtrials
"""

#
def getpearsonrtrials (dsig,nsamp,lstim=[1,6,12,15]):
  # get pearson correlation between individual signals within and across trials
  drtrials = {}
  for idx in range(len(lstim)):
    i = lstim[idx]
    for jdx in range(idx,len(lstim),1):
      j = lstim[jdx]
      drtrials[(i,j)] = []
      print(i,j,idx,jdx)
      for sdx1,sig1 in enumerate(dsig[i]):
        nsig1 = sig1[0:nsamp]
        for sdx2,sig2 in enumerate(dsig[j]):
          if i == j and sdx2 <= sdx1: continue
          nsig2 = sig2[0:nsamp]
          r = pearsonr(nsig1,nsig2)[0]
          drtrials[(i,j)].append(r)
  return drtrials

#
def plotdsigavg (dsig,dsigavg,trigIDs):
  for i in range(min(trigIDs),max(trigIDs)+1,1):
    subplot(2,8,i)
    for sig in dsig[i]:
      ttt = linspace(0,len(sig)*dt,len(sig))
      plot(ttt,normarr(sig),'gray')
    ttt = linspace(0,len(dsigavg[i])*dt,len(dsigavg[i]))
    plot(ttt,normarr(dsigavg[i]),linewidth=3,color='black')
    xlim((0,ttt[-1]))
    title('CSD for signal' + str(i)); xlabel('Time (s)'); ylabel('Amplitude')

#
def getdtwscore (ddtwtrials,lstim):
  dscore = {}
  dnum = {}
  dden = {}
  for i in lstim:
    dnum[i] = 0.0
    dden[i] = []
  for i in lstim:
    for j in lstim:
      if (i,j) in ddtwtrials:
        print(i,j,mean(ddtwtrials[i,j]),'+/-',std(ddtwtrials[i,j])/sqrt(len(ddtwtrials[i,j])))
        if i == j:
          dnum[i] = mean(ddtwtrials[i,j])
        else:
          m = mean(ddtwtrials[i,j])
          dden[i].append(m)
          dden[j].append(m)
  for i in lstim:
    dscore[i] = dnum[i] / mean(dden[i])
    print(i,dscore[i],dnum[i],dden[i])
  return dscore

#
def getrscore (ddtwtrials,lstim):
  dscore = {}
  dnum = {}
  dden = {}
  for i in lstim:
    dnum[i] = 0.0
    dden[i] = []
  for i in lstim:
    for j in lstim:
      if (i,j) in ddtwtrials:
        #print i,j,mean(ddtwtrials[i,j]),'+/-',std(ddtwtrials[i,j])/sqrt(len(ddtwtrials[i,j]))
        if i == j:
          dnum[i] = mean(ddtwtrials[i,j])
        else:
          m = mean(ddtwtrials[i,j])
          dden[i].append(m)
          dden[j].append(m)
  for i in lstim:
    dscore[i] = dnum[i] / mean(dden[i])
    print(i,dscore[i],dnum[i],dden[i])
  return dnum,dden,dscore

#
def loadwavedat (d,samprds=0):
  # load wave file data
  dout = {}
  for f in os.listdir(d):
    if f.endswith('.wav'):
      dout[f] = readwav(os.path.join(d,f))
      if samprds > 0.0:
        #dsfctr = dout[f][1]/samprds
        ldat = []
        #for ch in [0,1]: ldat.append(scipy.signal.decimate(dout[f][0][:,ch], int(dsfctr)))
        for ch in [0,1]: ldat.append(downsample(dout[f][0][:,ch], dout[f][1],samprds))
        dout[f] = np.array(ldat).T,samprds
  dwavedur = {}
  for k in dout.keys():
    wavedat,wavesampr = dout[k]
    wavedur = (len(wavedat) / float(wavesampr))  # duration in seconds
    dwavedur[k] = dwavedur[int(k.split('_')[0])] = wavedur, int(wavedur*samprds)
  return dout, dwavedur


def plotdsigavg (ddt, ddta, chan):
  # plots the signals, their averages
  for i in range(1,17,1):
    subplot(2,8,i)
    for sig in ddt[chan][i]:
      ttt = linspace(0,len(sig)*dt,len(sig))
      plot(ttt,normarr(sig),'gray')
    ttt = linspace(0,len(ddta[chan][i])*dt,len(ddta[chan][i]))
    plot(ttt,normarr(ddta[chan][i]),linewidth=3,color='black')
    xlim((0,ttt[-1]))
    title('CSD:signal' + str(i))
  
# get cochleogram
if int(sys.version[0])==2:
  def getcgram (sound,minf=0*Hz,maxf=1*kHz,nchan=100):
    cf = erbspace(minf,maxf,nchan)
    fb = Gammatone(sound, cf)
    output = fb.process()
    return cf,fb,output

if __name__ == '__main__':

  # load the auditory stimuli

  samprds = sampr = 2000 # (can use 22 kHz for wavelets on soundwaves, otherwise lower resolution)
  
  dwave, dwavedur = loadwavedat('data/robin_speech/stimuli',samprds=samprds)
  print(dwave.keys())
  wavedat,wavesampr = dwave['01_ba_peter.wav']
  
  # this one is for speech decoding
  fn = 'data/robin_speech/contproc/2-rb051052020@os.mat' #'data/robin_speech/contproc/2-rb049050022@os.mat'

  divby = getorigsampr(fn) / samprds
  trigtimes = [int(round(x)) for x in np.array(getTriggerTimes(fn)) / divby] # div by 22 since downsampled by factor of 22
  trigIDs = remaptrigIDs(getTriggerIDs(fn))

