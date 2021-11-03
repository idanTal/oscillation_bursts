from load import *
from netpyne import sim

#            File,               Area,S1a,S1b,S2a,S2b,G1a,G1b,I1a,I1b,I2a,I2b
# A1_v32_batch5_v32_batch5_0.pkl ,A1,  0,  1,  2,  9,  10,11, 12,  15,16, 19
dlyr = {'S1a':0,'S1b':1,'S2a':2,'S2b':9,'G1a':10,'G1b':11,'I1a':12,'I1b':15,'I2a':16,'I2b':19} # use S2a, G1a, I1b for supra, gran, infragran
dlyr[dlyr['S2a']] = 'supragranular'; dlyr[dlyr['G1a']] = 'granular'; dlyr[dlyr['I1b']] = 'infragranular'

def loadone (fn):  
  sim.load(fn,instantiate=False) # fn should be .pkl netpyne sim file 
  lfp_data = np.array(sim.allSimData['LFP']) # LFP data from sim
  dt = sim.cfg.recordStep/1000.0 # default unit is ms -- /1000 for conversion to seconds # recording time step (default is ms, divide by 1000 for sec -- like macaque data)
  sampr = 1./dt  # sampling rate (Hz) 
  spacing_um =  sim.cfg.recordLFP[1][1] - sim.cfg.recordLFP[0][1] # spacing between electrodes in microns
  CSD = getCSD(lfps=lfp_data,sampr=sampr,spacing_um=spacing_um,vaknin=False) # getCSD() in nhpdat.py 
  fullTimeRange = [0,(sim.cfg.duration/1000.0)] #[0,(sim.cfg.duration/1000.0)] # sim.cfg.duration is a float # timeRange should be the entire sim duration 
  tt = numpy.arange(fullTimeRange[0],fullTimeRange[1],dt)  # this is in MILLISECONDS -- not like MACAQUE DATA, which is in SECONDS
  dat = CSD
  return CSD, tt, sampr


def getspec (CSD, sampr, chan, freqmin, freqmax, freqstep, startt, endt):
  sidx,eidx = int(startt*sampr),int(endt*sampr)
  sig = CSD[chan,sidx:eidx]-mean(CSD[chan,sidx:eidx])
  ms = MorletSpec(sig,sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep)
  msn = mednorm(ms.TFR)
  return msn

def plotsigspec (fn, CSD, tt, sampr, lchan, lmsn, freqmin, freqmax, startt, endt):
  sidx,eidx = int(startt*sampr),int(endt*sampr)
  gdx = 1
  for chan , msn in zip(lchan, lmsn):
    subplot(3,2,gdx)
    plot(tt[sidx:eidx],CSD[chan,sidx:eidx],'k',linewidth=2); xlim((tt[sidx],tt[eidx])); title(fn)
    subplot(3,2,gdx+1)
    imshow(msn,extent=[tt[sidx],tt[eidx],freqmin,freqmax],aspect='auto',origin='lower',cmap=plt.get_cmap('jet'),interpolation=None);
    title(dlyr[chan] + ' CSD')
    xlim((tt[sidx],tt[eidx]))
    gdx+=2

    
