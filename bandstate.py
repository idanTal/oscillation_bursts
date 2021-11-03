"""
uses spectrogram and hilbert transform to determine states based on presence/absence (high/low) of power in specific frequency bands
"""

from modindex import *
from bbox import p2d

#
def getdspec (CSD,sampr,lchan,cutt=3):
  lfreq,lfwidth,lbwcfc = getlfreqwidths(step=0.5)
  dspec = {}
  dspec['lfreq'] = lfreq
  dspec['lfwidth'] = lfwidth
  dspec['lbwcfc'] = lbwcfc
  for chan in lchan:
    print('chan ' , chan)
    dspec[chan] = hilblist(CSD[chan],sampr,lfreq,lfwidth,cutt)
  tt = linspace(0,len(dspec[lchan[0]][0][0])/sampr,len(dspec[lchan[0]][0][0]))
  dspec['tt'] = tt
  return dspec

#
def getchanrelpowinrange (dspec,lfreq,lchan,tdur,sampr,norm=True):
  dlpr = {}
  for chan in lchan:
    for b in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'hgamma']:
      minF,maxF = dbands[b]; 
      print(chan,b,minF,maxF)
      dlpr[chan,b] = relpowinrange(dspec[chan][0],lfreq,minF,maxF,tdur,sampr,norm)
  lpow = dlpr[lchan[0],'alpha']
  dlpr['tt'] = linspace(0, len(lpow) * tdur , len(lpow) )
  return dlpr

#
def getdspk (dlpr,lchan,band,stdthresh):
  dspk = {}
  for chan in lchan:
    dspk[chan] = getlspk(dlpr[chan,band],mean(dlpr[chan,band])+stdthresh*std(dlpr[chan,band]))
  return dspk

#
def getbandbin (lchan,dspk,dlpr,band):
  dbandbin = {}
  sz = len(dlpr.values()[0])
  for chan in lchan:
    xx = zeros((sz,))
    for spk in dspk[chan]:
      #if spk.peak.x > 0: xx[spk.peak.x-1]=1
      xx[spk.peak.x]=1
      #if spk.peak.x+1 < len(xx): xx[spk.peak.x+1]=1
    dbandbin[chan] = xx
  return dbandbin

#
def getlbin (dat,lspk,rad):
  sz = len(dat)
  xx = zeros((sz,))
  for spk in lspk:
    xx[spk.peak.x]=1
    for idx in range(max(0,spk.peak.x-rad),min(spk.peak.x+rad,sz),1): xx[idx]=1
  return xx

def getnoisebin (fn,dsfctr,th,rad):
  csdavga,tt=getavgCSDhilbAMP(fn,dsfctr=dsfctr);
  lnoisespk = getlspk(csdavga,th)
  lbin = getlbin(csdavga,lnoisespk,int(rad*50/dsfctr))
  return lbin

# get the CFC in pairs of bands (npMI is mod index array vs time)
# lfreq is the list of frequencies used to generate the CFC
def getbandCFC (npMI,lfreq,minPF,minAF):
  dbidx = {}
  for k in dbands.keys(): dbidx[k]=minmaxIDX(k,lfreq)
  dsum = {}; pfsub,afsub = firstIDX(minPF,lfreq), firstIDX(minAF,lfreq);
  for b1 in ['delta', 'theta', 'alpha']:
    lo1,hi1 = dbidx[b1]; lo1-=pfsub; hi1-=pfsub;
    for b2 in ['gamma', 'hgamma']:
      lo2,hi2 = dbidx[b2]; lo2-=afsub; hi2-=afsub;
      dsum[b1+'_'+b2] = []
      for tt in range(npMI.shape[0]):
        dsum[b1+'_'+b2].append(mean(npMI[tt][lo2:hi2,lo1:hi1]))
  return dsum

# get the amplitude in bands (npAMP is hilbert transform amplitude time series)
# lfreq is the list of frequencies used to generate the hilbert transformed amplitudes
# winsz is number of samples to integrate over
def getbandAMP (npAMP,winsz,lfreq):
  dbidx = {}
  for k in dbands.keys(): dbidx[k]=minmaxIDX(k,lfreq)
  dsum = {}
  maxt = int(npAMP.shape[1] / winsz)
  sz=npAMP.shape[1]
  for b1 in ['delta', 'theta', 'alpha', 'gamma', 'hgamma']:
    lo1,hi1 = dbidx[b1]
    dsum[b1] = []
    sidx,eidx=0,winsz;
    for tt in range(maxt):
      dsum[b1].append(mean(npAMP[lo1:hi1+1,sidx:eidx]))
      sidx+=winsz; eidx+=winsz;
      if sidx >= sz or eidx >= sz: break
  return dsum

# get downsampled csd
def getCSDds (fn,dsfctr=20):
  sampr,dat,dt,tt=rdmat(fn);
  CSD=getCSD(dat,sampr);
  CSDds = zeros((CSD.shape[0],int(CSD.shape[1]/dsfctr))) # downsampled CSD
  tmax = tt[-1] # use original sampling rate for tmax - otherwise shifts phase
  #for i in range(CSD.shape[0]): CSDds[i,:]=scipy.signal.resample(CSD[i], len(CSD[i]) / dsfctr)
  samprds = sampr / dsfctr 
  for i in range(CSD.shape[0]): CSDds[i,:]=downsample(CSD[i], sampr, samprds)
  ttds = linspace(0,tmax,len(CSDds[0]))
  del CSD # free mem
  return samprds,CSDds,ttds

#
def getavgCSDhilbAMP (fn,dsfctr=1):
  if dsfctr == 1:
    sampr,dat,dt,tt=rdmat(fn); # read the data
    CSD = getCSD(dat,sampr)
    del dat,dt
  else:
    sampr,CSD,tt = getCSDds(fn,dsfctr)
  csdlhamp = []
  for i in range(CSD.shape[0]):
    print('ch ' , i , ' of ' , CSD.shape[0]-1)
    xf = CSD[i,:]  
    hamp,hang = hilb(xf)
    csdlhamp.append(hamp)
    del hang
  acsdlhamp = numpy.array(csdlhamp)
  del csdlhamp
  csdavga = numpy.mean(acsdlhamp,axis=0)
  del acsdlhamp,CSD
  return csdavga,tt

# 
def getnqmuaamp (lf,tdur,dsfctr,stdth,cutt):
  nqmuaamp = h.NQS('fidx','s2on','gon','i2on','s2deltatheta','gdeltatheta','i2deltatheta','s2alpha','galpha','i2alpha','s2gamma','ggamma','i2gamma','s2muaamp','gmuaamp','i2muaamp','s2state','gstate','i2state','noise')
  for fidx,fn in enumerate(lf):
    s1,s2,g,i1,i2=getflayers(fn.split('data/')[1])
    if s1 == -1:
      print('could not find layers for ', fn)
      continue
    try:
      samprds,CSDds,ttds = getCSDds(fn,dsfctr=dsfctr)
      print('read ' , fn)
    except:
      print('could not read ' , fn)
      continue
    noisebin = getnoisebin(fn,dsfctr,100.0,500)
    lfreq,lfwidth,lbwcfc = getlfreqwidths(step=0.5)
    dspec = getdspec(CSDds,samprds,[s2,g,i2])
    sampr,dat,dt,tt=rdmat(fn); # read the data
    MUA=getMUA(dat,sampr)
    dlpr = getchanrelpowinrange(dspec,lfreq,[s2,g,i2],tdur,samprds,False)
    dspk = getdspk(dlpr,[s2,g,i2],'alpha',stdth)
    dbandbin = getbandbin([s2,g,i2],dspk,dlpr,'alpha')
    sz = len(dbandbin[s2])
    for tdx in range(sz):
      s2on = dbandbin[s2][tdx]
      gon = dbandbin[g][tdx]
      i2on = dbandbin[i2][tdx]
      s2deltatheta = dlpr[s2,'delta'][tdx] + dlpr[s2,'theta'][tdx]
      gdeltatheta = dlpr[g,'delta'][tdx] + dlpr[g,'theta'][tdx]
      i2deltatheta = dlpr[i2,'delta'][tdx] + dlpr[i2,'theta'][tdx]
      s2alpha = dlpr[s2,'alpha'][tdx]
      galpha = dlpr[g,'alpha'][tdx]
      i2alpha = dlpr[i2,'alpha'][tdx]
      s2gamma = dlpr[s2,'gamma'][tdx] + dlpr[s2,'hgamma'][tdx]
      ggamma = dlpr[g,'gamma'][tdx] + dlpr[g,'hgamma'][tdx]
      i2gamma = dlpr[i2,'gamma'][tdx] + dlpr[i2,'hgamma'][tdx]
      s2muaamp=mean(MUA[s2-1,int(cutt*sampr+tdx*tdur*sampr):int(cutt*sampr+(tdx+1)*tdur*sampr)])
      gmuaamp=mean(MUA[g-1,int(cutt*sampr+tdx*tdur*sampr):int(cutt*sampr+(tdx+1)*tdur*sampr)])
      i2muaamp=mean(MUA[i2-1,int(cutt*sampr+tdx*tdur*sampr):int(cutt*sampr+(tdx+1)*tdur*sampr)])
      if s2deltatheta > s2alpha: 
        s2state = 0 
      else: 
        s2state = 1
      if gdeltatheta > galpha: 
        gstate = 0 
      else: 
        gstate = 1
      if i2deltatheta > galpha: 
        i2state = 0 
      else: 
        i2state = 1
      nqmuaamp.append(fidx,s2on,gon,i2on,s2deltatheta,gdeltatheta,i2deltatheta,s2alpha,galpha,i2alpha,s2gamma,ggamma,i2gamma,s2muaamp,gmuaamp,i2muaamp,s2state,gstate,i2state,noisebin[int(tdx*tdur*samprds)])
    del CSDds,MUA,dat,tt
  return nqmuaamp

#
def plotnqmcor (nqm,clr='k',band='alpha'):
  lc1 = ['s2'+band,'g'+band,'i2'+band]
  lc2 = ['s2muaamp','gmuaamp','i2muaamp']
  for i,c1 in enumerate(lc1):
   figure()
   for j,c2 in enumerate(lc2):
     subplot(1,3,j+1)
     pr = pearsonr(nqm.getcol(c1),nqm.getcol(c2))
     r,p = pr[0],pr[1]
     xlabel(c1); ylabel(c2); title('r = ' + str(round(r,3)))# + ' , p = ' + str(round(p,6)))
     plot(nqm.getcol(c1),nqm.getcol(c2),clr+'o')

# get minima,maxima in integrated power time-series
def getpowlocalMinMax (lpow,th = None):
  lpowMAX,lpowMIN = zeros(len(lpow)),zeros(len(lpow))
  sz = len(lpow)
  maxx,maxy,minx,miny=[],[],[],[]
  if th is None: th = median(lpow) # dynamic threshold
  for j in range(1,sz-1,1):
    if lpow[j] > th and lpow[j] > lpow[j-1] and lpow[j] > lpow[j+1]:
      lpowMAX[j]=1
      maxx.append(j)
      maxy.append(lpow[j])
    if lpow[j] <= th and lpow[j] < lpow[j-1] and lpow[j] < lpow[j+1]:
      lpowMIN[j]=1
      minx.append(j)
      miny.append(lpow[j])
  return lpowMIN,lpowMAX,minx,miny,maxx,maxy

# get indices splitting the data into low/high power for the band
def splitBYBandPow (ddat,minF,maxF):
  llpow,llpowMIN,llpowMAX = [],[],[]
  lminx,lminy,lmaxx,lmaxy = [],[],[],[]
  for fn in ddat.keys():
    print(fn)
    F = ddat[fn]['F']
    lspec = ddat[fn]['nplsp']
    lpow=powinrange(lspec,F,minF,maxF); llpow.append(lpow)
    lpowMIN,lpowMAX,minx,miny,maxx,maxy=getpowlocalMinMax(lpow)
    llpowMIN.append(lpowMIN); llpowMAX.append(lpowMAX); lminx.append(minx);
    lminy.append(miny); lmaxx.append(maxx); lmaxy.append(maxy);
  dout={}
  dout['llpow']=llpow;
  dout['llpowMIN']=llpowMIN; dout['llpowMAX']=llpowMAX;
  dout['lminx']=lminx; dout['lminy']=lminy;
  dout['lmaxx']=lmaxx; dout['lmaxy']=lmaxy;
  return dout

#
def getphtrigavg (fn,dsfctr,minf,maxf,wins=0.25):
  sampr,dat,dt,tt=rdmat(fn); MUA=getMUA(dat,sampr)
  s1,s2,g,i1,i2=getflayers(fn.split('data/')[1])
  samprds,CSDds,ttds = getCSDds(fn,dsfctr=dsfctr)
  dband = {}
  for ch in [s2,g,i2]: dband[ch] = bandpass(CSDds[ch,:],minf,maxf,df=samprds,zerophase=True)
  dminmax = {}
  for ch in dband.keys(): dminmax[ch] = getpowlocalMinMax(dband[ch])
  sz = int(sampr * 2.0 * wins) + 1
  szds = int(samprds * 2.0 * wins) + 1
  dphmin,dphmax = {},{}
  dphminband,dphmaxband = {},{}
  for ch1 in [s2,g,i2]:
    (lpowMIN,lpowMAX,minx,miny,maxx,maxy) = dminmax[ch1]
    for ch2 in [s2,g,i2]:
      dphmin[ch1,ch2] = zeros((sz,)); N = 0.0; myarr = dphmin[ch1,ch2]
      dphminband[ch1,ch2] = zeros((szds,)); NDS = 0.0; mycsd = dphminband[ch1,ch2]
      mua = MUA[ch2-1,:]; muasz = len(mua)
      CSD = CSDds[ch2,:]; csdsz = len(CSD)
      for x in minx:
        t0 = x*1.0/samprds
        sidx,eidx = int((t0-wins)*sampr), int((t0+wins)*sampr + 1)
        if eidx < muasz and sidx >= 0:
          myarr += mua[sidx:sidx+sz]; N += 1
        sidx,eidx = int((t0-wins)*samprds), int((t0+wins)*samprds + 1)
        if eidx < csdsz and sidx >= 0:
          mycsd += CSD[sidx:sidx+szds]; NDS += 1
      if N > 0: myarr /= N
      if NDS > 0: mycsd /= NDS
      dphmax[ch1,ch2] = zeros((sz,)); N = 0.0; myarr = dphmax[ch1,ch2]
      dphmaxband[ch1,ch2] = zeros((szds,)); NDS = 0.0; mycsd = dphmaxband[ch1,ch2]
      for x in maxx:
        t0 = x*1.0/samprds
        sidx,eidx = int((t0-wins)*sampr), int((t0+wins)*sampr + 1)
        if eidx < muasz and sidx >= 0:
          myarr += mua[sidx:sidx+sz]; N += 1
        sidx,eidx = int((t0-wins)*samprds), int((t0+wins)*samprds + 1)
        if eidx < csdsz and sidx >= 0:
          mycsd += CSD[sidx:sidx+szds]; NDS += 1
      if N > 0: myarr /= N
      if NDS > 0: mycsd /= NDS
  return dband,dminmax,dphmin,dphmax,dphminband,dphmaxband

#
def plotpha (tarr,dph,ch1,clr='b',norm=False):
  ltitle = ['s2','g','i2']
  yl = [1e9,-1e9]
  print(s2,g,i2)
  for ch2 in [s2,g,i2]:
    yl[0] = min(yl[0], amin(dph[ch1,ch2]))
    yl[1] = max(yl[1], amax(dph[ch1,ch2]))
  for i,ch2 in enumerate([s2,g,i2]): 
    subplot(3,1,i+1); ylabel(ltitle[i],fontsize=18)
    if norm:
      plot(tarr,dph[ch1,ch2]-mean(dph[ch1,ch2]),color=clr)
    else:
      plot(tarr,dph[ch1,ch2],color=clr)
    xlim((-250,250))
    #ylim((yl[0],yl[1]))
    if i == 2: xlabel('Time (ms)',fontsize=18)

# draws the CSD (or LFP) split by high/low power in a particular band. requires
# the lminx,lmaxx arrays from splitBYBandPow, and ddat (from rdspecgbatch)
def drawsplitbyBand (ddat,lminx,lmaxx,useCSD=True,ltit=['OFF','ON'],xls=(0,125),yls=(33,50)):
  csm=cm.ScalarMappable(cmap=cm.winter_r); csm.set_clim((0,1));
  avgON,avgOFF,cntON,cntOFF=[],[],[],[]; ii = 0; maxChan=19; minChan=1;
  chanSub,chanAdd=1,2; ylab='CSDSpec';
  if not useCSD: 
    chanSub=0; chanAdd=1; ylab='LFPSpec'; maxChan=23; minChan=0;
  for chan in range(minChan,maxChan,1):
    fn=ddat.keys()[0]; F=ddat[fn]['F'];
    print(chan)
    avgON.append(zeros((1,len(F)))); avgOFF.append(zeros((1,len(F))))
    cntON.append(0); cntOFF.append(0); fdx=0
    for fn in ddat.keys():
      ldat = ddat[fn]['nplsp']
      for i in range(chan-chanSub,chan+chanAdd,1):
        for j in lminx[fdx][i]:
          avgOFF[-1] += numpy.array(ldat[i][:,j]); cntOFF[-1] += 1
        for j in lmaxx[fdx][i]:
          avgON[-1] += numpy.array(ldat[i][:,j]); cntON[-1] += 1
      fdx += 1
    if cntON[-1]>0: avgON[-1] /= cntON[-1];
    if cntOFF[-1]>0: avgOFF[-1] /= cntOFF[-1];
    subplot(1,2,1); plot(F,avgOFF[-1].T,color=csm.to_rgba(float(chan)/(maxChan)),linewidth=1)
    xlabel('Frequency (Hz)'); ylabel(ylab); xlim(xls); ylim(yls); title(ltit[0]); grid(True)
    subplot(1,2,2); plot(F,avgON[-1].T,color=csm.to_rgba(float(chan)/(maxChan)),linewidth=1)
    xlabel('Frequency (Hz)'); ylabel(ylab); xlim(xls); ylim(yls); title(ltit[1]); grid(True)

    
INCS = 1.0 # increment in seconds
WINS = 1.0  # window size in seconds
#winsz = WINS*samprds; incsz = INCS*samprds; 
#lch = [s2,g,i2]
#s1,s2,g,i1,i2=getflayers(fn.split('data/')[1])

#
def hilblistmodArrVST (CSDds,samprds,lch,winsz,incsz,fstep=0.25):
  lfreq,lfwidth,lbw = getlfreqwidths(step=fstep)
  llhamp,llhang,lmodArr=[],[],[]
  for ch in lch:
    print('channel', ch)
    lhamp,lhang = hilblist(CSDds[ch],samprds,lfreq,lbw,3)
    llhamp.append(lhamp); llhang.append(lhang)
    lmodArr.append(modIndArrFromHilbListVST(lhamp,lhang,lfreq,winsz,incsz,0.5,14.0,25.0,115.0,18))
  return llhamp,llhang,lmodArr

# display a frame showing csd, power spec, cfc
def pframe (num,CSDds,llhamp,lmodArr,lch,samprds,yl2=(-45,45),vmaxamp=10,vmaxcfc=0.03):
  clf(); lt=['s2','g','i2']; 
  myt = linspace(num*INCS,num*INCS+WINS,samprds*WINS); 
  sidx = int(num*samprds*INCS)
  eidx = int(sidx + samprds*WINS)
  for i in range(len(lch)):
    subplot(3,3,i*3+1); ylabel(lt[i]);
    if i==0: title('CSD');
    plot(myt,CSDds[lch[i]][sidx:eidx]);
    xlim((myt[0],myt[-1])); ylim(yl2);
    if i==2: xlabel('t (s)'); 
    subplot(3,3,i*3+2);
    if i==0: title('specgram');
    imshow(llhamp[i][:,sidx:eidx],origin='lower',interpolation='None',extent=(myt[0],myt[-1],0,125),aspect='auto',vmax=vmaxamp);
    if i==2: xlabel('t (s)');
    colorbar(); ylim((0,115));
    subplot(3,3,i*3+3);
    if i==0: title('CFC');
    imshow(lmodArr[i][num,:,:],origin='lower',interpolation='None',extent=(.5,14,25,115),aspect='auto',vmax=vmaxcfc);
    if i==2: xlabel('Freq (Hz)');
    colorbar(); 

#
def getbandpowcfcnq (lch,lmodArr,llhamp,lfreq,winsz,minpf=0.5,minaf=25.0):
  dbidx,vtmp,dbandcfc,dbandpow,nqa = {},Vector(),{},{},NQS(); 
  for k in dbands.keys(): dbidx[k]=minmaxIDX(k,lfreq)
  for i in range(len(lch)):
    dbandcfc[lch[i]]=getbandCFC(lmodArr[i],lfreq,minpf,minaf)
    dbandpow[lch[i]]=getbandAMP(llhamp[i],int(winsz),lfreq)
  sch = ['s2','g','i2']
  for i in range(len(lch)):
   for k in dbandpow[lch[i]].keys():
     nqa.resize(sch[i]+'_'+k)
     vtmp.from_python(normarr(dbandpow[lch[i]][k]))
     nqa.v[int(nqa.m[0]-1)].copy(vtmp)
   for k in dbandcfc[lch[i]].keys():
     nqa.resize(sch[i]+'_'+k)
     vtmp.from_python(normarr(dbandcfc[lch[i]][k]))
     nqa.v[int(nqa.m[0]-1)].copy(vtmp)
  return dbandpow,dbandcfc,nqa

# print significant pearson correlations between columns of nqa
def prpearson (nqa,cutoff=0.01,negonly=False):
  for i in range(int(nqa.m[0])):
    xx = nqa.getcol(nqa.s[i].s).to_python()
    for j in range(i+1,int(nqa.m[0]),1):
      yy = nqa.getcol(nqa.s[j].s).to_python()
      pr = pearsonr(xx,yy)
      if pr[1] < cutoff:
        if negonly and pr[0] >= 0.0: continue
        print(nqa.s[i].s,' ',nqa.s[j].s,': r =',pr[0],', p =',pr[1])

#
class spike:
  def __init__ (self):
    self.left = p2d()
    self.peak = p2d()
    self.right = p2d()
  def __str__ (self):
    return 'L: ' + str(self.left) + ' PK: ' + str(self.peak) + ' R: ' + str(self.right)

#
def getlspk (dat, th):
  sz = len(dat)
  lspk = []
  for i in range(1,sz-1,1):
    if dat[i] >= th and dat[i] > dat[i-1] and dat[i] > dat[i+1]:
      spk = spike()
      spk.peak.x,spk.peak.y = i,dat[i]
      lx = i - 1
      while lx-1 > 0 and (dat[lx] > th or dat[lx]>spk.peak.y*0.5) and dat[lx] > dat[lx-1]: lx -= 1
      spk.left.x,spk.left.y = lx,dat[lx]
      rx = i + 1
      while rx+1 < sz and (dat[rx] > th or dat[rx]>spk.peak.y*0.5) and dat[rx] > dat[rx+1]: rx += 1
      spk.right.x,spk.right.y = rx,dat[rx]
      lspk.append(spk)
      i = rx + 1
  return lspk

# from /usr/site/nrniv/local/python/spectrogram.py - modified here
def getspec (tseries,rate=20000,window=1,maxfreq=125,tsmooth=0,fsmooth=0,winf=numpy.hanning,logit=False):
  from pylab import size, array, zeros, fft, convolve, r_    
  # Handle input arguments
  if maxfreq==0 or maxfreq>rate/2: maxfreq=rate/2 # Set maximum frequency if none entered or if too big
  npts=size(tseries,0) # How long the data are
  maxtime=npts/rate # Maximum time    
  ts = tseries - tseries.mean() # Remove mean
  print('Calculating spectra...')
  nwind=int(maxtime/window) # Number of windows
  lwind=int(window*rate) # Length of each window
  spectarr=zeros((lwind/2,nwind))
  if winf is None:
    for i in range(nwind):
      tstart=lwind*i # Initial time point
      tfinish=lwind*(i+1) # Final timepoint
      thists=ts[tstart:tfinish] # Pull out the part of the time series to make this spectrum
      spectarr[:,i]=abs(fft(thists))[0:lwind/2]
  else:
    winh = winf(lwind)
    for i in range(nwind):
      tstart=lwind*i # Initial time point
      tfinish=lwind*(i+1) # Final timepoint
      thists=ts[tstart:tfinish] # Pull out the part of the time series to make this spectrum
      tmp=winh*thists
      spectarr[:,i]=abs(fft(tmp))[0:lwind/2]
  if fsmooth > 0 or tsmooth > 0: smooth2D(spectarr,tsmooth,fsmooth) # Perform smoothing
  # Calculate time and frequency limits
  finalfreq=int(window*maxfreq)
  F=r_[0:finalfreq]/float(window)
  T=r_[0:nwind]*window
  if logit:
    return F,T,10*numpy.log10(spectarr[0:finalfreq,:])
  else:
    return F,T,spectarr[0:finalfreq,:]

# get CSD and associated specgrams (uses getspec). dat is a list of LFPs , eg from loadpair
def getCSDspec (dat,sampr,window=1,maxfreq=125,logit=True,tsmooth=0,fsmooth=0):
  CSD = getCSD(dat,sampr)
  lsp = []   # get specgrams
  F,T=None,None
  for i in range(CSD.shape[0]):
    F,T,sp = getspec(CSD[i,:],rate=sampr,window=window,maxfreq=maxfreq,tsmooth=tsmooth,fsmooth=fsmooth,logit=logit)
    lsp.append(sp)
  return CSD,F,T,lsp

# get spec from all channels in dat
def getALLspec (dat,sampr,window=1,maxfreq=125,logit=True,tsmooth=0,fsmooth=0):
  lF,lT,lspec = [],[],[]
  nchan = dat.shape[1]
  for cdx in range(nchan):
    print(cdx)
    F,T,spec = getspec(dat[:,cdx],rate=sampr,window=window,logit=logit,tsmooth=tsmooth,fsmooth=fsmooth)
    lF.append(F); lT.append(T); lspec.append(spec);
  return lF,lT,lspec

#
def smooth2D (arr,xsmooth,ysmooth):
  xblur=array([0.25,0.5,0.25])
  yblur=xblur
  nr = arr.shape[0]
  nc = arr.shape[1]
  if ysmooth > 0:
    print('Smoothing y...')
    for i in range(nc): # Smooth in frequency
      for j in range(ysmooth): 
        arr[:,i]=convolve(arr[:,i],yblur,'same')
  if xsmooth > 0:
    print('Smoothing x...')
    for i in range(nr): # Smooth in time
      for j in range(xsmooth): 
        arr[i,:]=convolve(arr[i,:],xblur,'same')

# calculates/saves spectrogram from the mat file (fn)
def savespecg (fn,csd=False,rate=20e3,window=1,maxfreq=300,tsmooth=0,fsmooth=0,logit=True):
  print(' ... ' + fn + ' ... ')
  sampr,dat,dt,tt=None,None,None,None
  try:
    sampr,dat,dt,tt = rdmat(fn)
  except:
    print('could not open ' , fn)
    return False
  print(dat.shape)
  fname = "/u/samn/plspont/data/specg/"+fn.split("/")[1]
  fname += "_window_"+str(window)+"_maxfreq_"+str(maxfreq)
  if csd:
    fname += "_CSD_specg.npz"
    CSD,F,T,lsp = getCSDspec(dat,sampr,window=window,maxfreq=maxfreq,logit=logit)
    nplsp = numpy.array(lsp)
    numpy.savez(fname,F=F,T=T,nplsp=lsp)
    del CSD,F,T,lsp,nplsp
  else:
    F,T,lsp=None,None,[]
    dat = dat.T
    for ts in dat:
      F,T,sp = getspec(ts,rate=sampr,window=window,logit=logit)
      lsp.append(sp)
    fname += "_specg.npz"
    nplsp = numpy.array(lsp)
    numpy.savez(fname,F=F,T=T,nplsp=lsp)
    del F,T,lsp,nplsp

# run mtspecg on files in lf (list of file paths)
def specgbatch (lf,csd=False,exbbn=True,rate=20e3,window=1,maxfreq=125,tsmooth=0,fsmooth=0):
  for fn in lf:
    if exbbn and fn.count("spont") < 1: continue
    savespecg(fn,csd,rate=rate,window=window,maxfreq=maxfreq,tsmooth=tsmooth,fsmooth=fsmooth)

#
def rdspecgbatch (lf,csd=False,exbbn=True,window=1,maxfreq=125):
  ddat = {}
  for fn in lf:
    if exbbn and fn.count("spont") < 1: continue
    fdat = '/u/samn/plspont/data/specg/'+fn.split('/')[1]
    fdat += "_window_"+str(window)+"_maxfreq_"+str(maxfreq)
    if csd: fdat += '_CSD'
    fdat += '_specg.npz'
    try:
      ddat[fn] = numpy.load(fdat)
    except:
      print('could not load ' , fdat)
  return ddat
        
# integrated power time-series -- gets power in range of minF,maxF frequencies
def powinrange (lspec,F,minF,maxF):
  nchan = len(lspec)
  lpow = numpy.zeros( (nchan,lspec[0].shape[1]) )
  F1idx,F2idx=-1,-1
  for i in range(len(F)):
    if minF <= F[i] and F1idx == -1: F1idx = i
    if maxF <= F[i] and F2idx == -1: F2idx = i
  # print F1idx,F[F1idx],F2idx,F[F2idx]
  rng = F2idx-F1idx+1
  for i in range(nchan): # channels
    for j in range(lspec[i].shape[1]): # time
      lpow[i][j] = numpy.sum(lspec[i][F1idx:F2idx+1,j])/rng
  return lpow

# integrated power time-series -- gets power in range of minF,maxF frequencies
def relpowinrange (lhamp,F,minF,maxF,tdur,sampr,norm=True):
  lpow = [] 
  winsz = int(tdur * sampr) # tdur in seconds
  F1idx,F2idx=-1,-1
  for i in range(len(F)):
    if minF <= F[i] and F1idx == -1: F1idx = i
    if maxF <= F[i] and F2idx == -1: F2idx = i
  N1,N2 = 55,65; N1idx,N2idx=-1,-1 # exclude noise
  for i in range(len(F)):
    if N1 <= F[i] and N1idx == -1: N1idx = i
    if N2 <= F[i] and N2idx == -1: N2idx = i
  # print F1idx,F[F1idx],F2idx,F[F2idx]
  tdx = 0
  while (tdx+1)*winsz < lhamp.shape[1]:
    if norm:
      num = sum(lhamp[F1idx:F2idx+1,tdx*winsz:(tdx+1)*winsz])
      den = sum(lhamp[0:N1idx,tdx*winsz:(tdx+1)*winsz])
      den += sum(lhamp[N1idx:-1,tdx*winsz:(tdx+1)*winsz])
      lpow.append(num / den)
    else:
      lpow.append(mean(lhamp[F1idx:F2idx+1,tdx*winsz:(tdx+1)*winsz]))
    tdx += 1
  return lpow

