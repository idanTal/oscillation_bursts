from neuron import h
from filt import hilblist,hilb
from nqs import *
from vector import *

Vector = h.Vector

# get frequencies of interest + bandwidths for filtering
def getlfreqwidths (minf=0.5,maxf=125.0,step=0.5,cfcAMPStart=20.0,cfcFctr=4.0):
  vfreq=Vector(); vfreq.indgen(minf,maxf,step); lfreq=vfreq.to_python(); lfwidth=[]
  off=0.0
  if minf < 1.0: off = 0.5 - log2(minf) # min freq get
  for f in lfreq: lfwidth.append(log2(f)+off) # logarithmic + shift
  lbwcfc = list(lfwidth) # copy
  for i in range(len(lfreq)): # bandwidths for CFC
    if lfreq[i] >= cfcAMPStart: lbwcfc[i] *= cfcFctr
  return lfreq,lfwidth,lbwcfc

# run a batch saving cfc vs time and overall power,cfc values over time, for each file listed in lf
def cfcNQAbatch (lf):
  lfreq,lfwidth,lbw = getlfreqwidths(step=0.25)
  npfreq,npwidth,nplbw=numpy.array(lfreq),numpy.array(lfwidth),numpy.array(lbw)
  for fn in lf:
    print('running cfc/nqa for ' , fn)
    foutbase = '/u/samn/plspont/data/nqa/13oct3_A_' + fn.split('data/')[1] 
    CSDds,samprds,ttds,llhamp,llhang,lmodArr,lch,dbandpow,dbandcfc,nqa=cfcnqaget(fn)
    if CSDds is None: continue
    fout = foutbase + '_nqa.nqs'
    nqa.sv(fout)
    lmodArr = numpy.array(lmodArr)
    fout = foutbase + '_lmodArr.npz'
    numpy.savez(fout,lmodArr=lmodArr,lfreq=npfreq,lwidth=npwidth,lbw=nplbw)
    nqsdel(nqa)
    del CSDds,ttds,llhamp,llhang,lmodArr,dbandpow,dbandcfc,nqa

# get cross-frequency coupling arrays  - vlfp is a vector
def getcfc (vlfp,sampr=20e3,lo1=0.5,hi1=16,lo2=25,hi2=125,step1=0.5,step2=1.0,phaseBW=1,ampBW=30,varbw=0,submean=False):
  v1 = h.Vector()
  v1.copy(vlfp)
  if submean: v1.sub(v1.mean())
  from_t = 1
  to_t = int( v1.size() / sampr - 1 )
  if varbw > 0:
    phaseFreq,ampFreq,modArr = varModIndArr(v1, sampr, from_t, to_t, lo1, hi1, lo2, hi2, step1 , step2, phaseBW, ampBWFctr=varbw)
  else:
    phaseFreq,ampFreq,modArr=modIndArr(v1,sampr,from_t,to_t,lo1,hi1,lo2,hi2,phaseStep=step1,phase_bandWidth=phaseBW,ampStep=step2, amp_bandWidth=ampBW)
  return phaseFreq, ampFreq, modArr

# get CFC vs time. winsz defaults to 3 since first/last seconds cut off in modIndArr.
def getcfcvstime (vec,winsz=3,incsz=1,sampr=20e3,lo1=0.5,hi1=16,lo2=25,hi2=125,step1=0.5,step2=1.0,phaseBW=1,ampBW=30,submean=False):
  nsec = int(vec.size() / sampr); lmodind,pf,af,lt=[],[],[],[]
  vtmp = h.Vector()
  for tt in range(1,nsec-winsz,incsz):
    print(tt , ' of ' , nsec)
    if submean:
      vtmp.copy(vec,tt*sampr,(tt+winsz)*sampr)
      vtmp.sub(vtmp.mean())
      pf,af,ma=modIndArr(vtmp,sampr,1,winsz-1,lo1,hi1,lo2,hi2,step1,phaseBW,step2,ampBW)
    else:
      pf,af,ma=modIndArr(vec,sampr,tt,tt+winsz,lo1,hi1,lo2,hi2,step1,phaseBW,step2,ampBW)
    lmodind.append(ma)
  return pf,af,lmodind,lt

# get cfc for a set of channels. performs downsampling first.
def getLCFC (ldat,lch,sampred=20.,lo1=0.5,hi1=16.0,lo2=25.0,hi2=125.0,step1=0.5,step2=1.0,varbw=0,submean=False,phaseBW=1.0,ampBW=30):
  vcsd=Vector(); lMA,pf,af=[],[],[]
  for ch in lch:
    print(ch)
    vcsd.from_python(ldat[ch])
    samprDS = sampr/sampred
    vcsdDS = downsamp(vcsd,int(sampred))  
    pf,af,ma = getcfc(vcsdDS,lo1=lo1,hi1=hi1,lo2=lo2,hi2=hi2,step1=step1,step2=step2,sampr=samprDS,varbw=varbw,submean=submean,phaseBW=phaseBW,ampBW=ampBW)
    lMA.append(ma)
  return pf,af,lMA

# run a batch saving cfc vs time for each file listed in lf
def cfcbatch (lf,exbbn=True,winsz=3,incsz=1,sampr=20e3,lo1=0.5,hi1=16,lo2=25,hi2=125,step1=0.5,step2=1.0,phaseBW=1,ampBW=30):
  vcsd = Vector()
  for fn in lf:
    if exbbn and fn.count("spont") < 1: continue
    print('running cfc vs time for ' , fn)
    sampr,dat,dt,tt=None,None,None,None
    try:
      sampr,dat,dt,tt = rdmat(fn)
    except:
      print('could not open ' , fn)
      continue
    print(dat.shape)
    CSD = getCSD(dat,sampr)
    foutbase = '/u/samn/plspont/data/cfc/13sep16_B_' + fn.split('data/')[1] 
    for i in range(len(CSD)): # go through the channels
      print('channel ' , i)
      vcsd.from_python(CSD[i])
      pf,af,lmodind,lt=getcfcvstime(vcsd,winsz=winsz,incsz=incsz,sampr=sampr,lo1=lo1,hi1=hi1,lo2=lo2,hi2=hi2,step1=step1,step2=step2,phaseBW=phaseBW,ampBW=ampBW)
      npMI = numpy.array(lmodind)
      fout = foutbase + '_ch_' + str(i) + '_cfcvst.npz'
      numpy.savez(fout,npMI=npMI,af=af,pf=pf)
      del pf,af,lmodind,lt,npMI
    del CSD,dat,tt

#
def cfcnqaget (fn):
  try:
    samprds,CSDds,ttds = getCSDds(fn)
  except:
    print('could not load:' , fn)
    return None,None,None,None,None,None,None,None,None,None
  lfreq,lfwidth,lbw = getlfreqwidths(step=0.25)
  s1,s2,g,i1,i2=getflayers(fn.split('data/')[1])
  lch=[s2,g,i2]
  llhamp,llhang,lmodArr = hilblistmodArrVST(CSDds,samprds,lch,WINS*samprds,INCS*samprds)
  dbandpow,dbandcfc,nqa=getbandpowcfcnq(lch,lmodArr,llhamp,lfreq,WINS*samprds,0.5,25.0)
  return CSDds,samprds,ttds,llhamp,llhang,lmodArr,lch,dbandpow,dbandcfc,nqa


