import matplotlib
from pylab import *

import numpy,scipy
from erp import *
from nhpdat import *

ion() # interactive drawing

#first
fn = 'data/ERP/A1/1-rb067068027@os.mat'
#second
#fn = 'data/1-rb067068029@os.mat'
#third
#fn = 'data/1-rb067068030@os.mat'
#fourth
#fn = 'data/1-rb067068031@os.mat'

spacing_um = 100.0
samprds = getdownsampr(fn)
divby = getorigsampr(fn) / samprds
trigtimes = [int(round(x)) for x in np.array(getTriggerTimes(fn)) / divby] 
trigIDs = getTriggerIDs(fn)

sampr,LFP,dt,tt,CSD,MUA = loadfile(fn, samprds)
labels = [ "S1a",  "S1b",  "S2a",  "S2b",  "G1a",  "G1b",  "I1a",  "I1b",  "I2a",  "I2b"]
layer = [ "Supragranular", "Supragranular", "Supragranular", "Supragranular", "Granular", "Granular", "Infragranular", "Infragranular", "Infragranular", "Infragranular"]
start_or_end = [ "start", "end", "start", "end", "start", "end", "start", "end", "start", "end" ]
sink_or_source = [ "Source", "Source", "Sink", "Sink",  "Sink", "Sink", "Sink", "Sink", "Source", "Source" ]
values = [1, 4, 5, 9, 10, 11, 12, 13, 14, 16]


# set epoch params
swindowms = 0
ewindowms = 50
windowms = ewindowms - swindowms

# clean bad LFP values and get CSD
sigmathresh=4

tts = trigtimes # removeBadEpochs(LFP, sampr, trigtimes, swindowms, ewindowms, sigmathresh)

# get averages
ttavg,avgLFP = getAvgERP(LFP, sampr, tts, swindowms, ewindowms)
ttavg,avgCSD = getAvgERP(CSD, sampr, tts, swindowms, ewindowms)

# set common extent for plots
xmin = 0
xmax = int(ttavg[-1])
ymin = 1
ymax = 24
extent_xy = [xmin, xmax, ymax, ymin]

# prepare the outer grid
#fig = plt.figure(figsize=(12, 8))
fig = plt.figure(figsize=(10, 8))

numplots=2
gs_outer = matplotlib.gridspec.GridSpec(2, 4, figure=fig, wspace=0.4, hspace=0.2, height_ratios = [20, 1])

fig.suptitle("Averaged Laminar CSD (n=%d) in A1 after 40 dB stimuli"%len(tts))

# create subplots common axis labels and tick marks
axs = []
for i in range(numplots):
  axs.append(plt.Subplot(fig,gs_outer[i*2:i*2+2]))
  fig.add_subplot(axs[i])
  axs[i].set_yticks(np.arange(1, 24, step=1))
  axs[i].set_ylabel('Contact', fontsize=12)
  axs[i].set_xlabel('Time (ms)',fontsize=12)
  axs[i].set_xticks(np.arange(0, 60, step=10))

# plot 1: CSD w/o interpolation
#raw = axs[0].imshow(avgCSD, extent=extent_xy, interpolation='none', aspect='auto', origin='upper', cmap='jet_r')
#axs[0].set_title('No interpolation',fontsize=12)

# plot 2: CSD w gaussian smoothing 
#axs[1].imshow(avgCSD, extent=extent_xy, interpolation='gaussian', aspect='auto', origin='upper', cmap='jet_r')
#axs[1].set_title('Gaussian',fontsize=12)

# plot 3: CSD w/ same smoothing as Sherman et al. 2016
X = ttavg
Y = range(avgCSD.shape[0])
CSD_spline=scipy.interpolate.RectBivariateSpline(Y, X, avgCSD)
Y_plot = np.linspace(0,avgCSD.shape[0],num=1000)
Z = CSD_spline(Y_plot, X)
#Z = np.clip(Z, -Z.max(), Z.max())

spline=axs[0].imshow(Z, extent=extent_xy, interpolation='none', aspect='auto', origin='upper', cmap='jet_r')
axs[0].set_title('RectBivariateSpline',fontsize=12)

height = axs[0].get_ylim()[0]
perlayer_height = int(height/avgCSD.shape[0])
xmin = axs[0].get_xlim()[0]
xmax = axs[0].get_xlim()[1]
for i,val in enumerate(values):
    if start_or_end[i] == "start":
      axs[0].hlines(values[i]+0.02, xmin, xmax,
                colors='black', linestyles='dashed')
      axs[0].text(2, values[i]+0.7, sink_or_source[i], fontsize=10)
    else:
      axs[0].hlines(values[i]+1.02, xmin, xmax,
                colors='black', linestyles='dashed')

# plot 4: CSD w/ LFP plots overlaid
axs[1].imshow(Z, extent=extent_xy, interpolation='none', aspect='auto', origin='upper', cmap='jet_r')
axs[1].set_title('LFP overlay',fontsize=12)

# trim first and last channel from LFPs to match CSD
#nrow = avgLFP.shape[0]
#avgLFP_trim = np.delete(avgLFP, [0, nrow-1], 0)

# grid for LFP plots
nrow = avgLFP.shape[0]
gs_inner = matplotlib.gridspec.GridSpecFromSubplotSpec(nrow, 1, subplot_spec=gs_outer[2:4], wspace=0.0, hspace=0.0)
clr = 'gray'
lw=0.5
subaxs = []


# go down grid and add LFP from each channel
for chan in range(nrow):
  subaxs.append(plt.Subplot(fig,gs_inner[chan],frameon=False))
  fig.add_subplot(subaxs[chan])
  subaxs[chan].margins(0.0,0.01)
  subaxs[chan].get_xaxis().set_visible(False)
  subaxs[chan].get_yaxis().set_visible(False)
  subaxs[chan].plot(X,avgLFP[chan,:],color=clr,linewidth=lw)

last_start_layer = None
for i,val in enumerate(values):
    if start_or_end[i] == "start":
      if last_start_layer == None:
        # draw first start
        axs[1].hlines(values[i]+0.03, xmin, xmax,
                      colors='black', linestyles='dashed')
      elif layer[i] == last_start_layer:
        continue
      else:
        axs[1].hlines(values[i], xmin, xmax,
                      colors='black', linestyles='dashed')        
      last_start_layer = layer[i]
      axs[1].text(35, values[i]+0.8, layer[i], fontsize=10)

# draw last end
axs[1].hlines(values[i]+0.98, xmin, xmax,
              colors='black', linestyles='dashed') 


# colorbar at the bottom using unsmoothed data for values
ax_bottom = plt.subplot(gs_outer[1,1:3])
fig.colorbar(spline,cax=ax_bottom,orientation='horizontal',use_gridspec=True)
ax_bottom.set_xlabel(r'CSD (mV/mm$^2$)', fontsize=12)

# annotate CSD colorbar
#ax_bottom.text(-0.82, -0.2, r'Sink')
#ax_bottom.text(0.68, -0.2, r'Source')

# annotate initial response at granular layer
#axs[2].text(9, 10.5, r'*')
#axs[2].text(7, 11, r'10.2 ms', fontsize=8)

#axs[2].text(12, 8.5, r'?')
#axs[2].text(9.2, 9.1, r'12.4 ms', fontsize=8)

plt.show()
