# Summarize spatial prediction results 
# Save as publication quality

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
import datetime
import sys

import matplotlib
from matplotlib import pyplot
from matplotlib import colors


def qsummary(df,vrlst):
    # Summarize with quantiles
    nmtch = df.shape[0] 
    dfout = pandas.DataFrame({'NSmp' : nmtch}, index=[0])
    for j in range(len(vrlst)):
        tmpdt = df[vrlst[j]]
        dtvld = tmpdt[numpy.isfinite(tmpdt)]
        dtvld = dtvld[dtvld != 0.0]
        vrnm = '%s_Mean' % (vrlst[j])
        dfout[vrnm] = numpy.mean(dtvld)

    return dfout

mndt = 20050401
dtlst = [20050411, 20050412, 20050413, 20050414, 20050415, 20050416]
lclst = ['A', 'B', 'C', 'D', 'E', 'F']
nlc = len(lclst)

c3 = ["#6929c4","#9f1853","#198038"]
rcls = ['#DDDDDD','#777777','#DDDDDD']
tsymb = ['d','v','o','x']
trnsz = [8,16,24, 32]
ntrn = len(trnsz)
bnsq = numpy.arange(0.0,1.05,0.05)
mdllst = ['NBTM','LBTM','GP']

fig, axs = pyplot.subplots(nrows=2,ncols=3,figsize=(9,6))
fig.subplots_adjust(bottom=0.075,top=0.875,left=0.1,right=0.9, \
                    hspace=0.45,wspace=0.45)
ftstr = 'Single Spatial Location Prediction PIT Histograms'
fig.suptitle(ftstr,fontsize=12)

# Loop through scoring results, add metrics to data frame
for j in range(nlc):

    rspt = int(j / 3)
    cspt = j % 3
    print(rspt)
    print(cspt)
    ax = axs[rspt][cspt]

    # BTM scoring, training size = 24
    qfnm = 'LENS_NAmer_H2OSNO_PredBlockScoring_%d.nc' % (dtlst[j])
    ncclm = Dataset(qfnm,'r')
    #crpsbtmn = ncclm.variables['btm_crps_nonlinear'][2,:,:,0]
    #crpsbtmnblk = ncclm.variables['btm_crps_nonlinear_blkavg'][2,:,:]
    pitbtmn = ncclm.variables['btm_pit_nonlinear'][2,:,:,0]
    pitbtml = ncclm.variables['btm_pit_linear'][2,:,:,0]
    ncclm.close()

    gfnm = 'LENS_NAmer_H2OSNO_PredBlockScoring_GP_%d.nc' % (dtlst[j])
    ncclm = Dataset(gfnm,'r')
    pitbtmg = ncclm.variables['gp_pit_nonstn'][2,:,:,0]
    ncclm.close()

    # PIT histograms
    #ax.hist(pitbtmn.flatten(), bins=bnsq, histtype='step', density=True, color=c3[0])
    #ax.hist(pitbtml.flatten(), bins=bnsq, histtype='step', density=True, color=c3[1])
    #ax.hist(pitbtmg.flatten(), bins=bnsq, histtype='step', density=True, color=c3[2])

    pitall = numpy.stack((pitbtmn.flatten(), pitbtml.flatten(), pitbtmg.flatten() ), axis=1 )
    print(pitall.shape)
    ax.hist(pitall, bins=bnsq, histtype='step', density=True, color=c3, label=mdllst)

    ax.set_xlabel('PIT',size=10)
    ax.set_ylabel('Density',size=10)
    
    for lb in ax.xaxis.get_ticklabels():
        lb.set_fontsize(9)
    for lb in ax.yaxis.get_ticklabels():
        lb.set_fontsize(9)

    if j == 5:
        leg = ax.legend(loc = 'upper left',\
                            labelspacing=0.2,borderpad=0.3,ncol=1,handletextpad=0.3,columnspacing=0.8)
        for t in leg.get_texts():
            t.set_fontsize(8)

    tstr = 'Location %s' % (lclst[j]) 
    ax.set_title(tstr,size=11)

fnm = 'LENS_NAmer_H2OSNO_PredBlock_PIT_%d.pdf' % (mndt)
pyplot.savefig(fnm)
pyplot.close()


# Block averages

fig, axs = pyplot.subplots(nrows=2,ncols=3,figsize=(9,6))
fig.subplots_adjust(bottom=0.075,top=0.875,left=0.1,right=0.9, \
                    hspace=0.45,wspace=0.45)
ftstr = 'Spatial Block Average Prediction PIT Histograms'
fig.suptitle(ftstr,fontsize=12)

# Loop through scoring results, add metrics to data frame
for j in range(nlc):

    rspt = int(j / 3)
    cspt = j % 3
    print(rspt)
    print(cspt)
    ax = axs[rspt][cspt]

    # BTM scoring, training size = 24
    qfnm = 'LENS_NAmer_H2OSNO_PredBlockScoring_%d.nc' % (dtlst[j])
    ncclm = Dataset(qfnm,'r')
    #crpsbtmn = ncclm.variables['btm_crps_nonlinear'][2,:,:,0]
    #crpsbtmnblk = ncclm.variables['btm_crps_nonlinear_blkavg'][2,:,:]
    pitbtmn = ncclm.variables['btm_pit_nonlinear_blkavg'][2,:,:]
    pitbtml = ncclm.variables['btm_pit_linear_blkavg'][2,:,:]
    ncclm.close()

    gfnm = 'LENS_NAmer_H2OSNO_PredBlockScoring_GP_%d.nc' % (dtlst[j])
    ncclm = Dataset(gfnm,'r')
    pitbtmg = ncclm.variables['gp_pit_nonstn_blkavg'][2,:,:]
    ncclm.close()

    # PIT histograms
    #ax.hist(pitbtmn.flatten(), bins=bnsq, histtype='step', density=True, color=c3[0])
    #ax.hist(pitbtml.flatten(), bins=bnsq, histtype='step', density=True, color=c3[1])
    #ax.hist(pitbtmg.flatten(), bins=bnsq, histtype='step', density=True, color=c3[2])

    pitall = numpy.stack((pitbtmn.flatten(), pitbtml.flatten(), pitbtmg.flatten() ), axis=1 )
    print(pitall.shape)
    ax.hist(pitall, bins=bnsq, histtype='step', density=True, color=c3, label=mdllst)

    ax.set_xlabel('PIT',size=10)
    ax.set_ylabel('Density',size=10)
    
    for lb in ax.xaxis.get_ticklabels():
        lb.set_fontsize(9)
    for lb in ax.yaxis.get_ticklabels():
        lb.set_fontsize(9)

    if j == 5:
        leg = ax.legend(loc = 'upper left',\
                            labelspacing=0.2,borderpad=0.3,ncol=1,handletextpad=0.3,columnspacing=0.8)
        for t in leg.get_texts():
            t.set_fontsize(8)

    tstr = 'Location %s' % (lclst[j]) 
    ax.set_title(tstr,size=11)

fnm = 'LENS_NAmer_H2OSNO_PredBlockAvg_PIT_%d.pdf' % (mndt)
pyplot.savefig(fnm)
pyplot.close()

