# Demonstrate the SWE transformation for two cities

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import random

import torch
import sys

import matplotlib
from matplotlib import pyplot
from matplotlib import colors
import pylab


mbrlst = []
fllst = []
flmsg = numpy.array([-9999.0], dtype=numpy.float32)

random.seed(411840)

for j in range(1,36):
    mbrstr = '%03d' % (j)
    mbrlst.append(mbrstr)
    if j == 1:
        dtsq = '18500101-20051231'
    else: 
        dtsq = '19200101-20051231'
    flcr = 'b.e11.B20TRC5CNBDRD.f09_g16.%s.clm2.h1.H2OSNO.%s.nc' % (mbrstr,dtsq)
    fllst.append(flcr)
for j in range(101,108):
    mbrstr = '%03d' % (j)
    mbrlst.append(mbrstr)
    dtsq = '19200101-20051231'
    flcr = 'b.e11.B20TRC5CNBDRD.f09_g16.%s.clm2.h1.H2OSNO.%s.nc' % (mbrstr,dtsq)
    fllst.append(flcr)

nmbr = len(mbrlst)

# Read Quantiles, mask, locations
tgtdt = 20050401
qfnm = 'LENS_NAmer_H2OSNO_%d_Quantile.nc' % (tgtdt)
ncclm = Dataset(qfnm,'r')
utmx = ncclm.variables['UTM_easting'][:,:]
utmy = ncclm.variables['UTM_northing'][:,:]
snomsk = ncclm.variables['LENS_mask'][:,:]
ltidx = ncclm.variables['lat_index'][:]
lnidx = ncclm.variables['lon_index'][:]
lat = ncclm.variables['latitude'][:]
lon = ncclm.variables['longitude'][:]
h2oqs = ncclm.variables['H2OSNO_quantile'][:,:,:]
prbs = ncclm.variables['probability'][:]
ncclm.close()

# Read ensemble members, re-structure
nlat = ltidx.shape[0]
nlon = lnidx.shape[0]

for j in range(nmbr):
    # Read H2OSNO
    ncclm = Dataset(fllst[j],'r')
    mcdt = ncclm.variables['mcdate'][:]
    dsq = numpy.arange(mcdt.shape[0])
    tgtidx = dsq[mcdt == tgtdt]
    snof = ncclm.variables['H2OSNO'][tgtidx[0],ltidx,lnidx]
    ncclm.close()

    print(tgtidx)

    # Set up multi-member array
    print(ma.count_masked(snof))
    if j == 0:
        snofl = ma.zeros((nmbr,snof.shape[0],snof.shape[1]),dtype=snof.dtype)
        snofl[j,:,:] = snof
    else:
        snofl[j,:,:] = snof


# Scatterplot of ensemble data
ctyfrm = pandas.read_csv("LENS_Cities_LocIdx.csv", \
                        dtype = {'Label':str, 'Location':str, 'Latitude':float, 'Longitude':float, \
                                 'LonIdx':int, 'LatIdx':int}, encoding='utf-8-sig')
ncity = ctyfrm.shape[0]

cty2ilc = ctyfrm['LonIdx'].values[1] - numpy.amin(lnidx)
cty2jlc = ctyfrm['LatIdx'].values[1] - numpy.amin(ltidx)
cty3ilc = ctyfrm['LonIdx'].values[2] - numpy.amin(lnidx)
cty3jlc = ctyfrm['LatIdx'].values[2] - numpy.amin(ltidx)

zscr2 = quantile_supp.std_norm_quantile_from_obs(snofl[:,cty2jlc,cty2ilc], h2oqs[cty2jlc,cty2ilc,:], prbs, msgval=-9999.)
zscr3 = quantile_supp.std_norm_quantile_from_obs(snofl[:,cty3jlc,cty3ilc], h2oqs[cty3jlc,cty3ilc,:], prbs, msgval=-9999.)

# Scatterplot
fig = pyplot.figure(figsize=(13,6))

p1 = pyplot.subplot(1,2,1)
cs = pyplot.scatter(snofl[:,cty2jlc,cty2ilc], snofl[:,cty3jlc,cty3ilc], marker='o', c='#3333CC', s=10)
p1.set_xlabel('Location B SWE [mm]')
p1.set_ylabel('Location C SWE [mm]')
p1.set_xlim(-5,530)
p1.set_ylim(-5,530)
p1.xaxis.grid(color='#777777',linestyle='dotted')
p1.yaxis.grid(color='#777777',linestyle='dotted')
ttlstr = 'Ensemble SWE'
pyplot.title(ttlstr)

fig.subplots_adjust(bottom=0.125,top=0.9,left=0.125,right=0.85,hspace=0.1,wspace=0.1)


pltnm = 'LENS_TwoCity_YZ.png'
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()


bnsq1 = numpy.arange(0,600,50)
bnsq2 = numpy.arange(-4.0,4.5,0.5)

fig = pyplot.figure(figsize=(9,8))

# Loc B Hist
p1 = pyplot.subplot(2,2,1)
# Histogram
n, bins, patches = pylab.hist(snofl[:,cty2jlc,cty2ilc], bnsq1, histtype='bar', rwidth=1.0,color='#3333CC')
p1.set_xlim(-5,555)
p1.set_xlabel('Location B SWE [mm]',size=12)
p1.set_ylabel('Count',size=12)
p1.xaxis.grid(color='#777777',linestyle='dotted')
p1.yaxis.grid(color='#777777',linestyle='dotted')
for lb in p1.xaxis.get_ticklabels():
    lb.set_fontsize(11)
for lb in p1.yaxis.get_ticklabels():
    lb.set_fontsize(11)

# Loc C Hist
p1 = pyplot.subplot(2,2,4)
# Histogram
n, bins, patches = pylab.hist(snofl[:,cty3jlc,cty3ilc], bnsq1, histtype='bar', rwidth=1.0,color='#3333CC')
p1.set_xlim(-5,555)
p1.set_xlabel('Location C SWE [mm]',size=12)
p1.set_ylabel('Count',size=12)
p1.xaxis.grid(color='#777777',linestyle='dotted')
p1.yaxis.grid(color='#777777',linestyle='dotted')
for lb in p1.xaxis.get_ticklabels():
    lb.set_fontsize(11)
for lb in p1.yaxis.get_ticklabels():
    lb.set_fontsize(11)

# Scatterplot
p1 = pyplot.subplot(2,2,3)
cs = pyplot.scatter(snofl[:,cty2jlc,cty2ilc], snofl[:,cty3jlc,cty3ilc], marker='o', c='#3333CC', s=10)
p1.set_xlabel('Location B SWE [mm]',size=12)
p1.set_ylabel('Location C SWE [mm]',size=12)
p1.set_xlim(-5,540)
p1.set_ylim(-5,540)
p1.xaxis.grid(color='#777777',linestyle='dotted')
p1.yaxis.grid(color='#777777',linestyle='dotted')
for lb in p1.xaxis.get_ticklabels():
    lb.set_fontsize(11)
for lb in p1.yaxis.get_ticklabels():
    lb.set_fontsize(11)

fig.subplots_adjust(bottom=0.1,top=0.9,left=0.1,right=0.9, \
                    hspace=0.3,wspace=0.3)
pyplot.suptitle(r'Ensemble SWE $\mathbf{y}$')
pltnm = 'LENS_TwoCity_Y_Matrix.png'
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()



# Z score

fig = pyplot.figure(figsize=(9,8))

# Loc B Hist
p1 = pyplot.subplot(2,2,1)
# Histogram
n, bins, patches = pylab.hist(zscr2, bnsq2, histtype='bar', rwidth=1.0,color='#229954')
p1.set_xlim(-4.1,4.1)
p1.set_xlabel('Location B Normal Score',size=12)
p1.set_ylabel('Count',size=12)
p1.xaxis.grid(color='#777777',linestyle='dotted')
p1.yaxis.grid(color='#777777',linestyle='dotted')
for lb in p1.xaxis.get_ticklabels():
    lb.set_fontsize(11)
for lb in p1.yaxis.get_ticklabels():
    lb.set_fontsize(11)

# Loc C Hist
p1 = pyplot.subplot(2,2,4)
# Histogram
n, bins, patches = pylab.hist(zscr3, bnsq2, histtype='bar', rwidth=1.0,color='#229954')
p1.set_xlim(-4.1,4.1)
p1.set_xlabel('Location C Normal Score',size=12)
p1.set_ylabel('Count',size=12)
p1.xaxis.grid(color='#777777',linestyle='dotted')
p1.yaxis.grid(color='#777777',linestyle='dotted')
for lb in p1.xaxis.get_ticklabels():
    lb.set_fontsize(11)
for lb in p1.yaxis.get_ticklabels():
    lb.set_fontsize(11)

# Scatterplot
p1 = pyplot.subplot(2,2,3)
p1.plot([0,0],[-4.1,4.1], 'k-', linewidth=0.8)
p1.plot([-4.1,4.1],[0,0], 'k-', linewidth=0.8)
cs = pyplot.scatter(zscr2, zscr3, marker='o', c='#229954', s=10)
p1.set_xlabel('Location B Normal Score',size=12)
p1.set_ylabel('Location C Normal Score',size=12)
p1.set_xlim(-4.1,4.1)
p1.set_ylim(-4.1,4.1)
p1.xaxis.grid(color='#777777',linestyle='dotted')
p1.yaxis.grid(color='#777777',linestyle='dotted')
for lb in p1.xaxis.get_ticklabels():
    lb.set_fontsize(11)
for lb in p1.yaxis.get_ticklabels():
    lb.set_fontsize(11)

fig.subplots_adjust(bottom=0.1,top=0.9,left=0.1,right=0.9, \
                    hspace=0.3,wspace=0.3)
pyplot.suptitle(r'Ensemble Normal Score $\mathbf{z}^{*}$')
pltnm = 'LENS_TwoCity_Z_Matrix.png'
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

print(numpy.amax(zscr2))
print(numpy.amax(zscr3))
print(numpy.corrcoef(zscr2,zscr3))