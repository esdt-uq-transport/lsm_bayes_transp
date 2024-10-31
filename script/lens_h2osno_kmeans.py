# Quick clustering of LENS SWE

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import random
from datetime import datetime


import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.cluster import KMeans

mbrlst = []
fllst = []
flmsg = numpy.array([-9999.0], dtype=numpy.float32)

#random.seed(624431)
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

# Loop through and transform
mskchk = snomsk[snomsk == 1]
nvld = mskchk.shape[0]
print(mskchk.shape)
print(snomsk.shape)

zout = numpy.zeros((nmbr,nvld), dtype=numpy.float32)
lcsout = numpy.zeros((nvld,2), dtype=numpy.float32)
xidx = numpy.zeros((nvld,), dtype=numpy.int16)
yidx = numpy.zeros((nvld,), dtype=numpy.int16)
lcctr = 0
for q in range(nlat):
    for p in range(nlon):
        if snomsk[q,p] == 1:
            ztmp = quantile_supp.std_norm_quantile_from_obs(snofl[:,q,p], h2oqs[q,p,:], prbs, msgval=-9999.)
            zout[:,lcctr] = ztmp[:]
            lcsout[lcctr,0] = utmx[q,p]
            lcsout[lcctr,1] = utmy[q,p]
            xidx[lcctr] = p
            yidx[lcctr] = q
            lcctr = lcctr + 1

print(lcctr)
print(numpy.amin(ztmp))
print(numpy.amax(ztmp))

# Cluster
inertias = []
zclst = numpy.transpose(zout)

clchc = 6
for i in range(1,21):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(zclst)
    inertias.append(kmeans.inertia_)
    if i == clchc:
        lblout = kmeans.labels_

print(inertias)
print(lblout[100:120])

# Plot inertia
c3 = ["#77B0F1","#81BB4E","#E991A1"]
c6 = ["#C7657B","#AC7A23","#718E00","#009A68","#0098A9","#6881C9"]
fig = pyplot.figure(figsize=(6,5))

p1 = pyplot.subplot(1,1,1)
p1.plot(numpy.arange(1,21),inertias,'-',linewidth=1.2,color=c3[0])
p1.set_xticks(numpy.arange(4,24,4))
p1.set_xlabel('Clusters',size=10)
p1.set_ylabel('Inertia',size=10)
p1.xaxis.grid(color='#777777',linestyle='dotted')
p1.yaxis.grid(color='#777777',linestyle='dotted')
for lb in p1.xaxis.get_ticklabels():
    lb.set_fontsize(10)
for lb in p1.yaxis.get_ticklabels():
    lb.set_fontsize(10)
tstr = 'LENS SWE K-Means Clustering' 
pyplot.title(tstr,size=12)

pltnm = 'LENS_KMeans_Inertia_%d.pdf' % (tgtdt)
pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()

# Plot fit results
#fig, axs = pyplot.subplots(1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.5})
#res.plot_loss(axs[0], use_inset=False)
#res.plot_loss(axs[1], use_inset=True)
#pltnm = 'LENS_H2OSNO_Loss.png'
#pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
#pyplot.close()

# Plot params
#fig, axs = pyplot.subplots(1, 1, figsize=(6.5, 5))
#res.plot_params()
#pltnm = 'LENS_H2OSNO_Params.png'
#pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
#pyplot.close()

# Plot neighbors
#fig, axs = pyplot.subplots(1, 1, figsize=(6.5, 5))
#res.plot_neighbors()
#pltnm = 'LENS_H2OSNO_Neighbors.png'
#pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
#pyplot.close()

