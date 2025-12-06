# Quick clustering of LENS SWE

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import random
from datetime import datetime


import matplotlib
from matplotlib import pyplot, colors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

mbrlst = []
fllst = []
flmsg = numpy.array([-9999.0], dtype=numpy.float32)

random.seed(411840)

# Read Quantiles, mask, locations
tgtdt = 19850101
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
snofl = ncclm.variables['H2OSNO_ensemble'][:,:,:]
ncclm.close()

nmbr = snofl.shape[0]

# Read ensemble members, re-structure
nlat = ltidx.shape[0]
nlon = lnidx.shape[0]
lndv = (lon[1] - lon[0]) / 2.0
ltdv = (lat[1] - lat[0]) / 2.0

# Loop through and transform
mskchk = snomsk[snomsk == 1]
nvld = mskchk.shape[0]
print(mskchk.shape)
print(snomsk.shape)

zout = numpy.zeros((nmbr,nvld), dtype=numpy.float32)
yout = numpy.zeros((nmbr,nvld), dtype=numpy.float32)
lcsout = numpy.zeros((nvld,2), dtype=numpy.float32)
xidx = numpy.zeros((nvld,), dtype=numpy.int16)
yidx = numpy.zeros((nvld,), dtype=numpy.int16)
lcctr = 0
for q in range(nlat):
    for p in range(nlon):
        if snomsk[q,p] == 1:
            ztmp = quantile_supp.std_norm_quantile_from_obs(snofl[:,q,p], h2oqs[q,p,:], prbs, msgval=-9999.)
            zout[:,lcctr] = ztmp[:]
            yout[:,lcctr] = snofl[:,q,p]
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
    wards = AgglomerativeClustering(n_clusters=i,linkage="ward")
    kmeans.fit(zclst)
    inertias.append(kmeans.inertia_)
    wards.fit(zclst)
    if i == clchc:
        lblout = wards.labels_

print(inertias)
print(lblout[100:120])

clsout = numpy.zeros((nlat,nlon), dtype=numpy.int16) - 99

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

# Map clusters
asq = numpy.arange(nvld)
fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
for k in range(clchc):
    asb = asq[lblout == k]
    nct = asb.shape[0]
    patches = []
    for j in range(nct):
        ltcr = lat[yidx[asb[j]]]
        lncr = lon[xidx[asb[j]]]
        clsout[yidx[asb[j]], xidx[asb[j]]] = k
        rectmn = numpy.array([[lncr-lndv,ltcr-ltdv],[lncr-lndv,ltcr+ltdv],[lncr+lndv,ltcr+ltdv],[lncr+lndv,ltcr-ltdv]])
        patches.append(Polygon(rectmn))
    pmp.add_collection(PatchCollection(patches, facecolor=c6[k], edgecolor='#444444', linewidth=0, transform=trnsfrm))
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
#cbar = pyplot.colorbar(cs,extend='max',orientation='horizontal',shrink=0.5,pad=0.06)
#cbar.set_label('SWE [mm]',size=10)
#cbar.ax.tick_params(labelsize=9)
tstr = 'Cluster Membership %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS_H2OSNO_Clusters_%d_%d.pdf' % (clchc,tgtdt)
pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()


# Write cluster results
ncclm = Dataset(qfnm,'r+')
if ('cluster' in ncclm.variables):
    varclst = ncclm.variables['cluster']
    varclst[:] = clsout
else:
    varclst = ncclm.createVariable('cluster','i2',['latitude','longitude'], fill_value = -99)
    varclst[:] = clsout
    varclst.long_name = 'KMeans cluster number'
    varclst.units = 'none'
    varclst.missing_value = -99
ncclm.close()

