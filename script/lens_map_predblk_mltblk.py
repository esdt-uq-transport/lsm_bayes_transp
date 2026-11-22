# Generate map of prediction locations for BTM spatial missing blocks, multiple input files

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import pandas

import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


dtlst = [20050411, 20050412, 20050413, 20050414, 20050415, 20050416]
ndt = len(dtlst)
rcpscn = False

### Ensemble preparation
# Read Quantiles, mask, locations, clusters

for i in range(ndt):
    qfnm = 'LENS_NAmer_H2OSNO_%d_Quantile.nc' % (dtlst[i])
    ncclm = Dataset(qfnm,'r')
    prdmsk = ncclm.variables['predict_mask'][:,:]
    if i == 0:
        utmx = ncclm.variables['UTM_easting'][:,:]
        utmy = ncclm.variables['UTM_northing'][:,:]
        snomsk = ncclm.variables['LENS_mask'][:,:]
        ltidx = ncclm.variables['lat_index'][:]
        lnidx = ncclm.variables['lon_index'][:]
        lat = ncclm.variables['latitude'][:]
        lon = ncclm.variables['longitude'][:]

        mskout = numpy.zeros(prdmsk.shape,prdmsk.dtype)
        mskout[:,:] = prdmsk[:,:]
    else:
        mskout = mskout * prdmsk
    # Read prediction blocks 
    ncclm.close()

    mskflt = prdmsk.flatten()
    snoflt = snomsk.flatten()
    msk2 = mskflt[(mskflt == -1) & (snoflt == 1)]
    nprdcr = msk2.shape[0]
    strct = 'Date: %d, Held out locs: %d' % (dtlst[i],nprdcr)
    print(strct)


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

mskprd = mskout[mskout == 1]
ncnd = mskprd.shape[0]
print(nvld)
print(ncnd)

lcsout = numpy.zeros((nvld,2), dtype=numpy.float32)
xidx = numpy.zeros((nvld,), dtype=numpy.int16)
yidx = numpy.zeros((nvld,), dtype=numpy.int16)
prdmvc = numpy.zeros((nvld,), dtype=numpy.int16)
lcctr = 0
for q in range(nlat):
    for p in range(nlon):
        if snomsk[q,p] == 1:
            lcsout[lcctr,0] = lon[p]
            lcsout[lcctr,1] = lat[q]
            prdmvc[lcctr] = mskout[q,p]
            xidx[lcctr] = p
            yidx[lcctr] = q
            lcctr = lcctr + 1

# Subset locations
lcsq = numpy.arange(nvld)
lcscnd = lcsq[prdmvc == 1]
lcsprd = lcsq[prdmvc == -1]
nprd = lcsprd.shape[0]

### Point locations for diagnostics, identify their location indices
ctyfrm = pandas.read_csv("LENS_Cities_LocIdx.csv", \
                        dtype = {'Label':str, 'Location':str, 'Latitude':float, 'Longitude':float, \
                                 'LonIdx':int, 'LatIdx':int}, encoding='utf-8-sig')
ncity = ctyfrm.shape[0]

ctymsk = ['A','B','C','D','E','F']


## Map locations

c6 = ["#C7657B","#AC7A23","#718E00","#009A68","#0098A9","#6881C9"]
ctyfrm['LblColor'] = c6

fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
p1 = pyplot.scatter(lcsout[lcscnd,0], lcsout[lcscnd,1], \
                    s=12, color='#3333CC', marker='o', transform=trnsfrm)
p2 = pyplot.scatter(lcsout[lcsprd,0], lcsout[lcsprd,1], \
                    s=12, color='#CDCDCD', marker='o', transform=trnsfrm)
for k in range(len(ctymsk)):
    crfrm = ctyfrm[ ctyfrm['Label'] == ctymsk[k]]
    xspt = crfrm['LonIdx'].values[0] - lnidx[0]
    yspt = crfrm['LatIdx'].values[0] - ltidx[0]
    pyplot.text(crfrm['Longitude'].values[0], crfrm['Latitude'].values[0], crfrm['Label'].values[0], \
                fontsize=16, color=crfrm['LblColor'].values[0], horizontalalignment='center', verticalalignment='center', transform=trnsfrm)
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
tstr = 'Spatial Prediction Locations' 
pyplot.title(tstr)

#pltnm = 'LENS_NAmer_MultBlock_Locs.png' 
#pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
#pyplot.close()

fnm = 'LENS_NAmer_MultBlock_Locs.pdf'
pyplot.tight_layout()
pyplot.savefig(fnm)
pyplot.close()

