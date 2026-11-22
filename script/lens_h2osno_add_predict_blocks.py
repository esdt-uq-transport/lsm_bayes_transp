# Add flags for blocks of locations for spatial prediction 

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import random
from datetime import datetime

flmsg = numpy.array([-9999.0], dtype=numpy.float32)

random.seed(411840)

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
ncclm.close()

# Read in city locations for prediction
# Read city list
ctyfrm = pandas.read_csv("LENS_Cities_LocIdx.csv", \
                        dtype = {'Label':str, 'Location':str, 'Latitude':float, 'Longitude':float, 
                                 'LonIdx': int, 'LatIdx': int},
                        encoding='utf-8-sig')
ncity = ctyfrm.shape[0]

print(ctyfrm)

nlat = ltidx.shape[0]
nlon = lnidx.shape[0]
ltsq = numpy.arange(nlat)
lnsq = numpy.arange(nlon)

# Mask near cities A and C
ctymsk = ['A', 'C']
prdmsk = numpy.zeros( snomsk.shape, dtype=snomsk.dtype)
prdmsk[:,:] = snomsk[:,:]

for p in range(len(ctymsk)):
    crfrm = ctyfrm[ ctyfrm['Label'] == ctymsk[p]]
    xspt = crfrm['LonIdx'].values[0] - lnidx[0]
    yspt = crfrm['LatIdx'].values[0] - ltidx[0]
    
    xsq = numpy.arange(xspt-4,xspt+5)
    ysq = numpy.arange(yspt-4,yspt+5)
    print(xsq)
    print(ysq)

    # Slice by index specs
    xst = xspt-4
    xfn = xspt+5
    yst = yspt-4
    yfn = yspt+5
    prdmsk[yst:yfn,xst:xfn] = -1


# Write mask results
ncclm = Dataset(qfnm,'r+')
if ('predict_mask' in ncclm.variables):
    varmsk = ncclm.variables['predict_mask']
    varmsk[:] = prdmsk
else:
    varmsk= ncclm.createVariable('predict_mask','i2',['latitude','longitude'], fill_value = -99)
    varmsk[:] = prdmsk
    varmsk.long_name = 'LENS spatial prediction mask'
    varmsk.units = '-1=held out; 1=valid for conditioning'
    varmsk.missing_value = -99
    varmsk.comment = 'Masking of spatial locations for prediction; -1 indicates location is masked for prediction; 1 indicates location is used for fitting and conditioning'
ncclm.close()

