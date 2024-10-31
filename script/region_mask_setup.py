# Match Transcom regions to CESM grid, save results
from netCDF4 import Dataset
import numpy
import pandas
from pyproj import Proj


# LENS file
lsmfl = 'b.e11.B20TRC5CNBDRD.f09_g16.001.clm2.h1.H2OSNO.18500101-20051231.nc'
ncclm = Dataset(lsmfl,'r')
latclm = ncclm.variables['lat'][:]
lonclm = ncclm.variables['lon'][:]
lndfrc = ncclm.variables['landfrac'][:,:]
ncclm.close()
nltlsm = latclm.shape[0]
nlnlsm = lonclm.shape[0]
lonclm[lonclm > 180] = lonclm[lonclm > 180] - 360
ltsqclm = numpy.arange(nltlsm)
lnsqclm = numpy.arange(nlnlsm)

# Transcom File
mipfl = 'oco2_regions_l4mip_v7.nc'
ncmip = Dataset(mipfl)
latmip = ncmip.variables['latitude'][:]
lonmip = ncmip.variables['longitude'][:]
rgnmip = ncmip.variables['transcom_regions'][:,:]
ncmip.close()
nltrf = latmip.shape[0]
nlnrf = lonmip.shape[0]
ltsqrf = numpy.arange(nltrf)
lnsqrf = numpy.arange(nlnrf)


# Full collection
rgnclm = numpy.zeros((nltlsm,nlnlsm),dtype=numpy.int32)
print(rgnclm.shape)
for j in range(nltlsm):
    for i in range(nlnlsm):
        dlon = numpy.absolute(lonmip-lonclm[i])
        dlon[dlon > 180.0] = 360.0 - dlon[dlon > 180.0]
        #print(dlon.shape)
        mnln = numpy.amin(dlon)
        #print(mnln)
        ispt = lnsqrf[dlon == mnln]

        dlat = numpy.absolute(latmip-latclm[j])
        mnlt = numpy.amin(dlat)
        jspt = ltsqrf[dlat == mnlt]

        #sstr = 'LnIdx %d: %d,  LtIdx %d: %d' % (i,ispt[0],j,jspt[0])
        #print(sstr)

        rgnclm[j,i] = int(rgnmip[jspt[0],ispt[0]])

# Output to NetCDF
outfl = 'LENS_Transcom_Matchup.nc'
zout = Dataset(outfl,'w')

dimlon = zout.createDimension('lon',nlnlsm)
dimlat = zout.createDimension('lat',nltlsm)

varlon = zout.createVariable('lon','f4',['lon'])
varlon[:] = lonclm
varlon.long_name = 'Longitude'
varlon.units = 'degrees_east'

varlat = zout.createVariable('lat','f4',['lat'])
varlat[:] = latclm
varlat.long_name = 'Latitude'
varlat.units = 'degrees_north'

varfrc = zout.createVariable('landfrac','f4',['lat','lon'])
varfrc[:,:] = lndfrc
varfrc.long_name = 'Land fraction'

varrgn = zout.createVariable('transcom_regions','i4',['lat','lon'])
varrgn[:,:] = rgnclm
varrgn.long_name = 'Transcom region number'

zout.close()

# North America Subset
ltrp = numpy.repeat(latclm,nlnlsm)
lnrp = numpy.tile(lonclm,nltlsm)

jsqrp = numpy.repeat(ltsqclm,nlnlsm)
isqrp = numpy.tile(lnsqclm,nltlsm)
rgnfrm = pandas.DataFrame({'LonIdx':isqrp, 'LatIdx':jsqrp, \
                           'Longitude':lnrp, 'Latitude':ltrp, \
                           'LandFrac':lndfrc.flatten(), 'Region':rgnclm.flatten()}) 

nafrm = rgnfrm[ ((rgnfrm['Region'] == 1) | (rgnfrm['Region'] == 2)) & (rgnfrm['LandFrac'] >= 0.8) ]
print(nafrm.shape)
print(nafrm[0:10])

# UTM
# Get UTM coordinates
#cord.dec <- SpatialPoints(cbind(frmsb$Longitude, frmsb$Latitude), proj4string=CRS("+proj=longlat"))
#cord.UTM <- spTransform(cord.dec, CRS("+proj=utm +lon_0=93w +zone=15 +ellps=GRS80"))
#myProj = Proj("+proj=utm +zone=23K, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
myProj = Proj("+proj=utm +lon_0=93w +zone=15 +ellps=GRS80")
UTMx, UTMy = myProj(nafrm['Longitude'], nafrm['Latitude'])
nafrm['UTMx'] = UTMx / 1.0e3
nafrm['UTMy'] = UTMy / 1.0e3

print(UTMx.shape)
csvnamer = 'LENS_NAmer_Locs.csv'
nafrm.to_csv(csvnamer,index=False)
