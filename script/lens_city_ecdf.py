# Selected "city" empirical distributions/locations for LENS

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
import datetime

import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Read city list
ctyfrm = pandas.read_csv("LENS_Cities.csv", \
                        dtype = {'Label':str, 'Location':str, 'Latitude':float, 'Longitude':float}, \
                        encoding='utf-8-sig')
ncity = ctyfrm.shape[0]

# Quantile info
tgtdtfl = datetime.date(2005,4,1)
dtstr = tgtdtfl.strftime('%Y-%m-%d')
tgtdt = 20050401
qfnm = 'LENS_NAmer_H2OSNO_%d_Quantile.nc' % (tgtdt)
ncclm = Dataset(qfnm,'r')
ltidx = ncclm.variables['lat_index'][:]
lnidx = ncclm.variables['lon_index'][:]
lat = ncclm.variables['latitude'][:]
lon = ncclm.variables['longitude'][:]
h2oqs = ncclm.variables['H2OSNO_quantile'][:,:,:]
prbs = ncclm.variables['probability'][:]
ncclm.close()

nlat = ltidx.shape[0]
nlon = lnidx.shape[0]
# Loop through and match location index
xidxcty = numpy.zeros((ncity,),dtype=numpy.int16)
yidxcty = numpy.zeros((ncity,),dtype=numpy.int16)
for k in range(ncity):
    latdif = numpy.absolute(lat - ctyfrm['Latitude'].values[k])
    ycr = ltidx[latdif == numpy.amin(latdif)]
    londif = numpy.absolute(lon - ctyfrm['Longitude'].values[k] - 360)
    xcr = lnidx[londif == numpy.amin(londif)]
    xidxcty[k] = xcr[0]
    yidxcty[k] = ycr[0]

ctyfrm['LonIdx'] = xidxcty
ctyfrm['LatIdx'] = yidxcty
print(ctyfrm)

# Save frame


#c6 = ['#092177', '#76177d', '#b90a70', '#e83557', '#ff6c36', '#ffa600']
c6 = ["#C7657B","#AC7A23","#718E00","#009A68","#0098A9","#6881C9"]
# Map cities
fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
for k in range(ncity):
    pyplot.text(ctyfrm['Longitude'].values[k], ctyfrm['Latitude'].values[k], ctyfrm['Label'].values[k], \
                fontsize=18, color=c6[k], verticalalignment='center', transform=trnsfrm)
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
tstr = 'LENS Example Locations'
pyplot.title(tstr)

pltnm = 'LENS_City_Map.png'
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Empirical CDFs
fig = pyplot.figure(figsize=(9,8))

plst = []
p7 = pyplot.subplot(1,1,1)
for k in range(ncity):
    xspt = xidxcty[k] - lnidx[0]
    yspt = yidxcty[k] - ltidx[0]
    pl1, = p7.plot(h2oqs[yspt,xspt,:],prbs,'-',c=c6[k],linewidth=2)
    plst.append(pl1)
p7.set_ylim(-0.05,1.05)
p7.set_xlim(-30,550)
p7.set_xlabel('SWE [mm]',size=11)
p7.set_ylabel('CDF',size=11)
p7.xaxis.grid(color='#777777',linestyle='dotted')
p7.yaxis.grid(color='#777777',linestyle='dotted')
for lb in p7.xaxis.get_ticklabels():
    lb.set_fontsize(10)
for lb in p7.yaxis.get_ticklabels():
    lb.set_fontsize(10)
leg = pyplot.legend(plst,['A','B','C','D','E','F'], loc = 'lower right',labelspacing=0.6,borderpad=0.6)
for t in leg.get_texts():
    t.set_fontsize(14)
         
tstr = 'Single Site SWE CDF %s' % (dtstr)
pyplot.title(tstr,size=12)
pltnm = 'LENS_City_CDF_H2OSNO_%d.png' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()


#lcsout = "LENS_Cities_LocIdx.csv"
#ctyfrm.to_csv(lcsout,index=False)
