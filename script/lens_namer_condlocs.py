# Map North America locations as ordered by veccs

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import veccs.orderings
import pandas

import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Use mask information previously assembled
# Read continent mask
nafrm = pandas.read_csv("LENS_NAmer_Locs.csv", \
                        dtype = {'LonIdx':int, 'LatIdx':int, 'Longitude':float, 'Latitude':float, \
                                 'LandFrac':float, 'Region':int, 'UTMx':float, 'UTMy':float}, \
                        encoding='utf-8-sig')
nvld = nafrm.shape[0]
lcsout = numpy.zeros((nvld,2), dtype=numpy.float32)
for q in range(nvld):
    lcsout[q,0] = nafrm['UTMx'].values[q]
    lcsout[q,1] = nafrm['UTMy'].values[q]

order = veccs.orderings.maxmin_cpp(lcsout)
#lcsord = lcsout[order, ...]
print(order.shape)

lcs015 = numpy.sort(order[0:15])
lcs100 = numpy.sort(order[0:100])

# Map 15 locations
fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
p1 = pyplot.scatter(nafrm['Longitude'].values[lcs015], nafrm['Latitude'].values[lcs015], \
                    s=20, color='#3333CC', marker='x', transform=trnsfrm)
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
tstr = 'First 15 Ordered Locations'
pyplot.title(tstr)

pltnm = 'NAmer_OrderedLocs_015.png' 
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Map 100 locations
fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
p1 = pyplot.scatter(nafrm['Longitude'].values[lcs100], nafrm['Latitude'].values[lcs100], \
                    s=20, color='#3333CC', marker='x', transform=trnsfrm)
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
tstr = 'First 100 Ordered Locations'
pyplot.title(tstr)

pltnm = 'NAmer_OrderedLocs_100.png' 
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()


