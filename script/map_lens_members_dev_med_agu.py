# Map some of the snow water equivalent output from LENS
# Median map
# Individual member deviations

import numpy
import numpy.ma as ma
from netCDF4 import Dataset

import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


mbrlst = ['001','002','003','004','005','006']
fllst = ['b.e11.B20TRC5CNBDRD.f09_g16.001.clm2.h1.H2OSNO.18500101-20051231.nc', \
         'b.e11.B20TRC5CNBDRD.f09_g16.002.clm2.h1.H2OSNO.19200101-20051231.nc', \
         'b.e11.B20TRC5CNBDRD.f09_g16.003.clm2.h1.H2OSNO.19200101-20051231.nc', \
         'b.e11.B20TRC5CNBDRD.f09_g16.004.clm2.h1.H2OSNO.19200101-20051231.nc', \
         'b.e11.B20TRC5CNBDRD.f09_g16.005.clm2.h1.H2OSNO.19200101-20051231.nc', \
         'b.e11.B20TRC5CNBDRD.f09_g16.006.clm2.h1.H2OSNO.19200101-20051231.nc' ]

nmbr = len(mbrlst)

# Target date to map
tgtdt = 20050401

# Impose flag from quantile effort
qfnm = 'LENS_NAmer_H2OSNO_%d_Quantile.nc' % (tgtdt)
ncclm = Dataset(qfnm,'r')
snomsk = ncclm.variables['LENS_mask'][:,:]
ltidx = ncclm.variables['lat_index'][:]
lnidx = ncclm.variables['lon_index'][:]
prbs = ncclm.variables['probability'][:]
sweqs = ncclm.variables['H2OSNO_quantile'][:,:,:]
ncclm.close()

qsq = numpy.arange(prbs.shape[0])
qspt = qsq[prbs == 0.5]

xmn = numpy.amin(lnidx)
xmx = numpy.amax(lnidx) + 1
ymn = numpy.amin(ltidx)
ymx = numpy.amax(ltidx) + 1

# Grid setup
ncclm = Dataset(fllst[0],'r')
latclm = ncclm.variables['lat'][ymn:ymx]
lonclm = ncclm.variables['lon'][xmn:xmx]
ncclm.close()

nlat = latclm.shape[0]
nlon = lonclm.shape[0]
print(nlat)
print(nlon)

ltrp = numpy.repeat(latclm,nlon)
lnrp = numpy.tile(lonclm,nlat)
lnarr = numpy.reshape(lnrp,(nlat,nlon))
ltarr = numpy.reshape(ltrp,(nlat,nlon))

for j in range(nmbr):
    # Read H2OSNO
    ncclm = Dataset(fllst[j],'r')
    mcdt = ncclm.variables['mcdate'][:]
    dsq = numpy.arange(mcdt.shape[0])
    tgtidx = dsq[mcdt == tgtdt]
    h2osno = ncclm.variables['H2OSNO'][tgtidx[0],ymn:ymx,xmn:xmx]
    ncclm.close()

    print(tgtidx)
    print(numpy.amin(h2osno))
    print(numpy.amax(h2osno))

    # Set up multi-member array
    print(ma.count_masked(h2osno))
    if j == 0:
        h2osnofl = ma.zeros((nmbr,h2osno.shape[0],h2osno.shape[1]),dtype=h2osno.dtype)
        h2osnofl[j,:,:] = h2osno
    else:
        h2osnofl[j,:,:] = h2osno

h2osnomd = sweqs[:,:,qspt[0]]
print(h2osnomd.shape)
print(ma.count_masked(h2osnomd))
h2osnomd =  ma.masked_where(snomsk < 0,h2osnomd)

# Plot median
fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
cs = pyplot.pcolormesh(lnarr,ltarr,h2osnomd,vmin=0,vmax=550,transform = trnsfrm,cmap=pyplot.get_cmap('Blues'))
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
cbar = pyplot.colorbar(cs,extend='max',orientation='horizontal',shrink=0.5,pad=0.06)
cbar.set_label('H2OSNO [mm]',size=10)
cbar.ax.tick_params(labelsize=9)
tstr = 'Ensemble Median Snow Water Equivalent %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS_H2OSNO_EnsMedian_NAmer_%d.pdf' % (tgtdt)
pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()

# Plot devs
fig = pyplot.figure(figsize=(9,6))
trnsfrm = ccrs.PlateCarree()

for j in range(nmbr):
    crdv = h2osnofl[j,:,:] - h2osnomd
    crdv = ma.masked_where(snomsk < 0,crdv)
    pmp = fig.add_subplot(2,3,j+1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33))
    cs = pyplot.pcolormesh(lnarr,ltarr,crdv,vmin=-550,vmax=550,transform = trnsfrm,cmap=pyplot.get_cmap('bwr'))
    pmp.coastlines(color='#777777',linewidth=0.5)
    pmp.set_extent([-140, -70, 10, 75])
    cbar = pyplot.colorbar(cs,extend='both',orientation='horizontal',shrink=0.8,pad=0.06)
    cbar.set_label('H2OSNO [mm]',size=9)
    cbar.ax.tick_params(labelsize=9)
    tstr = 'Ens Member %s Deviation' % mbrlst[j]
    pyplot.title(tstr,fontsize=10)

fig.subplots_adjust(bottom=0.08,top=0.88,left=0.15,right=0.9, \
                    hspace=0.25,wspace=0.2)
ftstr = 'Snow Water Equivalent %s' % (tgtdt)
fig.suptitle(ftstr,fontsize=12)

pltnm = 'LENS_H2OSNO_EnsDev_NAmer_%d.pdf' % (tgtdt)
pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()
