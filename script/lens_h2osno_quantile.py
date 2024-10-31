# Compute quantiles from LENS ensemble

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas

import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


mbrlst = []
fllst = []
flmsg = numpy.array([-9999.0], dtype=numpy.float32)

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


# Use mask information previously assembled
# Read continent mask
nafrm = pandas.read_csv("LENS_NAmer_Locs.csv", \
                        dtype = {'LonIdx':int, 'LatIdx':int, 'Longitude':float, 'Latitude':float, \
                                 'LandFrac':float, 'Region':int, 'UTMx':float, 'UTMy':float}, \
                        encoding='utf-8-sig')
mnx = numpy.amin(nafrm['LonIdx'])
mxx = numpy.amax(nafrm['LonIdx']) + 1
mny = numpy.amin(nafrm['LatIdx'])
mxy = numpy.amax(nafrm['LatIdx']) + 1
nlc = nafrm.shape[0]

# Grid setup
ncclm = Dataset(fllst[0],'r')
latclm = ncclm.variables['lat'][mny:mxy]
lonclm = ncclm.variables['lon'][mnx:mxx]
ncclm.close()

nlat = latclm.shape[0]
nlon = lonclm.shape[0]
print(nlat)
print(nlon)

ltrp = numpy.repeat(latclm,nlon)
lnrp = numpy.tile(lonclm,nlat)
lnarr = numpy.reshape(lnrp,(nlat,nlon))
ltarr = numpy.reshape(ltrp,(nlat,nlon))
dtout = numpy.zeros( (nmbr,nlat,nlon), dtype=numpy.float32)

mskarr = numpy.zeros( (nlat,nlon), dtype=numpy.int16)
utmxout = numpy.zeros( (nlat,nlon), dtype=numpy.float32) + flmsg[0]
utmyout = numpy.zeros( (nlat,nlon), dtype=numpy.float32) + flmsg[0]
# Assemble ragged locs
for i in range(nlc):
    ltspt = nafrm['LatIdx'].values[i] - mny
    lnspt = nafrm['LonIdx'].values[i] - mnx
    mskarr[ltspt,lnspt] = 1
    utmxout[ltspt,lnspt] = nafrm['UTMx'].values[i]
    utmyout[ltspt,lnspt] = nafrm['UTMy'].values[i]

# Target date to map
tgtdt = 20050401

for j in range(nmbr):
    # Read H2OSNO
    ncclm = Dataset(fllst[j],'r')
    mcdt = ncclm.variables['mcdate'][:]
    dsq = numpy.arange(mcdt.shape[0])
    tgtidx = dsq[mcdt == tgtdt]
    snof = ncclm.variables['H2OSNO'][tgtidx[0],mny:mxy,mnx:mxx]
    ncclm.close()

    print(tgtidx)

    # Set up multi-member array
    print(ma.count_masked(snof))
    if j == 0:
        snofl = ma.zeros((nmbr,snof.shape[0],snof.shape[1]),dtype=snof.dtype)
        snofl[j,:,:] = snof
    else:
        snofl[j,:,:] = snof

snofmn = ma.mean(snofl,axis=(0))
print(snofmn.shape)
print(ma.count_masked(snofmn))
snofmn = ma.masked_where(mskarr == 0,snofmn)
print(ma.count_masked(snofmn))

# Quantiles
prbfrm = pandas.read_csv("Probs.csv", dtype = {'Prob':float}, encoding='utf-8-sig')
prbs = prbfrm['Prob']

nprb = prbs.shape[0]
prbsq = numpy.arange(nprb)
q1spt = prbsq[prbs == 0.25]
q2spt = prbsq[prbs == 0.50]
q3spt = prbsq[prbs == 0.75]


qsout = numpy.zeros((nlat,nlon,nprb),dtype=numpy.float32) - 9999.0
for q in range(nlat):
    for p in range(nlon):
        if (ma.count_masked(snofl[:,q,p]) < nmbr):
            qscr = quantile_supp.quantile_msgdat(snofl[:,q,p],prbs)
            qsout[q,p,:] = qscr
            print(qscr[104])

# Additional flagging for 0/1000 cases
# quartile coefficient of dispersion
print(q1spt)
print(q3spt)
qcd = (qsout[:,:,q3spt[0]] - qsout[:,:,q1spt[0]]) / (qsout[:,:,q3spt[0]] + qsout[:,:,q1spt[0]])
qcd = ma.masked_invalid(qcd)
qcd = ma.masked_where(qsout[:,:,q3spt[0]] < 0.0, qcd)
qcd = ma.masked_where(mskarr == 0, qcd)
qcd = ma.masked_where(qsout[:,:,0] > 980.0, qcd)
# Grand mask
nmsk = ma.count_masked(qcd)
nvld = nlat * nlon - nmsk
print(nvld)
print(numpy.amin(qcd))
print(numpy.amax(qcd))

# Map QCD
fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
cs = pyplot.pcolormesh(lnarr,ltarr,qcd,vmin=0,vmax=1.0,transform = trnsfrm,cmap=pyplot.get_cmap('Blues'))
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
cbar = pyplot.colorbar(cs,extend='max',orientation='horizontal',shrink=0.5,pad=0.06)
cbar.set_label('Quartile Coefficient of Dispersion',size=10)
cbar.ax.tick_params(labelsize=9)
tstr = 'Ensemble QCD %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS_H2OSNO_EnsQCD_%d.png' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Map ensemble median
qmed = qsout[:,:,q2spt[0]]
qmed = ma.masked_where(ma.getmask(qcd), qmed)
print(numpy.amax(qmed))

fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
cs = pyplot.pcolormesh(lnarr,ltarr,qmed,vmin=0,vmax=500,transform = trnsfrm,cmap=pyplot.get_cmap('Blues'))
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
cbar = pyplot.colorbar(cs,extend='max',orientation='horizontal',shrink=0.5,pad=0.06)
cbar.set_label('SWE [mm]',size=10)
cbar.ax.tick_params(labelsize=9)
tstr = 'Ensemble Median SWE %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS_H2OSNO_EnsMedian_%d.png' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Map IQR
qiqr = qsout[:,:,q3spt[0]] - qsout[:,:,q1spt[0]] 
qiqr = ma.masked_where(ma.getmask(qcd), qiqr)
print(numpy.amax(qiqr))

fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
cs = pyplot.pcolormesh(lnarr,ltarr,qmed,vmin=0,vmax=500,transform = trnsfrm,cmap=pyplot.get_cmap('Blues'))
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
cbar = pyplot.colorbar(cs,extend='max',orientation='horizontal',shrink=0.5,pad=0.06)
cbar.set_label('SWE [mm]',size=10)
cbar.ax.tick_params(labelsize=9)
tstr = 'Ensemble IQR SWE %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS_H2OSNO_EnsIQR_%d.png' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Mask array for saving
mskarrout = numpy.zeros( (nlat,nlon), dtype=numpy.int16) + 1
mskarrout = ma.masked_where(ma.getmask(qcd), mskarrout)
mskarrout = ma.filled(mskarrout, fill_value = -1)

xlcout = numpy.arange(mnx,mxx, dtype=numpy.int16)
ylcout = numpy.arange(mny,mxy, dtype=numpy.int16)

print(latclm.dtype)
## Output quantiles
#  2-dim location array output
qfnm = 'LENS_NAmer_H2OSNO_%d_Quantile.nc' % (tgtdt)
qout = Dataset(qfnm,'w') 

dimp = qout.createDimension('probability',nprb)
dimlon = qout.createDimension('longitude',nlon)
dimlat = qout.createDimension('latitude',nlat)

varprb = qout.createVariable('probability','f4',['probability'], fill_value = -9999)
varprb[:] = prbs
varprb.long_name = 'Probability break points'
varprb.units = 'none'
varprb.missing_value = -9999

varlon = qout.createVariable('longitude','f4',['longitude'], fill_value = -9999)
varlon[:] = lonclm
varlon.long_name = 'Longitude'
varlon.units = 'degrees_east'
varlon.missing_value = -9999

varlat = qout.createVariable('latitude','f4',['latitude'], fill_value = -9999)
varlat[:] = latclm
varlat.long_name = 'Latitude'
varlat.units = 'degrees_north'
varlat.missing_value = -9999

varxlc = qout.createVariable('lon_index','i2',['longitude'], fill_value = -99)
varxlc[:] = xlcout
varxlc.long_name = 'LENS file longitude index'
varxlc.units = 'none'
varxlc.missing_value = -99

varylc = qout.createVariable('lat_index','i2',['latitude'], fill_value = -99)
varylc[:] = ylcout
varylc.long_name = 'LENS file latitude index'
varylc.units = 'none'
varylc.missing_value = -99

varutmx = qout.createVariable('UTM_easting','f4',['latitude','longitude'], fill_value = -9999)
varutmx[:] = utmxout
varutmx.long_name = 'UTM easting coordinate'
varutmx.units = 'm'
varutmx.missing_value = -9999

varutmy = qout.createVariable('UTM_northing','f4',['latitude','longitude'], fill_value = -9999)
varutmy[:] = utmyout
varutmy.long_name = 'UTM northing coordinate'
varutmy.units = 'm'
varutmy.missing_value = -9999

varmsk = qout.createVariable('LENS_mask','i2',['latitude','longitude'], fill_value = -99)
varmsk[:] = mskarrout
varmsk.long_name = 'LENS land/snow mask'
varmsk.units = '-1=mask;1=valid'
varmsk.missing_value = -99
varmsk.comment = 'Suggested location mask for LENS results based on land areas and permanent ice; -1 indicates location is masked; 1 indicates location is usable'

varqsno = qout.createVariable('H2OSNO_quantile','f4',['latitude','longitude','probability'], fill_value = -9999)
varqsno[:] = qsout
varqsno.long_name = 'Snow water equivalent ensemble quantiles'
varqsno.units = 'mm'
varqsno.missing_value = -9999

qout.close()
