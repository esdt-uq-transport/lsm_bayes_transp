# Summarize SWE quantiles and gamma distribution fits for nonzero cases

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from scipy.stats import gamma

import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# RCP or 20cent
rcpscn = True
dtsq = '20750101-20841231'
flhdr = 'BSSP370cmip6'

# Target date to map
tgtdt = 20750401

# Members by src yr, perturbation
srcyrlst = [1231, 1251, 1281, 1301]
nprtmbr = 10

mbrlst = []
fllst = []
flmsg = numpy.array([-9999.0], dtype=numpy.float32)

if rcpscn:
    for j in range(len(srcyrlst)):
        for i in range(nprtmbr):
            mbrstr = '%03d' % (i+1)
            mbrnbr = '%d.%s' % (srcyrlst[j],mbrstr)
            mbrlst.append(mbrnbr)
            flcr = 'b.e21.%s.f09_g17.LE2-%d.%s.clm2.h5.H2OSNO.%s.nc' % (flhdr,srcyrlst[j],mbrstr,dtsq)
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


for j in range(nmbr):
    # Read H2OSNO
    ncclm = Dataset(fllst[j],'r')
    mcdt = ncclm.variables['mcdate'][:]
    dsq = numpy.arange(mcdt.shape[0])
    tgtidx = dsq[mcdt == tgtdt]
    snof = ncclm.variables['H2OSNO'][tgtidx[0],mny:mxy,mnx:mxx]
    #if j == 0:
    #    topo = ncclm.variables['topo'][mny:mxy,mnx:mxx]
    ncclm.close()

    # Set up multi-member array
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
q95spt = prbsq[prbs == 0.95]
q99spt = prbsq[prbs == 0.99]

qsout = numpy.zeros((nlat,nlon,nprb),dtype=numpy.float32) - 9999.0
qsoutpar = numpy.zeros((nlat,nlon,nprb),dtype=numpy.float32) - 9999.0
nonzrs = numpy.zeros((nlat,nlon),dtype=numpy.float32) - 9999.0
nzmean = numpy.zeros((nlat,nlon),dtype=numpy.float32) - 9999.0
nzvar = numpy.zeros((nlat,nlon),dtype=numpy.float32) - 9999.0
gamshp = numpy.zeros((nlat,nlon),dtype=numpy.float32) - 9999.0
gamscl = numpy.zeros((nlat,nlon),dtype=numpy.float32) - 9999.0
for q in range(nlat):
    for p in range(nlon):
        if (ma.count_masked(snofl[:,q,p]) < nmbr):
            ytmp = snofl[:,q,p]
            qscr = quantile_supp.quantile_msgdat(snofl[:,q,p],prbs)
            qsout[q,p,:] = qscr
            sprb = prbs[qscr >= 2.5]
            qscrpar = numpy.zeros(qscr.shape[0], dtype=qscr.dtype)
            if sprb.shape[0] > 0:
                ypos = ytmp[ytmp > 2.5]
                #nonzrs[q,p] = 1.0 - numpy.amin(sprb)
                nonzrs[q,p] = 1.0 * ypos.shape[0] / nmbr
                nzmean[q,p] = numpy.mean(ypos)
                nzvar[q,p] = numpy.var(ypos,ddof=1)
                if ypos.shape[0] > 1:
                    # Fit gamma distribution
                    try:
                        shape_fit, loc_fit, scale_fit = gamma.fit(ypos,  floc=0, method="MLE")
                    except ValueError:
                        strexcpt = 'MLE failure at %.2f, %.2f, Sample size %d' % (lonclm[p],latclm[q],ypos.shape[0])
                        print(strexcpt)
                        shape_fit, loc_fit, scale_fit = gamma.fit(ypos,  floc=0, method="MM")
                        #shape_fit = -9999.0
                        #scale_fit = -9999.0
                    gamshp[q,p] = shape_fit
                    gamscl[q,p] = scale_fit
                    gprbcr = prbs - (1.0 - nonzrs[q,p])
                    qscrpar[gprbcr >= 0] = gamma.ppf(gprbcr[gprbcr >= 0] / nonzrs[q,p], a=shape_fit, scale=scale_fit)
            qsoutpar[q,p,:] = qscrpar

# Additional flagging for 0/10000 cases
# quartile coefficient of dispersion
print(q1spt)
print(q3spt)
qcd = (qsout[:,:,q3spt[0]] - qsout[:,:,q1spt[0]]) / (qsout[:,:,q3spt[0]] + qsout[:,:,q1spt[0]])
qcd = ma.masked_invalid(qcd)
qcd = ma.masked_where(qsout[:,:,q3spt[0]] < 0.0, qcd)
qcd = ma.masked_where(mskarr == 0, qcd)
qcd = ma.masked_where(qsout[:,:,0] > 9500.0, qcd)
# Grand mask
nmsk = ma.count_masked(qcd)
nvld = nlat * nlon - nmsk
print(nvld)
print(numpy.amin(qcd))
print(numpy.amax(qcd))

# Latitude, Elevation summary
phisfl = 'b.e21.BSSP370cmip6.f09_g17.LE2-1281.001.cam.h0.PHIS.207501-208412.nc'
ncphi = Dataset(phisfl,'r')
phidt = ncphi.variables['date'][:]
phdsq = numpy.arange(phidt.shape[0])
ptgtidx = phdsq[phidt == tgtdt]
phis = ncphi.variables['PHIS'][ptgtidx[0],mny:mxy,mnx:mxx]
ncphi.close()
print(ptgtidx)
topo = phis / 9.8

topomsk = ma.masked_where(ma.getmask(qcd), topo)
latmsk = ma.masked_where(ma.getmask(qcd), ltarr)
lonmsk = ma.masked_where(ma.getmask(qcd), lnarr)
print(topomsk.shape)
print(latmsk.shape)

locfrm = pandas.DataFrame({'Latitude': latmsk.flatten(), 'Longitude': lonmsk.flatten(), 'Topog': topomsk.flatten()})
print(locfrm.describe())

# Map ensemble median
qmed = qsout[:,:,q2spt[0]]
qmed = ma.masked_where(ma.getmask(qcd), qmed)
print(numpy.amax(qmed))

fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
cs = pyplot.pcolormesh(lnarr,ltarr,qmed,vmin=0,vmax=550,transform = trnsfrm,cmap=pyplot.get_cmap('Blues'))
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
cbar = pyplot.colorbar(cs,extend='max',orientation='horizontal',shrink=0.5,pad=0.06)
cbar.set_label('SWE [mm]',size=10)
cbar.ax.tick_params(labelsize=9)
tstr = 'Ensemble Median Snow Water Equivalent %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS2_H2OSNO_EnsMedian_NAmer_%d.pdf' % (tgtdt)
pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()


nonzrs =  ma.masked_where(ma.getmask(qcd),nonzrs)
# Plot Nonzero prop
fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
cs = pyplot.pcolormesh(lnarr,ltarr,nonzrs,vmin=0,vmax=1,transform = trnsfrm,cmap=pyplot.get_cmap('RdPu'))
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
cbar = pyplot.colorbar(cs,extend='neither',orientation='horizontal',shrink=0.5,pad=0.06)
cbar.set_label('Proportion',size=10)
cbar.ax.tick_params(labelsize=9)
tstr = 'Ensemble Proportion Nonzero SWE %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS2_H2OSNO_EnsNonzeroProp_NAmer_%d.pdf' % (tgtdt)
pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()

# Map Gamma dist params
gamshp =  ma.masked_where(ma.getmask(qcd),gamshp)
gamshp = ma.masked_where(gamshp < -1000,gamshp)

gamscl =  ma.masked_where(ma.getmask(qcd),gamscl)
gamscl = ma.masked_where(gamscl < -1000,gamscl)

fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
cs = pyplot.pcolormesh(lnarr,ltarr,gamshp,vmin=0,vmax=50,transform = trnsfrm)
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
cbar = pyplot.colorbar(cs,extend='max',orientation='horizontal',shrink=0.5,pad=0.06)
cbar.set_label('Shape',size=10)
cbar.ax.tick_params(labelsize=9)
tstr = 'SWE Gamma Distribution Shape Parameter %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS2_H2OSNO_SWEGammaShape_NAmer_%d.pdf' % (tgtdt)
pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()

fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
cs = pyplot.pcolormesh(lnarr,ltarr,gamscl,vmin=0,vmax=50,transform = trnsfrm)
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
cbar = pyplot.colorbar(cs,extend='max',orientation='horizontal',shrink=0.5,pad=0.06)
cbar.set_label('Scale',size=10)
cbar.ax.tick_params(labelsize=9)
tstr = 'SWE Gamma Distribution Scale Parameter %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS2_H2OSNO_SWEGammaScale_NAmer_%d.pdf' % (tgtdt)
pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()


# Map surface geopotential height
fig = pyplot.figure(figsize=(9,9))
trnsfrm = ccrs.PlateCarree()
pmp = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33) )
cs = pyplot.pcolormesh(lnarr,ltarr,topomsk,transform = trnsfrm)
pmp.coastlines(color='#777777',linewidth=0.5)
pmp.set_extent([-140, -70, 10, 75])
cbar = pyplot.colorbar(cs,extend='neither',orientation='horizontal',shrink=0.5,pad=0.06)
cbar.set_label('Geopotential Height [m]',size=10)
cbar.ax.tick_params(labelsize=9)
tstr = 'Mean Surface Geopotential Height %d' % (tgtdt) 
pyplot.title(tstr)

pltnm = 'LENS2_H2OSNO_SfcGeopHt_NAmer_%d.pdf' % (tgtdt)
pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()

# Mask array for saving
mskarrout = numpy.zeros( (nlat,nlon), dtype=numpy.int16) + 1
mskarrout = ma.masked_where(ma.getmask(qcd), mskarrout)
mskarrout = ma.filled(mskarrout, fill_value = -1)

xlcout = numpy.arange(mnx,mxx, dtype=numpy.int16)
ylcout = numpy.arange(mny,mxy, dtype=numpy.int16)

# Original ensemble mask
for j in range(nmbr):
    snofl[j,:,:] = ma.masked_where(ma.getmask(qcd), snofl[j,:,:])
snofl = ma.filled(snofl, fill_value = -9999)

# Latitude partitions: < 52, 52-62, > 62
# Height partitions: < 300, 300-700, > 700
nzmean =  ma.masked_where(ma.getmask(qcd),nzmean)
nzmean = ma.masked_where(nzmean < -1000,nzmean)

nzvar =  ma.masked_where(ma.getmask(qcd),nzvar)
nzvar = ma.masked_where(nzmean < -1000,nzvar)

# Upper quantiles summary
q95 = qsout[:,:,q95spt[0]]
q95 = ma.masked_where(ma.getmask(qcd), q95)
q95 = ma.masked_where(q95 < -1000,q95)

q95gam = qsoutpar[:,:,q95spt[0]]
q95gam = ma.masked_where(ma.getmask(qcd), q95gam)
q95gam = ma.masked_where(q95gam < -1000,q95gam)

# Median summary
q50 = qsout[:,:,q2spt[0]]
q50 = ma.masked_where(ma.getmask(qcd), q50)
q50 = ma.masked_where(q50 < -1000,q50)

q50gam = qsoutpar[:,:,q2spt[0]]
q50gam = ma.masked_where(ma.getmask(qcd), q50gam)
q50gam = ma.masked_where(q50gam < -1000,q50gam)


locfrm['NonZeroProp'] = nonzrs.flatten()
locfrm['NonZeroMean'] = nzmean.flatten()
locfrm['NonZeroVar'] = nzvar.flatten()
locfrm['GammaShape'] = gamshp.flatten()
locfrm['GammaScale'] = gamscl.flatten()
locfrm['Q95'] = q95.flatten()
locfrm['Q95Gam'] = q95gam.flatten()
locfrm['Q50'] = q50.flatten()
locfrm['Q50Gam'] = q50gam.flatten()

print(locfrm.describe())

latlwr = numpy.array([0,52,62])
latupr = numpy.array([52,62,90])
elvlwr = numpy.array([0,300,700])
elvupr = numpy.array([300,700,9999])
latprt = ['< 52', '52-62', '> 62']
elvprt = [ '< 300', '300-700', '> 700']

fig = pyplot.figure(figsize=(10,11))

for q in range(3):
    for p in range(3):
        cspt = p*3 + q + 1
        cfrm = locfrm[ (locfrm['Latitude'] > latlwr[q]) & (locfrm['Latitude'] <= latupr[q]) & (locfrm['Topog'] > elvlwr[p]) & (locfrm['Topog'] <= elvupr[p])]
        cfrm = cfrm[numpy.isfinite(cfrm['NonZeroProp'])]

        p1 = pyplot.subplot(3,3,cspt)
        cs = pyplot.scatter(cfrm['NonZeroMean'], cfrm['NonZeroVar'], marker='o', c=cfrm['NonZeroProp'], s=4, vmin=0, vmax=1, cmap=pyplot.get_cmap('RdPu'))
        p1.set_xlabel('SWE Mean')
        p1.set_ylabel('SWE Variance')
        #if p == 2:
        cbar = pyplot.colorbar(cs,extend='neither',orientation='horizontal',shrink=0.6,pad=0.18)
        cbar.set_label('Prop Nonzero',size=9)
        cbar.ax.tick_params(labelsize=9)

        #p1.set_xlim(-5,530)
        #p1.set_ylim(-5,530)
        p1.xaxis.grid(color='#777777',linestyle='dotted')
        p1.yaxis.grid(color='#777777',linestyle='dotted')
        ttlstr = 'Latitude %s\nElevation %s\n n=%d' % (latprt[q],elvprt[p],cfrm.shape[0])
        pyplot.title(ttlstr,size=10)

fig.subplots_adjust(bottom=0.08,top=0.94,left=0.1,right=0.9,hspace=0.3,wspace=0.4)

pltnm = 'LENS2_H2OSNO_MeanVar_NAmer_%d.pdf' % (tgtdt)
#pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()


# Summary of empirical and fitted quantiles
locmx = locfrm[locfrm['GammaShape'] > 1000]
print(locmx)


fig = pyplot.figure(figsize=(10,11))

for q in range(3):
    for p in range(3):
        cspt = p*3 + q + 1
        cfrm = locfrm[ (locfrm['Latitude'] > latlwr[q]) & (locfrm['Latitude'] <= latupr[q]) & (locfrm['Topog'] > elvlwr[p]) & (locfrm['Topog'] <= elvupr[p])]
        cfrm = cfrm[numpy.isfinite(cfrm['NonZeroProp'])]
        mxswe = 1.05 * numpy.amax(cfrm['Q95Gam'])

        p1 = pyplot.subplot(3,3,cspt)
        p1.plot([0,mxswe],[0,mxswe],'-',color='#565656')
        cs = pyplot.scatter(cfrm['Q95'], cfrm['Q95Gam'], marker='o', c=cfrm['NonZeroProp'], s=4, vmin=0, vmax=1, cmap=pyplot.get_cmap('RdPu'))
        p1.set_xlabel('Empirical 0.95 Quantile')
        p1.set_ylabel('Gamma 0.95 Quantile')
        #if p == 2:
        cbar = pyplot.colorbar(cs,extend='neither',orientation='horizontal',shrink=0.6,pad=0.22)
        cbar.set_label('Prop Nonzero',size=9)
        cbar.ax.tick_params(labelsize=9)

        p1.set_xlim(-5,mxswe)
        p1.set_ylim(-5,mxswe)
        p1.xaxis.grid(color='#777777',linestyle='dotted')
        p1.yaxis.grid(color='#777777',linestyle='dotted')
        ttlstr = 'Latitude %s\nElevation %s\n n=%d' % (latprt[q],elvprt[p],cfrm.shape[0])
        pyplot.title(ttlstr,size=10)

fig.subplots_adjust(bottom=0.05,top=0.9,left=0.1,right=0.9,hspace=0.3,wspace=0.4)

titlestr = 'Extreme Quantile Comparison %d' % (tgtdt)
pyplot.suptitle(titlestr)

pltnm = 'LENS2_H2OSNO_Q95Pairs_NAmer_%d.pdf' % (tgtdt)
#pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()


fig = pyplot.figure(figsize=(10,11))

for q in range(3):
    for p in range(3):
        cspt = p*3 + q + 1
        cfrm = locfrm[ (locfrm['Latitude'] > latlwr[q]) & (locfrm['Latitude'] <= latupr[q]) & (locfrm['Topog'] > elvlwr[p]) & (locfrm['Topog'] <= elvupr[p])]
        cfrm = cfrm[numpy.isfinite(cfrm['NonZeroProp'])]
        mxswe = 1.05 * numpy.amax(cfrm['Q50Gam'])

        p1 = pyplot.subplot(3,3,cspt)
        p1.plot([0,mxswe],[0,mxswe],'-',color='#565656')
        cs = pyplot.scatter(cfrm['Q50'], cfrm['Q50Gam'], marker='o', c=cfrm['NonZeroProp'], s=4, vmin=0, vmax=1, cmap=pyplot.get_cmap('RdPu'))
        p1.set_xlabel('Empirical 0.50 Quantile')
        p1.set_ylabel('Gamma 0.50 Quantile')
        #if p == 2:
        cbar = pyplot.colorbar(cs,extend='neither',orientation='horizontal',shrink=0.6,pad=0.22)
        cbar.set_label('Prop Nonzero',size=9)
        cbar.ax.tick_params(labelsize=9)

        p1.set_xlim(-5,mxswe)
        p1.set_ylim(-5,mxswe)
        p1.xaxis.grid(color='#777777',linestyle='dotted')
        p1.yaxis.grid(color='#777777',linestyle='dotted')
        ttlstr = 'Latitude %s\nElevation %s\n n=%d' % (latprt[q],elvprt[p],cfrm.shape[0])
        pyplot.title(ttlstr,size=10)

fig.subplots_adjust(bottom=0.05,top=0.9,left=0.1,right=0.9,hspace=0.3,wspace=0.4)

titlestr = 'Median Comparison %d' % (tgtdt)
pyplot.suptitle(titlestr)

pltnm = 'LENS2_H2OSNO_Q50Pairs_NAmer_%d.pdf' % (tgtdt)
#pyplot.tight_layout()
pyplot.savefig(pltnm)
pyplot.close()


## Save results

## Output quantiles
#  2-dim location array output
gamshp = ma.filled(gamshp, fill_value = -9999)
gamscl = ma.filled(gamscl, fill_value = -9999)
nonzrs = ma.filled(nonzrs, fill_value = -9999)
qsoutpar = ma.filled(qsoutpar, fill_value = -9999)

qfnm = 'LENS2_NAmer_H2OSNO_%d_Quantile_ParamFit.nc' % (tgtdt)
qout = Dataset(qfnm,'w') 

dimp = qout.createDimension('probability',nprb)
dimlon = qout.createDimension('longitude',nlon)
dimlat = qout.createDimension('latitude',nlat)
dimens = qout.createDimension('member',nmbr)

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

varqsnop = qout.createVariable('H2OSNO_quantile_gamfit','f4',['latitude','longitude','probability'], fill_value = -9999)
varqsnop[:] = qsoutpar
varqsnop.long_name = 'Snow water equivalent ensemble quantiles based on estimated gamma distribution parameters'
varqsnop.units = 'mm'
varqsnop.missing_value = -9999

varswe = qout.createVariable('H2OSNO_ensemble','f4',['member','latitude','longitude'], fill_value = -9999)
varswe[:] = snofl
varswe.long_name = 'Snow water equivalent ensemble fields'
varswe.units = 'mm'
varswe.missing_value = -9999

vargamshp = qout.createVariable('H2OSNO_gamma_shape','f4',['latitude','longitude'], fill_value = -9999)
vargamshp[:] = gamshp
vargamshp.long_name = 'Estimated gamma distribution shape parameter'
vargamshp.units = 'none'
vargamshp.missing_value = -9999

vargamscl = qout.createVariable('H2OSNO_gamma_scale','f4',['latitude','longitude'], fill_value = -9999)
vargamscl[:] = gamscl
vargamscl.long_name = 'Estimated gamma distribution scale parameter'
vargamscl.units = 'none'
vargamscl.missing_value = -9999

varnonzero = qout.createVariable('H2OSNO_prop_nonzero','f4',['latitude','longitude'], fill_value = -9999)
varnonzero[:] = nonzrs
varnonzero.long_name = 'Estimated SWE nonzero proportion'
varnonzero.units = 'none'
varnonzero.missing_value = -9999

qout.close()

