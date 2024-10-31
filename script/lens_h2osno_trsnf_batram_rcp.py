# Implement Batram on quantile-transformed H2OSNO

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import random
from datetime import datetime

import torch
import sys

# Packages for building transport maps
import veccs.orderings
from batram.legmods import Data, SimpleTM

import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature



mbrlst = []
fllst = []
flmsg = numpy.array([-9999.0], dtype=numpy.float32)

#random.seed(624431)
random.seed(411840)

for j in range(1,36):
    mbrstr = '%03d' % (j)
    mbrlst.append(mbrstr)
    if j < 34:
        dtsq = '20060101-20801231'
    else:
        dtsq = '20060101-21001231'
    flcr = 'b.e11.BRCP85C5CNBDRD.f09_g16.%s.clm2.h1.H2OSNO.%s.nc' % (mbrstr,dtsq)
    fllst.append(flcr)
for j in range(101,106):
    mbrstr = '%03d' % (j)
    mbrlst.append(mbrstr)
    dtsq = '20060101-21001231'
    flcr = 'b.e11.BRCP85C5CNBDRD.f09_g16.%s.clm2.h1.H2OSNO.%s.nc' % (mbrstr,dtsq)
    fllst.append(flcr)

nmbr = len(mbrlst)

# Read Quantiles, mask, locations
tgtdt = 20750401
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

order = veccs.orderings.maxmin_cpp(lcsout)
lcsord = lcsout[order, ...]
obs = torch.as_tensor(zout)
obs = obs[..., order]
print(order)

# Timing
train_start = datetime.now()

# Conditioning set
largest_conditioning_set = 30
nn = veccs.orderings.find_nns_l2(lcsord, largest_conditioning_set)

# Create a `Data` object for use with the `SimpleTM` model.
# All objects passed to this class must be torch tensors, so we type convert
# the numpy arrays in this step.
data = Data.new(
    torch.as_tensor(lcsord),
    obs,
    torch.as_tensor(nn)
)

tm = SimpleTM(data, theta_init = None, linear=False, smooth=1.5, nug_mult=4.0)
nsteps = 200
opt = torch.optim.Adam(tm.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps)
res = tm.fit(nsteps, 0.1, test_data=tm.data, optimizer=opt, scheduler=sched, batch_size=128)

train_end = datetime.now()

train_time = train_end - train_start
train_rslt = 'Total training time: %.3f' % (train_time.total_seconds())
print(train_rslt)

# Plot fit results
fig, axs = pyplot.subplots(1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.5})
res.plot_loss(axs[0], use_inset=False)
res.plot_loss(axs[1], use_inset=True)
pltnm = 'LENS_H2OSNO_Loss_%d.png' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Plot params
fig, axs = pyplot.subplots(1, 1, figsize=(6.5, 5))
res.plot_params()
pltnm = 'LENS_H2OSNO_Params_%d.png' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Plot neighbors
fig, axs = pyplot.subplots(1, 1, figsize=(6.5, 5))
res.plot_neighbors()
pltnm = 'LENS_H2OSNO_Neighbors_%d.png' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Sample some fields
nrps = 16
smpsv = numpy.zeros((nrps,nvld), dtype=numpy.float32)
samp_start = datetime.now()
for i in range(nrps):
    new_sample = tm.cond_sample()[0, ...]
    re_ordered_sample = torch.zeros_like(new_sample)
    re_ordered_sample[..., order] = new_sample
    print(re_ordered_sample.shape)
    smpsv[i,:] = re_ordered_sample

samp_end = datetime.now()

samp_time = samp_end - samp_start
samp_rslt = 'Total sampling time for %d replicates: %.3f' % (nrps,samp_time.total_seconds())
print(samp_rslt)


print(smpsv[:,511:513])
# Ensemble and inverse transform
mpzout = numpy.zeros((nrps,nlat,nlon),dtype=numpy.float32) - 9999.0
mpyout = numpy.zeros((nrps,nlat,nlon),dtype=numpy.float32) - 9999.0
for j in range(nvld):
    mpzout[:,yidx[j],xidx[j]] = smpsv[:,j]
    mpyout[:,yidx[j],xidx[j]] = quantile_supp.data_quantile_from_std_norm(mpzout[:,yidx[j],xidx[j]], h2oqs[yidx[j],xidx[j],:], prbs, \
                                                                          minval=0.0, maxval=1050.0, msgval=-9999.)
mpzout = ma.masked_where(mpzout < -9000,mpzout)
mpyout = ma.masked_where(mpyout < -9000,mpyout)

# Map fields
ltrp = numpy.repeat(lat,nlon)
lnrp = numpy.tile(lon,nlat)
lnarr = numpy.reshape(lnrp,(nlat,nlon))
ltarr = numpy.reshape(ltrp,(nlat,nlon))

fig = pyplot.figure(figsize=(12,12))
trnsfrm = ccrs.PlateCarree()

for j in range(nrps):
    pmp = fig.add_subplot(4,4,j+1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33))
    cs = pyplot.pcolormesh(lnarr,ltarr,mpzout[j,:],vmin=-4,vmax=4,transform = trnsfrm,cmap=pyplot.get_cmap('PRGn'))
    pmp.coastlines(color='#777777',linewidth=0.5)
    pmp.set_extent([-140, -70, 10, 75])
    cbar = pyplot.colorbar(cs,extend='both',orientation='horizontal',shrink=0.8,pad=0.06)
    cbar.set_label('Normal Score',size=9)
    cbar.ax.tick_params(labelsize=9)
    tstr = 'BTM Normal Score Sample %d' % (j+1)
    pyplot.title(tstr,fontsize=10)

fig.subplots_adjust(bottom=0.08,top=0.94,left=0.15,right=0.9, \
                    hspace=0.25,wspace=0.2)
ftstr = 'Normalized Realizations' 
#fig.suptitle(ftstr,fontsize=12)

pltnm = 'LENS_BTM_NormScore_Sample_Maps_%d.png' % (tgtdt) 
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# 12 realizations
fig = pyplot.figure(figsize=(12,9))
trnsfrm = ccrs.PlateCarree()

for j in range(12):
    pmp = fig.add_subplot(3,4,j+1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33))
    cs = pyplot.pcolormesh(lnarr,ltarr,mpzout[j,:],vmin=-4,vmax=4,transform = trnsfrm,cmap=pyplot.get_cmap('PRGn'))
    pmp.coastlines(color='#777777',linewidth=0.5)
    pmp.set_extent([-140, -70, 10, 75])
    cbar = pyplot.colorbar(cs,extend='both',orientation='horizontal',shrink=0.8,pad=0.06)
    cbar.set_label('Normal Score',size=9)
    cbar.ax.tick_params(labelsize=9)
    tstr = 'BTM Normal Score Sample %d' % (j+1)
    pyplot.title(tstr,fontsize=10)

fig.subplots_adjust(bottom=0.08,top=0.94,left=0.15,right=0.9, \
                    hspace=0.25,wspace=0.2)
ftstr = 'Normalized Realizations' 
#fig.suptitle(ftstr,fontsize=12)

pltnm = 'LENS_BTM_NormScore_Sample_Maps_12_%d.png' % (tgtdt) 
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()


# Inverse transforms
fig = pyplot.figure(figsize=(12,12))
trnsfrm = ccrs.PlateCarree()

for j in range(nrps):
    pmp = fig.add_subplot(4,4,j+1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33))
    cs = pyplot.pcolormesh(lnarr,ltarr,mpyout[j,:],vmin=0,vmax=500,transform = trnsfrm,cmap=pyplot.get_cmap('Blues'))
    pmp.coastlines(color='#777777',linewidth=0.5)
    pmp.set_extent([-140, -70, 10, 75])
    cbar = pyplot.colorbar(cs,extend='both',orientation='horizontal',shrink=0.8,pad=0.06)
    cbar.set_label('SWE [mm]',size=9)
    cbar.ax.tick_params(labelsize=9)
    tstr = 'BTM SWE Sample %d' % (j+1)
    pyplot.title(tstr,fontsize=10)

fig.subplots_adjust(bottom=0.08,top=0.94,left=0.15,right=0.9, \
                    hspace=0.25,wspace=0.2)
ftstr = 'Normalized Realizations' 
#fig.suptitle(ftstr,fontsize=12)

pltnm = 'LENS_BTM_Sample_Maps_%d.png' % (tgtdt) 
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# 12 realizations
fig = pyplot.figure(figsize=(12,9))
trnsfrm = ccrs.PlateCarree()

for j in range(12):
    pmp = fig.add_subplot(3,4,j+1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33))
    cs = pyplot.pcolormesh(lnarr,ltarr,mpyout[j,:],vmin=0,vmax=500,transform = trnsfrm,cmap=pyplot.get_cmap('Blues'))
    pmp.coastlines(color='#777777',linewidth=0.5)
    pmp.set_extent([-140, -70, 10, 75])
    cbar = pyplot.colorbar(cs,extend='both',orientation='horizontal',shrink=0.8,pad=0.06)
    cbar.set_label('SWE [mm]',size=9)
    cbar.ax.tick_params(labelsize=9)
    tstr = 'BTM SWE Sample %d' % (j+1)
    pyplot.title(tstr,fontsize=10)

fig.subplots_adjust(bottom=0.08,top=0.94,left=0.15,right=0.9, \
                    hspace=0.25,wspace=0.2)
ftstr = 'Normalized Realizations' 
#fig.suptitle(ftstr,fontsize=12)

pltnm = 'LENS_BTM_Sample_Maps_12_%d.png' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()



# Large ensemble
nensout = 100
enssv = numpy.zeros((nensout,nvld), dtype=numpy.float32)
samp_start = datetime.now()
for i in range(nensout):
    new_sample = tm.cond_sample()[0, ...]
    re_ordered_sample = torch.zeros_like(new_sample)
    re_ordered_sample[..., order] = new_sample
    enssv[i,:] = re_ordered_sample

crens = numpy.corrcoef(numpy.transpose(enssv))
print(crens.shape)

samp_end = datetime.now()

samp_time = samp_end - samp_start
samp_rslt = 'Total sampling time for %d replicates: %.3f' % (nensout,samp_time.total_seconds())
print(samp_rslt)


# Map selected locs correlations
ctyfrm = pandas.read_csv("LENS_Cities_LocIdx.csv", \
                        dtype = {'Label':str, 'Location':str, 'Latitude':float, 'Longitude':float, \
                                 'LonIdx':int, 'LatIdx':int}, encoding='utf-8-sig')
ncity = ctyfrm.shape[0]

lcsq = numpy.arange(nvld,)
crmpout = numpy.zeros((ncity,nlat,nlon),dtype=numpy.float32) - 9999.0
for i in range(ncity):
    ctyiidx = ctyfrm['LonIdx'].values[i] - numpy.amin(lnidx)
    ctyjidx = ctyfrm['LatIdx'].values[i] - numpy.amin(ltidx)
    ispt = lcsq[(yidx == ctyjidx) & (xidx == ctyiidx)]
    print(ispt)
    if len(ispt) > 0:
        for j in range(nvld):
            crmpout[i,yidx[j],xidx[j]] = crens[ispt[0],j]
crmpout = ma.masked_where(crmpout < -9000,crmpout)

# Maps
fig = pyplot.figure(figsize=(9,6))
trnsfrm = ccrs.PlateCarree()

for i in range(ncity):
    pmp = fig.add_subplot(2,3,i+1, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=33))
    cs = pyplot.pcolormesh(lnarr,ltarr,crmpout[i,:,:],vmin=-1,vmax=1,transform = trnsfrm,cmap=pyplot.get_cmap('bwr'))
    pmp.coastlines(color='#777777',linewidth=0.5)
    pmp.set_extent([-140, -70, 10, 75])
    cbar = pyplot.colorbar(cs,orientation='horizontal',shrink=0.8,pad=0.06)
    cbar.set_label('Correlation',size=9)
    cbar.ax.tick_params(labelsize=9)
    tstr = 'Location %s Correlation' % (ctyfrm['Label'].values[i])
    pyplot.title(tstr,fontsize=10)

fig.subplots_adjust(bottom=0.08,top=0.88,left=0.15,right=0.9, \
                    hspace=0.25,wspace=0.2)
#ftstr = 'Snow Water Equivalent %s' % (tgtdt)
#fig.suptitle(ftstr,fontsize=12)

pltnm = 'LENS_City_Corr_H2OSNO_%d.png' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()
 