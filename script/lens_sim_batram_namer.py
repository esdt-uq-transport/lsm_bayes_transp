# Demo/test Batram on simulated fields over North America land areas

import numpy
import numpy.ma as ma
import quantile_supp
import pandas
from numpy import random, linalg
from netCDF4 import Dataset

import torch
import gpytorch
import sys

# Packages for building transport maps
import veccs.orderings
from batram.legmods import Data, SimpleTM

import matplotlib
from matplotlib import pyplot
from matplotlib import colors
import pylab

import cartopy.crs as ccrs
import cartopy.feature as cfeature



nmbr = 40
flmsg = numpy.array([-9999.0], dtype=numpy.float32)

#random.seed(624431)
random.seed(522342)


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

# Read ensemble members, re-structure
nlat = ltidx.shape[0]
nlon = lnidx.shape[0]

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
            #zout[:,lcctr] = ztmp[:]
            lcsout[lcctr,0] = utmx[q,p]
            lcsout[lcctr,1] = utmy[q,p]
            xidx[lcctr] = p
            yidx[lcctr] = q
            lcctr = lcctr + 1

print(lcctr)
print(nvld)

# Draw realizations from stationary model
covar_module = gpytorch.kernels.MaternKernel(nu=1.5)
covar_module.lengthscale = 500.0
xlc = torch.tensor(lcsout)
covar = covar_module(xlc) 
corrij = covar.numpy()
crchk = numpy.tril(corrij) + numpy.triu(corrij.T, 1)
print(crchk.shape)

cvvl, cvvc = linalg.eig(crchk)
print(numpy.amin(cvvl))
print(crchk[200:205,200:205])

dtz = random.multivariate_normal(numpy.zeros((nvld)), crchk,size=nmbr)
print(dtz.shape)
print(dtz.dtype)
zout[:,:] = dtz[:,:]

order = veccs.orderings.maxmin_cpp(lcsout)
lcsord = lcsout[order, ...]
obs = torch.as_tensor(zout)
obs = obs[..., order]
print(order)

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

# Plot fit results
fig, axs = pyplot.subplots(1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.5})
res.plot_loss(axs[0], use_inset=False)
res.plot_loss(axs[1], use_inset=True)
pltnm = 'Batram_Sim_NAmer_Loss.png'
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Plot params
fig, axs = pyplot.subplots(1, 1, figsize=(6.5, 5))
res.plot_params()
pltnm = 'Batram_Sim_NAmer_Params.png'
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Plot neighbors
fig, axs = pyplot.subplots(1, 1, figsize=(6.5, 5))
res.plot_neighbors()
pltnm = 'Batram_Sim_NAmer_Neighbors.png'
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Sample some fields
nrps = 16
smpsv = numpy.zeros((nrps,nvld), dtype=numpy.float32)
for i in range(nrps):
    new_sample = tm.cond_sample()[0, ...]
    re_ordered_sample = torch.zeros_like(new_sample)
    re_ordered_sample[..., order] = new_sample
    print(re_ordered_sample.shape)
    smpsv[i,:] = re_ordered_sample
    #smpsv[i,:] = zout[i,:]

# Ensemble on grid
mpzout = numpy.zeros((nrps,nlat,nlon),dtype=numpy.float32) - 9999.0
for j in range(nvld):
    mpzout[:,yidx[j],xidx[j]] = smpsv[:,j]
mpzout = ma.masked_where(mpzout < -9000,mpzout)

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
fig.suptitle(ftstr,fontsize=12)

pltnm = 'Batram_Sim_NAmer_Sample_Maps.png' 
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Correlation structure via large ensemble
nensout = 100
enssv = numpy.zeros((nensout,nvld), dtype=numpy.float32)
ordsv = numpy.zeros((nensout,4), dtype=numpy.float32)
for i in range(nensout):
    new_sample = tm.cond_sample()[0, ...]
    re_ordered_sample = torch.zeros_like(new_sample)
    re_ordered_sample[..., order] = new_sample
    enssv[i,:] = re_ordered_sample
    ordsv[i,:] = new_sample[0:4]

crens = numpy.corrcoef(numpy.transpose(enssv))
print(crens.shape)

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

pltnm = 'Batram_Sim_NAmer_City_Corr.png' 
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()


# Samples for first four ordered locs
bnsq2 = numpy.arange(-4.0,4.5,0.5)
fig = pyplot.figure(figsize=(13,12))

for a1 in range(4):
    for b1 in range(4):
        pspt = a1*4 + b1 + 1
        if a1 == b1:
            p1 = pyplot.subplot(4,4,pspt)
            # Histogram
            n, bins, patches = pylab.hist(ordsv[:,a1], bnsq2, histtype='bar', rwidth=1.0,color='#229954')
            p1.set_xlim(-4.1,4.1)
            p1.set_xlabel('Normal Score',size=12)
            p1.set_ylabel('Count',size=12)
            p1.xaxis.grid(color='#777777',linestyle='dotted')
            p1.yaxis.grid(color='#777777',linestyle='dotted')
            for lb in p1.xaxis.get_ticklabels():
                lb.set_fontsize(11)
            for lb in p1.yaxis.get_ticklabels():
                lb.set_fontsize(11)
        elif a1 > b1:
            # Scatterplot
            p1 = pyplot.subplot(4,4,pspt)
            p1.plot([0,0],[-4.1,4.1], 'k-', linewidth=0.8)
            p1.plot([-4.1,4.1],[0,0], 'k-', linewidth=0.8)
            cs = pyplot.scatter(ordsv[:,b1], ordsv[:,a1], marker='o', c='#229954', s=10)
            xstr = 'Location %d Normal Score' % (b1+1)
            ystr = 'Location %d Normal Score' % (a1+1)
            p1.set_xlabel(xstr,size=12)
            p1.set_ylabel(ystr,size=12)
            p1.set_xlim(-4.1,4.1)
            p1.set_ylim(-4.1,4.1)
            p1.xaxis.grid(color='#777777',linestyle='dotted')
            p1.yaxis.grid(color='#777777',linestyle='dotted')
            for lb in p1.xaxis.get_ticklabels():
                lb.set_fontsize(11)
            for lb in p1.yaxis.get_ticklabels():
                lb.set_fontsize(11)

fig.subplots_adjust(bottom=0.1,top=0.9,left=0.1,right=0.9, \
                    hspace=0.3,wspace=0.3)
pyplot.suptitle(r'Ensemble Normal Score $\mathbf{z}^{*}$')
pltnm = 'Batram_Sim_NAmer_ScatMat.png'
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()



