# Implement Batram on quantile-transformed H2OSNO
# Output figures for SWE manuscript

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import random
import datetime

import torch
import sys

# Packages for building transport maps
from veccs.orderings2 import farthest_first_ordering, preceding_neighbors
from batram.legmods import Data, SimpleTM

import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Command line argument is name of case file
rffl = sys.argv[1]
csfrm = pandas.read_csv(rffl, dtype = {'DateFmt8':int, 'RCP':str})
rwidx = int(sys.argv[2])

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

tgtdt = csfrm['DateFmt8'].values[rwidx]
rcptxt = csfrm['RCP'].values[rwidx]
if rcptxt == 'Yes':
    rcpscn = True
else:
    rcpscn = False

case_str = 'BTM scoring for %d, RCP: %s' % (tgtdt,rcptxt)
print(case_str)

# Loop neighbor choices
nbrsq = numpy.array([8,12,16,20,30], dtype=numpy.int16)
nbrfrm = pandas.read_csv('Batram_mNbr_Thetaq.csv', dtype = {'Neighbors':int, 'ThetaQThrsh':float})


# Read Quantiles, mask, locations
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
snofl = ncclm.variables['H2OSNO_ensemble'][:,:,:]
prbs = ncclm.variables['probability'][:]
ncclm.close()

# Read ensemble members, re-structure
nlat = ltidx.shape[0]
nlon = lnidx.shape[0]
nmbr = snofl.shape[0]

ltrp = numpy.repeat(lat,nlon)
lnrp = numpy.tile(lon,nlat)
lnarr = numpy.reshape(lnrp,(nlat,nlon))
ltarr = numpy.reshape(ltrp,(nlat,nlon))

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

# Loction ordering
order, dsts = farthest_first_ordering(lcsout, start_index=None) 
lcsord = lcsout[order, ...]
obs = torch.as_tensor(zout)
obs = obs[..., order]
# Conditioning set
largest_conditioning_set = 30
nn, nbrdst = preceding_neighbors(lcsord, numpy.arange(lcsord.shape[0]), largest_conditioning_set) 

# Set up fully optimized TM fit
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


# Batram params
hprfrm = pandas.read_csv("batram_hyperparam.csv", \
                        dtype = {'ParamName':str, 'FitLoc':str, 'Elements':int,}, encoding='utf-8-sig')

for j in range(hprfrm.shape[0]):
    totpr = hprfrm['Elements'].values[j]
    curnm = hprfrm['ParamName'].values[j]
    if hprfrm['FitLoc'].values[j] == 'param_chain':
        if totpr > 1:
            curprs = res.param_chain[curnm][-1,:]
        else:
            curprs = res.param_chain[curnm][-1]
    else:
        if totpr > 1:
            curprs = res.tracked_chain[curnm][-1,:]
        else:
            curprs = res.tracked_chain[curnm][-1]
    idx = numpy.arange(totpr)
    cmpfrm = pandas.DataFrame({'ParamValue': curprs}, index=idx.tolist())
    cmpfrm['ParamName'] = curnm
    cmpfrm['ParamElement'] = numpy.arange(totpr)
    if j == 0:
        prlstout = cmpfrm
    else:
        prlstout = pandas.concat([prlstout,cmpfrm],ignore_index=True)
prlstout['CaseName'] = 'Opt'
print(prlstout)


# Large ensemble
nensout = 100
enssv = numpy.zeros((nensout,nvld), dtype=numpy.float32)
for i in range(nensout):
    new_sample = tm.cond_sample()[0, ...]
    re_ordered_sample = torch.zeros_like(new_sample)
    re_ordered_sample[..., order] = new_sample
    enssv[i,:] = re_ordered_sample

crens = numpy.corrcoef(numpy.transpose(enssv))
print(crens.shape)

# Check std devs
strstd = 'Opt case loc 0 std dev: %.4f\n       loc 200 std dev: %.4f' % (numpy.std(enssv[:,order[0]]), numpy.std(enssv[:,order[200]]))
print(strstd)

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
    cbar.set_label('Correlation',size=10)
    cbar.ax.tick_params(labelsize=10)
    tstr = 'Location %s Correlation' % (ctyfrm['Label'].values[i])
    pyplot.title(tstr,fontsize=12)

fig.subplots_adjust(bottom=0.08,top=0.85,left=0.15,right=0.9, \
                    hspace=0.25,wspace=0.2)
titlestr = 'Normal Score Correlation %d\nOptimized Hyperparameters' % (tgtdt)
pyplot.suptitle(titlestr)

pltnm = 'LENS_City_Corr_H2OSNO_%d_Opt.pdf' % (tgtdt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=300)
pyplot.close()

### Fix neighbor params
### Note that theta_q in the software is implemented differently than K&S paper see batram/legmods _scale


for n1 in range(nbrsq.shape[0]):
    thres = res.parameters
    cnbrfrm = nbrfrm[nbrfrm['Neighbors'] == nbrsq[n1]]
    thres['kernel.theta_q'] = cnbrfrm['ThetaQThrsh'].values[0] - 0.03
    th0nbr = numpy.append(thres['nugget.nugget_params'], thres['kernel.theta_q']) 
    th0nbr = numpy.append(th0nbr,thres['kernel.sigma_params'])
    th0nbr = numpy.append(th0nbr,thres['kernel.lengthscale'])
    thinbr = torch.tensor(th0nbr)
    print(th0nbr)

    tmnbr = SimpleTM(data, theta_init = thinbr, linear=False, smooth=1.5, nug_mult=4.0)
    tmnbr.kernel.theta_q.requires_grad = False
    nsteps = 130
    optnbr = torch.optim.Adam(tmnbr.parameters(), lr=0.01)
    schednbr = torch.optim.lr_scheduler.CosineAnnealingLR(optnbr, nsteps)
    resnbr = tmnbr.fit(nsteps, 0.1, test_data=tmnbr.data, optimizer=optnbr, scheduler=schednbr, batch_size=128)


    for j in range(hprfrm.shape[0]):
        totpr = hprfrm['Elements'].values[j]
        curnm = hprfrm['ParamName'].values[j]
        if hprfrm['FitLoc'].values[j] == 'param_chain':
            if totpr > 1:
                curprs = resnbr.param_chain[curnm][-1,:]
            else:
                curprs = resnbr.param_chain[curnm][-1]
        else:
            if totpr > 1:
                curprs = resnbr.tracked_chain[curnm][-1,:]
            else:
                curprs = resnbr.tracked_chain[curnm][-1]
        idx = numpy.arange(totpr)
        cmpfrm2 = pandas.DataFrame({'ParamValue': curprs}, index=idx.tolist())
        cmpfrm2['ParamName'] = curnm
        cmpfrm2['ParamElement'] = numpy.arange(totpr)
        if j == 0:
            prlstout2 = cmpfrm2
        else:
            prlstout2 = pandas.concat([prlstout2,cmpfrm2],ignore_index=True)
    prlstout2['CaseName'] = 'Nbr%02d' % (nbrsq[n1])

    if n1 == 0:
        prlstall = pandas.concat([prlstout,prlstout2],ignore_index=True)
    else:
        prlstall = pandas.concat([prlstall,prlstout2],ignore_index=True)

    # Correlation fields
    enssv = numpy.zeros((nensout,nvld), dtype=numpy.float32)
    for i in range(nensout):
        new_sample = tmnbr.cond_sample()[0, ...]
        re_ordered_sample = torch.zeros_like(new_sample)
        re_ordered_sample[..., order] = new_sample
        enssv[i,:] = re_ordered_sample

    # Check std devs
    strstd = 'Nbr%02d case loc 0 std dev: %.4f\n       loc 200 std dev: %.4f' % (nbrsq[n1], numpy.std(enssv[:,order[0]]), numpy.std(enssv[:,order[200]]))
    print(strstd)

    crens = numpy.corrcoef(numpy.transpose(enssv))

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
        cbar.set_label('Correlation',size=10)
        cbar.ax.tick_params(labelsize=10)
        tstr = 'Location %s Correlation' % (ctyfrm['Label'].values[i])
        pyplot.title(tstr,fontsize=12)

    fig.subplots_adjust(bottom=0.08,top=0.85,left=0.15,right=0.9, \
                        hspace=0.25,wspace=0.2)
    titlestr = 'Normal Score Correlation %d\n%d Neighbors' % (tgtdt,nbrsq[n1])
    pyplot.suptitle(titlestr)

    pltnm = 'LENS_City_Corr_H2OSNO_%d_Nbr%02d.pdf' % (tgtdt,nbrsq[n1])
    pyplot.savefig(pltnm,bbox_inches="tight",dpi=300)
    pyplot.close()




# Output to CSV
prflout = 'BTM_HyperPars_NbrSeq_%s.csv' % (tgtdt)
prlstall.to_csv(prflout, index=False)

