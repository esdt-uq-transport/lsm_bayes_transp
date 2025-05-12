# Compute log scores linear and nonlinear BTM fits across multiple epochs

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import random
import datetime

from numpy import linalg
import scipy

import torch
import gpytorch

import veccs.orderings
from batram.legmods import Data, SimpleTM

random.seed(623114)
rcpscn = True
tgtdt = 20750401

mbrlst = []
fllst = []
shmsg = numpy.array([-99], dtype=numpy.int16)

if rcpscn:
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
else:
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
print(nmbr)

### Read in training sequences
sqfnm = 'LENS_NAmer_H2OSNO_TestSets_%d.nc' % (tgtdt)
ncseq = Dataset(sqfnm,'r')
trnchc = ncseq.variables['trainsize'][:]
trnseq = ncseq.variables['member_sequence'][:,:,:]
ncseq.close()
print(trnchc.shape)
print(trnchc)
nepoch = trnseq.shape[1]

### Ensemble preparation
# Read Quantiles, mask, locations, clusters
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
clstid = ncclm.variables['cluster'][:,:]
ncclm.close()

# Read ensemble members, re-structure
nlat = ltidx.shape[0]
nlon = lnidx.shape[0]
lndv = (lon[1] - lon[0]) / 2.0
ltdv = (lat[1] - lat[0]) / 2.0

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
yout = numpy.zeros((nmbr,nvld), dtype=numpy.float32)
lcsout = numpy.zeros((nvld,2), dtype=numpy.float32)
xidx = numpy.zeros((nvld,), dtype=numpy.int16)
yidx = numpy.zeros((nvld,), dtype=numpy.int16)
clstvc = numpy.zeros((nvld,), dtype=numpy.int16)
lcctr = 0
for q in range(nlat):
    for p in range(nlon):
        if snomsk[q,p] == 1:
            ztmp = quantile_supp.std_norm_quantile_from_obs(snofl[:,q,p], h2oqs[q,p,:], prbs, msgval=-9999.)
            zout[:,lcctr] = ztmp[:]
            yout[:,lcctr] = snofl[:,q,p]
            lcsout[lcctr,0] = utmx[q,p]
            lcsout[lcctr,1] = utmy[q,p]
            clstvc[lcctr] = clstid[q,p]
            xidx[lcctr] = p
            yidx[lcctr] = q
            lcctr = lcctr + 1
nclst = int(numpy.amax(clstvc)) + 1

# Scale locations
lcsadj = numpy.zeros(lcsout.shape,dtype=lcsout.dtype)
lcsadj[:,0] = (lcsout[:,0] + 150.0) / 6500.0
lcsadj[:,1] = (lcsout[:,1] - 6750.0) / 6500.0

# Location ordering
order = veccs.orderings.maxmin_cpp(lcsout)
lcsord = lcsout[order, ...]
# Conditioning set
largest_conditioning_set = 30
nn = veccs.orderings.find_nns_l2(lcsord, largest_conditioning_set)


### Training sequences
ntrncs =  trnchc.shape[0]
flsq = numpy.arange(nmbr)
flst = flsq.tolist()
for j in range(ntrncs):
    ntrncr = trnchc[j]
    ntst = nmbr - ntrncr
    for k in range(nepoch): 
        trnlst = trnseq[j,k,0:ntrncr].tolist()
        tstlst = []
        for i in range(nmbr):
            if i not in trnlst:
                tstlst.append(i)

        obs = torch.as_tensor(zout[trnlst,:])
        obs = obs[..., order]

        # Create a `Data` object for use with the `SimpleTM` model.
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


        # Score the nonlinear cases with fitted map
        obstst = torch.as_tensor(zout[tstlst,:])
        obstst = obstst[..., order]

        scroutnln = numpy.zeros((ntst,), dtype=numpy.float32)
        for i in range(ntst):
            with torch.no_grad():
                scroutnln[i] = tm.score(obstst[i,:])
                scrtxt = 'Sample %d log score: %.3f' % (i, scroutnln[i])
                print(scrtxt)
        scmntxt = 'Average nonlinear score: %.3f' % (numpy.mean(scroutnln))
        print(scmntxt)
        print(res.parameters)

        ## Set up linear TM
        thres = res.parameters
        thres['kernel.sigma_params'][0] = -9999.0
        th0lin = numpy.append(thres['nugget.nugget_params'], thres['kernel.theta_q']) 
        th0lin = numpy.append(th0lin,thres['kernel.sigma_params'])
        th0lin = numpy.append(th0lin,thres['kernel.lengthscale'])
        thilin = torch.tensor(th0lin)

        tmlin = SimpleTM(data, theta_init = thilin, linear=False, smooth=1.5, nug_mult=4.0)
        tmlin.kernel.sigma_params.requires_grad = False
        nsteps = 130
        optlin = torch.optim.Adam(tmlin.parameters(), lr=0.01)
        schedlin = torch.optim.lr_scheduler.CosineAnnealingLR(optlin, nsteps)
        reslin = tmlin.fit(nsteps, 0.1, test_data=tmlin.data, optimizer=optlin, scheduler=schedlin, batch_size=128)

        # Score test cases with linear map
        scroutlin = numpy.zeros((ntst,), dtype=numpy.float32)
        for i in range(ntst):
            with torch.no_grad():
                scroutlin[i] = tmlin.score(obstst[i,:])
                scrtxt = 'Sample %d log score: %.3f' % (i, scroutlin[i])
                print(scrtxt)
        scmntxt = 'Average linear score: %.3f' % (numpy.mean(scroutlin))
        print(scmntxt)

        # Output to frame
        scrfrm = pandas.DataFrame({'Epoch': k+1, 'TrainSize': ntrncr, 'NonLinScore': numpy.mean(scroutnln), 'LinearScore': numpy.mean(scroutlin)}, index=[k])

        if ( (k == 0) and (j == 0)):
            prlstout = scrfrm
        else:
            prlstout = pandas.concat([prlstout, scrfrm], ignore_index=True)

scrfl = 'LENS_NAmer_H2OSNO_%d_BTMScoring.csv' % (tgtdt)
prlstout.to_csv(scrfl, index=False)

def qsummary(df,vrlst):
    # Summarize with quantiles
    nmtch = df.shape[0] 
    dfout = pandas.DataFrame({'NSmp' : nmtch}, index=[0])
    for j in range(len(vrlst)):
        tmpdt = df[vrlst[j]]
        dtvld = tmpdt[numpy.isfinite(tmpdt)]
        dtvld = dtvld[dtvld != 0.0]
        vrnm = '%s_Mean' % (vrlst[j])
        dfout[vrnm] = numpy.mean(dtvld)

    return dfout

grpbtm = prlstout.groupby(['TrainSize'])
btmqs = grpbtm.apply(qsummary,vrlst=['NonLinScore','LinearScore'],include_groups=False)
btmqs.reset_index(drop=False,inplace=True)

print(btmqs)
