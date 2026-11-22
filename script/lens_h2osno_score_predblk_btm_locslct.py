# Compute log scores linear and nonlinear BTM fits across multiple epochs

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import random
import datetime
import sys

from numpy import linalg
import scipy
import properscoring

import torch
import gpytorch

from veccs.orderings2 import farthest_first_ordering, preceding_neighbors
from batram.legmods import Data, SimpleTM

srng = random.default_rng(623114)

# Command line argument is name of case file
rffl = sys.argv[1]
csfrm = pandas.read_csv(rffl, dtype = {'DateFmt8':int, 'RCP':str, 'LocName':str})
rwidx = int(sys.argv[2])

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

tgtdt = csfrm['DateFmt8'].values[rwidx]
rcptxt = csfrm['RCP'].values[rwidx]
stntxt = csfrm['LocName'].values[rwidx]
if rcptxt == 'Yes':
    rcpscn = True
else:
    rcpscn = False

case_str = 'BTM scoring for %d, RCP: %s\n Prediction location %s' % (tgtdt,rcptxt,stntxt)
print(case_str)

shmsg = numpy.array([-99], dtype=numpy.int16)

### Read in training sequences
sqfnm = 'LENS_NAmer_H2OSNO_TestSets_%d.nc' % (tgtdt)
ncseq = Dataset(sqfnm,'r')
trnchc = ncseq.variables['trainsize'][:]
trnseq = ncseq.variables['member_sequence'][:,:,:]
ncseq.close()
print(trnchc.shape)
print(trnchc)
nepoch = trnseq.shape[1]
ntrncs =  trnchc.shape[0]


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
snofl = ncclm.variables['H2OSNO_ensemble'][:,:,:]
# Read prediction blocks 
prdmsk = ncclm.variables['predict_mask'][:,:]
ncclm.close()

nmbr = snofl.shape[0]
print(nmbr)

# Read ensemble members, re-structure
nlat = ltidx.shape[0]
nlon = lnidx.shape[0]
lndv = (lon[1] - lon[0]) / 2.0
ltdv = (lat[1] - lat[0]) / 2.0

# Loop through and transform
mskchk = snomsk[snomsk == 1]
nvld = mskchk.shape[0]
print(mskchk.shape)
print(snomsk.shape)

mskprd = prdmsk[prdmsk == 1]
ncnd = mskprd.shape[0]
print(nvld)
print(ncnd)


zout = numpy.zeros((nmbr,nvld), dtype=numpy.float32)
yout = numpy.zeros((nmbr,nvld), dtype=numpy.float32)
lcsout = numpy.zeros((nvld,2), dtype=numpy.float32)
xidx = numpy.zeros((nvld,), dtype=numpy.int16)
yidx = numpy.zeros((nvld,), dtype=numpy.int16)
clstvc = numpy.zeros((nvld,), dtype=numpy.int16)
prdmvc = numpy.zeros((nvld,), dtype=numpy.int16)
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
            prdmvc[lcctr] = prdmsk[q,p]
            xidx[lcctr] = p
            yidx[lcctr] = q
            lcctr = lcctr + 1
nclst = int(numpy.amax(clstvc)) + 1

# Scale locations
lcsadj = numpy.zeros(lcsout.shape,dtype=lcsout.dtype)
lcsadj[:,0] = (lcsout[:,0] + 150.0) / 6500.0
lcsadj[:,1] = (lcsout[:,1] - 6750.0) / 6500.0

# Subset locations
lcsq = numpy.arange(nvld)
lcscnd = lcsq[prdmvc == 1]
lcsprd = lcsq[prdmvc == -1]

# Location ordering
order_cnd, dsts_cnd = farthest_first_ordering(lcsout[lcscnd,:], start_index=None) 
print(numpy.amax(order_cnd))
ncnd = order_cnd.shape[0]
order_prd, dsts_prd = farthest_first_ordering(lcsout[lcsprd,:], start_index=None) 
print(numpy.amax(order_prd))
nprd = order_prd.shape[0]

print(type(order_prd))

# 'Master' order
order = numpy.concatenate((lcscnd[order_cnd], lcsprd[order_prd]))
lcsord = lcsout[order, ...]


# Conditioning set
largest_conditioning_set = 30
nn, nbrdst = preceding_neighbors(lcsord, numpy.arange(lcsord.shape[0]), largest_conditioning_set) 


### Point locations for diagnostics, identify their location indices
ctyfrm = pandas.read_csv("LENS_Cities_LocIdx.csv", \
                        dtype = {'Label':str, 'Location':str, 'Latitude':float, 'Longitude':float, \
                                 'LonIdx':int, 'LatIdx':int}, encoding='utf-8-sig')
ncity = ctyfrm.shape[0]

ctymsk = [stntxt]
icty = []
pcty = []
for p in range(len(ctymsk)):
    crfrm = ctyfrm[ ctyfrm['Label'] == ctymsk[p]]
    ctyiidx = crfrm['LonIdx'].values[0] - lnidx[0]
    ctyjidx = crfrm['LatIdx'].values[0] - ltidx[0]
    ispt = lcsq[(yidx == ctyjidx) & (xidx == ctyiidx)]
    print(ctymsk[p])
    print(ispt)
    print(lcsq[order == ispt])
    pspt = lcsq[order == ispt]

    icty.append(ispt[0])
    pcty.append(pspt[0])

# Create output file for results
# Need a dimension for training member: 8, 16, 24, 32
flflt = -9999
prbsv = numpy.array([0.05,0.50,0.95],dtype=numpy.float32)
nprbout = prbsv.shape[0]

prdfnm = 'LENS_NAmer_H2OSNO_PredBlockScoring_%d.nc' % (tgtdt)
qout = Dataset(prdfnm,'w') 

dimep = qout.createDimension('epoch',nepoch)
dimtrn = qout.createDimension('trainsize',ntrncs)
dimloc = qout.createDimension('location',len(ctymsk))
dimmbr = qout.createDimension('member',nmbr)
dimpct = qout.createDimension('probability',nprbout)

vartrn = qout.createVariable('trainsize','i2',['trainsize'], fill_value = -99)
vartrn[:] = trnchc
vartrn.long_name = 'Ensemble training size scenarios'
vartrn.units = 'none'
vartrn.missing_value = -99

ctynparr = numpy.array(ctymsk, dtype='object')
varloc = qout.createVariable('location_abbrev', str, ['location']) # use Python's str type
varloc[:] = ctynparr
varloc.long_name = 'Location abbreviation'

varprb = qout.createVariable('probability','f4',['probability'], fill_value = flflt)
varprb[:] = prbsv
varprb.long_name = 'Probability levels for predictive distribution quantiles'
varprb.units = ''
varprb.missing_value = flflt

varobs = qout.createVariable('observed_swe','f4',['trainsize','epoch','member','location'], fill_value = flflt)
varobs.long_name = 'Observed SWE corresponding to prediction location'
varobs.units = 'mm'
varobs.missing_value = flflt

varpit1 = qout.createVariable('btm_pit_nonlinear','f4',['trainsize','epoch','member','location'], fill_value = flflt)
varpit1.long_name = 'Validation PIT for nonlinear BTM prediction'
varpit1.units = ''
varpit1.missing_value = flflt

varpit2 = qout.createVariable('btm_pit_linear','f4',['trainsize','epoch','member','location'], fill_value = flflt)
varpit2.long_name = 'Validation PIT for linear BTM prediction'
varpit2.units = ''
varpit2.missing_value = flflt

varcrps1 = qout.createVariable('btm_crps_nonlinear','f4',['trainsize','epoch','member','location'], fill_value = flflt)
varcrps1.long_name = 'Validation CRPS for nonlinear BTM prediction'
varcrps1.units = 'mm'
varcrps1.missing_value = flflt

varcrps2 = qout.createVariable('btm_crps_linear','f4',['trainsize','epoch','member','location'], fill_value = flflt)
varcrps2.long_name = 'Validation CRPS for linear BTM prediction'
varcrps2.units = 'mm'
varcrps2.missing_value = flflt

varprd1 = qout.createVariable('btm_pred_quantile_nonlinear','f4',['trainsize','epoch','member','location','probability'], fill_value = flflt)
varprd1.long_name = 'Predictive distribution quantiles from nonlinear BTM'
varprd1.units = 'mm'
varprd1.missing_value = flflt

varprd2 = qout.createVariable('btm_pred_quantile_linear','f4',['trainsize','epoch','member','location','probability'], fill_value = flflt)
varprd2.long_name = 'Predictive distribution quantiles from linear BTM'
varprd2.units = 'mm'
varprd2.missing_value = flflt

# Block average quantities
varobsblk = qout.createVariable('observed_swe_blkavg','f4',['trainsize','epoch','member'], fill_value = flflt)
varobsblk.long_name = 'Block average Observed SWE for held-out locations'
varobsblk.units = 'mm'
varobsblk.missing_value = flflt

varpit1blk = qout.createVariable('btm_pit_nonlinear_blkavg','f4',['trainsize','epoch','member'], fill_value = flflt)
varpit1blk.long_name = 'Validation PIT for block average nonlinear BTM prediction'
varpit1blk.units = ''
varpit1blk.missing_value = flflt

varpit2blk = qout.createVariable('btm_pit_linear_blkavg','f4',['trainsize','epoch','member'], fill_value = flflt)
varpit2blk.long_name = 'Validation PIT for block average linear BTM prediction'
varpit2blk.units = ''
varpit2blk.missing_value = flflt

varcrps1blk = qout.createVariable('btm_crps_nonlinear_blkavg','f4',['trainsize','epoch','member'], fill_value = flflt)
varcrps1blk.long_name = 'Validation CRPS for block average nonlinear BTM prediction'
varcrps1blk.units = 'mm'
varcrps1blk.missing_value = flflt

varcrps2blk = qout.createVariable('btm_crps_linear_blkavg','f4',['trainsize','epoch','member'], fill_value = flflt)
varcrps2blk.long_name = 'Validation CRPS for block average linear BTM prediction'
varcrps2blk.units = 'mm'
varcrps2blk.missing_value = flflt

varprd1blk = qout.createVariable('btm_pred_quantile_nonlinear_blkavg','f4',['trainsize','epoch','member','probability'], fill_value = flflt)
varprd1blk.long_name = 'Block average predictive distribution quantiles from nonlinear BTM'
varprd1blk.units = 'mm'
varprd1blk.missing_value = flflt

varprd2blk = qout.createVariable('btm_pred_quantile_linear_blkavg','f4',['trainsize','epoch','member','probability'], fill_value = flflt)
varprd2blk.long_name = 'Block average Predictive distribution quantiles from linear BTM'
varprd2blk.units = 'mm'
varprd2blk.missing_value = flflt

qout.close()

# Set up output arrays
obststout = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk)), dtype=numpy.float32) + flflt
prdqsout1 = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk),prbsv.shape[0]), dtype=numpy.float32) + flflt
prdqsout2 = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk),prbsv.shape[0]), dtype=numpy.float32) + flflt
prdpitout1 = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk)), dtype=numpy.float32) + flflt
prdpitout2 = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk)), dtype=numpy.float32) + flflt
prdcrpsout1 = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk)), dtype=numpy.float32) + flflt
prdcrpsout2 = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk)), dtype=numpy.float32) + flflt

obststoutblk = numpy.zeros( (ntrncs,nepoch,nmbr), dtype=numpy.float32) + flflt
prdqsout1blk = numpy.zeros( (ntrncs,nepoch,nmbr,prbsv.shape[0]), dtype=numpy.float32) + flflt
prdqsout2blk = numpy.zeros( (ntrncs,nepoch,nmbr,prbsv.shape[0]), dtype=numpy.float32) + flflt
prdpitout1blk = numpy.zeros( (ntrncs,nepoch,nmbr), dtype=numpy.float32) + flflt
prdpitout2blk = numpy.zeros( (ntrncs,nepoch,nmbr), dtype=numpy.float32) + flflt
prdcrpsout1blk = numpy.zeros( (ntrncs,nepoch,nmbr), dtype=numpy.float32) + flflt
prdcrpsout2blk = numpy.zeros( (ntrncs,nepoch,nmbr), dtype=numpy.float32) + flflt


### Training sequences
nsmprep = 100   # Number of conditional samples to simulate
smprpseq = numpy.arange(nsmprep)

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
        scroutnln_cnd = numpy.zeros((ntst,), dtype=numpy.float32)
        for i in range(ntst):
            with torch.no_grad():
                scroutnln[i] = tm.score(obstst[i,:])
                scroutnln_cnd[i] = tm.score(obstst[i,:],x_fix=obstst[i,0:ncnd])
                new_sample = tm.cond_sample(x_fix=obstst[i,0:ncnd],num_samples=nsmprep)
                smpstr = 'New sample %d dims: ' % (i)
                print(smpstr)
                print(new_sample.shape) # [num_samples,nvld]
                # For conditioning locations, all values are constant
                #print('Conditioning location')
                #print(new_sample[50:55,200])
                print('Predition location')
                #print(new_sample[50:55,1645])
                print(torch.std(new_sample[:,1645]))

                # SWE predictions for selected locations (icty,pcty)
                for p in range(len(ctymsk)):
                    crfrm = ctyfrm[ ctyfrm['Label'] == ctymsk[p]]
                    ctyiidx = crfrm['LonIdx'].values[0] - lnidx[0]
                    ctyjidx = crfrm['LatIdx'].values[0] - ltidx[0]

                    cndyout = quantile_supp.data_quantile_from_std_norm(new_sample[:,pcty[p]], h2oqs[ctyjidx,ctyiidx,:], prbs, \
                                                                        minval=0.0, maxval=1050.0, msgval=-9999.)
                    cndy90 = numpy.quantile(cndyout,q=prbsv)
                    strprd = 'Location %s prediction interval: [%.2f, %.2f],  Test obs: %.2f' % (ctymsk[p], cndy90[0], cndy90[2], yout[tstlst[i],icty[p]])
                    print(strprd)
                    obststout[j,k,i,p] = yout[tstlst[i],icty[p]]
                    prdqsout1[j,k,i,p,:] = cndy90[:]

                    # PIT - need to handle zeros
                    ssblwr = smprpseq[ cndyout < yout[tstlst[i],icty[p]]]
                    ssbupr = smprpseq[ cndyout <= yout[tstlst[i],icty[p]]]

                    smlw = ssblwr.shape[0]
                    smhi = ssbupr.shape[0]

                    pltlw = (smlw + 0.5) / (nsmprep + 1.0)
                    plthi = (smhi + 0.5) / (nsmprep + 1.0)
                    u1 = srng.uniform(size=1)

                    prbdif = plthi - pltlw
                    prdpitout1[j,k,i,p] = pltlw + prbdif * u1[0]
                    prdcrpsout1[j,k,i,p] = properscoring.crps_ensemble(yout[tstlst[i],icty[p]], cndyout)
 
                ## Block average diagnostics 
                # Data quantiles for all samples
                cndblky = numpy.zeros((nsmprep,nprd),dtype=numpy.float32)
                for p0 in range(nprd):
                    cndblky[:,p0] = quantile_supp.data_quantile_from_std_norm(new_sample[:,ncnd+p0], \
                                                                              h2oqs[yidx[lcsprd[order_prd[p0]]],xidx[lcsprd[order_prd[p0]]],:],\
                                                                              prbs, minval=0.0, maxval=1050.0, msgval=-9999.)
                cndblkmn = numpy.mean(cndblky, axis=1)
                cndblk90 = numpy.quantile(cndblkmn,q=prbsv)
                    
                obststoutblk[j,k,i] = numpy.mean(yout[tstlst[i],lcsprd])
                prdqsout1blk[j,k,i,:] = cndblk90[:]

                # PIT - need to handle zeros
                ssblwr = smprpseq[ cndblkmn < obststoutblk[j,k,i] ]
                ssbupr = smprpseq[ cndblkmn <= obststoutblk[j,k,i] ]

                smlw = ssblwr.shape[0]
                smhi = ssbupr.shape[0]

                pltlw = (smlw + 0.5) / (nsmprep + 1.0)
                plthi = (smhi + 0.5) / (nsmprep + 1.0)
                u1 = srng.uniform(size=1)

                prbdif = plthi - pltlw
                prdpitout1blk[j,k,i] = pltlw + prbdif * u1[0]
                prdcrpsout1blk[j,k,i] = properscoring.crps_ensemble(obststoutblk[j,k,i], cndblkmn)

        scmntxt = 'Average marginal nonlinear score: %.3f' % (numpy.mean(scroutnln))
        print(scmntxt)
        scndtxt = 'Average conditional nonlinear score: %.3f' % (numpy.mean(scroutnln_cnd))
        print(scndtxt)
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
        scroutlin_cnd = numpy.zeros((ntst,), dtype=numpy.float32)
        for i in range(ntst):
            with torch.no_grad():
                scroutlin[i] = tmlin.score(obstst[i,:])
                scroutlin_cnd[i] = tmlin.score(obstst[i,:],x_fix=obstst[i,0:ncnd])
                new_sample_lin = tmlin.cond_sample(x_fix=obstst[i,0:ncnd],num_samples=100)

                # SWE predictions for selected locations (icty,pcty)
                for p in range(len(ctymsk)):
                    crfrm = ctyfrm[ ctyfrm['Label'] == ctymsk[p]]
                    ctyiidx = crfrm['LonIdx'].values[0] - lnidx[0]
                    ctyjidx = crfrm['LatIdx'].values[0] - ltidx[0]

                    cndyout = quantile_supp.data_quantile_from_std_norm(new_sample_lin[:,pcty[p]], h2oqs[ctyjidx,ctyiidx,:], prbs, \
                                                                        minval=0.0, maxval=1050.0, msgval=-9999.)
                    cndy90 = numpy.quantile(cndyout,q=prbsv)
                    prdqsout2[j,k,i,p,:] = cndy90[:]

                    # PIT - need to handle zeros
                    ssblwr = smprpseq[ cndyout < yout[tstlst[i],icty[p]]]
                    ssbupr = smprpseq[ cndyout <= yout[tstlst[i],icty[p]]]

                    smlw = ssblwr.shape[0]
                    smhi = ssbupr.shape[0]

                    pltlw = (smlw + 0.5) / (nsmprep + 1.0)
                    plthi = (smhi + 0.5) / (nsmprep + 1.0)
                    u2 = srng.uniform(size=1)

                    prbdif = plthi - pltlw
                    prdpitout2[j,k,i,p] = pltlw + prbdif * u2[0]
                    prdcrpsout2[j,k,i,p] = properscoring.crps_ensemble(yout[tstlst[i],icty[p]], cndyout)

                ## Block average diagnostics 
                # Data quantiles for all samples
                cndblky = numpy.zeros((nsmprep,nprd),dtype=numpy.float32)
                for p0 in range(nprd):
                    cndblky[:,p0] = quantile_supp.data_quantile_from_std_norm(new_sample_lin[:,ncnd+p0], \
                                                                              h2oqs[yidx[lcsprd[order_prd[p0]]],xidx[lcsprd[order_prd[p0]]],:],\
                                                                              prbs, minval=0.0, maxval=1050.0, msgval=-9999.)
                cndblkmn = numpy.mean(cndblky, axis=1)
                cndblk90 = numpy.quantile(cndblkmn,q=prbsv)
                    
                obststoutblk[j,k,i] = numpy.mean(yout[tstlst[i],lcsprd])
                prdqsout2blk[j,k,i,:] = cndblk90[:]

                # PIT - need to handle zeros
                ssblwr = smprpseq[ cndblkmn < obststoutblk[j,k,i] ]
                ssbupr = smprpseq[ cndblkmn <= obststoutblk[j,k,i] ]

                smlw = ssblwr.shape[0]
                smhi = ssbupr.shape[0]

                pltlw = (smlw + 0.5) / (nsmprep + 1.0)
                plthi = (smhi + 0.5) / (nsmprep + 1.0)
                u2 = srng.uniform(size=1)

                prbdif = plthi - pltlw
                prdpitout2blk[j,k,i] = pltlw + prbdif * u2[0]
                prdcrpsout2blk[j,k,i] = properscoring.crps_ensemble(obststoutblk[j,k,i], cndblkmn)

        scmntxt = 'Average marginal linear score: %.3f' % (numpy.mean(scroutlin))
        print(scmntxt)
        scndtxt = 'Average conditional nonlinear score: %.3f' % (numpy.mean(scroutlin_cnd))
        print(scndtxt)

        # Output to frame
        scrfrm = pandas.DataFrame({'Epoch': k+1, 'TrainSize': ntrncr, 'NonLinScore': numpy.mean(scroutnln), 'NonLinScoreCond': numpy.mean(scroutnln_cnd), \
                                   'LinearScore': numpy.mean(scroutlin), 'LinearScoreCond': numpy.mean(scroutlin_cnd)}, index=[k])

        if ( (k == 0) and (j == 0)):
            prlstout = scrfrm
        else:
            prlstout = pandas.concat([prlstout, scrfrm], ignore_index=True)

        # Save update
        scrfl = 'LENS_NAmer_H2OSNO_%d_BTMScoring_PredBlock.csv' % (tgtdt)
        prlstout.to_csv(scrfl, index=False)

        # Save prediction results
        qout = Dataset(prdfnm,'r+') 

        varobs = qout.variables['observed_swe']
        varobs[:,:,:,:] = obststout[:,:,:,:]

        varpit1 = qout.variables['btm_pit_nonlinear']
        varpit1[:,:,:,:] = prdpitout1[:,:,:,:]

        varpit2 = qout.variables['btm_pit_linear']
        varpit2[:,:,:,:] = prdpitout2[:,:,:,:]

        varcrps1 = qout.variables['btm_crps_nonlinear']
        varcrps1[:,:,:,:] = prdcrpsout1[:,:,:,:]

        varcrps2 = qout.variables['btm_crps_linear']
        varcrps2[:,:,:,:] = prdcrpsout2[:,:,:,:]

        varprd1 = qout.variables['btm_pred_quantile_nonlinear']
        varprd1[:,:,:,:,:] = prdqsout1[:,:,:,:,:]

        varprd2 = qout.variables['btm_pred_quantile_linear']
        varprd2[:,:,:,:,:] = prdqsout2[:,:,:,:,:]

        # Block average
        varobsblk = qout.variables['observed_swe_blkavg']
        varobsblk[:,:,:] = obststoutblk[:,:,:]

        varpit1blk = qout.variables['btm_pit_nonlinear_blkavg']
        varpit1blk[:,:,:] = prdpitout1blk[:,:,:]

        varpit2blk = qout.variables['btm_pit_linear_blkavg']
        varpit2blk[:,:,:] = prdpitout2blk[:,:,:]

        varcrps1blk = qout.variables['btm_crps_nonlinear_blkavg']
        varcrps1blk[:,:,:] = prdcrpsout1blk[:,:,:]

        varcrps2blk = qout.variables['btm_crps_linear_blkavg']
        varcrps2blk[:,:,:] = prdcrpsout2blk[:,:,:]

        varprd1blk = qout.variables['btm_pred_quantile_nonlinear_blkavg']
        varprd1blk[:,:,:,:] = prdqsout1blk[:,:,:,:]

        varprd2blk = qout.variables['btm_pred_quantile_linear_blkavg']
        varprd2blk[:,:,:,:] = prdqsout2blk[:,:,:,:]

        qout.close()

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
