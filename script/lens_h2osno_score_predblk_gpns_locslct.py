# Compute log scores for nonstationary GP fits across multiple epochs

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
from torch.distributions import MultivariateNormal
import math

srng = random.default_rng(623114)

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

case_str = 'GP scoring for %d, RCP: %s\n Prediction location %s' % (tgtdt,rcptxt,stntxt)
print(case_str)

# Set Matern smoothness
mtrnnu = 1.5

def matern_kernel_from_distance(
    dist_matrix: torch.Tensor, lengthscale: float = 1.0, nu: float = 2.5
) -> torch.Tensor:
  """Computes a Matern covariance matrix directly from a distance matrix."""
  scaled_dist = dist_matrix / lengthscale

  if nu == 0.5:
    return torch.exp(-scaled_dist)
  elif nu == 1.5:
    sqrt3_d = math.sqrt(3.0) * scaled_dist
    return (1.0 + sqrt3_d) * torch.exp(-sqrt3_d)
  elif nu == 2.5:
    sqrt5_d = math.sqrt(5.0) * scaled_dist
    return (1.0 + sqrt5_d + (5.0 / 3.0) * (scaled_dist**2)) * torch.exp(-sqrt5_d)
  else:
    raise ValueError("nu must be one of {0.5, 1.5, 2.5}")

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
nprd = lcsprd.shape[0]

order = numpy.concatenate((lcscnd,lcsprd))
prdsq = numpy.arange(nprd)


# Find cluster centroids
lsq = numpy.arange(nvld)

xictr = numpy.zeros( (nclst,), dtype=numpy.int32)
yictr = numpy.zeros( (nclst,), dtype=numpy.int32)
iictr = numpy.zeros( (nclst,), dtype=numpy.int32)

for q in range(nclst):

    lcr = lsq[clstvc == q]
    xcr = torch.tensor(lcsadj[lcr,:])

    # Distance matrix for each cluster
    nlcclst = lcr.shape[0]
    urps = numpy.tile(xcr[:,0], (nlcclst,1))
    vrps = numpy.tile(xcr[:,1], (nlcclst,1))

    udif = urps - numpy.transpose(urps)
    vdif = vrps - numpy.transpose(vrps)
    dstmt = numpy.sqrt( (udif * udif) + (vdif * vdif) )

    dstmd = numpy.median(dstmt,axis=0)

    dstminmd = numpy.amin(dstmd)
    lcmin = lcr[dstmd == dstminmd]
    xictr[q] = xidx[lcmin[0]]
    yictr[q] = yidx[lcmin[0]]
    iictr[q] = lcmin[0]

    strclst = 'Cluster %d Distances. Smallest Median: %.4f,  Median Median: %.4f' % (q+1, numpy.amin(dstmd), numpy.median(dstmd))
    print(strclst)

# Compute distances to each cluster centroid
# mdst = 0.5 (specifies distance threshold for cluster members in GP fitting
dstarr = numpy.zeros( (nclst,nvld), dtype=numpy.float32)
mdst = 0.5           

for q in range(nclst):

    # Distance array to cluster center
    udif = lcsadj[:,0] - lcsadj[iictr[q],0]
    vdif = lcsadj[:,1] - lcsadj[iictr[q],1]
    dstarr[q,:] = numpy.sqrt( (udif*udif + vdif*vdif) )

    lcr = lsq[clstvc == q]
    lcr2 = lsq[ (clstvc == q) & (dstarr[q,:] <= mdst) ]

    strcl = 'Cluster %d:  %d members, %d within %.2f' % (q+1, lcr.shape[0], lcr2.shape[0], mdst)
    print(strcl)

# mxdstbw = 0.01     bandwidth param for rate of decay for mixture weights
mxdstbw = 0.01
expfct = numpy.exp( -1.0 * dstarr * dstarr / (2.0 * mxdstbw))
sumfct = numpy.sum(expfct, axis=0)
sumtl = numpy.tile(sumfct, (nclst,1))
mixwts = expfct / sumtl

# Squared distances in Euclidean
urps = numpy.tile(lcsadj[:,0], (nvld,1))
vrps = numpy.tile(lcsadj[:,1], (nvld,1))
udif = urps - numpy.transpose(urps)
vdif = vrps - numpy.transpose(vrps)
dstmtsqr = (udif * udif) + (vdif * vdif)


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
    pspt = prdsq[lcsprd == ispt]    

    icty.append(ispt[0])
    pcty.append(pspt[0])

# Create output file for results
# Need a dimension for training member: 8, 16, 24, 32
flflt = -9999
prbsv = numpy.array([0.05,0.50,0.95],dtype=numpy.float32)
nprbout = prbsv.shape[0]

prdfnm = 'LENS_NAmer_H2OSNO_PredBlockScoring_GP_%d.nc' % (tgtdt)
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

varpit1 = qout.createVariable('gp_pit_nonstn','f4',['trainsize','epoch','member','location'], fill_value = flflt)
varpit1.long_name = 'Validation PIT for nonstationary GP prediction'
varpit1.units = ''
varpit1.missing_value = flflt

varcrps1 = qout.createVariable('gp_crps_nonstn','f4',['trainsize','epoch','member','location'], fill_value = flflt)
varcrps1.long_name = 'Validation CRPS for nonstationary GP prediction'
varcrps1.units = 'mm'
varcrps1.missing_value = flflt

varprd1 = qout.createVariable('gp_pred_quantile_nonstn','f4',['trainsize','epoch','member','location','probability'], fill_value = flflt)
varprd1.long_name = 'Predictive distribution quantiles from nonstationary GP'
varprd1.units = 'mm'
varprd1.missing_value = flflt

# Block average quantities
varobsblk = qout.createVariable('observed_swe_blkavg','f4',['trainsize','epoch','member'], fill_value = flflt)
varobsblk.long_name = 'Block average Observed SWE for held-out locations'
varobsblk.units = 'mm'
varobsblk.missing_value = flflt

varpit1blk = qout.createVariable('gp_pit_nonstn_blkavg','f4',['trainsize','epoch','member'], fill_value = flflt)
varpit1blk.long_name = 'Validation PIT for block average nonstationary GP prediction'
varpit1blk.units = ''
varpit1blk.missing_value = flflt

varcrps1blk = qout.createVariable('gp_crps_nonstn_blkavg','f4',['trainsize','epoch','member'], fill_value = flflt)
varcrps1blk.long_name = 'Validation CRPS for block average nonstationary GP prediction'
varcrps1blk.units = 'mm'
varcrps1blk.missing_value = flflt

varprd1blk = qout.createVariable('gp_pred_quantile_nonstn_blkavg','f4',['trainsize','epoch','member','probability'], fill_value = flflt)
varprd1blk.long_name = 'Block average predictive distribution quantiles from nonstationary GP'
varprd1blk.units = 'mm'
varprd1blk.missing_value = flflt

qout.close()

# Set up output arrays
obststout = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk)), dtype=numpy.float32) + flflt
prdqsout1 = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk),prbsv.shape[0]), dtype=numpy.float32) + flflt
prdpitout1 = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk)), dtype=numpy.float32) + flflt
prdcrpsout1 = numpy.zeros( (ntrncs,nepoch,nmbr,len(ctymsk)), dtype=numpy.float32) + flflt

obststoutblk = numpy.zeros( (ntrncs,nepoch,nmbr), dtype=numpy.float32) + flflt
prdqsout1blk = numpy.zeros( (ntrncs,nepoch,nmbr,prbsv.shape[0]), dtype=numpy.float32) + flflt
prdpitout1blk = numpy.zeros( (ntrncs,nepoch,nmbr), dtype=numpy.float32) + flflt
prdcrpsout1blk = numpy.zeros( (ntrncs,nepoch,nmbr), dtype=numpy.float32) + flflt

# We will use the simplest form of GP model, exact inference
xlc = torch.tensor(lcsadj)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, xlcs, zdat, likelihood, smooth=0.5):
        super(ExactGPModel, self).__init__(xlcs, zdat, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=smooth))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


### Training sequences
nsmprep = 100   # Number of conditional samples to simulate
smprpseq = numpy.arange(nsmprep)
optchc = "LBGFS"
ntrncs =  trnchc.shape[0]
flsq = numpy.arange(nmbr)
flst = flsq.tolist()

# GP can use Gaussian CRPS evaluation
fitct = 0
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

        ztrn = zout[trnlst,:]
        ztst = zout[tstlst,:]

        # Loop through clusters
        lsq = numpy.arange(nvld)

        for q in range(nclst):
            strst = 'Optimizing Cluster %d' % (q+1)
            print(strst)

            lcr = lsq[(clstvc == q) & (dstarr[q,:] <= mdst)]
            xcr = torch.tensor(lcsadj[lcr,:])
            zchk = torch.from_numpy(ztrn[0,lcr]).float()

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(xcr, zchk, likelihood, smooth=mtrnnu)

            model.covar_module.base_kernel.lengthscale = torch.tensor(0.2)
            model.likelihood.noise = torch.tensor(0.1)

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
  
            if optchc == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Includes GaussianLikelihood parameters
                training_iter = 500


                for i in range(training_iter):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = model(xcr)
                    # Calc loss and backprop gradients
                    loss = 0
                    for t in range(ntrncr):
                        zchk = torch.from_numpy(ztrn[t,lcr]).float()
                        loss = loss-mll(output, zchk)
                    loss.backward()

                    if ( (i % 50) == 10):
                        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                            i + 1, training_iter, loss.item(),
                            model.covar_module.base_kernel.lengthscale.item(),
                            model.likelihood.noise.item()
                        ))
                    optimizer.step()

            elif optchc == "LBGFS":
                training_iter = 10
                optimizer = torch.optim.LBFGS(model.parameters(), history_size=20, max_iter=training_iter, \
                                              line_search_fn='strong_wolfe')

                # Define the closure function
                def closure():
                    optimizer.zero_grad()
                    output = model(xcr)
                    loss = 0
                    for t in range(ntrncr):
                        zchk = torch.from_numpy(ztrn[t,lcr]).float()
                        loss = loss-mll(output, zchk)
                    loss.backward()
                    return loss

                # Optimization loop
                for i in range(training_iter):
                    optimizer.step(closure)
                    loss = closure()
                    #if ( (i % 2) == 1):
                    print('Iter %d/%d - Loss: %.5f   lengthscale: %.4f   gpvar: %.4f   noise: %.4f' % (
                        i + 1, training_iter, loss.item(),
                        model.covar_module.base_kernel.lengthscale.item(),
                        model.covar_module.outputscale.item(),
                        model.likelihood.noise.item()
                    ))

            covarfn = model.covar_module(xcr) 
 
            # Output to frame
            lkfrm = pandas.DataFrame({'Cluster': q+1, 'Locations': covarfn.shape[0], 'Loss': loss.item(), 'Lengthscale': model.covar_module.base_kernel.lengthscale.item(), \
                                      'VarScale': model.covar_module.outputscale.item(), 'Nugget': model.likelihood.noise.item()}, index=[q])
            if q == 0:
                prlstout = lkfrm
            else:
                prlstout = pandas.concat([prlstout, lkfrm], ignore_index=True)

        ## Score test cases

        # Set up nonstationary covariance using mixture approach
        # Assemble weighted parameters
        wtsqrrng = numpy.zeros( (nvld,), dtype=numpy.float32)
        wtgpvar = numpy.zeros( (nvld,), dtype=numpy.float32)
        wtnugget = numpy.zeros( (nvld,), dtype=numpy.float32)

        for q in range(nclst):
            wtsqrrng = wtsqrrng + prlstout['Lengthscale'].values[q] * prlstout['Lengthscale'].values[q] * mixwts[q,:]
            wtgpvar = wtgpvar + prlstout['VarScale'].values[q] * mixwts[q,:]
            wtnugget = wtnugget + prlstout['Nugget'].values[q] * mixwts[q,:]

        # Matrix of range parameters
        rngrps = numpy.tile(wtsqrrng, (nvld,1))
        rangeij = 0.5 * (rngrps + numpy.transpose(rngrps))

        # Q_ij "distances"
        qdsts = numpy.sqrt(dstmtsqr / rangeij)
        qtns = torch.tensor(qdsts)

        # Matern evaluation, unit range param
        K = matern_kernel_from_distance(qtns, lengthscale=1.0, nu=mtrnnu)

        # Re-normalize by Paciorek eq. 7
        nrmcr = numpy.sqrt(rngrps) * numpy.sqrt(numpy.transpose(rngrps)) / rangeij
        cornstn = nrmcr * K.numpy()

        # Incorporate variances, nuggets
        gpsdrp = numpy.tile( numpy.sqrt(wtgpvar), (nvld,1))
        cvfl = gpsdrp * numpy.transpose(gpsdrp) * cornstn + wtnugget * numpy.eye(nvld)

        # Force symmetric
        ctrcvsym = 0.5 * (cvfl + numpy.transpose(cvfl)) 
        cvvl, cvvc = linalg.eig(ctrcvsym)
        print(numpy.amin(cvvl))
        print(scipy.linalg.issymmetric(ctrcvsym))


        # Set up conditional distribution
        if (numpy.amin(cvvl) > 0):

            cvcnd = numpy.zeros((ncnd,ncnd), dtype=numpy.float32)
            cvprd = numpy.zeros((nprd,nprd), dtype=numpy.float32)
            cv12 = numpy.zeros((nprd,ncnd), dtype=numpy.float32)
            for p in range(ncnd):
                for q in range(p,ncnd):
                    cvcnd[p,q] = ctrcvsym[lcscnd[p], lcscnd[q]]
                    cvcnd[q,p] = cvcnd[p,q]
                for q in range(nprd):
                    cv12[q,p] = ctrcvsym[lcsprd[q], lcscnd[p]]

            for p in range(nprd):
                for q in range(p,nprd):
                    cvprd[p,q] = ctrcvsym[lcsprd[p], lcsprd[q]]
                    cvprd[q,p] = cvprd[p,q]
            cndcvsym = 0.5 * (cvcnd + numpy.transpose(cvcnd)) 
            cvcndvl, cvcndvc = linalg.eig(cndcvsym)
            print(numpy.amin(cvcndvl))
            print(scipy.linalg.issymmetric(cndcvsym))

            prdcvsym = 0.5 * (cvprd + numpy.transpose(cvprd)) 
            cvprdvl, cvprdvc = linalg.eig(prdcvsym)
            print(numpy.amin(cvprdvl))
            print(scipy.linalg.issymmetric(prdcvsym))

            # Inverse
            sg2_eps = 0.005
            prccnd = linalg.inv(cndcvsym)
            wtmt = numpy.dot(cv12,prccnd)
            cvadj = numpy.dot(wtmt,numpy.transpose(cv12))
            cndcv = prdcvsym - cvadj
            prdcvsym = 0.5 * (cndcv + numpy.transpose(cndcv)) + numpy.diag(numpy.zeros(nprd) + sg2_eps)   # Incorporate nugget

            print('Conditional cov')
            print(numpy.amin(prdcvsym))
            print(scipy.linalg.issymmetric(prdcvsym))
            print(prdcvsym[40:44,40:44])
   

            # Conditional mean (note marginal is zero)
            zcnd = numpy.dot(wtmt,numpy.transpose(ztst[:,lcscnd]))

            print(zcnd.shape)

            # Score with log likelihood for held out locs, ens members
            mvnns = MultivariateNormal(torch.zeros(nvld), torch.tensor(ctrcvsym)) 
            scroutnst = numpy.zeros((ntst,), dtype=numpy.float32)
            scroutnst_cnd = numpy.zeros((ntst,), dtype=numpy.float32)
            for q in range(ntst):
                scrtst = mvnns.log_prob(torch.tensor(ztst[q,:]))
                scroutnst[q] = scrtst
                print('Avg cond mean')
                print(numpy.mean(zcnd[:,q]))
                zmncr = zcnd[:,q].flatten()
                mvnns_cnd = MultivariateNormal(torch.tensor(zmncr,dtype=torch.float64), torch.tensor(prdcvsym,dtype=torch.float64)) 
                scrtst_cnd = mvnns_cnd.log_prob(torch.tensor(ztst[q,lcsprd])) 
                scroutnst_cnd[q] = scrtst_cnd

                # Sample and score selected locations
                new_sample = mvnns_cnd.sample((nsmprep,))

                # SWE predictions for selected locations (icty,pcty)
                for p in range(len(ctymsk)):
                    crfrm = ctyfrm[ ctyfrm['Label'] == ctymsk[p]]
                    ctyiidx = crfrm['LonIdx'].values[0] - lnidx[0]
                    ctyjidx = crfrm['LatIdx'].values[0] - ltidx[0]

                    cndyout = quantile_supp.data_quantile_from_std_norm(new_sample[:,pcty[p]].numpy(), h2oqs[ctyjidx,ctyiidx,:], prbs, \
                                                                        minval=0.0, maxval=1050.0, msgval=-9999.)
                    cndy90 = numpy.quantile(cndyout,q=prbsv)
                    strprd = 'Location %s prediction interval: [%.2f, %.2f],  Test obs: %.2f' % (ctymsk[p], cndy90[0], cndy90[2], yout[tstlst[q],icty[p]])
                    print(strprd)
                    obststout[j,k,q,p] = yout[tstlst[q],icty[p]]
                    prdqsout1[j,k,q,p,:] = cndy90[:]

                    # PIT - need to handle zeros
                    ssblwr = smprpseq[ cndyout < yout[tstlst[q],icty[p]]]
                    ssbupr = smprpseq[ cndyout <= yout[tstlst[q],icty[p]]]

                    smlw = ssblwr.shape[0]
                    smhi = ssbupr.shape[0]

                    pltlw = (smlw + 0.5) / (nsmprep + 1.0)
                    plthi = (smhi + 0.5) / (nsmprep + 1.0)
                    u1 = srng.uniform(size=1)

                    prbdif = plthi - pltlw
                    prdpitout1[j,k,q,p] = pltlw + prbdif * u1[0]
                    prdcrpsout1[j,k,q,p] = properscoring.crps_ensemble(yout[tstlst[q],icty[p]], cndyout)

                ## Block average diagnostics
                # Data quantiles for all samples 
                cndblky = numpy.zeros((nsmprep,nprd),dtype=numpy.float32)
                for p0 in range(nprd):
                    cndblky[:,p0] = quantile_supp.data_quantile_from_std_norm(new_sample[:,p0], \
                                                                              h2oqs[yidx[lcsprd[p0]],xidx[lcsprd[p0]],:],\
                                                                              prbs, minval=0.0, maxval=1050.0, msgval=-9999.)
                cndblkmn = numpy.mean(cndblky, axis=1)
                cndblk90 = numpy.quantile(cndblkmn,q=prbsv)
                    
                obststoutblk[j,k,q] = numpy.mean(yout[tstlst[q],lcsprd])
                prdqsout1blk[j,k,q,:] = cndblk90[:]

                # PIT - need to handle zeros
                ssblwr = smprpseq[ cndblkmn < obststoutblk[j,k,q] ]
                ssbupr = smprpseq[ cndblkmn <= obststoutblk[j,k,q] ]

                smlw = ssblwr.shape[0]
                smhi = ssbupr.shape[0]

                pltlw = (smlw + 0.5) / (nsmprep + 1.0)
                plthi = (smhi + 0.5) / (nsmprep + 1.0)
                u1 = srng.uniform(size=1)

                prbdif = plthi - pltlw
                prdpitout1blk[j,k,q] = pltlw + prbdif * u1[0]
                prdcrpsout1blk[j,k,q] = properscoring.crps_ensemble(obststoutblk[j,k,q], cndblkmn)

            scmntxt = 'Average marg log like: %.3f' % (numpy.mean(scroutnst))
            print(scmntxt)
            scndtxt = 'Average cond log like: %.3f' % (numpy.mean(scroutnst_cnd))
            print(scndtxt)

            # Output to frame
            scrfrm = pandas.DataFrame({'Epoch': k+1, 'TrainSize': ntrncr, 'NonStatScore': numpy.mean(scroutnst), 'NonStatScoreCond': numpy.mean(scroutnst_cnd)}, index=[k])

            if ( fitct == 0):
                scrlstout = scrfrm
            else:
                scrlstout = pandas.concat([scrlstout, scrfrm], ignore_index=True)

            fitct = fitct + 1
            # Save update
            scrfl = 'LENS_NAmer_H2OSNO_%d_GPScoring_PredBlock.csv' % (tgtdt)
            scrlstout.to_csv(scrfl, index=False)

        else:
            scmntxt = 'Nonstationary cov not positive definite'
            print(scmntxt)

        # Save prediction results
        qout = Dataset(prdfnm,'r+') 

        varobs = qout.variables['observed_swe']
        varobs[:,:,:,:] = obststout[:,:,:,:]

        varpit1 = qout.variables['gp_pit_nonstn']
        varpit1[:,:,:,:] = prdpitout1[:,:,:,:]

        varcrps1 = qout.variables['gp_crps_nonstn']
        varcrps1[:,:,:,:] = prdcrpsout1[:,:,:,:]

        varprd1 = qout.variables['gp_pred_quantile_nonstn']
        varprd1[:,:,:,:,:] = prdqsout1[:,:,:,:,:]

        # Block average
        varobsblk = qout.variables['observed_swe_blkavg']
        varobsblk[:,:,:] = obststoutblk[:,:,:]

        varpit1blk = qout.variables['gp_pit_nonstn_blkavg']
        varpit1blk[:,:,:] = prdpitout1blk[:,:,:]

        varcrps1blk = qout.variables['gp_crps_nonstn_blkavg']
        varcrps1blk[:,:,:] = prdcrpsout1blk[:,:,:]

        varprd1blk = qout.variables['gp_pred_quantile_nonstn_blkavg']
        varprd1blk[:,:,:,:] = prdqsout1blk[:,:,:,:]

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

grpbtm = scrlstout.groupby(['TrainSize'])
btmqs = grpbtm.apply(qsummary,vrlst=['NonStatScore','NonStatScoreCond'],include_groups=False)
btmqs.reset_index(drop=False,inplace=True)

print(btmqs)
