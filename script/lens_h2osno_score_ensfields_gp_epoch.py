# Fit separate stationary GPs to clusters

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import random, linalg

import scipy

import torch
import gpytorch
from torch.distributions import MultivariateNormal

random.seed(623114)
tgtdt = 20050401

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
snofl = ncclm.variables['H2OSNO_ensemble'][:,:,:]
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
print(nclst)

# Scale locations
lcsadj = numpy.zeros(lcsout.shape,dtype=lcsout.dtype)
lcsadj[:,0] = (lcsout[:,0] + 150.0) / 6500.0
lcsadj[:,1] = (lcsout[:,1] - 6750.0) / 6500.0

# We will use the simplest form of GP model, exact inference
xlc = torch.tensor(lcsadj)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, xlcs, zdat, likelihood):
        super(ExactGPModel, self).__init__(xlcs, zdat, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def cornonstat(locs, rng1, rng2):
    # Compute nonstationary exponential correlation 
    # locs:     location array
    # mat1:     covariance centered at location 1
    # mat2:     covariance centered at location 2

    rngcmb = 0.5 * (rng1 + rng2)

    dstdv = (locs[0,:] - locs[1,:])
    #qij = numpy.dot(numpy.transpose(dstdv),dstdv)
    qij = linalg.multi_dot([numpy.transpose(dstdv),dstdv])
    
    cvout = numpy.sqrt(rng1) * numpy.sqrt(rng2) * numpy.exp(-1.0 * numpy.sqrt(qij) / rngcmb) / rngcmb

    return cvout


### Training sequences
optchc = "Adam"
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


        ztrn = zout[trnlst,:]
        ztst = zout[tstlst,:]

        # Loop through clusters
        lsq = numpy.arange(nvld)

        for q in range(nclst):
            strst = 'Optimizing Cluster %d' % (q+1)
            print(strst)

            lcr = lsq[clstvc == q]
            xcr = torch.tensor(lcsadj[lcr,:])
            zchk = torch.from_numpy(ztrn[0,lcr]).float()

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(xcr, zchk, likelihood)

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
                optimizer = torch.optim.LBFGS(model.parameters(), lr=0.005)
                training_iter = 60

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
                    if ( (i % 10) == 5):
                        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                            i + 1, training_iter, loss.item(),
                            model.covar_module.base_kernel.lengthscale.item(),
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

        # Loop through clusters
        lsq = numpy.arange(nvld)
        cvall = numpy.zeros((nclst,nvld,nvld), dtype=numpy.float32)
        lcstns = torch.tensor(lcsadj)

        for c1 in range(nclst):
            lcr = lsq[clstvc == c1]
            llst = lcr.tolist()
            xcr = torch.tensor(lcsadj[lcr,:])

            covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
            covar_module.lengthscale = prlstout['Lengthscale'].values[c1]
            covar = covar_module(lcstns) 
            cvmt = prlstout['VarScale'].values[c1] * covar.numpy()
            crmt = covar.numpy()

            cvall[c1,:,:] = cvmt[:,:] 

        # Loop locations for nonstationary
        cvfl = numpy.zeros((nvld,nvld), dtype=numpy.float32)
        for p in range(nvld):
            for q in range(p,nvld):
                clstp = clstvc[p]
                clstq = clstvc[q]
                if p == q:
                    cvfl[p,q] = cvall[clstp,p,p] + prlstout['Nugget'].values[clstp]
                elif clstp == clstq:
                    cvfl[p,q] = cvall[clstp,p,q]
                    cvfl[q,p] = cvall[clstp,q,p]
                else:
                    # nonstationary case, need quad form, etc.
                    lclst = [p,q]
                    lcstmp = lcstns[lclst,:]    
                    crtmp = cornonstat(lcstmp,prlstout['Lengthscale'].values[clstp],prlstout['Lengthscale'].values[clstq])
                    cvfl[p,q] = numpy.sqrt(prlstout['VarScale'].values[clstp] * prlstout['VarScale'].values[clstq]) * crtmp
                    cvfl[q,p] = cvfl[p,q]

        # Force symmetric
        ctrcvsym = 0.5 * (cvfl + numpy.transpose(cvfl)) 
        cvvl, cvvc = linalg.eig(ctrcvsym)
        print(numpy.amin(cvvl))
        print(scipy.linalg.issymmetric(ctrcvsym))

        # Log likelihood for sampled locs
        mvnns = MultivariateNormal(torch.zeros(nvld), torch.tensor(ctrcvsym))
        scroutnst = numpy.zeros((ntst,), dtype=numpy.float32)

        for q in range(ntst):
            scrtst = mvnns.log_prob(torch.tensor(ztst[q,:]))
            scrtxt = 'Sample %d log like: %.3f' % (tstlst[q], scrtst)
            print(scrtxt)
            scroutnst[q] = scrtst
        scmntxt = 'Average log like: %.3f' % (numpy.mean(scroutnst))
        print(scmntxt)

        # Output to frame
        scrfrm = pandas.DataFrame({'Epoch': k+1, 'TrainSize': ntrncr, 'NonStatScore': numpy.mean(scroutnst)}, index=[k])

        if ( (k == 0) and (j == 0)):
            scrlstout = scrfrm
        else:
            scrlstout = pandas.concat([scrlstout, scrfrm], ignore_index=True)

scrfl = 'LENS_NAmer_H2OSNO_%d_GPScoring.csv' % (tgtdt)
scrlstout.to_csv(scrfl, index=False)

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
btmqs = grpbtm.apply(qsummary,vrlst=['NonStatScore'],include_groups=False)
btmqs.reset_index(drop=False,inplace=True)

print(btmqs)
