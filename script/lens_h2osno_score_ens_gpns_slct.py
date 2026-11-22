# Fit separate stationary GPs to clusters
# Stitch into global nonstationary model with the mixture approach of Risser (2017)

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
from numpy import linalg
import sys
import os
import scipy

import torch
import gpytorch
from torch.distributions import MultivariateNormal
import math

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(os.environ['OMP_NUM_THREADS'])

# Command line argument is name of case file
rffl = sys.argv[1]
csfrm = pandas.read_csv(rffl, dtype = {'DateFmt8':int, 'RCP':str})
rwidx = int(sys.argv[2])


tgtdt = csfrm['DateFmt8'].values[rwidx]
rcptxt = csfrm['RCP'].values[rwidx]
if rcptxt == 'Yes':
    rcpscn = True
else:
    rcpscn = False

case_str = 'Nonstationary GP scoring for %d, RCP: %s' % (tgtdt,rcptxt)
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
optchc = "LBGFS"
ntrncs =  trnchc.shape[0]
flsq = numpy.arange(nmbr)
flst = flsq.tolist()

fitct = 0
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

            lcr = lsq[ (clstvc == q) & (dstarr[q,:] <= mdst)]
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
                    if ( (i % 2) == 1):
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

        # Log likelihood for sampled locs
        if (numpy.amin(cvvl) > 0):
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

            if ( fitct == 0):
                scrlstout = scrfrm
            else:
                scrlstout = pandas.concat([scrlstout, scrfrm], ignore_index=True)

            fitct = fitct + 1
            # Save update
            scrfl = 'LENS_NAmer_H2OSNO_%d_GPScoring.csv' % (tgtdt)
            scrlstout.to_csv(scrfl, index=False)

        else:
            scmntxt = 'Nonstationary cov not positive definite'
            print(scmntxt)

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
