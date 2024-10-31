import numpy as np
import numpy.ma as ma
from scipy import stats
from numpy import random, ndarray, linalg

def quantile_msgdat(vcdat, probs, msgval=-9999.):
    # Compute quantiles with missing data
    if (np.amax(probs) <= 1.0):
        prb100 = 100.0 * probs
    else:
        prb100 = probs

    dtsb = vcdat[vcdat != msgval]

    if dtsb.shape[0] > 0:
        qsout = np.percentile(dtsb,q=prb100)
    else:
        qsout = np.zeros(probs.shape) + msgval
    return qsout

def quantile_msgdat_discrete(vcdat, probs, msgval=-99):
    # Compute quantiles with missing data, discrete version
    if (np.amax(probs) <= 1.0):
        prb100 = 100.0 * probs
    else:
        prb100 = probs

    dtsb = vcdat[vcdat != msgval]

    if dtsb.shape[0] > 0:
        qsout = np.percentile(dtsb,q=prb100,interpolation='nearest')
    else:
        qsout = np.zeros(probs.shape) + msgval
    return qsout

def std_norm_quantile_from_obs(vcdat, obsqs, probs, msgval=-9999.):
    # Compute transform from observed quantiles to standard normal quantiles
    # Based on R function nrmrng from hydrology applications in vic_spec_fns.R

    nprb = probs.shape[0]
    qprb = stats.norm.ppf(probs)
    ndat = vcdat.shape[0]
    dsq = np.arange(ndat)
    vsq = dsq[vcdat != msgval]
    nvld = vsq.shape[0]
    zout = np.zeros((ndat,)) + msgval

    if (obsqs.shape != probs.shape):
        print("Input and output quantile lengths must match")
    elif (nvld == 0):
        print("All observations missing, no transformation performed")
    else:
        ptst = np.append(0.0,np.append(probs,1.0))
        etst = np.append(-np.inf,np.append(obsqs,np.inf))
        qsq = np.arange(ptst.shape[0])

        # Matrices
        ntst = etst.shape[0]
        dtmt = np.tile(vcdat[vsq],(ntst,1))
        etmt = np.transpose(np.tile(etst,(nvld,1)))

        # Indicators for breakpoints of empirical CDF
        lwind = (etmt < dtmt)
        hiind = (dtmt < etmt)

        smlw = np.sum(lwind,axis=0) - 1
        smhi = ntst - np.sum(hiind,axis=0)
        #if smlw.shape[0] > 520:
        #    for j in range(505,510):
        #        str1 = 'Data[%d]: %.3f,  Lwind: %d, Hiind: %d' % (j,vcdat[vsq[j]],smlw[j],smhi[j])
        #        str2 = '  Quantiles: %.3f, %.3f' % (etst[smlw[j]],etst[smhi[j]])
        #        print(str1)
        #        print(str2)

        # Find probability spot
        prbdif = ptst[smhi] - ptst[smlw]
        pspt = ptst[smlw] + prbdif * random.uniform(size=nvld)
        #print(pspt[505:510])

        zout[vsq] = stats.norm.ppf(pspt)
    return zout


def data_quantile_from_std_norm(zdat, obsqs, probs, minval=-np.inf, maxval=np.inf, msgval=-9999.):
    # Inverse quantile transform: Transform from z-score back to data scale
    # Based on R function nrmrng from hydrology applications in vic_spec_fns.R

    nprb = probs.shape[0]
    qprb = stats.norm.ppf(probs)
    ndat = zdat.shape[0]
    dsq = np.arange(ndat)
    vsq = dsq[zdat != msgval]
    nvld = vsq.shape[0]
    qout = np.zeros((ndat,)) + msgval

    if (obsqs.shape != probs.shape):
        print("Input and output quantile lengths must match")
    elif (nvld == 0):
        print("All observations missing, no transformation performed")
    else:
        # qtst, practical limits of z-score
        qtst = np.append(-99.0,np.append(qprb,99.0))
        etst = np.append(minval,np.append(obsqs,maxval))
        qsq = np.arange(qtst.shape[0])

        # Matrices
        ntst = etst.shape[0]
        dtmt = np.tile(zdat[vsq],(ntst,1))
        qtmt = np.transpose(np.tile(qtst,(nvld,1)))

        # Indicators for breakpoints of empirical CDF
        lwind = (qtmt < dtmt)
        hiind = (dtmt < qtmt)

        smlw = np.sum(lwind,axis=0) - 1
        smhi = ntst - np.sum(hiind,axis=0)

        #print('Sum Low')
        #print(smlw[0:10])
        #print('Sum High')
        #print(smhi[0:10])

        # Interpolate
        wtvc = (zdat[vsq] - qtst[smlw]) / (qtst[smhi] - qtst[smlw])
        qtmp = (1.0-wtvc) * etst[smlw] + wtvc * etst[smhi]

        qout[vsq] = qtmp[:]
    return qout

def data_quantile_from_std_norm_discrete(zdat, obsqs, probs, minval=-np.inf, maxval=np.inf, msgval=-99):
    # Inverse quantile transform: Transform from z-score back to data scale, discrete case
    # Based on R function nrmrng from hydrology applications in vic_spec_fns.R

    nprb = probs.shape[0]
    qprb = stats.norm.ppf(probs)
    ndat = zdat.shape[0]
    dsq = np.arange(ndat)
    vsq = dsq[zdat != msgval]
    nvld = vsq.shape[0]
    qout = np.zeros((ndat,)) + msgval

    if (obsqs.shape != probs.shape):
        print("Input and output quantile lengths must match")
    elif (nvld == 0):
        print("All observations missing, no transformation performed")
    else:
        # qtst, practical limits of z-score
        qtst = np.append(-99.0,np.append(qprb,99.0))
        etst = np.append(minval,np.append(obsqs,maxval))
        qsq = np.arange(qtst.shape[0])

        # Matrices
        ntst = etst.shape[0]
        dtmt = np.tile(zdat[vsq],(ntst,1))
        qtmt = np.transpose(np.tile(qtst,(nvld,1)))

        # Indicators for breakpoints of empirical CDF
        lwind = (qtmt < dtmt)
        hiind = (dtmt < qtmt)

        smlw = np.sum(lwind,axis=0) - 1
        smhi = ntst - np.sum(hiind,axis=0)

        #print('Sum Low')
        #print(smlw[0:10])
        #print('Sum High')
        #print(smhi[0:10])

        # Assign at random
        wtvc = (zdat[vsq] - qtst[smlw]) / (qtst[smhi] - qtst[smlw])
        u1 = random.uniform(size=nvld)

        #vsqlw = vsq[wtvc < u1]
        #vsqhi = vsq[wtvc > u1]
        qtmplw = etst[smlw]
        qtmphi = etst[smhi]
        qtmp = qtmplw
        qtmp[wtvc > u1] = qtmphi[wtvc > u1]
        qout[vsq] = qtmp[:]

    return qout
