# Generate ensemble member sequences for computing average log scores

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import pandas
from numpy import random
import datetime

random.seed(623114)
rcpscn = False
tgtdt = 20050401
trnlst = numpy.array([24, 32], dtype=numpy.int16)
ntrncs = trnlst.size
nepoch = 30

mbrlst = []
fllst = []
shmsg = numpy.array([-99], dtype=numpy.int16)

qfnm = 'LENS_NAmer_H2OSNO_%d_Quantile.nc' % (tgtdt)
ncclm = Dataset(qfnm,'r')
snofl = ncclm.variables['H2OSNO_ensemble'][:,:,:]
ncclm.close()

nmbr = snofl.shape[0]
print(nmbr)

smpsq = numpy.zeros( (ntrncs,nepoch,nmbr), dtype=numpy.int16 ) - 99
msq = numpy.arange(nmbr)
for j in range(ntrncs):
    ntrncr = trnlst[j]
    for k in range(nepoch): 
        csq = random.choice(msq,size=ntrncr,replace=False)
        print(numpy.sort(csq))
        smpsq[j,k,0:ntrncr] = numpy.sort(csq) 

# Set up output and sample
qfnm = 'LENS_NAmer_H2OSNO_TestSets_%d.nc' % (tgtdt)
qout = Dataset(qfnm,'w') 

dimep = qout.createDimension('epoch',nepoch)
dimtrn = qout.createDimension('trainsize',ntrncs)
dimens = qout.createDimension('member',nmbr)

vartrn = qout.createVariable('trainsize','i2',['trainsize'], fill_value = -99)
vartrn[:] = trnlst
vartrn.long_name = 'Ensemble training size scenarios'
vartrn.units = 'none'
vartrn.missing_value = -99

varseq = qout.createVariable('member_sequence','i2',['trainsize','epoch','member'], fill_value = -99)
varseq[:] = smpsq
varseq.long_name = 'Ensemble member training sequences'
varseq.units = 'none'
varseq.missing_value = -99

qout.close()