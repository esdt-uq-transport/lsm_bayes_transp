# Generate ensemble member sequences for computing average log scores

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import pandas
from numpy import random
import datetime

random.seed(623114)
rcpscn = True
tgtdt = 20750401
trnlst = numpy.array([24, 32], dtype=numpy.int16)
ntrncs = trnlst.size
nepoch = 30

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