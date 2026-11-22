# Selected "city" empirical distributions/locations for LENS
# Summarize across multiple times/scenarios
# Save as publication quality

import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import quantile_supp
import pandas
import datetime
import sys

import matplotlib
from matplotlib import pyplot
from matplotlib import colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Read scenarios
rffl = sys.argv[1]
csfrmfl = pandas.read_csv(rffl, dtype = {'DateFmt8':int, 'RCP':str})

# Subset case list
csfrm = csfrmfl[ (csfrmfl['DateFmt8'] == 20750101) | (csfrmfl['DateFmt8'] == 20750201) | (csfrmfl['DateFmt8'] == 20750301) | (csfrmfl['DateFmt8'] == 20750401)] 
ncase = csfrm.shape[0]
cssq = numpy.arange(ncase)

# Date sequences
cryr = 1950
yrlst = []
yrsts = []
yrseq = []
mnseq = []
lblst = []
for j in range(ncase):
    dstr = str(csfrm['DateFmt8'].values[j])
    crdt = datetime.datetime.strptime(dstr, '%Y%m%d')
    dstr = crdt.strftime('%Y-%m-%d')
    lblst.append(dstr)
    yrseq.append(crdt.year)
    mnseq.append(crdt.month)
    if cryr != crdt.year:
        yrsts.append(j)
        yrlst.append(crdt.year)
        cryr = crdt.year
print(yrlst)
print(yrseq)
nyr = len(yrlst)
yrarr = numpy.array(yrseq,dtype=numpy.int32)
mnarr = numpy.array(mnseq,dtype=numpy.int32)
yrsts = numpy.array(yrsts,dtype=numpy.int32)
yrlst = numpy.array(yrlst,dtype=numpy.int32)

#yrsts[yrlst > 2040] = yrsts[yrlst > 2040] + 1
#tspts = numpy.array([yrsts[1], yrsts[2]+4, yrsts[4]],numpy.int32)
#rwdth = numpy.array([4.0, 1.0, 4.0])

# Read city list
ctyfrm = pandas.read_csv("LENS_Cities.csv", \
                        dtype = {'Label':str, 'Location':str, 'Latitude':float, 'Longitude':float}, \
                        encoding='utf-8-sig')
ncity = ctyfrm.shape[0]
xidxcty = numpy.zeros((ncity,),dtype=numpy.int16)
yidxcty = numpy.zeros((ncity,),dtype=numpy.int16)

nonzrs = numpy.zeros((ncase,ncity), dtype=numpy.float32)
qmed = numpy.zeros((ncase,ncity), dtype=numpy.float32)
q95 = numpy.zeros((ncase,ncity), dtype=numpy.float32)

# Loop through quantiles, add metrics to data frame
for j in range(ncase):

    # Quantile info
    tgtdtfl = datetime.date(2075,4,1)
    dtstr = tgtdtfl.strftime('%Y-%m-%d')
    tgtdt = csfrm['DateFmt8'].values[j]
    qfnm = 'LENS2_NAmer_H2OSNO_%d_Quantile_ParamFit.nc' % (tgtdt)
    ncclm = Dataset(qfnm,'r')
    h2oqs = ncclm.variables['H2OSNO_quantile'][:,:,:]
    pnzro = ncclm.variables['H2OSNO_prop_nonzero'][:,:]
    ncclm.close()

    if j == 0:
        ncclm = Dataset(qfnm,'r')
        ltidx = ncclm.variables['lat_index'][:]
        lnidx = ncclm.variables['lon_index'][:]
        lat = ncclm.variables['latitude'][:]
        lon = ncclm.variables['longitude'][:]
        prbs = ncclm.variables['probability'][:]
        ncclm.close()

        nprb = prbs.shape[0]
        prbsq = numpy.arange(nprb)
        q50spt = prbsq[prbs == 0.50]
        q95spt = prbsq[(prbs >= 0.949) & (prbs <= 0.951)]
        print(q50spt)
        print(q95spt)

        nlat = ltidx.shape[0]
        nlon = lnidx.shape[0]
        # Loop through and match location index
        for k in range(ncity):
            latdif = numpy.absolute(lat - ctyfrm['Latitude'].values[k])
            ycr = ltidx[latdif == numpy.amin(latdif)]
            londif = numpy.absolute(lon - ctyfrm['Longitude'].values[k] - 360)
            xcr = lnidx[londif == numpy.amin(londif)]
            xidxcty[k] = xcr[0]
            yidxcty[k] = ycr[0]

        ctyfrm['LonIdx'] = xidxcty
        ctyfrm['LatIdx'] = yidxcty

    # Compute summaries
    for k in range(ncity):
        xspt = xidxcty[k] - lnidx[0]
        yspt = yidxcty[k] - ltidx[0]
        qmed[j,k] = h2oqs[yspt,xspt,q50spt]
        q95[j,k] = h2oqs[yspt,xspt,q95spt]
        #nonzrs[j,k] = pnzro[yspt,xspt]

        qscr = h2oqs[yspt,xspt,:]
        sprb = prbs[qscr >= 2.5]
        print(sprb.shape)
        if sprb.shape[0] > 0:
            nonzrs[j,k] = 1.0 - numpy.amin(sprb) 

print(ctyfrm)

c6 = ["#C7657B","#AC7A23","#718E00","#009A68","#0098A9","#6881C9"]
rcls = ['#DDDDDD','#777777','#DDDDDD']

fig = pyplot.figure(figsize=(6,10.5))

p1 = pyplot.subplot(3,1,1)


for k in range(ncity):
    plt2 = p1.plot(cssq+0.5,nonzrs[:,k],'-', c=c6[k])

#p1.xaxis.grid(color='#898989',linestyle='dotted')
p1.yaxis.grid(color='#898989',linestyle='dotted')
p1.set_ylim(-0.05,1.05)
p1.set_xlim(0,ncase)
p1.set_xlabel('Date',size=11)
p1.set_ylabel('Nonzero Proportion',size=11)
p1.set_xticks(cssq+0.5)
p1.set_xticklabels(lblst)
    
for lb in p1.xaxis.get_ticklabels():
    lb.set_fontsize(9)
for lb in p1.yaxis.get_ticklabels():
    lb.set_fontsize(9)
tstr = 'Ensemble Proportion Nonzero SWE' 
pyplot.title(tstr,size=12)

# Median
p2 = pyplot.subplot(3,1,2)

for k in range(ncity):
    plt2 = p2.plot(cssq+0.5,qmed[:,k],'-', c=c6[k])

#p2.xaxis.grid(color='#898989',linestyle='dotted')
p2.yaxis.grid(color='#898989',linestyle='dotted')
p2.set_ylim(-10,260)
p2.set_xlim(0,ncase)
p2.set_xlabel('Date',size=11)
p2.set_ylabel('Ens Median [mm]',size=11)
p2.set_xticks(cssq+0.5)
p2.set_xticklabels(lblst)
tstr = 'Ensemble Median SWE' 
pyplot.title(tstr,size=12)
    
for lb in p2.xaxis.get_ticklabels():
    lb.set_fontsize(9)
for lb in p2.yaxis.get_ticklabels():
    lb.set_fontsize(9)


# 95th percentile
p3 = pyplot.subplot(3,1,3)

plst = []
for k in range(ncity):
    plt2, = p3.plot(cssq+0.5,q95[:,k],'-', c=c6[k])
    plst.append(plt2)

p3.yaxis.grid(color='#898989',linestyle='dotted')
p3.set_ylim(-10,310)
p3.set_xlim(0,ncase)
p3.set_xlabel('Date',size=11)
p3.set_ylabel('Ens 95th Percentile [mm]',size=11)
p3.set_xticks(cssq+0.5)
p3.set_xticklabels(lblst)
    
for lb in p3.xaxis.get_ticklabels():
    lb.set_fontsize(9)
for lb in p3.yaxis.get_ticklabels():
    lb.set_fontsize(9)

leg = pyplot.legend(plst,['A','B','C','D','E','F'], loc = 'upper left',labelspacing=0.6,borderpad=0.6,ncol=2)
for t in leg.get_texts():
    t.set_fontsize(10)
tstr = 'Ensemble 95th Percentile SWE' 
pyplot.title(tstr,size=12)

fig.subplots_adjust(bottom=0.08,top=0.96,left=0.12,right=0.95, \
                    hspace=0.3,wspace=0.3)
fnm = 'LENS2_NAmer_H2OSNO_City_QuantileSummary_%d.pdf' % (yrlst[0])
pyplot.savefig(fnm)
pyplot.close()


