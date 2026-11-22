# Summarize log scores across multiple scenarios
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


# Read scenarios
rffl = sys.argv[1]
csfrmfl = pandas.read_csv(rffl, dtype = {'DateFmt8':int, 'RCP':str})

# Subset case list
csfrm = csfrmfl[ (csfrmfl['DateFmt8'] == 19950401) | (csfrmfl['DateFmt8'] == 20650401) ] 
ncase = csfrm.shape[0]
cssq = numpy.arange(ncase)

# Date sequences
cryr = 1950
yrlst = []
yrsts = []
yrseq = []
mnseq = []
for j in range(ncase):
    dstr = str(csfrm['DateFmt8'].values[j])
    crdt = datetime.datetime.strptime(dstr, '%Y%m%d')
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


# Loop through scoring results, add metrics to data frame
# List of scoring variables
scrvrlst = ['NonLinScore','Score_Nbr08','Score_Nbr12','Score_Nbr16','Score_Nbr20','Score_Nbr30']
scrvrlbs = ['Opt','8','12','16','20','30']

for j in range(ncase):

    # Quantile info - count number of locations used per case
    tgtdt = csfrm['DateFmt8'].values[j]
    qfnm = 'LENS_NAmer_H2OSNO_%d_Quantile.nc' % (tgtdt)
    ncclm = Dataset(qfnm,'r')
    snomsk = ncclm.variables['LENS_mask'][:,:]
    ncclm.close()

    snoflt = snomsk.flatten()
    snovld = snoflt[snoflt == 1]

    # Read scoring results - BTM only
    #LENS_NAmer_H2OSNO_20650401_BTMScoring_NbrSeq.csv
    #Epoch,TrainSize,NonLinScore,OptThetaQ,OptNbrM,Score_Nbr08,Score_Nbr12,Score_Nbr16,Score_Nbr20,Score_Nbr30
    btmscrfl = 'LENS_NAmer_H2OSNO_%d_BTMScoring_NbrSeq.csv' % (tgtdt)
    btmfrm = pandas.read_csv(btmscrfl, \
                        dtype = {'Epoch':int, 'TrainSize':int, 'NonLinScore':float, 'OptThetaQ':float, 'OptNbrM':int, \
                                 'Score_Nbr08':float, 'Score_Nbr12':float, 'Score_Nbr16':float, 'Score_Nbr20':float, 'Score_Nbr30':float}, encoding='utf-8-sig')
    grpbtm = btmfrm.groupby(['TrainSize'])
    btmqs = grpbtm.apply(qsummary,vrlst=scrvrlst,include_groups=False)
    btmqs.reset_index(drop=False,inplace=True)
    btmqs.drop(['level_1'],axis=1,inplace=True)

    btmqs['Year'] = yrarr[j]
    btmqs['Month'] = mnarr[j]
    btmqs['NLoc'] = snovld.shape[0]

    if j == 0:
        scrfrmfl = btmqs
    else:
        scrfrmfl = pandas.concat([scrfrmfl, btmqs], ignore_index=True)


print(btmqs)

for i in range(len(scrvrlst)):
    vrnmi = '%s_Norm' % (scrvrlst[i])
    vrnmimn = '%s_Mean' % (scrvrlst[i])
    scrfrmfl[vrnmi] = -1.0 * scrfrmfl[vrnmimn] / scrfrmfl['NLoc']

#c6 = ["#C7657B","#AC7A23","#718E00","#009A68","#0098A9","#6881C9"]
c3 = ["#6929c4","#9f1853","#198038"]
rcls = ['#DDDDDD','#777777','#DDDDDD']
tsymb = ['d','v','o','x']
trnsz = [8,16,24, 32]
ntrn = len(trnsz)

lbllst = []
for p in range(len(trnsz)):
    lblstr = 'J=%d' % (trnsz[p])
    lbllst.append(lblstr)

pltht = ncase*5
fig = pyplot.figure(figsize=(6,pltht))

for q in range(ncase):
    p3 = pyplot.subplot(ncase,1,q+1)

    plst = []
    for p in range(ntrn):
        frmsb = scrfrmfl[(scrfrmfl['Year'] == yrlst[q]) & (scrfrmfl['TrainSize'] == trnsz[p])]
        for i in range(len(scrvrlst)):
            crvr = '%s_Norm' % (scrvrlst[i]) 
            plt1, = p3.plot(i+0.5,frmsb[crvr],tsymb[p], ms=4, c=c3[0])
            if i == 0:
                plst.append(plt1)

    p3.yaxis.grid(color='#898989',linestyle='dotted')
    #p3.xaxis.grid(color='#898989',linestyle='dotted')
    p3.set_ylim(0.68,0.88)
    p3.set_xlim(0,len(scrvrlst))
    p3.set_xlabel('Neighbors',size=11)
    p3.set_ylabel('Normalized Log Score',size=11)
    p3.set_xticks(numpy.arange(len(scrvrlst))+0.5)
    p3.set_xticklabels(scrvrlbs)
    
    for lb in p3.xaxis.get_ticklabels():
        lb.set_fontsize(9)
    for lb in p3.yaxis.get_ticklabels():
        lb.set_fontsize(9)

    leg = pyplot.legend(plst,lbllst, loc = 'upper left',\
                       labelspacing=0.2,borderpad=0.6,ncol=1,handletextpad=0.4,columnspacing=1.0)
    for t in leg.get_texts():
        t.set_fontsize(8)

    tstr = 'Log Score Summary %d0401' % (yrlst[q]) 
    pyplot.title(tstr,size=12)

fig.subplots_adjust(bottom=0.08,top=0.94,left=0.15,right=0.95, \
                    hspace=0.3,wspace=0.3)
fnm = 'LENS_NAmer_H2OSNO_LogScoreSummary_NbrSeq.pdf'
pyplot.savefig(fnm)
pyplot.close()

