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

# Loop through quantiles, add metrics to data frame
for j in range(ncase):

    # Quantile info
    tgtdt = csfrm['DateFmt8'].values[j]
    qfnm = 'LENS2_NAmer_H2OSNO_%d_Quantile_ParamFit.nc' % (tgtdt)
    ncclm = Dataset(qfnm,'r')
    snomsk = ncclm.variables['LENS_mask'][:,:]
    ncclm.close()

    snoflt = snomsk.flatten()
    snovld = snoflt[snoflt == 1]

    # Read scoring results
    btmscrfl = 'LENS2_NAmer_H2OSNO_%d_BTMScoring.csv' % (tgtdt)
    btmfrm = pandas.read_csv(btmscrfl, \
                        dtype = {'Epoch':int, 'TrainSize':int, \
                                 'NonLinScore':float, 'LinearScore':float}, encoding='utf-8-sig')
    grpbtm = btmfrm.groupby(['TrainSize'])
    btmqs = grpbtm.apply(qsummary,vrlst=['NonLinScore','LinearScore'],include_groups=False)
    btmqs.reset_index(drop=False,inplace=True)
    btmqs.drop(['level_1'],axis=1,inplace=True)
    btmqs.rename(columns={'NSmp':'NSmpBTM'})

    gpscrfl = 'LENS2_NAmer_H2OSNO_%d_GPScoring.csv' % (tgtdt)
    gpfrm = pandas.read_csv(gpscrfl, \
                        dtype = {'Epoch':int, 'TrainSize':int, \
                                 'NonStatScore':float}, encoding='utf-8-sig')
    grpgp = gpfrm.groupby(['TrainSize'])
    gpqs = grpgp.apply(qsummary,vrlst=['NonStatScore'],include_groups=False)
    gpqs.reset_index(drop=False,inplace=True)
    gpqs.drop(['level_1'],axis=1,inplace=True)
    gpqs.rename(columns={'NSmp':'NSmpGP'})

    scrmrg = pandas.merge(btmqs,gpqs,how='inner',on=['TrainSize'])
    scrmrg['Year'] = yrarr[j]
    scrmrg['Month'] = mnarr[j]
    scrmrg['NLoc'] = snovld.shape[0]

    if j == 0:
        scrfrmfl = scrmrg
    else:
        scrfrmfl = pandas.concat([scrfrmfl, scrmrg], ignore_index=True)

scrfrmfl['NonLinScore_Norm'] = scrfrmfl['NonLinScore_Mean'] / scrfrmfl['NLoc']
scrfrmfl['LinearScore_Norm'] = scrfrmfl['LinearScore_Mean'] / scrfrmfl['NLoc']
scrfrmfl['NonStatScore_Norm'] = scrfrmfl['NonStatScore_Mean'] / scrfrmfl['NLoc']

#c6 = ["#C7657B","#AC7A23","#718E00","#009A68","#0098A9","#6881C9"]
c3 = ["#6929c4","#9f1853","#198038"]
rcls = ['#DDDDDD','#777777','#DDDDDD']
tsymb = ['d','v','o','x']
trnsz = [8,16,24, 32]
ntrn = len(trnsz)

fig = pyplot.figure(figsize=(6,3.5))

p3 = pyplot.subplot(1,1,1)

plst = []
for p in range(ntrn):
    frmsb = scrfrmfl[(scrfrmfl['TrainSize'] == trnsz[p])]
    plt1, = p3.plot(frmsb['Month']-0.5+p*0.1-0.15,frmsb['NonLinScore_Norm'],tsymb[p], ms=3, c=c3[0])
    plt2, = p3.plot(frmsb['Month']-0.5+p*0.1-0.15,frmsb['LinearScore_Norm'],tsymb[p], ms=3, c=c3[1])
    plt3, = p3.plot(frmsb['Month']-0.5+p*0.1-0.15,frmsb['NonStatScore_Norm'],tsymb[p], ms=3, c=c3[2])
    plst.append(plt1)
    plst.append(plt2)
    plst.append(plt3)

plstswp = []

mdllst = ['NBTM','LBTM','GP']
lbllst = []
for j in range(3):
    for p in range(4):
        plstswp.append(plst[p*3+j])
        lblstr = '%s, J=%d' % (mdllst[j],trnsz[p])
        lbllst.append(lblstr)

p3.yaxis.grid(color='#898989',linestyle='dotted')
p3.set_ylim(-0.95,-0.6)
p3.set_xlim(0,ncase)
p3.set_xlabel('Date',size=11)
p3.set_ylabel('Normalized Log Score',size=11)
p3.set_xticks(cssq+0.5)
p3.set_xticklabels(lblst)    

for lb in p3.xaxis.get_ticklabels():
    lb.set_fontsize(9)
for lb in p3.yaxis.get_ticklabels():
    lb.set_fontsize(9)

leg = pyplot.legend(plstswp,lbllst, loc = 'upper left',\
                    labelspacing=0.2,borderpad=0.6,ncol=3,handletextpad=0.4,columnspacing=1.0)
for t in leg.get_texts():
    t.set_fontsize(8)

tstr = 'CESM-LE2 Log Score Summary' 
pyplot.title(tstr,size=12)

fig.subplots_adjust(bottom=0.08,top=0.88,left=0.12,right=0.95, \
                    hspace=0.3,wspace=0.3)
fnm = 'LENS2_NAmer_H2OSNO_LogScoreSummary_%d.pdf' % (yrlst[0])
pyplot.savefig(fnm)
pyplot.close()


