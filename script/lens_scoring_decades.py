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
csfrm = pandas.read_csv(rffl, dtype = {'DateFmt8':int, 'RCP':str})
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

yrsts[yrlst > 2040] = yrsts[yrlst > 2040] + 1
tspts = numpy.array([yrsts[1], yrsts[2]+4, yrsts[4]],numpy.int32)
rwdth = numpy.array([4.0, 1.0, 4.0])


# Loop through scoring results, add metrics to data frame
for j in range(ncase):

    # Quantile info - count number of locations used per case
    tgtdtfl = datetime.date(2075,4,1)
    dtstr = tgtdtfl.strftime('%Y-%m-%d')
    tgtdt = csfrm['DateFmt8'].values[j]
    qfnm = 'LENS_NAmer_H2OSNO_%d_Quantile.nc' % (tgtdt)
    ncclm = Dataset(qfnm,'r')
    snomsk = ncclm.variables['LENS_mask'][:,:]
    ncclm.close()

    snoflt = snomsk.flatten()
    snovld = snoflt[snoflt == 1]

    # Read scoring results
    btmscrfl = 'LENS_NAmer_H2OSNO_%d_BTMScoring.csv' % (tgtdt)
    btmfrm = pandas.read_csv(btmscrfl, \
                        dtype = {'Epoch':int, 'TrainSize':int, \
                                 'NonLinScore':float, 'LinearScore':float}, encoding='utf-8-sig')
    grpbtm = btmfrm.groupby(['TrainSize'])
    btmqs = grpbtm.apply(qsummary,vrlst=['NonLinScore','LinearScore'],include_groups=False)
    btmqs.reset_index(drop=False,inplace=True)
    btmqs.drop(['level_1'],axis=1,inplace=True)
    btmqs.rename(columns={'NSmp':'NSmpBTM'})

    gpscrfl = 'LENS_NAmer_H2OSNO_%d_GPScoring.csv' % (tgtdt)
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


print(btmqs)
print(gpqs)
print(scrmrg)

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

# 95th percentile
p3 = pyplot.subplot(1,1,1)

for j in range(len(tspts)):
    rxspt = tspts[j]
    rect1 = pyplot.Rectangle((rxspt,-0.9),rwdth[j],0.3, \
                              facecolor=rcls[j],edgecolor='none')
    pyplot.gca().add_patch(rect1)

plst = []
for q in range(nyr):
    for p in range(ntrn):
        frmsb = scrfrmfl[(scrfrmfl['Year'] == yrlst[q]) & (scrfrmfl['TrainSize'] == trnsz[p])]
        #asq = cssq[yrarr == yrlst[q]]
        if yrlst[q] < 2040:
            plt1, = p3.plot(q*4+frmsb['Month']-0.5+p*0.2-0.3,frmsb['NonLinScore_Norm'],tsymb[p], ms=1.6, c=c3[0])
            plt2, = p3.plot(q*4+frmsb['Month']-0.5+p*0.2-0.3,frmsb['LinearScore_Norm'],tsymb[p], ms=1.6, c=c3[1])
            plt3, = p3.plot(q*4+frmsb['Month']-0.5+p*0.2-0.3,frmsb['NonStatScore_Norm'],tsymb[p], ms=1.6, c=c3[2])
        else: 
            plt1, = p3.plot(q*4+frmsb['Month']+0.5+p*0.2-0.3,frmsb['NonLinScore_Norm'],tsymb[p], ms=1.6, c=c3[0])
            plt2, = p3.plot(q*4+frmsb['Month']+0.5+p*0.2-0.3,frmsb['LinearScore_Norm'],tsymb[p], ms=1.6, c=c3[1])
            plt3, = p3.plot(q*4+frmsb['Month']+0.5+p*0.2-0.3,frmsb['NonStatScore_Norm'],tsymb[p], ms=1.6, c=c3[2])
        if q == 0:
            plst.append(plt1)
            plst.append(plt2)
            plst.append(plt3)
    #for j in range(ncase):
    #    pyplot.text(cssq[j]+0.2+0.1*k,qmed[j,k],s=ctyfrm['Label'].values[k],c=c6[k], fontsize=10)
plstswp = []

mdllst = ['NBTM','LBTM','GP']
lbllst = []
for j in range(3):
    for p in range(4):
        plstswp.append(plst[p*3+j])
        lblstr = '%s, J=%d' % (mdllst[j],trnsz[p])
        lbllst.append(lblstr)

p3.yaxis.grid(color='#898989',linestyle='dotted',linewidth=0.7)
# Manual y grid, month labels
mthstr = ['J','F','M','A']
for q in range(nyr):
    for m1 in range(1,5):
        if yrlst[q] < 2040:
            xspt = q*4+m1-1
            pyplot.text(xspt+0.5,-0.89,mthstr[m1-1],fontsize=8,ha='center')
        else:
            xspt = q*4+m1
            pyplot.text(xspt+0.5,-0.89,mthstr[m1-1],fontsize=8,ha='center')
        p3.plot( [xspt, xspt], [-0.9,-0.6], linestyle='dotted', c='#898989',linewidth=0.7)

xspt = (nyr/2)*4
p3.plot( [xspt, xspt], [-0.9,-0.6], linestyle='dotted', c='#898989',linewidth=0.7)
xspt = nyr*4+1
p3.plot( [xspt, xspt], [-0.9,-0.6], linestyle='dotted', c='#898989',linewidth=0.7)


p3.set_ylim(-0.9,-0.6)
p3.set_xlim(-0.5,ncase+1.5)
p3.set_xlabel('Year',size=11)
p3.set_ylabel('Normalized Log Score',size=11)
p3.set_xticks(yrsts+2.0)
p3.set_xticklabels(yrlst)
    
for lb in p3.xaxis.get_ticklabels():
    lb.set_fontsize(9)
for lb in p3.yaxis.get_ticklabels():
    lb.set_fontsize(9)

leg = pyplot.legend(plstswp,lbllst, loc = 'upper right',\
                    labelspacing=0.2,borderpad=0.6,ncol=3,handletextpad=0.4,columnspacing=1.0)
for t in leg.get_texts():
    t.set_fontsize(8)

tstr = 'Log Score Summary' 
pyplot.title(tstr,size=12)

fig.subplots_adjust(bottom=0.08,top=0.88,left=0.12,right=0.95, \
                    hspace=0.3,wspace=0.3)
fnm = 'LENS_NAmer_H2OSNO_LogScoreSummary.pdf'
pyplot.savefig(fnm)
pyplot.close()

print(len(plst))

# Output summary
scrfl = 'LENS_NAmer_H2OSNO_Decades_Scoring.csv'
scrfrmfl.to_csv(scrfl, index=False)

