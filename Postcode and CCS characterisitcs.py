import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

############### Read data
start_date='2017-02-01 0:00:00'
end_date='2020-02-01 0:00:00'
Path='../iord_extract_20220325/'
# read inpatient_epidsodes
inpatient_epidsodes = pd.read_hdf(Path+'inpatient_epidsodes.h5', key='df')
inpatient_epidsodes['AdmissionDate']=pd.to_datetime(inpatient_epidsodes['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
inpatient_epidsodes['DischargeDate']=pd.to_datetime(inpatient_epidsodes['DischargeDate'], dayfirst=True).dt.tz_localize(None)
inpatient_epidsodes=inpatient_epidsodes[inpatient_epidsodes['AdmissionDate']>=pd.to_datetime(start_date)]
inpatient_epidsodes=inpatient_epidsodes[inpatient_epidsodes['AdmissionDate']<=pd.to_datetime(end_date)]
inpatient_epidsodes=inpatient_epidsodes[~inpatient_epidsodes['AdmissionDate'].isnull()]
inpatient_epidsodes=inpatient_epidsodes.sort_values(['ClusterID','AdmissionDate']).drop_duplicates()
print(inpatient_epidsodes)

Path='../iord_extract_20220325/'
# Admission method
AdmissionMethodMapping=pd.read_csv(Path+"ref_admissionmethod.csv")
inpatient_epidsodes['Planned']=np.where(inpatient_epidsodes['AdmissionMethodCode'].isin(AdmissionMethodMapping['AdmissionMethodCode'][AdmissionMethodMapping['AdmissionMethodType']=='Elective']), 1, 0)
inpatient_epidsodes['Emergency']=np.where(inpatient_epidsodes['AdmissionMethodCode'].isin(AdmissionMethodMapping['AdmissionMethodCode'][AdmissionMethodMapping['AdmissionMethodType']=='Emergency']), 1, 0)
cols = ['Planned', 'Emergency']
inpatient_epidsodes['Admission method']=inpatient_epidsodes[cols].idxmax(1)
print(inpatient_epidsodes)

# PostCodes
PostCodes=pd.read_csv(Path+"PostCodesMapping.csv")
for code in list(set(PostCodes["Fixed_Postcode"])):
    inpatient_epidsodes[code+' (Yes/No)'] = np.where(inpatient_epidsodes['PostcodeStub'].isin(PostCodes['PostcodeStub'][PostCodes["Fixed_Postcode"]==code].values.tolist()), 1, 0)
cols = [x+' (Yes/No)' for x in list(set(PostCodes["Fixed_Postcode"]))]
inpatient_epidsodes['Postcode region']=inpatient_epidsodes[cols].idxmax(1)
print(inpatient_epidsodes)

inpatient_epidsodes['Postcode region']=inpatient_epidsodes['Postcode region'].apply(lambda x: x.replace(' (Yes/No)',''))
pd.DataFrame(inpatient_epidsodes['Postcode region'].value_counts().reset_index())['index'].values.tolist()

medianLOSbyRegion=pd.DataFrame(inpatient_epidsodes[inpatient_epidsodes['Admission method']=='Emergency'].groupby('Postcode region')['LOS, day'].median().reset_index())
medianLOSbyRegion.sort_values(by=['LOS, day'], inplace=True,ascending=False)
print(medianLOSbyRegion)
print(pd.DataFrame(inpatient_epidsodes[inpatient_epidsodes['Admission method']=='Emergency'].groupby('Postcode region')['LOS, day'].median().reset_index())['Postcode region'].values.tolist())


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

LOSStatsbyRegion=pd.DataFrame(inpatient_epidsodes[inpatient_epidsodes['Admission method']=='Planned'].groupby('Postcode region')['LOS, day'].agg([np.count_nonzero, np.mean, np.std, np.median, percentile(25), percentile(75),np.var, np.min, np.max]).reset_index())
LOSStatsbyRegion.sort_values(by=['count_nonzero'], inplace=True,ascending=False)
LOSStatsbyRegion.columns=['Postcode region', 'Regional total number of episodes', 'Regional mean LOS, days', 
                          'Regional standard deviation in LOS', 'Regional median LOS, days', 'Regional 25% percentile in LOS','Regional 75% percentile in LOS', 'Regional variance in LOS', 'Regional minimum LOS, days', 'Regional maximum LOS, days']
LOSStatsbyRegion['Admission method']='Emergency patients'
print(LOSStatsbyRegion)

SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"
LOSStatsbyRegion.to_csv(SavePath+'Postcode region LOS for Planned Patients.csv')
   
import numpy as np
sns.set_style("whitegrid")
plt.figure(figsize=(10,7), dpi=300)

medianLOSbyRegion=pd.DataFrame(inpatient_epidsodes[inpatient_epidsodes['Admission method']=='Planned'].groupby('Postcode region')['LOS, day'].median().reset_index())
medianLOSbyRegion.sort_values(by=['LOS, day'], inplace=True,ascending=False)
myorder=medianLOSbyRegion['Postcode region'].values.tolist()


# Plot the orbital period with horizontal boxes
ax = sns.boxplot(x="LOS, day", y="Postcode region", data=inpatient_epidsodes[inpatient_epidsodes['Admission method']=='Planned'], 
                 order=myorder, width=0.6,linewidth=0.4, showfliers = False)
ax.set_xlabel('Episode LOS, days',weight='bold')
ax.set_ylabel('Admission postcode source',weight='bold')
ax.set_title('Planned patients',weight='bold')
SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"
ax.figure.savefig(SavePath+'Postcode LOS for Planned Patients.svg', dpi=300, bbox_inches='tight',transparent=True)


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

LOSStatsbyRegion=pd.DataFrame(inpatient_epidsodes[inpatient_epidsodes['Admission method']=='Emergency'].groupby('Postcode region')['LOS, day'].agg([np.count_nonzero, np.mean, np.std, np.median, percentile(25), percentile(75),np.var, np.min, np.max]).reset_index())
LOSStatsbyRegion.sort_values(by=['count_nonzero'], inplace=True,ascending=False)
LOSStatsbyRegion.columns=['Postcode region', 'Regional total number of episodes', 'Regional mean LOS, days', 
                          'Regional standard deviation in LOS', 'Regional median LOS, days', 'Regional 25% percentile in LOS','Regional 75% percentile in LOS', 'Regional variance in LOS', 'Regional minimum LOS, days', 'Regional maximum LOS, days']
print(LOSStatsbyRegion)
SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"
LOSStatsbyRegion.to_csv(SavePath+'Postcode region LOS for Emergency Patients.csv')
  
    
import numpy as np
import seaborn as sns
plt.figure(figsize=(10,7), dpi=300)

medianLOSbyRegion=pd.DataFrame(inpatient_epidsodes[inpatient_epidsodes['Admission method']=='Emergency'].groupby('Postcode region')['LOS, day'].median().reset_index())
medianLOSbyRegion.sort_values(by=['LOS, day'], inplace=True,ascending=False)
myorder=medianLOSbyRegion['Postcode region'].values.tolist()

# Plot the orbital period with horizontal boxes
ax = sns.boxenplot(x="LOS, day", y="Postcode region", data=inpatient_epidsodes[inpatient_epidsodes['Admission method']=='Emergency'], 
                 order=myorder,width=0.6,linewidth=0.4,
                 showfliers = False)
ax.set_xlabel('Episode LOS, days',weight='bold')
ax.set_ylabel('Admission postcode source',weight='bold')
ax.set_title('Emergency patients',weight='bold')
SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"
ax.figure.savefig(SavePath+'Postcode LOS for Emergency Patients.svg', dpi=300, bbox_inches='tight',transparent=True)

 
# combine episodes with diagnosis
ccsCodes=pd.read_csv(Path+"ccs_lookup.csv")
ccsCodes=ccsCodes[ccsCodes['Disease']!='perinatal, congenital and birth disorders']
diseasesSet=list(set(ccsCodes['Disease'].dropna()))
inpt_diagnosis=pd.read_hdf(Path+"inpt_diagnosis.h5", key='df')
inpt_diagnosis=inpt_diagnosis[['ClusterID', 'EpisodeID', 'DiagCode']].drop_duplicates()
inpt_diagnosis=inpt_diagnosis[~inpt_diagnosis['DiagCode'].isnull()]
inpt_diagnosis_epis = pd.merge(inpt_diagnosis, inpatient_epidsodes[['ClusterID', 'Admission method', 'AdmissionDate', 'DischargeDate', 'EpisodeID']],  how='left', on=['ClusterID','EpisodeID'])
inpt_diagnosis_epis['AdmissionDate']=pd.to_datetime(inpt_diagnosis_epis['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
inpt_diagnosis_epis['DischargeDate']=pd.to_datetime(inpt_diagnosis_epis['DischargeDate'], dayfirst=True).dt.tz_localize(None)
inpt_diagnosis_epis=inpt_diagnosis_epis[~inpt_diagnosis_epis['AdmissionDate'].isnull()]
inpt_diagnosis_epis=inpt_diagnosis_epis.sort_values(['ClusterID','AdmissionDate'], ascending=[True, False]).drop_duplicates()
inpt_diagnosis_epis=pd.merge(inpt_diagnosis_epis, ccsCodes[['DiagCode','Disease']][ccsCodes['DiagCode'].isin(list(set(inpt_diagnosis_epis['DiagCode'])))], how='left', on='DiagCode')
inpt_diagnosis_epis=inpt_diagnosis_epis[~inpt_diagnosis_epis['Disease'].isnull()]


# CCS characterisitcs
start_date='2017-02-01 0:00:00'
end_date='2020-02-01 0:00:00'
Path='../iord_extract_20220325/'
# read inpatient_epidsodes
inpt_diagnosis_epis['AdmissionDate']=pd.to_datetime(inpt_diagnosis_epis['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
inpt_diagnosis_epis['DischargeDate']=pd.to_datetime(inpt_diagnosis_epis['DischargeDate'], dayfirst=True).dt.tz_localize(None)
inpt_diagnosis_epis=inpt_diagnosis_epis[inpt_diagnosis_epis['AdmissionDate']>=pd.to_datetime(start_date)]
inpt_diagnosis_epis=inpt_diagnosis_epis[inpt_diagnosis_epis['AdmissionDate']<=pd.to_datetime(end_date)]
inpt_diagnosis_epis=inpt_diagnosis_epis[~inpt_diagnosis_epis['AdmissionDate'].isnull()]
inpt_diagnosis_epis=inpt_diagnosis_epis[~inpt_diagnosis_epis['DischargeDate'].isnull()]
inpt_diagnosis_epis['LOS, day']=divmod((inpt_diagnosis_epis['DischargeDate']-inpt_diagnosis_epis['AdmissionDate']).dt.total_seconds(), 3600)[0]/24
print(inpt_diagnosis_epis)

# Planned
plt.rcParams["font.family"] = "Arial"
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_
CCSStatsbyDisease=pd.DataFrame(inpt_diagnosis_epis[inpt_diagnosis_epis['Admission method']=='Planned'].groupby('Disease')['LOS, day'].agg([np.count_nonzero, np.mean, np.std, np.median, percentile(25), percentile(75),np.var, np.min, np.max]).reset_index())
CCSStatsbyDisease.sort_values(by=['count_nonzero'], inplace=True,ascending=False)
CCSStatsbyDisease.columns=['Disease', 'Total number of episodes', 'Mean LOS, days', 
                          'Standard deviation in LOS', 'Median LOS, days', '25% percentile in LOS','75% percentile in LOS', 'Variance in LOS', 'Minimum LOS, days', 'Maximum LOS, days']
CCSStatsbyDisease['Admission method']='Planned patients'
SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"
CCSStatsbyDisease.to_csv(SavePath+'CCSStatsbyDisease LOS for Planned Patients.csv')
    
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
plt.figure(figsize=(8,8), dpi=300)
medianLOSbyDisease=pd.DataFrame(inpt_diagnosis_epis[inpt_diagnosis_epis['Admission method']=='Planned'].groupby('Disease')['LOS, day'].median().reset_index())
medianLOSbyDisease.sort_values(by=['LOS, day'], inplace=True,ascending=False)
myorder=medianLOSbyDisease['Disease'].values.tolist()
# Plot the orbital period with horizontal boxes
ax = sns.boxplot(x="LOS, day", y="Disease", data=inpt_diagnosis_epis[inpt_diagnosis_epis['Admission method']=='Planned'], 
                 order=myorder, width=0.6,linewidth=0.4, showfliers = False)
ax.set_xlabel('Episode LOS, days',weight='bold', fontsize=10)
ax.set_ylabel('Diagnostic category',weight='bold', fontsize=10)
ax.set_title('Planned patients',weight='bold', fontsize=10)
SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"
ax.figure.savefig(SavePath+'CCSStatsbyDisease LOS for Planned Patients.svg', dpi=300, bbox_inches='tight',transparent=True)
ax.figure.savefig(SavePath+'CCSStatsbyDisease LOS for Planned Patients.pdf', dpi=300, bbox_inches='tight',transparent=True)


# Emergency
plt.rcParams["font.family"] = "Arial"
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_
CCSStatsbyDisease=pd.DataFrame(inpt_diagnosis_epis[inpt_diagnosis_epis['Admission method']=='Emergency'].groupby('Disease')['LOS, day'].agg([np.count_nonzero, np.mean, np.std, np.median, percentile(25), percentile(75),np.var, np.min, np.max]).reset_index())
CCSStatsbyDisease.sort_values(by=['count_nonzero'], inplace=True,ascending=False)
CCSStatsbyDisease.columns=['Disease', 'Total number of episodes', 'Mean LOS, days', 
                          'Standard deviation in LOS', 'Median LOS, days', '25% percentile in LOS','75% percentile in LOS', 'Variance in LOS', 'Minimum LOS, days', 'Maximum LOS, days']
CCSStatsbyDisease['Admission method']='Emergency patients'
SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"
CCSStatsbyDisease.to_csv(SavePath+'CCSStatsbyDisease LOS for Emergency Patients.csv')
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
plt.figure(figsize=(8,8), dpi=300)
medianLOSbyDisease=pd.DataFrame(inpt_diagnosis_epis[inpt_diagnosis_epis['Admission method']=='Emergency'].groupby('Disease')['LOS, day'].median().reset_index())
medianLOSbyDisease.sort_values(by=['LOS, day'], inplace=True,ascending=False)
myorder=medianLOSbyDisease['Disease'].values.tolist()
# Plot the orbital period with horizontal boxes
ax = sns.boxplot(x="LOS, day", y="Disease", data=inpt_diagnosis_epis[inpt_diagnosis_epis['Admission method']=='Emergency'], 
                 order=myorder, width=0.6,linewidth=0.4, showfliers = False)
ax.set_xlabel('Episode LOS, days',weight='bold', fontsize=10)
ax.set_ylabel('Diagnostic category',weight='bold', fontsize=10)
ax.set_title('Emergency patients',weight='bold', fontsize=10)
SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"
ax.figure.savefig(SavePath+'CCSStatsbyDisease LOS for Emergency Patients.svg', dpi=300, bbox_inches='tight',transparent=True)
ax.figure.savefig(SavePath+'CCSStatsbyDisease LOS for Emergency Patients.pdf', dpi=300, bbox_inches='tight',transparent=True)





 

