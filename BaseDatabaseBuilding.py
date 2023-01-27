import datetime
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from multiprocessing import Pool
import timeit
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from line_profiler_pycharm import profile 

# =============================================================================
# Feature engineering
# =============================================================================
############### Read data
start_date='2017-02-01 6:00:00'
end_date='2020-02-01 6:00:00'
Path='/Users/jiandong/Oxford/Data/iord_extract_20220325/'
# read inpatient_epidsodes
inpatient_epidsodes = pd.read_hdf(Path+'inpatient_epidsodes.h5', key='df')
inpatient_epidsodes['AdmissionDate']=pd.to_datetime(inpatient_epidsodes['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
inpatient_epidsodes['DischargeDate']=pd.to_datetime(inpatient_epidsodes['DischargeDate'], dayfirst=True).dt.tz_localize(None)
inpatient_epidsodes=inpatient_epidsodes[inpatient_epidsodes['AdmissionDate']>=(pd.to_datetime(start_date)-datetime.timedelta(hours=360*24))]
inpatient_epidsodes=inpatient_epidsodes[~inpatient_epidsodes['AdmissionDate'].isnull()]
inpatient_epidsodes=inpatient_epidsodes.sort_values(['ClusterID','AdmissionDate']).drop_duplicates()
# read ConsultantSpecifialty mapping
ConsultantSpecifialty=pd.read_csv(Path+"ConsultantSpecifialty.csv")
ConsultantSpecifialty=ConsultantSpecifialty[ConsultantSpecifialty['Specialty']!='Procedural']
inpatient_epidsodes=pd.merge(inpatient_epidsodes, ConsultantSpecifialty[['ConsultantMainSpecialtyCode','Specialty']][ConsultantSpecifialty['ConsultantMainSpecialtyCode'].isin(list(set(inpatient_epidsodes['ConsultantMainSpecialtyCode'])))], how='left', on='ConsultantMainSpecialtyCode')
# create a baseDatabase (To save RAM memory when reading other big files)
def getIndexDatePatients(inpatient_epidsodes, index_date):
    Database_day_t=inpatient_epidsodes[(inpatient_epidsodes['AdmissionDate']<=pd.to_datetime(index_date, dayfirst=True)) &
                                       (inpatient_epidsodes['DischargeDate']>=pd.to_datetime(index_date, dayfirst=True))]
    Database_day_t['ClusterID']=Database_day_t['ClusterID'].astype(str)
    return ','.join(Database_day_t['ClusterID'].values.tolist())
start = timeit.default_timer()
Database=pd.DataFrame(Pool(10).map(int, [str(i) for i in list(filter(None, list(set(','.join([getIndexDatePatients(inpatient_epidsodes, index_date) for index_date in pd.date_range(start_date, end_date, freq='1440T')]).split(',')))))]), columns=['ClusterID'])
stop = timeit.default_timer()
print('Time to create basedatabase', stop - start)
# read Demographics and Geographics
Demographics=pd.read_hdf(Path+"demographics.h5", key='df')
Demographics=Demographics[['ClusterID', 'date_admitted', 'LinkedSex', 'EthnicGroupCode', 'record_type', 'episode_number', 'GPPracticeCode', 'GPCode']]
Demographics=Demographics[Demographics['LinkedSex']!='U'] 
Demographics['date_admitted']=pd.to_datetime(Demographics['date_admitted'], dayfirst=True).dt.tz_localize(None)
Demographics=Demographics[Demographics['date_admitted']<=pd.to_datetime(end_date)]
Demographics=Demographics.sort_values(['ClusterID','date_admitted'], ascending=[True, False]).drop_duplicates()
# combine episodes with diagnosis
ccsCodes=pd.read_csv(Path+"ccs_lookup.csv")
ccsCodes=ccsCodes[ccsCodes['Disease']!='perinatal, congenital and birth disorders']
diseasesSet=list(set(ccsCodes['Disease'].dropna()))
inpt_diagnosis=pd.read_hdf(Path+"inpt_diagnosis.h5", key='df')
inpt_diagnosis=inpt_diagnosis[['ClusterID', 'EpisodeID', 'DiagCode']].drop_duplicates()
inpt_diagnosis=inpt_diagnosis[~inpt_diagnosis['DiagCode'].isnull()]
inpt_diagnosis_epis = pd.merge(inpt_diagnosis, inpatient_epidsodes[['ClusterID', 'AdmissionDate', 'DischargeDate', 'EpisodeID']],  how='left', on=['ClusterID','EpisodeID'])
inpt_diagnosis_epis['AdmissionDate']=pd.to_datetime(inpt_diagnosis_epis['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
inpt_diagnosis_epis['DischargeDate']=pd.to_datetime(inpt_diagnosis_epis['DischargeDate'], dayfirst=True).dt.tz_localize(None)
inpt_diagnosis_epis=inpt_diagnosis_epis[~inpt_diagnosis_epis['AdmissionDate'].isnull()]
inpt_diagnosis_epis=inpt_diagnosis_epis.sort_values(['ClusterID','AdmissionDate'], ascending=[True, False])
inpt_diagnosis_epis=inpt_diagnosis_epis[inpt_diagnosis_epis['ClusterID'].isin(Database['ClusterID'].values.tolist())]
inpt_diagnosis_epis=pd.merge(inpt_diagnosis_epis, ccsCodes[['DiagCode','Disease']][ccsCodes['DiagCode'].isin(list(set(inpt_diagnosis_epis['DiagCode'])))], how='left', on='DiagCode')
inpt_diagnosis_epis=inpt_diagnosis_epis[~inpt_diagnosis_epis['Disease'].isnull()]
# read weight height BMI
weight_height=pd.read_hdf(Path+"weight_height.h5", key='df')
weight_height=weight_height[['ClusterID', 'EventName', 'EventResult', 'PerformedDateTime']].drop_duplicates()
weight_height=weight_height[weight_height['ClusterID'].isin(Database['ClusterID'].values.tolist())]
weight_height['PerformedDateTime']=pd.to_datetime(weight_height['PerformedDateTime'], dayfirst=True).dt.tz_localize(None)
weight_height['EventName']=weight_height['EventName'].replace('Height/Length Measured','Height, cm')
weight_height['EventName']=weight_height['EventName'].replace('Weight [Working]','Weight, kg')
weight_height['EventName']=weight_height['EventName'].replace('Weight Estimated','Weight, kg')
weight_height['EventName']=weight_height['EventName'].replace('Weight [Unknown]','Weight, kg')
weight_height['EventName']=weight_height['EventName'].replace('Weight [Measured]','Weight, kg')
weight_height['EventName']=weight_height['EventName'].replace('BMI Score (MUST)','BMI')
weight_height_num=weight_height[~pd.to_numeric(weight_height['EventResult'], errors='coerce').isnull()]
weight_height=weight_height.sort_values(['ClusterID','PerformedDateTime'], ascending=[True, False])
WHBMI=list(set(weight_height['EventName']))
# read procedure data       
inpt_procedures=pd.read_hdf(Path+'inpt_procedures.h5', key='df')
inpt_procedures=inpt_procedures[['ClusterID', 'EpisodeID', 'ProcDate', 'ProcCode','ProcNumber']].drop_duplicates()
inpt_procedures=inpt_procedures[inpt_procedures['ClusterID'].isin(Database['ClusterID'].values.tolist())]
inpt_procedures['ProcDate']=pd.to_datetime(inpt_procedures['ProcDate'], dayfirst=True).dt.tz_localize(None)
inpt_procedures=inpt_procedures[~inpt_procedures['ProcDate'].isnull()]
inpt_procedures=inpt_procedures.sort_values(['ClusterID','ProcDate'], ascending=[True, False])
procedures_radiology=inpt_procedures[inpt_procedures['ProcCode'].isin(['Y98','Y981','Y982','Y983','Y984','Y985','Y986','Y987','Y988','Y989'])]
# read theatre data
theatres=pd.read_hdf(Path+'theatres.h5', key='df')
theatres=theatres[['ClusterID', 'arrived_time', 'left_time']].drop_duplicates()
theatres=theatres[theatres['ClusterID'].isin(Database['ClusterID'].values.tolist())]
theatres['arrived_time']=pd.to_datetime(theatres['arrived_time'], dayfirst=True).dt.tz_localize(None)
theatres=theatres[~theatres['arrived_time'].isnull()]
theatres=theatres[~theatres['left_time'].isnull()]
theatres=theatres.sort_values(['ClusterID','arrived_time'], ascending=[True, False])
# read ward data
wards=pd.read_hdf(Path+'wards.h5', key='df')
wards=wards[['ClusterID', 'WardStartDate']].drop_duplicates()
wards=wards[wards['ClusterID'].isin(Database['ClusterID'].values.tolist())]
wards['WardStartDate']=pd.to_datetime(wards['WardStartDate'], dayfirst=True).dt.tz_localize(None)
wards=wards[~wards['WardStartDate'].isnull()]
wards=wards.sort_values(['ClusterID','WardStartDate'], ascending=[True, False])
# read antibiotics prescription
antibiotics=pd.read_hdf(Path+'antibiotics_epr_admin.h5', key='df')
antibiotics=antibiotics[['ClusterID', 'PrescriptionDateTime', 'Drug','OrderStatus']].drop_duplicates()
antibiotics=antibiotics[antibiotics['ClusterID'].isin(Database['ClusterID'].values.tolist())]
antibiotics=antibiotics[antibiotics['OrderStatus'].isin(['Ordered'])]
antibiotics['PrescriptionDateTime']=pd.to_datetime(antibiotics['PrescriptionDateTime'], dayfirst=True).dt.tz_localize(None)
antibiotics=antibiotics[~antibiotics['PrescriptionDateTime'].isnull()]
antibiotics=antibiotics.sort_values(['ClusterID','PrescriptionDateTime'], ascending=[True, False])
# read labtests
labtests=pd.read_hdf(Path+'labtestsSelected.h5', key='df')
labtests=labtests[['ClusterID', 'TestName', 'Value', 'CollectionDateTime']].drop_duplicates()
labtests=labtests[labtests['ClusterID'].isin(Database['ClusterID'].values.tolist())]
labtests['CollectionDateTime']=pd.to_datetime(labtests['CollectionDateTime'], dayfirst=True).dt.tz_localize(None)
labtests=labtests[~labtests['CollectionDateTime'].isnull()]
labtests=labtests.sort_values(['ClusterID','CollectionDateTime'], ascending=[True, False])
labmapping=pd.read_csv(Path+"LabMapping.csv")
labmapping=labmapping[labmapping['Selected']=='Yes']
labtests=labtests[labtests['TestName'].isin(labmapping['TestName'].values.tolist())]
tests=labmapping['TestName'].values.tolist()
# read vitals for O2Devices
vitals=pd.read_csv(Path+"vitalsO2Devices.csv") 
vitals=vitals[['ClusterID', 'PerformedDateTime', 'EventName','EventResult']]
vitals=vitals[vitals['ClusterID'].isin(Database['ClusterID'].values.tolist())]
vitals['PerformedDateTime']=pd.to_datetime(vitals['PerformedDateTime'], dayfirst=True).dt.tz_localize(None)
vitals=vitals[~vitals['PerformedDateTime'].isnull()]
vitals=vitals.sort_values(['ClusterID','PerformedDateTime'], ascending=[True, False]).drop_duplicates()
vitals=vitals[vitals['EventName']=='Delivery device used']
O2Devices=pd.read_csv(Path+"OxDevices.csv")
vitals.to_csv(Path+"vitalsO2Devices.csv", index=False)
vitals=pd.merge(vitals, O2Devices[['EventResult','O2Device']][O2Devices['EventResult'].isin(list(set(vitals['EventResult'])))], how='left', on='EventResult')
vitals=vitals[~vitals['O2Device'].isnull()]
# read NEWS2 scores
NEWS2=pd.read_hdf(Path+'vitals_wide_news2_selected.h5', key='df') # already being processed
NEWS2.columns=['ClusterID', 'PerformedDateTime', 'Oxygen saturation, %', 'Diastolic blood pressure, mmHg', 'AVPU', 'Heart rate, bpm', 'Oxygen, L/min',
       'Systolic blood pressure, mmHg', 'Respiratory rate, br/min', 'Temperature, °C', 'NEWS2 score']
NEWS2=NEWS2[NEWS2['ClusterID'].isin(Database['ClusterID'].values.tolist())]
NEWS2['PerformedDateTime']=pd.to_datetime(NEWS2['PerformedDateTime'], dayfirst=True).dt.tz_localize(None)
NEWS2=NEWS2[~NEWS2['PerformedDateTime'].isnull()]
NEWS2=NEWS2.sort_values(['ClusterID','PerformedDateTime'], ascending=[True, False]).drop_duplicates()
# read DNACPR
resus=pd.read_hdf(Path+'resus.h5', key='df')
resus=resus[resus['ClusterID'].isin(Database['ClusterID'].values.tolist())]
resus['PerformedDateTime']=pd.to_datetime(resus['PerformedDateTime'], dayfirst=True).dt.tz_localize(None)
resus=resus[~resus['PerformedDateTime'].isnull()]
resus=resus.sort_values(['ClusterID','PerformedDateTime'], ascending=[True, False]).drop_duplicates()
resus=resus[resus['EventTag'].isin(['DNACPR', 'dnacpr'])]
# read microbiology data
micro=pd.read_hdf(Path+'microSelected.h5', key='df')
micro=micro[['ClusterID', 'CollectionDateTime', 'BatTestName', 'BugCode', 'BugName']]
micro=micro[micro['ClusterID'].isin(Database['ClusterID'].values.tolist())]
micro['CollectionDateTime']=pd.to_datetime(micro['CollectionDateTime'], dayfirst=True).dt.tz_localize(None)
micro=micro[~micro['CollectionDateTime'].isnull()]
micro=micro[~micro['BugName'].isnull()]
micro=micro[~micro['BugCode'].isnull()]
micro['BatTestName']=micro['BatTestName'].replace('URINE CULTURE','urine culture')
micro['BatTestName']=micro['BatTestName'].replace('URINE CULTURE(GP)','urine culture')
micro['BatTestName']=micro['BatTestName'].replace('URINE CULTURE(HOSP)','urine culture')
micro['BatTestName']=micro['BatTestName'].replace('BLOOD CULTURE','blood culture')
micro['BatTestName']=micro['BatTestName'].replace('BLOOD BOTTLE CULTURE','blood culture')
micro=micro.sort_values(['ClusterID','CollectionDateTime'], ascending=[True, False]).drop_duplicates()
BugsIsolatedMapping=pd.read_csv(Path+"BugsIsolatedMapping.csv")
###############  Extract and save features
Database=[]
for index_date in pd.date_range(start_date, end_date, freq='1440T'):
    print(index_date)
    start_whole = timeit.default_timer()
    # Identify patients in hospital on index date
    Database_day_t=inpatient_epidsodes[(inpatient_epidsodes['AdmissionDate']<=pd.to_datetime(index_date, dayfirst=True)) & (inpatient_epidsodes['DischargeDate']>=pd.to_datetime(index_date, dayfirst=True))]
    Database_day_t['Index date']=pd.to_datetime(index_date)
    Database_day_t['Latest admission to decision day, days']=(Database_day_t['Index date']-Database_day_t['AdmissionDate']).dt.days
    Database_day_t=Database_day_t.iloc[Database_day_t.reset_index().groupby(['ClusterID'])['AdmissionDate'].idxmax(),:]
    col=Database_day_t.pop("Index date")
    Database_day_t.insert(1, col.name, col)
    col=Database_day_t.pop("Latest admission to decision day, days")
    Database_day_t.insert(2, col.name, col)
    instances_day_t=Database_day_t[['ClusterID','AdmissionDate']].drop_duplicates()
    # demographics and geographics
    Demographics_day_t=Demographics[Demographics['ClusterID'].isin(list(set(Database_day_t['ClusterID'])))]
    Demographics_day_t=Demographics_day_t[Demographics_day_t['date_admitted']<=pd.to_datetime(index_date, dayfirst=True)]
    if len(Demographics_day_t.reset_index().groupby('ClusterID')['date_admitted'].idxmax())>0:
        Demographics_day_t=Demographics_day_t.iloc[Demographics_day_t.reset_index().groupby('ClusterID')['date_admitted'].idxmax(),:].reset_index()
    for item in ['date_admitted', 'record_type', 'episode_number', 'GPPracticeCode', 'GPCode']:
        Database_day_t=pd.merge(Database_day_t, Demographics_day_t[['ClusterID',item]], how = 'left', on='ClusterID')
    # height, weight, and BMI 
    weight_height_day_t=weight_height[weight_height['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    weight_height_day_t['PerformedDateTime']=pd.to_datetime(weight_height_day_t['PerformedDateTime'], dayfirst=True).dt.tz_localize(None)
    weight_height_day_t=weight_height_day_t[weight_height_day_t['PerformedDateTime']<=pd.to_datetime(index_date)]
    for test in ['Height, cm', 'Weight, kg']:
        weight_height_day_t_test=weight_height_day_t[weight_height_day_t['EventName']==test].drop_duplicates()
        TestValue_day_t_test=pd.DataFrame(weight_height_day_t_test.groupby(['ClusterID'])['EventResult'].first())
        TestValue_day_t_test.columns=[test]
        Database_day_t=pd.merge(Database_day_t,TestValue_day_t_test, how='left', on='ClusterID')
    Database_day_t['Height, cm']=np.where(Database_day_t['Height, cm']>=217, 217, Database_day_t['Height, cm'])
    Database_day_t['Height, cm']=np.where(Database_day_t['Height, cm']==0, 170, Database_day_t['Height, cm'])
    Database_day_t['Weight, kg']=np.where(Database_day_t['Weight, kg']>=300, 300, Database_day_t['Weight, kg'])
    Database_day_t['Weight, kg']=np.where(Database_day_t['Weight, kg']==0, 70, Database_day_t['Weight, kg'])
    Database_day_t['BMI']=Database_day_t['Weight, kg']/(Database_day_t['Height, cm']/100*Database_day_t['Height, cm']/100)
    Database_day_t['BMI']=np.where(Database_day_t['BMI']>=39.9, np.nan, Database_day_t['BMI'])
    # LOS features 365 days before index date    
    inpatient_epidsodes_day_t=inpatient_epidsodes[inpatient_epidsodes['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    inpatient_epidsodes_day_t['AdmissionDate']=pd.to_datetime(inpatient_epidsodes_day_t['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
    inpatient_epidsodes_day_t=inpatient_epidsodes_day_t[inpatient_epidsodes_day_t['AdmissionDate']<=pd.to_datetime(index_date)]
    inpatient_epidsodes_day_t_previous365=inpatient_epidsodes_day_t[inpatient_epidsodes_day_t['AdmissionDate']>=(pd.to_datetime(index_date)-datetime.timedelta(days=365))]
    LOS_features=inpatient_epidsodes_day_t_previous365.groupby(['ClusterID'])['LOS, hour'].agg(['count', 'sum','max', 'min', 'mean', 'median', 'std'])
    LOS_features.columns=['Numnber of admissions within 365 days before index date', 'Overall LOS within 365 days before index date, hour', 'Maximum LOS of prior admissions within 365 days before index date, hour', 'Minimum LOS of prior admissions within 365 days before index date, hour', 'Mean LOS of prior admissions within 365 days before index date, hour', 'Median LOS of prior admissions within 365 days before index date, hour', 'LOS SD of prior admissions within 365 days before index date']
    Database_day_t=pd.merge(Database_day_t,LOS_features, how='left', on='ClusterID')
    # readmission features
    Readm30Num=pd.DataFrame(inpatient_epidsodes_day_t_previous365[['ClusterID', 'AdmissionDate', 'DischargeDate', 'Readmit30', 'Readmit180']].drop_duplicates().groupby(['ClusterID'])['Readmit30'].sum())
    Readm30Num.columns=['Number of early 30-day readmissions within 365 days before index date']
    Database_day_t=pd.merge(Database_day_t,Readm30Num, how='left', on='ClusterID')
    Readm30Duration=pd.DataFrame(inpatient_epidsodes_day_t_previous365[['ClusterID', 'AdmissionDate', 'DischargeDate', 'Readmit30', 'Readmit180']].groupby(['ClusterID'])['AdmissionDate'].last())
    Readm30Duration['Time elapsed from most recent early 30-day readmission within 365 days before index date, day']=[(pd.to_datetime(index_date)-Readm30Duration['AdmissionDate'].iloc[x]).days for x in range(Readm30Duration.shape[0])]
    Database_day_t=pd.merge(Database_day_t,Readm30Duration.iloc[: , 1:], how='left', on='ClusterID')
    # diagnosis features
    start = timeit.default_timer()
    inpt_diagnosis_epis_day_t=inpt_diagnosis_epis[inpt_diagnosis_epis['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    inpt_diagnosis_epis_day_t['AdmissionDate']=pd.to_datetime(inpt_diagnosis_epis_day_t['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
    inpt_diagnosis_epis_day_t=inpt_diagnosis_epis_day_t[inpt_diagnosis_epis_day_t['AdmissionDate']<=pd.to_datetime(index_date)]
    inpt_diagnosis_epis_day_t_current=inpt_diagnosis_epis_day_t[inpt_diagnosis_epis_day_t['EpisodeID'].isin(Database_day_t['EpisodeID'].values.tolist())]
    inpt_diagnosis_current=pd.get_dummies(inpt_diagnosis_epis_day_t_current, columns=['Disease'], prefix='').groupby(['ClusterID'], as_index=True).sum().T.drop_duplicates().T
    inpt_diagnosis_current[inpt_diagnosis_current>0]=1
    inpt_diagnosis_current.columns=['Diagnosis of '+x.split('_')[1].lower()+' within current admission (Yes/No)' for x in inpt_diagnosis_current.columns.tolist()]
    Database_day_t=pd.merge(Database_day_t,inpt_diagnosis_current, how='left', on='ClusterID') 
    inpt_diagnosis_epis_day_t_prior24=inpt_diagnosis_epis_day_t[inpt_diagnosis_epis_day_t['AdmissionDate']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=24))]
    inpt_diagnosis_prior24=pd.get_dummies(inpt_diagnosis_epis_day_t_prior24, columns=['Disease'], prefix='').groupby(['ClusterID'], as_index=True).sum().T.drop_duplicates().T
    inpt_diagnosis_prior24[inpt_diagnosis_prior24>0]=1
    inpt_diagnosis_prior24.columns=['Diagnosis of '+x.split('_')[1].lower()+' within 24 hours before index date (Yes/No)' for x in inpt_diagnosis_prior24.columns.tolist()]
    Database_day_t=pd.merge(Database_day_t,inpt_diagnosis_prior24, how='left', on='ClusterID') 
    inpt_diagnosis_epis_day_t_prior365=inpt_diagnosis_epis_day_t[inpt_diagnosis_epis_day_t['AdmissionDate']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=24*365))]
    inpt_diagnosis_prior365=pd.get_dummies(inpt_diagnosis_epis_day_t_prior365, columns=['Disease'], prefix='').groupby(['ClusterID'], as_index=True).sum().T.drop_duplicates().T
    inpt_diagnosis_prior365.columns=['Number of '+x.split('_')[1].lower()+' diagnosis within 365 days before index date (Yes/No)' for x in inpt_diagnosis_prior365.columns.tolist()]
    Database_day_t=pd.merge(Database_day_t,inpt_diagnosis_prior365, how='left', on='ClusterID') 
    inpt_diagnosis_prior365=pd.get_dummies(inpt_diagnosis_epis_day_t_prior365, columns=['Disease'], prefix='').groupby(['ClusterID'], as_index=True).sum().T.drop_duplicates().T
    inpt_diagnosis_prior365[inpt_diagnosis_prior365>0]=1
    inpt_diagnosis_prior365.columns=['Diagnosis of '+x.split('_')[1].lower()+' within 365 days before index date (Yes/No)' for x in inpt_diagnosis_prior365.columns.tolist()]
    Database_day_t=pd.merge(Database_day_t,inpt_diagnosis_prior365, how='left', on='ClusterID')     
    stop = timeit.default_timer()
    print('Time to extract diagnosis features: ', stop - start)
    # specialty features
    start = timeit.default_timer()
    inpt_diagnosis_epis_day_t_current=inpatient_epidsodes_day_t[inpatient_epidsodes_day_t['EpisodeID'].isin(Database_day_t['EpisodeID'].values.tolist())]
    inpatient_epidsodes_specialty_current=pd.get_dummies(inpt_diagnosis_epis_day_t_current[['ClusterID','Specialty']], columns=['Specialty'], prefix='').groupby(['ClusterID'], as_index=True).sum().T.drop_duplicates().T
    inpatient_epidsodes_specialty_current[inpatient_epidsodes_specialty_current>0]=1
    inpatient_epidsodes_specialty_current.columns=['Had '+x.split('_')[1].lower()+' within current admission (Yes/No)' for x in inpatient_epidsodes_specialty_current.columns.tolist()]
    Database_day_t=pd.merge(Database_day_t,inpatient_epidsodes_specialty_current, how='left', on='ClusterID')     
    inpatient_epidsodes_specialty_prior365=pd.get_dummies(inpatient_epidsodes_day_t_previous365[['ClusterID','Specialty']], columns=['Specialty'], prefix='').groupby(['ClusterID'], as_index=True).sum().T.drop_duplicates().T
    inpatient_epidsodes_specialty_prior365.columns=['Number of '+x.split('_')[1].lower()+' 365 days before index date (Yes/No)' for x in inpatient_epidsodes_specialty_prior365.columns.tolist()]
    Database_day_t=pd.merge(Database_day_t,inpatient_epidsodes_specialty_prior365, how='left', on='ClusterID')     
    inpatient_epidsodes_specialty_prior365[inpatient_epidsodes_specialty_prior365>0]=1
    inpatient_epidsodes_specialty_prior365=pd.get_dummies(inpatient_epidsodes_day_t_previous365[['ClusterID','Specialty']], columns=['Specialty'], prefix='').groupby(['ClusterID'], as_index=True).sum().T.drop_duplicates().T
    inpatient_epidsodes_specialty_prior365.columns=['Had '+x.split('_')[1].lower()+' 365 days before index date (Yes/No)' for x in inpatient_epidsodes_specialty_prior365.columns.tolist()]
    Database_day_t=pd.merge(Database_day_t,inpatient_epidsodes_specialty_prior365, how='left', on='ClusterID')     
    inpatient_epidsodes_specialty_duration=pd.DataFrame(inpatient_epidsodes_day_t_previous365.groupby(['ClusterID','Specialty'])['AdmissionDate'].last()).reset_index()
    inpatient_epidsodes_specialty_duration['Duration']=[(pd.to_datetime(index_date)-inpatient_epidsodes_specialty_duration['AdmissionDate'].iloc[x]).total_seconds()/3600/24 for x in range(inpatient_epidsodes_specialty_duration.shape[0])]
    for specialty in list(set(ConsultantSpecifialty['Specialty'])):
        inpatient_epidsodes_specialty_duration_sub=inpatient_epidsodes_specialty_duration[['ClusterID','Duration']][inpatient_epidsodes_specialty_duration['Specialty']==specialty]
        inpatient_epidsodes_specialty_duration_sub.columns=['ClusterID','Time elapsed since most recent '+specialty.lower()+' 365 days before index date, day']
        Database_day_t=pd.merge(Database_day_t,inpatient_epidsodes_specialty_duration_sub, how='left', on='ClusterID')        
    stop = timeit.default_timer()
    print('Time to extract specialty features: ', stop - start)
    # procedure duration
    procedures_day_t=inpt_procedures[inpt_procedures['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    procedures_day_t['ProcDate']=pd.to_datetime(procedures_day_t['ProcDate'], dayfirst=True).dt.tz_localize(None)
    procedures_day_t=procedures_day_t[procedures_day_t['ProcDate']<=pd.to_datetime(index_date)]
    Procedure_duration=pd.DataFrame(procedures_day_t.groupby(['ClusterID'])['ProcDate'].last())
    Procedure_duration['Time elapsed since most recent procedure before index date, day']=[(pd.to_datetime(index_date)-Procedure_duration['ProcDate'].iloc[x]).total_seconds()/3600/24 for x in range(Procedure_duration.shape[0])]
    Procedure_duration=Procedure_duration.iloc[: , 1:]
    Database_day_t=pd.merge(Database_day_t,Procedure_duration, how='left', on='ClusterID')
    # radiology procedure duration
    procedures_radiology_day_t=procedures_radiology[procedures_radiology['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    procedures_radiology_day_t['ProcDate']=pd.to_datetime(procedures_radiology_day_t['ProcDate'], dayfirst=True).dt.tz_localize(None)
    procedures_radiology_day_t=procedures_radiology_day_t[procedures_radiology_day_t['ProcDate']<=pd.to_datetime(index_date)]
    Procedure_radiology_duration=pd.DataFrame(procedures_radiology_day_t.groupby(['ClusterID'])['ProcDate'].last())
    Procedure_radiology_duration['Time elapsed since most recent radiology procedure before index date, day']=[(pd.to_datetime(index_date)-Procedure_radiology_duration['ProcDate'].iloc[x]).total_seconds()/3600/24 for x in range(Procedure_radiology_duration.shape[0])]
    Procedure_radiology_duration=Procedure_radiology_duration.iloc[: , 1:]
    Database_day_t=pd.merge(Database_day_t,Procedure_radiology_duration, how='left', on='ClusterID')
    # theatre duration
    theatres_day_t=theatres[theatres['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    theatres_day_t['arrived_time']=pd.to_datetime(theatres_day_t['arrived_time'], dayfirst=True).dt.tz_localize(None)
    theatres_day_t=theatres_day_t[theatres_day_t['arrived_time']<=pd.to_datetime(index_date)]
    Theatre_duration=pd.DataFrame(theatres_day_t.groupby(['ClusterID'])['arrived_time'].last())
    Theatre_duration['Time elapsed since most recent theatre use before index date, day']=[(pd.to_datetime(index_date)-Theatre_duration['arrived_time'].iloc[x]).total_seconds()/3600/24 for x in range(Theatre_duration.shape[0])]
    Theatre_duration=Theatre_duration.iloc[: , 1:]
    Database_day_t=pd.merge(Database_day_t,Theatre_duration, how='left', on='ClusterID')
    # ward duration
    wards_day_t=wards[wards['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    wards_day_t['WardStartDate']=pd.to_datetime(wards_day_t['WardStartDate'], dayfirst=True).dt.tz_localize(None)
    wards_day_t=wards_day_t[wards_day_t['WardStartDate']<=pd.to_datetime(index_date)]
    Ward_duration=pd.DataFrame(wards_day_t.groupby(['ClusterID'])['WardStartDate'].last())
    Ward_duration['Time elapsed since most recent ward start before index date, day']=[(pd.to_datetime(index_date)-Ward_duration['WardStartDate'].iloc[x]).total_seconds()/3600/24 for x in range(Ward_duration.shape[0])]
    Ward_duration=Ward_duration.iloc[: , 1:]
    Database_day_t=pd.merge(Database_day_t,Ward_duration, how='left', on='ClusterID')
    hourMarkers=[24, 48, 365*24]
    for hourMarker in hourMarkers:
        # procedure features
        procedures_day_t_temp=procedures_day_t[procedures_day_t['ProcDate']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
        Procedure_sum=pd.DataFrame(procedures_day_t_temp.groupby(['ClusterID'])['ProcNumber'].sum())
        colname='Number of procedures '+str(hourMarker)+' hours before index date'
        Procedure_sum.columns=[colname]
        Database_day_t=pd.merge(Database_day_t,Procedure_sum, how='left', on='ClusterID')
        Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
        Database_day_t['Had procedure '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
        # radiology procedure features
        procedures_radiology_day_t_temp=procedures_radiology_day_t[procedures_radiology_day_t['ProcDate']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
        Procedure_radiology_sum=pd.DataFrame(procedures_radiology_day_t_temp.groupby(['ClusterID'])['ProcNumber'].sum())
        colname='Number of radiology procedures '+str(hourMarker)+' hours before index date'
        Procedure_radiology_sum.columns=[colname]
        Database_day_t=pd.merge(Database_day_t,Procedure_radiology_sum, how='left', on='ClusterID')
        Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
        Database_day_t['Had radiology procedure '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
        # theatre features
        theatres_day_t_temp=theatres_day_t[theatres_day_t['arrived_time']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
        theatres_day_t_temp['Theatre use']=1
        Theatre_sum=pd.DataFrame(theatres_day_t_temp.groupby(['ClusterID'])['Theatre use'].sum())
        colname='Number of theatre use '+str(hourMarker)+' hours before index date'
        Theatre_sum.columns=[colname]
        Database_day_t=pd.merge(Database_day_t,Theatre_sum, how='left', on='ClusterID')
        Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
        Database_day_t['Had theatre use '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
        # ward features
        wards_day_t_temp=wards_day_t[wards_day_t['WardStartDate']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
        wards_day_t_temp['Ward use']=1
        Ward_sum=pd.DataFrame(wards_day_t_temp.groupby(['ClusterID'])['Ward use'].sum())
        colname='Number of ward use '+str(hourMarker)+' hours before index date'
        Ward_sum.columns=[colname]
        Database_day_t=pd.merge(Database_day_t,Ward_sum, how='left', on='ClusterID')
        Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
        Database_day_t['Had ward use '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
    # DNACPR features
    DNACPR_day_t=resus[resus['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    DNACPR_day_t['PerformedDateTime']=pd.to_datetime(DNACPR_day_t['PerformedDateTime'], dayfirst=True).dt.tz_localize(None)
    DNACPR_day_t=DNACPR_day_t[DNACPR_day_t['PerformedDateTime']<=pd.to_datetime(index_date)]
    DNACPR_duration=pd.DataFrame(DNACPR_day_t.groupby(['ClusterID'])['PerformedDateTime'].last())
    DNACPR_duration['Time elapsed since most recent DNACPR before index date, hour']=[(pd.to_datetime(index_date)-DNACPR_duration['PerformedDateTime'].iloc[x]).total_seconds()/3600 for x in range(DNACPR_duration.shape[0])]
    DNACPR_duration=DNACPR_duration.iloc[: , 1:]
    Database_day_t=pd.merge(Database_day_t,DNACPR_duration, how='left', on='ClusterID')
    Database_day_t['Had DNACPR before index date (Yes/No)']=np.where(Database_day_t['Time elapsed since most recent DNACPR before index date, hour']>0,1,0)  
    # news2 features
    start = timeit.default_timer()
    NEWS2_day_t=NEWS2[NEWS2['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    NEWS2_day_t['PerformedDateTime']=pd.to_datetime(NEWS2_day_t['PerformedDateTime'], dayfirst=True).dt.tz_localize(None)
    NEWS2_day_t=NEWS2_day_t[NEWS2_day_t['PerformedDateTime']<=pd.to_datetime(index_date)]
    for testcol in ['Oxygen saturation, %', 'Diastolic blood pressure, mmHg', 'Systolic blood pressure, mmHg',
                    'Heart rate, bpm', 'Respiratory rate, br/min', 'Temperature, °C', 'NEWS2 score']:
        NEWS2_day_t_test=NEWS2_day_t[~NEWS2_day_t[testcol].isnull()]
        NEWS2_duration_test=pd.DataFrame(NEWS2_day_t_test.groupby(['ClusterID'])['PerformedDateTime'].last())
        NEWS2_duration_test['Time elapsed since most recent '+testcol.split(',')[0]+' before index date, hour']=[(pd.to_datetime(index_date)-NEWS2_duration_test['PerformedDateTime'].iloc[x]).total_seconds()/3600 for x in range(NEWS2_duration_test.shape[0])]
        Database_day_t=pd.merge(Database_day_t,NEWS2_duration_test.iloc[: , 1:], how='left', on='ClusterID')
        for hourMarker in [24, 48]:
            NEWS2_day_t_temp=NEWS2_day_t_test[NEWS2_day_t_test['PerformedDateTime']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
            # number
            NEWS2_day_t_temp_sum=pd.DataFrame(NEWS2_day_t_temp.groupby(['ClusterID'])['PerformedDateTime'].count())
            colname='Number of '+testcol.split(',')[0]+' tested '+str(hourMarker)+' hours before index date'
            NEWS2_day_t_temp_sum.columns=[colname]
            Database_day_t=pd.merge(Database_day_t,NEWS2_day_t_temp_sum, how='left', on='ClusterID')
            Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
            Database_day_t['Had '+testcol.split(',')[0]+' tested '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
            NEWS2_features=NEWS2_day_t_temp.groupby(['ClusterID'])[testcol].agg(['max', 'min', 'mean', 'median', 'std'])
            NEWS2_features.columns=['Maximum '+testcol.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'Minimum '+testcol.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'Mean '+testcol.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'Median '+testcol.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'SD of '+testcol.split(',')[0]+' '+str(hourMarker)+' hours before index date']
            Database_day_t=pd.merge(Database_day_t,NEWS2_features, how='left', on='ClusterID') 
        for hourMarker in [365*24]:
            NEWS2_day_t_temp=NEWS2_day_t_test[NEWS2_day_t_test['PerformedDateTime']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
            # number
            NEWS2_day_t_temp_sum=pd.DataFrame(NEWS2_day_t_temp.groupby(['ClusterID'])['PerformedDateTime'].count())
            colname='Number of '+testcol.split(',')[0]+' tested '+str(hourMarker)+' hours before index date'
            NEWS2_day_t_temp_sum.columns=[colname]
            Database_day_t=pd.merge(Database_day_t,NEWS2_day_t_temp_sum, how='left', on='ClusterID')
            Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
            Database_day_t['Had '+testcol.split(',')[0]+' tested '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)    
    stop = timeit.default_timer()
    print('Time to extract news2 features: ', stop - start)
    # O2Devices features
    start = timeit.default_timer()
    O2Devices_day_t=vitals[vitals['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    O2Devices_day_t=O2Devices_day_t[O2Devices_day_t['PerformedDateTime']<=pd.to_datetime(index_date)]
    for hourMarker in [24, 48, 365*24]:
        O2Devices_day_t_temp=O2Devices_day_t[O2Devices_day_t['PerformedDateTime']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
        O2Devices_day_t_temp_stats=pd.DataFrame(O2Devices_day_t_temp.groupby(['ClusterID','O2Device'])['PerformedDateTime'].count()).reset_index(level=1)
        for testcol in O2Devices['O2Device'].dropna().unique().tolist():
            O2Devices_day_t_temp_stats_test=O2Devices_day_t_temp_stats[O2Devices_day_t_temp_stats['O2Device']==test].iloc[:,1:O2Devices_day_t_temp_stats.shape[1]]
            colname='Number of '+testcol+' '+str(hourMarker)+' hours before index date'
            O2Devices_day_t_temp_stats_test.columns=[colname]
            Database_day_t=pd.merge(Database_day_t,O2Devices_day_t_temp_stats_test, how='left', on='ClusterID')
            Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
            Database_day_t['Had '+testcol+' '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
    O2Devices_day_t_current=pd.merge(O2Devices_day_t, Database_day_t[['ClusterID', 'AdmissionDate']], how='left', on='ClusterID')
    O2Devices_day_t_current=O2Devices_day_t_current[O2Devices_day_t_current['AdmissionDate']<=O2Devices_day_t_current['PerformedDateTime']]
    O2Devices_day_t_current_stats=O2Devices_day_t_current.groupby(['ClusterID','O2Device'])['PerformedDateTime'].count().reset_index(level=1) 
    for testcol in O2Devices['O2Device'].dropna().unique().tolist():
        O2Devices_day_t_current_stats_test=O2Devices_day_t_current_stats[O2Devices_day_t_current_stats['O2Device']==test].iloc[:,1:O2Devices_day_t_current_stats.shape[1]]
        colname=['Number of '+testcol+' within current admission']
        O2Devices_day_t_current_stats_test.columns=colname
        Database_day_t=pd.merge(Database_day_t,O2Devices_day_t_current_stats_test, how='left', on='ClusterID')  
        Database_day_t['Had '+testcol+' within current admission (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
    stop = timeit.default_timer()
    print('Time to extract O2Devices features: ', stop - start) 
    # labtests features
    start = timeit.default_timer()
    labs_day_t=labtests[labtests['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    labs_day_t=labs_day_t[labs_day_t['CollectionDateTime']<=pd.to_datetime(index_date)]
    hourMarkers=[24, 48, 365*24]
    for hourMarker in hourMarkers:
        labs_day_t_temp=labs_day_t[labs_day_t['CollectionDateTime']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
        labs_day_t_temp_stats=labs_day_t_temp.groupby(['ClusterID','TestName'])['Value'].agg(['count', 'max', 'min', 'mean', 'median', 'std']).reset_index(level=1) 
        for test in tests:
            labs_day_t_temp_stats_test=labs_day_t_temp_stats[labs_day_t_temp_stats['TestName']==test].iloc[:,1:labs_day_t_temp_stats.shape[1]]
            if len(test.split(','))>1:
                labs_day_t_temp_stats_test.columns=['Number of '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'Maximum '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date,'+test.split(',')[1], 'Minimum '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date,'+test.split(',')[1], 'Mean '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date,'+test.split(',')[1], 'Median '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date,'+test.split(',')[1], 'SD of '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date']
            else:
                labs_day_t_temp_stats_test.columns=['Number of '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'Maximum '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'Minimum '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'Mean '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'Median '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date', 'SD of '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date']
            if hourMarker==8760:
                labs_day_t_temp_stats_test=labs_day_t_temp_stats_test.loc[:, [x.split(' ')[0] not in ['Maximum', 'Minimum', 'Mean', 'Median', 'SD'] for x in labs_day_t_temp_stats_test.columns.tolist()]]
            Database_day_t=pd.merge(Database_day_t,labs_day_t_temp_stats_test, how='left', on='ClusterID')
            Database_day_t['Had '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t['Number of '+test.split(',')[0]+' '+str(hourMarker)+' hours before index date']>0,1,0)
    labs_day_t_current=pd.merge(labs_day_t, Database_day_t[['ClusterID', 'AdmissionDate']], how='left', on='ClusterID')
    labs_day_t_current=labs_day_t_current[labs_day_t_current['AdmissionDate']<=labs_day_t_current['CollectionDateTime']]
    labs_day_t_current_stats=labs_day_t_current.groupby(['ClusterID','TestName'])['Value'].agg(['count', 'max', 'min', 'mean', 'median', 'std']).reset_index(level=1) 
    for test in tests:
        labs_day_t_current_stats_test=labs_day_t_current_stats[labs_day_t_current_stats['TestName']==test].iloc[:,1:labs_day_t_current_stats.shape[1]]
        if len(test.split(','))>1:
            labs_day_t_current_stats_test.columns=['Number of '+test.split(',')[0]+' within current admission', 'Maximum '+test.split(',')[0]+' within current admission,'+test.split(',')[1], 'Minimum '+test.split(',')[0]+' within current admission,'+test.split(',')[1], 'Mean '+test.split(',')[0]+' within current admission,'+test.split(',')[1], 'Median '+test.split(',')[0]+' within current admission,'+test.split(',')[1], 'SD of '+test.split(',')[0]+' within current admission']
        else:
            labs_day_t_current_stats_test.columns=['Number of '+test.split(',')[0]+' within current admission', 'Maximum '+test.split(',')[0]+' within current admission', 'Minimum '+test.split(',')[0]+' within current admission', 'Mean '+test.split(',')[0]+' within current admission', 'Median '+test.split(',')[0]+' within current admission', 'SD of '+test.split(',')[0]+' within current admission']
        Database_day_t=pd.merge(Database_day_t,labs_day_t_current_stats_test, how='left', on='ClusterID')
        Database_day_t['Had '+test.split(',')[0]+' within current admission (Yes/No)']=np.where(Database_day_t['Number of '+test.split(',')[0]+' within current admission']>0,1,0)
    stop = timeit.default_timer()
    print('Time to extract labtests features: ', stop - start) 
    # positive cultures
    start = timeit.default_timer()
    micro_day_t=micro[micro['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    micro_day_t['CollectionDateTime']=pd.to_datetime(micro_day_t['CollectionDateTime'], dayfirst=True).dt.tz_localize(None)
    micro_day_t=micro_day_t[micro_day_t['CollectionDateTime']<=pd.to_datetime(index_date)]
    for culture in ['urine culture', 'blood culture']:
        micro_day_t_culture=micro_day_t[micro_day_t['BatTestName']==culture]
        micro_day_t_culture['Culture number']=1
        Culture_duration=pd.DataFrame(micro_day_t_culture.groupby(['ClusterID'])['CollectionDateTime'].last())
        Culture_duration['Time elapsed since most recent positive '+culture+' before index date, hour']=[(pd.to_datetime(index_date)-Culture_duration['CollectionDateTime'].iloc[x]).total_seconds()/3600 for x in range(Culture_duration.shape[0])]
        Database_day_t=pd.merge(Database_day_t,Culture_duration.iloc[: , 1:], how='left', on='ClusterID')
        hourMarkers=[24, 48, 365*24]
        for hourMarker in hourMarkers:
            micro_day_t_culture_temp=micro_day_t_culture[micro_day_t_culture['CollectionDateTime']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
            micro_day_t_culture_sum=pd.DataFrame(micro_day_t_culture_temp.groupby(['ClusterID'])['Culture number'].sum())
            colname='Number of positive '+culture+' within '+str(hourMarker)+' hours before index date'
            micro_day_t_culture_sum.columns=[colname]
            Database_day_t=pd.merge(Database_day_t,micro_day_t_culture_sum, how='left', on='ClusterID')
            Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
            Database_day_t['Positive '+culture+' tested within '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
    stop = timeit.default_timer()
    print('Time to extract positive cultures features: ', stop - start)
    # other microbiology tests
    start = timeit.default_timer()
    micro_day_t=micro[micro['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    micro_day_t['CollectionDateTime']=pd.to_datetime(micro_day_t['CollectionDateTime'], dayfirst=True).dt.tz_localize(None)
    micro_day_t=micro_day_t[micro_day_t['CollectionDateTime']<=pd.to_datetime(index_date)]
    for bug in ['Staphylococcus aureus', 'Enterococcus','ESBL/CPE']:
        micro_day_t_bug=micro_day_t[micro_day_t['BugName'].isin(BugsIsolatedMapping[bug].dropna().tolist())]
        micro_day_t_bug['Bug number']=1
        Bug_duration=pd.DataFrame(micro_day_t_bug.groupby(['ClusterID'])['CollectionDateTime'].last())
        Bug_duration['Time elapsed since most recent tested '+bug+' before index date, hour']=[(pd.to_datetime(index_date)-Bug_duration['CollectionDateTime'].iloc[x]).total_seconds()/3600 for x in range(Bug_duration.shape[0])]
        Database_day_t=pd.merge(Database_day_t, Bug_duration , how='left', on='ClusterID')
        hourMarkers=[24, 48, 365*24]
        for hourMarker in hourMarkers:
            micro_day_t_bug_temp=micro_day_t_bug[micro_day_t_bug['CollectionDateTime']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))]
            micro_day_t_bug_sum=pd.DataFrame(micro_day_t_bug_temp.groupby(['ClusterID'])['Bug number'].sum())
            colname='Number of tested '+bug+' within '+str(hourMarker)+' hours before index date'
            micro_day_t_bug_sum.columns=[colname]
            Database_day_t=pd.merge(Database_day_t,micro_day_t_bug_sum, how='left', on='ClusterID')
            Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
            Database_day_t[bug+' tested within '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
    stop = timeit.default_timer()
    print('Time to extract tested bug features: ', stop - start)
    # antibiotics prescriptions
    start = timeit.default_timer()
    antibiotics_day_t=antibiotics[antibiotics['ClusterID'].isin(Database_day_t['ClusterID'].values.tolist())]
    antibiotics_day_t['PrescriptionDateTime']=pd.to_datetime(antibiotics_day_t['PrescriptionDateTime'], dayfirst=True).dt.tz_localize(None)
    antibiotics_day_t=antibiotics_day_t[antibiotics_day_t['PrescriptionDateTime']<=pd.to_datetime(index_date)]
    antibiotics_day_t=antibiotics_day_t[antibiotics_day_t['PrescriptionDateTime']>=(pd.to_datetime(index_date)-datetime.timedelta(days=365))]
    Antibiotics_duration=pd.DataFrame(antibiotics_day_t.groupby(['ClusterID'])['PrescriptionDateTime'].last())
    colname='Time elapsed since most recent antibiotics prescription 365 days before index date, hour'
    Antibiotics_duration[colname]=[(pd.to_datetime(index_date)-Antibiotics_duration['PrescriptionDateTime'].iloc[x]).total_seconds()/3600 for x in range(Antibiotics_duration.shape[0])]
    Database_day_t=pd.merge(Database_day_t,Antibiotics_duration.iloc[: , 1:], how='left', on='ClusterID')
    Database_day_t['Had antibiotics use 365 days before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
    hourMarkers=[24, 48]
    for hourMarker in hourMarkers:
        antibiotics_day_t_temp=antibiotics_day_t[antibiotics_day_t['PrescriptionDateTime']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=hourMarker))].drop_duplicates()
        antibiotics_day_t_temp['Antibiotics number']=1
        antibiotics_day_t_temp_sum=pd.DataFrame(antibiotics_day_t_temp.groupby(['ClusterID'])['Antibiotics number'].sum())
        colname='Number of antibiotics prescription within '+str(hourMarker)+' hours before index date'
        antibiotics_day_t_temp_sum.columns=[colname]
        Database_day_t=pd.merge(Database_day_t,antibiotics_day_t_temp_sum, how='left', on='ClusterID')
        Database_day_t[colname]=np.where(Database_day_t[colname]>0,Database_day_t[colname],0)
        Database_day_t['Antibiotics prescription within '+str(hourMarker)+' hours before index date (Yes/No)']=np.where(Database_day_t[colname]>0,1,0)
    HadNewAntibiotics=pd.DataFrame(np.zeros((instances_day_t.shape[0],1)), columns=['New antibiotics prescribed 24 hours before index date (Yes/No)'])
    HadNewAntibiotics.index=instances_day_t['ClusterID']
    for i in range(instances_day_t.shape[0]):
        antibiotics_day_t_pat=antibiotics_day_t[antibiotics_day_t['ClusterID']==instances_day_t.iloc[i,0]]
        antibiotics_day_t_pat['PrescriptionDateTime']=pd.to_datetime(antibiotics_day_t_pat['PrescriptionDateTime'], dayfirst=True).dt.tz_localize(None)
        antibiotics_day_t_pat_prior=antibiotics_day_t_pat[antibiotics_day_t_pat['PrescriptionDateTime']<(pd.to_datetime(index_date)-datetime.timedelta(hours=24))].drop_duplicates()
        antibiotics_day_t_pat_current=antibiotics_day_t_pat[antibiotics_day_t_pat['PrescriptionDateTime']>=(pd.to_datetime(index_date)-datetime.timedelta(hours=24))].drop_duplicates()
        newdrugs= [x for x in list(set(antibiotics_day_t_pat_current['Drug'].str.lower())) if x not in list(set(antibiotics_day_t_pat_prior['Drug'].str.lower()))]
        if len(newdrugs)>0:
            HadNewAntibiotics.iloc[i,0]=1
    Database_day_t=pd.merge(Database_day_t,HadNewAntibiotics, how='left', on='ClusterID')   
    stop = timeit.default_timer()
    print('Time to extract antibiotics features: ', stop - start)
    # aggregate all above extracted features
    Database.append(Database_day_t)
    stop_whole = timeit.default_timer()
    print('Time to extract features for patients being in hospital on index date: ', stop_whole - start_whole)

Database=pd.concat(Database, axis=0)

# Gender
Database['Male']=np.where(Database['LinkedSex']=='M', 1, 0)
Database['Female']=np.where(Database['LinkedSex']=='F', 1, 0)
cols = ['Male', 'Female']
Database['Gender']=Database[cols].idxmax(1)
# Age
Database['<16'] = np.where(Database['Age']<=16, 1, 0)
Database['16-40'] = np.where( (Database['Age']>16) & (Database['Age']<=40), 1, 0)
Database['40-60'] = np.where( (Database['Age']>40) & (Database['Age']<=60), 1, 0)
Database['60-80'] = np.where( (Database['Age']>60) & (Database['Age']<=80), 1, 0)
Database['>80'] = np.where(Database['Age']>80, 1, 0)
cols = ['16-40', '40-60', '60-80', '>80']
Database['Age group']=Database[cols].idxmax(1)
# EthnicGroup
Database['White']=np.where(Database['EthnicGroupCode'].isin(['A ','B ','C ']), 1, 0)
Database['Mixed']=np.where(Database['EthnicGroupCode'].isin(['D ','E ','F ','G ']), 1, 0)
Database['Asian or Asian British']=np.where(Database['EthnicGroupCode'].isin(['H ','J ','K ','L ']), 1, 0)
Database['Black or Black British']=np.where(Database['EthnicGroupCode'].isin(['M ','N ','P ']), 1, 0)
Database['Other Ethnicity Groups']=np.where(Database['EthnicGroupCode'].isin(['R ','S ']), 1, 0)
Database['Not Stated/Not Known']=np.where(Database['EthnicGroupCode'].isin(['Z ',99]), 1, 0)
cols = ['White', 'Mixed', 'Asian or Asian British', 'Black or Black British',
       'Other Ethnicity Groups', 'Not Stated/Not Known']
Database['Ethnicity group']=Database[cols].idxmax(1)
# IMDScore
Database['IMD<8.5'] = np.where(Database['IMDScore']<8.5, 1, 0)
Database['IMD 8.5-13.8'] = np.where( (Database['IMDScore']>=8.5) & (Database['IMDScore']<13.8), 1, 0)
Database['IMD 13.8-21.36'] = np.where( (Database['IMDScore']>=13.8) & (Database['IMDScore']<21.36), 1, 0)
Database['IMD 21.36-34.18'] = np.where( (Database['IMDScore']>=21.36) & (Database['IMDScore']<=34.18), 1, 0)
Database['IMD>34.18'] = np.where(Database['IMDScore']>34.18, 1, 0)
cols = ['IMD<8.5', 'IMD 8.5-13.8', 'IMD 13.8-21.36', 'IMD 21.36-34.18', 'IMD>34.18']
Database['IMD group']=Database[cols].idxmax(1)
# PostCodes
PostCodes=pd.read_csv(Path+"PostCodesMapping.csv")
for code in list(set(PostCodes["Fixed_Postcode"])):
    Database[code] = np.where(Database['PostcodeStub'].isin(PostCodes['PostcodeStub'][PostCodes["Fixed_Postcode"]==code].values.tolist()), 1, 0)
cols = [x for x in list(set(PostCodes["Fixed_Postcode"]))]
Database['Postcode region']=Database[cols].idxmax(1)
# Admission date features
# Admission daytime
Database['Admission daytime'] = (Database['AdmissionDate'].dt.hour % 24 + 4) // 4
Database['Admission daytime'].replace({1: 'Late night', 2: 'Early morning', 3: 'Morning', 4: 'Noon', 5: 'Evening', 6: 'Night'}, inplace=True)
# Weekday
Database['Weekday of admission date']=Database['AdmissionDate'].dt.dayofweek
Database['Admission_Week_Day']=Database['AdmissionDate'].dt.day_name().tolist()
Database['Monday admission'] = np.where(Database['Admission_Week_Day']=='Monday', 1, 0)
Database['Tuesday admission'] = np.where(Database['Admission_Week_Day']=='Tuesday', 1, 0)
Database['Wednesday admission'] = np.where(Database['Admission_Week_Day']=='Wednesday', 1, 0)
Database['Thursday admission'] = np.where(Database['Admission_Week_Day']=='Thursday', 1, 0)
Database['Friday admission'] = np.where(Database['Admission_Week_Day']=='Friday', 1, 0)
Database['Saturday admission'] = np.where(Database['Admission_Week_Day']=='Saturday', 1, 0)
Database['Sunday admission'] = np.where(Database['Admission_Week_Day']=='Sunday', 1, 0)
cols = ['Monday admission', 'Tuesday admission', 'Wednesday admission', 'Thursday admission', 'Friday admission', 'Saturday admission', 'Sunday admission']
Database['Admission weekday']=Database[cols].idxmax(1)
# Season
Database['Admission_Season'] = Database['AdmissionDate'].apply(lambda x: pd.to_datetime(x).month)
Database['Spring admission'] = np.where(Database['Admission_Season'].isin([3,4,5]), 1, 0)
Database['Summer admission'] = np.where(Database['Admission_Season'].isin([6,7,8]), 1, 0)
Database['Fall admission'] = np.where(Database['Admission_Season'].isin([9,10,11]), 1, 0)
Database['Winter admission'] = np.where(Database['Admission_Season'].isin([12,1,2]), 1, 0)
cols = ['Spring admission', 'Summer admission', 'Fall admission', 'Winter admission']
Database['Admission season']=Database[cols].idxmax(1)
# Admission day features
Database['Admission day of year'] = [Database['AdmissionDate'].iloc[x].timetuple().tm_yday for x in range(Database.shape[0])]
Database['Admission day time of month'] = Database['AdmissionDate'].dt.day
Database['Admission hour time of day'] = Database['AdmissionDate'].dt.round('H').dt.hour
# Index day features
Database['Index day of year'] = [Database['Index date'].iloc[x].timetuple().tm_yday for x in range(Database.shape[0])]
Database['Index hour time of day'] = Database['Index date'].dt.round('H').dt.hour
Database['Weekday of index date']=Database['Index date'].dt.dayofweek
from datetime import date 
import holidays 
holidayset=[]
for year in [2017, 2018, 2019, 2020]:
    [holidayset.append(x) for x in [str(date[0]) for date in holidays.UnitedKingdom(years=year).items()]]
holidayset=pd.DataFrame(holidayset, columns=['Date'])
holidayset=holidayset[(holidayset['Date']>=date(2017, 2, 1).strftime("%Y-%m-%d")) &
                                (holidayset['Date']<=date(2020, 2, 1).strftime("%Y-%m-%d"))]
Database['Index date public holiday (Yes/No)']=np.where(Database['Index date'].isin(holidayset['Date'].values.tolist()), 1, 0)
# Admission source
Database['Usual place of residence']=np.where(Database['AdmissionSourceCode'].isin([29,'29', 19,'19']), 1, 0)
Database['Non-NHS institutional care']=np.where(Database['AdmissionSourceCode'].isin([66, '66', 69,'69', 65, '66', 54, '54', 88, '88', 87, '87', 89, '89', 86, '86', 85, '85']), 1, 0)
Database['Other NHS Provider']=np.where(Database['AdmissionSourceCode'].isin([51, '51', 52,'52', 53, '53', 49, '49']), 1, 0)
cols = ['Usual place of residence', 'Non-NHS institutional care', 'Other NHS Provider']
Database['Admission source']=Database[cols].idxmax(1)
# Admission method
AdmissionMethodMapping=pd.read_csv(Path+"ref_admissionmethod.csv")
Database['Planned']=np.where(Database['AdmissionMethodCode'].isin(AdmissionMethodMapping['AdmissionMethodCode'][AdmissionMethodMapping['AdmissionMethodType']=='Elective']), 1, 0)
Database['Emergency']=np.where(Database['AdmissionMethodCode'].isin(AdmissionMethodMapping['AdmissionMethodCode'][AdmissionMethodMapping['AdmissionMethodType']=='Emergency']), 1, 0)
cols = ['Planned', 'Emergency']
Database['Admission method']=Database[cols].idxmax(1)
# Latest admission to decision day
Database['Adm2index, hours']=divmod((Database['Index date']-Database['AdmissionDate']).dt.total_seconds(), 3600)[0]
conditions=[(Database['Adm2index, hours']>=24*0) & (Database['Adm2index, hours']<24*1),
            (Database['Adm2index, hours']>=24*1) & (Database['Adm2index, hours']<24*2),
            (Database['Adm2index, hours']>=24*2) & (Database['Adm2index, hours']<24*3),
            (Database['Adm2index, hours']>=24*3) & (Database['Adm2index, hours']<24*4),
            (Database['Adm2index, hours']>=24*4) & (Database['Adm2index, hours']<24*5),
            (Database['Adm2index, hours']>=24*5) & (Database['Adm2index, hours']<24*6),
            (Database['Adm2index, hours']>=24*6) & (Database['Adm2index, hours']<24*11),
            (Database['Adm2index, hours']>=24*11) & (Database['Adm2index, hours']<24*14),
            (Database['Adm2index, hours']>=24*14) & (Database['Adm2index, hours']<24*21),
            (Database['Adm2index, hours']>=24*21) & (Database['Adm2index, hours']<24*28),
            (Database['Adm2index, hours']>=24*28)]
choices=['Being in hospital 0-24 hours (D1)', 'Being in hospital 24-48 hours (D2)',
         'Being in hospital 48-72 hours (D3)', 'Being in hospital 72-96 hours (D4)',
         'Being in hospital 96-120 hours (D5)', 'Being in hospital 120-144 hours (D6)',
         'Being in hospital 144-264 hours (D7-D10)', 'Being in hospital 264-336 hours (D11-D13)',
         'Being in hospital 336-504 hours (D13-D20)', 'Being in hospital 504-672 hours (D21-D27)',
         'Being in hospital >672 hours (D28+)']
Database['Being in hospital group'] = np.select(conditions, choices)
# Hospital stay outcomes 
Database['LOS after index date, hour']=divmod((Database['DischargeDate']-Database['Index date']).dt.total_seconds(), 3600)[0]
Database['LOS after index date, day']=Database['LOS after index date, hour']/24
Database['Imminent discharge within 24 hours'] = np.where(Database['LOS after index date, hour']<=24, 1, 0)
Database['Imminent discharge within 48 hours'] = np.where(Database['LOS after index date, hour']<=48, 1, 0)
Database['7-days longer stay'] = np.where(Database['LOS after index date, hour']>=24*7, 1, 0)
Database['7-days stay'] = np.where( (Database['LOS after index date, hour']>24) & (Database['LOS after index date, hour']<=24*7), 1, 0)
Database['7-14 days stay'] = np.where( (Database['LOS after index date, hour']>24*7) & (Database['LOS after index date, hour']<=24*14), 1, 0)
Database['14-30 days stay'] = np.where( (Database['LOS after index date, hour']>24*14) & (Database['LOS after index date, hour']<=24*30), 1, 0)
Database['Stay beyond 30 days'] = np.where(Database['LOS after index date, hour']>24*30, 1, 0)
# Mortality outcomes
Database['LinkedDeathdate']=pd.to_datetime(Database['LinkedDeathdate'], dayfirst=True).dt.tz_localize(None)
Database['ToMortalityHourDuration']=divmod((Database['LinkedDeathdate']-Database['Index date']).dt.total_seconds(), 3600)[0]
Database['30-day mortality'] = np.where(Database['ToMortalityHourDuration']<=24*30, 1, 0)
# Save
Columns=pd.read_csv(Path+"SelectedVariables.csv")
Database=Database.reindex(Columns['Selected'].dropna().values.tolist(), axis=1)
Database=Database[Database['Age']>=16]
cut_date='2019-02-01 6:00:00'
Database['Index date']=pd.to_datetime(Database['Index date'], dayfirst=True).dt.tz_localize(None)
Database['TrainingTesting']=np.where(Database['Index date']<=pd.to_datetime(cut_date, dayfirst=True),'Training', 'Testing')
Database['Patient group'] = Database[['Admission method', 'TrainingTesting']].apply(','.join, axis=1)
SavePath=Path+"Databases/0919/WithSplitingAdults/Index6/"
Database.to_csv(SavePath+"DatabaseIndex6.csv", index=False)
DataStatistics=[]
for admissionmethod in ['Planned', 'Emergency']:
    for modeluse in ['Testing', 'Training']:
        print(admissionmethod, modeluse)
        Data=Database[(Database['Admission method']==admissionmethod) & (Database['TrainingTesting']==modeluse)]
        Data[(Data['Adm2index, hours']>=24*0) & (Data['Adm2index, hours']<24*1)].to_csv(SavePath+"Database"+admissionmethod+"D1"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*1) & (Data['Adm2index, hours']<24*2)].to_csv(SavePath+"Database"+admissionmethod+"D2"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*2) & (Data['Adm2index, hours']<24*3)].to_csv(SavePath+"Database"+admissionmethod+"D3"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*3) & (Data['Adm2index, hours']<24*4)].to_csv(SavePath+"Database"+admissionmethod+"D4"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*4) & (Data['Adm2index, hours']<24*5)].to_csv(SavePath+"Database"+admissionmethod+"D5"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*5) & (Data['Adm2index, hours']<24*6)].to_csv(SavePath+"Database"+admissionmethod+"D6"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*6) & (Data['Adm2index, hours']<24*11)].to_csv(SavePath+"Database"+admissionmethod+"D7D10"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*11) & (Data['Adm2index, hours']<24*14)].to_csv(SavePath+"Database"+admissionmethod+"D11D13"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*14) & (Data['Adm2index, hours']<24*21)].to_csv(SavePath+"Database"+admissionmethod+"D14D20"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*21) & (Data['Adm2index, hours']<24*28)].to_csv(SavePath+"Database"+admissionmethod+"D21D27"+modeluse+".csv", index=False)
        Data[(Data['Adm2index, hours']>=24*28)].to_csv(SavePath+"Database"+admissionmethod+"D28+"+modeluse+".csv", index=False)

        

    
    

    


    
    

    












