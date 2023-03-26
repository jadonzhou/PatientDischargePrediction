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
        
# read basedatabase
Path="../iord_extract_20220325/"
Database = pd.read_csv(Path+'BaseDatabase for index date hour 0.csv', 
                 usecols = ['EpisodeID','Index date','Adm2index, hours','Admission method',
                            'Imminent discharge within 24 hours',
                            'Imminent discharge within 48 hours'], 
                 low_memory = True)
print(Database.head())

# read inpatient_epidsodes
inpatient_epidsodes = pd.read_hdf(Path+'inpatient_epidsodes.h5', key='df')
inpatient_epidsodes=inpatient_epidsodes[['ClusterID', 'EpisodeID','AdmissionDate','DischargeDate']]
Database=pd.merge(Database, inpatient_epidsodes, how='left', on='EpisodeID')
print(Database.head())


cut_date='2019-01-31 2:00:00'
Database['AdmissionDate']=pd.to_datetime(Database['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
Database['DischargeDate']=pd.to_datetime(Database['DischargeDate'], dayfirst=True).dt.tz_localize(None)
Database['Index date']=pd.to_datetime(Database['Index date'], dayfirst=True).dt.tz_localize(None)
Database['Hospital stay after index date, hours']=divmod((Database['DischargeDate']-Database['Index date']).dt.total_seconds(), 3600)[0]
Database['Hospital stay after index date, days']=Database['Hospital stay after index date, hours']/24
Database['TrainingTesting']=np.where(Database['Index date']<=pd.to_datetime(cut_date, dayfirst=True),'Training (2017/2/1-2019/1/31)', 'Testing (2019/2/1-2020/1/31)')
Database['Patient group'] = Database[['Admission method', 'TrainingTesting']].apply(','.join, axis=1)
print(Database.head())
Database['Patient group'].value_counts()


Database['Hospital stay after index date, days'][Database['Admission method']=='Emergency admission'].mean()
Database['Hospital stay after index date, days'][Database['Admission method']=='Emergency admission'].median()
quantile1=np.percentile(Database['Hospital stay after index date, days'][Database['Admission method']=='Planned admission'].tolist(), (25, 50, 75), interpolation='midpoint')
quantile1
    
   
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
choices=['D1', 'D2',
         'D3', 'D4',
         'D5', 'D6',
         'D7-D10', 'D11-D13',
         'D13-D20', 'D21-D27',
         'D28+']
Database['Being in hospital group (short)'] = np.select(conditions, choices)


# Weekday
Database['Admission_Week_Day']=Database['AdmissionDate'].dt.day_name().tolist()
Database['Monday admission'] = np.where(Database['Admission_Week_Day']=='Monday', 1, 0)
Database['Tuesday admission'] = np.where(Database['Admission_Week_Day']=='Tuesday', 1, 0)
Database['Wednesday admission'] = np.where(Database['Admission_Week_Day']=='Wednesday', 1, 0)
Database['Thursday admission'] = np.where(Database['Admission_Week_Day']=='Thursday', 1, 0)
Database['Friday admission'] = np.where(Database['Admission_Week_Day']=='Friday', 1, 0)
Database['Saturday admission'] = np.where(Database['Admission_Week_Day']=='Saturday', 1, 0)
Database['Sunday admission'] = np.where(Database['Admission_Week_Day']=='Sunday', 1, 0)
# Weekday
cols = ['Monday admission', 'Tuesday admission', 'Wednesday admission', 'Thursday admission', 'Friday admission', 'Saturday admission', 'Sunday admission']
Database['Admission weekday']=Database[cols].idxmax(1)



AdmissionWeekdayResult=pd.DataFrame(Database[['Admission weekday','Admission method']].value_counts()).reset_index()
AdmissionWeekdayResult.columns=['Weekday of admission date', 'Admission method', 'Number of patient-admissions']
AdmissionWeekdayResult

temp=Database[['Admission weekday','Admission method']][Database['Hospital stay after index date, days']<=1].value_counts().reset_index()
temp.columns=['Weekday of admission date', 'Admission method', 'Number of patient-admissions discharging in 24 hours after index date']
temp

AdmissionWeekdayResult=pd.merge(AdmissionWeekdayResult, temp,
                           how='left', on=['Weekday of admission date','Admission method'])

AdmissionWeekdayResult

AdmissionWeekdayResult.to_csv("../Patients flow prediction/Codes/Imbalance XGBoost/Admission date weekday summary plots.csv")


# Weekday
Database['Admission_Week_Day']=Database['Index date'].dt.day_name().tolist()
Database['Monday'] = np.where(Database['Admission_Week_Day']=='Monday', 1, 0)
Database['Tuesday'] = np.where(Database['Admission_Week_Day']=='Tuesday', 1, 0)
Database['Wednesday'] = np.where(Database['Admission_Week_Day']=='Wednesday', 1, 0)
Database['Thursday'] = np.where(Database['Admission_Week_Day']=='Thursday', 1, 0)
Database['Friday'] = np.where(Database['Admission_Week_Day']=='Friday', 1, 0)
Database['Saturday'] = np.where(Database['Admission_Week_Day']=='Saturday', 1, 0)
Database['Sunday'] = np.where(Database['Admission_Week_Day']=='Sunday', 1, 0)
# Weekday
cols = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
Database['Index weekday']=Database[cols].idxmax(1)

    
IndexWeekdayResult=pd.DataFrame(Database[['Index weekday','Admission method']].value_counts()).reset_index()
IndexWeekdayResult.columns=['Weekday of index date', 'Admission method', 'Number of patient-admissions']

temp=Database[['Index weekday','Admission method']][Database['Hospital stay after index date, days']<=1].value_counts().reset_index()
temp.columns=['Weekday of index date', 'Admission method', 'Number of patient-admissions discharging in 24 hours after index date']
temp

IndexWeekdayResult=pd.merge(IndexWeekdayResult, temp,
                           how='left', on=['Weekday of index date','Admission method'])
IndexWeekdayResult.to_csv("../Patients flow prediction/Codes/Imbalance XGBoost/Index weekday summary plots.csv")


Database['Admission method']=Database['Admission method'].apply(lambda x: x.replace(' admission',''))
Database["Hospital stay after index date, days"]=Database["Hospital stay after index date, hours"]/24
Database["Previous LOS before index date, days"]=Database["Adm2index, hours"]/24
Database.to_csv("../iord_extract_20220325/Database.csv", index=False)
   
import numpy as np
import seaborn as sns
plt.figure(figsize=(12,5), dpi=300)
medianLOSbyRegion=pd.DataFrame(Database[Database['Admission method']=='Emergency'].groupby("Being in hospital group (short)")["Hospital stay after index date, days"].median().reset_index())
medianLOSbyRegion.sort_values(by=["Hospital stay after index date, days"], inplace=True,ascending=False)
myorder=medianLOSbyRegion["Being in hospital group (short)"].values.tolist()
# Plot the orbital period with horizontal boxes
ax = sns.boxplot(x="Hospital stay after index date, days", y="Being in hospital group (short)", hue="Admission method", 
                 palette={'Emergency': '#0533ff', 'Planned': '#ff2602'}, 
                 data=Database, 
                 order=myorder, showfliers = False)
ax.set_xlabel("Hospital stay after index date, days",weight='bold')
ax.set_ylabel('Being in hospital before index date, days',weight='bold')
SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"
ax.figure.savefig(SavePath+'Hospital stay after index date by previous hospital exposure.svg', dpi=300, bbox_inches='tight',transparent=True)


import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

def boxplot_sorted(df, by, column):
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    meds = df2.median().sort_values()
    ax=df2[meds.index].boxplot(showfliers=False, rot=0,vert=False,return_type='axes')
    ax.set_xlabel('Hospital stay after index date, days',weight='bold', size=10)
    ax.set_ylabel('Being in hospital before index date, days',size=10)
    ax.set_title('Emergency patients',weight='bold',size=10)
    return ax
    
Data=Database[Database['Admission method']=='Emergency admission']
myFig = plt.figure(dpi=100)
ax=boxplot_sorted(Data, by=["Being in hospital group (short)"], column="Hospital stay after index date, days")
ax.figure.savefig('../Emergency11.svg', dpi=300, bbox_inches='tight',transparent=True)
    
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

def boxplot_sorted(df, by, column):
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    meds = df2.median().sort_values()
    ax=df2[meds.index].boxplot(showfliers=False, rot=0,vert=False,return_type='axes')
    ax.set_xlabel('Hospital stay after index date, days',weight='bold', size=10)
    ax.set_ylabel('Being in hospital before index date, days',size=10)
    ax.set_title('Planned patients',weight='bold',size=10)
    return ax
    
Data=Database[Database['Admission method']=='Planned admission']
myFig = plt.figure(dpi=100)
ax=boxplot_sorted(Data, by=["Being in hospital group (short)"], column="Hospital stay after index date, days")
ax.figure.savefig('..Planned11.svg', dpi=300, bbox_inches='tight',transparent=True)



from matplotlib.ticker import StrMethodFormatter
fig = plt.figure(dpi=300)
Database['Adm2index, days']=Database['Adm2index, hours']/24
ax = Database.hist(column='Hospital stay after index date, days', by='Admission method', bins=300, grid=False, figsize=(16,20), layout=(4,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
for i,x in enumerate(ax):
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
    # Set x-axis label
    x.set_xlabel("Hospital stay length after index date, days", labelpad=20, weight='bold', size=12)
    x.set_xlim(left=0.0,right=50)
    # Set y-axis label
    if i == 1:
        x.set_ylabel("Number of admissions", labelpad=50, weight='bold', size=12)
    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    x.tick_params(axis='x', rotation=0)
    x.figure.savefig('../Patient group histogram already being in hospital Admission method.svg', dpi=300, bbox_inches='tight', transparent=True)


from matplotlib.ticker import StrMethodFormatter
fig = plt.figure(dpi=300)
Database['Adm2index, days']=Database['Adm2index, hours']/24
ax = Database.hist(column='Adm2index, days', by='Patient group', bins=300, grid=False, figsize=(16,20), layout=(4,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
for i,x in enumerate(ax):
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
    # Set x-axis label
    x.set_xlabel("Already being in hospital before index date, days", labelpad=20, weight='bold', size=12)
    x.set_xlim(left=0.0,right=50)
    # Set y-axis label
    if i == 1:
        x.set_ylabel("Number of admissions", labelpad=50, weight='bold', size=12)
    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    x.tick_params(axis='x', rotation=0)
    
    x.figure.savefig('../Patient group histogram already being in hospital.svg', dpi=300, bbox_inches='tight', transparent=True)


from matplotlib.ticker import StrMethodFormatter
fig = plt.figure(dpi=300)
ax = Database.hist(column='Hospital stay after index date, hours', by='Patient group', bins=300, grid=False, figsize=(16,20), layout=(4,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
for i,x in enumerate(ax):
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
    # Set x-axis label
    x.set_xlabel("Hospital stay after index date, hours", labelpad=20, weight='bold', size=12)
    x.set_xlim(left=0.0,right=1000)
    # Set y-axis label
    if i == 1:
        x.set_ylabel("Number of admissions", labelpad=50, weight='bold', size=12)
    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    x.tick_params(axis='x', rotation=0)
    x.figure.savefig('../Patient group histogram hospital stay after index date, hours.svg', dpi=300, bbox_inches='tight', transparent=True)


from matplotlib.ticker import StrMethodFormatter
fig = plt.figure(dpi=300)
ax = Database.hist(column='Hospital stay after index date, days', by='Patient group', bins=300, grid=False, figsize=(16,20), layout=(4,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
for i,x in enumerate(ax):
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
    # Set x-axis label
    x.set_xlabel("Hospital stay after index date, days", labelpad=20, weight='bold', size=12)
    x.set_xlim(left=0.0,right=80)
    # Set y-axis label
    if i == 1:
        x.set_ylabel("Number of admissions", labelpad=50, weight='bold', size=12)
    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    x.tick_params(axis='x', rotation=0)
    
    x.figure.savefig('../Patient group histogram hospital stay after index date, days.svg', dpi=300, bbox_inches='tight', transparent=True)


