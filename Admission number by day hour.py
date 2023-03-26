import pandas as pd
import plotly.offline as py
py.init_notebook_mode(connected=True)      
from matplotlib import pyplot as plt
from datetime import date, timedelta
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
        
Path='.../iord_extract_20220325/'
Data=pd.read_csv(Path+'inpt_episode_fixed.csv')
Data['AdmissionDate'] = pd.to_datetime(Data['AdmissionDate'])
start_date = date(2017, 2, 1)
end_date = date(2020, 1, 31)
dateVar='AdmissionDate'
Data=Data[~Data['AdmissionDate'].isnull()]
Data=Data[Data[dateVar]>=pd.to_datetime(start_date)]
Data=Data[Data[dateVar]<=pd.to_datetime(end_date)]
print(Data)

Data['AdmissionMethodCategory'].value_counts()


Data['AdmissionDate']=pd.to_datetime(Data['AdmissionDate'])
# Assign Day and Hour of latest admission date 
Data['Admission_Day'] = Data['AdmissionDate'].dt.day
Data['Admission_Day_Hour'] = Data['AdmissionDate'].dt.round('H').dt.hour
print(Data.columns)

# Admission method
DataPlanned=Data[Data['AdmissionMethodCategory'].isin(['Elective'])]
DataEmergency=Data[Data['AdmissionMethodCategory'].isin(['Emergency'])]
print(DataPlanned)
print(DataEmergency)


SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization"

hourFreq=pd.DataFrame(DataPlanned['Admission_Day_Hour'].value_counts())
hourFreq['Number']=hourFreq.iloc[:,0]
hourFreq.iloc[:,0]=[int(x) for x in hourFreq.index]
hourFreq=hourFreq.sort_values(by=['Admission_Day_Hour'],ascending=True)

plt.figure(figsize=(10,4), dpi=300)
hourFreq['Number'].plot(kind='bar',grid=False)
plt.xticks(rotation=25)
plt.xlabel('Admission date hour time',fontweight='bold')
plt.ylabel('Number of episodes \n(1 Feb 2017- 31 Jan 2020)',fontweight='bold')
plt.show()
plt.savefig(SavePath+'Admission episodes number for planned patients by admission hour time.svg', bbox_inches='tight', transparent=True, dpi=300)


SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/"

hourFreq=pd.DataFrame(DataEmergency['Admission_Day_Hour'].value_counts())
hourFreq['Number']=hourFreq.iloc[:,0]
hourFreq.iloc[:,0]=[int(x) for x in hourFreq.index]
hourFreq=hourFreq.sort_values(by=['Admission_Day_Hour'],ascending=True)

plt.figure(figsize=(10,4), dpi=300)
hourFreq['Number'].plot(kind='bar',grid=False)
plt.xticks(rotation=25)
plt.xlabel('Admission date hour time',fontweight='bold')
plt.ylabel('Number of episodes \n(1 Feb 2017- 31 Jan 2020)',fontweight='bold')
plt.show()
plt.savefig(SavePath+'Admission episodes number for planned patients by admission hour time.svg', bbox_inches='tight', transparent=True, dpi=300)

















