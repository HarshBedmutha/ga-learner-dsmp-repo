# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
print(data.shape)
# plt.hist(data['Rating'])
# plt.show()
data = data[data['Rating']< 6]
plt.hist(data['Rating'],bins=20)
# plt.show()

#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())


missing_data = pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])
print(missing_data)

data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())

missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------

#Code starts here
sns.catplot(x='Category',y='Rating',data=data,kind='box',height=10)
plt.xticks(rotation=90)
plt.title('Rating vs Category [Boxplot]')
plt.show()

#Code ends here



# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].value_counts())
data['Installs'] = data['Installs'].str.replace(',','')
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].astype('int')

le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])

sns.regplot(x='Installs',y='Rating',data=data)
plt.title('Rating vs Installs [Regplot]')
plt.show()
#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())
data['Price'] = data['Price'].str.replace('$','')
data['Price'] = data['Price'].astype(float)
sns.regplot(x='Price',y='Rating',data=data)
plt.title('Rating vs Price [Regplot] ')
plt.show()
print(data.tail())

#Code ends here


# --------------
#Code starts here
# print(data['Genres'].unique())
# print(data['Genres'].nunique())

data['Genres'] = data['Genres'].str.split(';',n=1,expand=True)
print(data['Genres'].nunique())
gr = data.loc[:,['Genres','Rating']]
gr_mean = gr.groupby('Genres',as_index=False).mean()
# print(gr_mean)
print(gr_mean.describe())
gr_mean = gr_mean.sort_values(by='Rating')
print(gr_mean.iloc[0],gr_mean.iloc[-1])
#Code ends here


# --------------

#Code starts here
# print(data['Last Updated'])
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
print(data.info())
max_date = max(data['Last Updated'])
print(max_date)
data['Last Updated Days'] = (max_date-data['Last Updated']).dt.days
print(['Last Updated Days'])
sns.regplot(x='Last Updated Days',y='Rating',data=data)
plt.title('Rating vs Last Updated [RegPlot]')
plt.show()
#Code ends here


