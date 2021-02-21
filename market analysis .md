
 # Business Analysis with EDA & Statistics


 ### This project is documented using markdown and then uploaded on git hub
 ### Data-set for this project is taken from kaggle datasets
 ##### Libraries used
 - Shap
 - Eli5 (Permutation)
 - OneHotEncode (assignning values)
 - Clustermap
 - Linear Regression
 - 


 *  let's load that initial required libraries

```python
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

* lets look into data first we'll read a file and get a gist of how data is presented
### Reading data
```python
data_path='../input/marketing-data/marketing_data.csv'
df_d= pd.read_csv(data_path)
df=pd.read_csv(data_path,parse_dates=['Year_Birth','Dt_Customer'])
```
```python
df.head(5)
```
```python
df.info()
```

```python
type(df['Year_Birth'][1])
# Dt_Customer & Year birth are converted in to time stamp above
```
 - removing garbage values such as '$'  & ',' from the Income column.

```python
df.columns = df.columns.str.strip()
df.Income = df.Income.str.replace('$','')
df.Income = df.Income.str.replace(',','').astype('float')
```

- We can see that garbage values are removed

```python
df.Income.head(6)
```

*  lets check if data is clean and if there are any null values present in the dataset
```python
df.isnull().sum()
```
- Looking at the data we can see tah tthere are 24 null values present and we will fill it with median values of 
 income
```python
df.Income.isnull().sum()
#we need to fill the Nans by median value (Assumption)
# 24 values are null
# check for outliers using boxplot
```

### Exploring data


- From below figure we can see that most data is present between 0-1000
- By taking mean we are avoiding outliers(Imputing)


```python
sns.set_style("white")
plt.figure(figsize=(10,5))
sns.distplot(df['Income'],bins=50,kde=True,hist =True)
plt.title('Density of Income')
plt.ylabel('COUNT')
```

```python
# Income column is clean and we can use it for analysis
df.Income = df.Income.fillna(df.Income.median())
```

- let's find outliers 
-for year of birth will have seperate plot
* Year_Birth <= 1900 (Ignored due to errors)

```python
z = pd.DatetimeIndex(df.Year_Birth).year
sns.boxplot(z, orient='v')
```
```python
plot = df.drop(columns=['ID','Education', 'Marital_Status', 'AcceptedCmp1', 'AcceptedCmp2',
               'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
               'Response','Complain']).select_dtypes(include=np.number)
plot.plot(subplots=True, layout=(5,4), kind='box', figsize=(12,14), patch_artist=True)
plt.subplots_adjust(wspace=0.2);
````
```python
Total = [col for col in df.columns if 'Purchases' in col]
print(Total)
```
 ### Useful data
 * dependents I.e no of kids or teen
 * Total purchases in column (adding all columns contaning purchase)
 * total campagin (adding all columns where there is CMP)
 * customer year
```python
# Adding depends
df['Dependents']= df['Teenhome'] + df['Kidhome']
#Adding total purchase by creating new column and adding them with columns containing purches
Total = [col for col in df.columns if 'Purchases' in col]
df['Total_purchases'] = df[Total].sum(axis =1)
#adding total mnts
total_mnts = [mnt for mnt in df.columns if 'Mnt' in mnt]
df['Total_mnts']= df[total_mnts].sum(axis=1)
#ADDING cmp 
campagain = [camp for camp in df.columns if 'Cmp' in camp]
df['Total_camps'] = df[campagain].sum(axis=1)
#costumer 
df['year_cust'] = pd.DatetimeIndex(df.Dt_Customer).year
print(Total+total_mnts+campagain)
```
```python
df.head(5)
```
```python
df[['ID', 'Dependents', 'year_cust', 'Total_purchases', 'Total_mnts', 'Total_camps']].head()
```
```python
df.drop(columns='ID').select_dtypes(include=np.number).corr(method = 'kendall')
```
```python
type(df.ID[1])
```

* lets find correlation between all data using heatmap to get general idea of how data are dependent on each other

```python
df.head(0)

```python
corr_rex =df.drop(columns = ['ID','Kidhome','Teenhome','NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                             'NumStorePurchases', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                             'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'AcceptedCmp3', 
                             'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 
                             'AcceptedCmp2']).select_dtypes(include=np.number).corr(method = 'kendall')
```
```python
sns.clustermap(corr_rex,cmap='Greys')
```
```python
corrs = df.drop(columns='ID').select_dtypes(include=np.number).corr(method = 'kendall')
```
```python
sns.clustermap(corrs, cmap='Greys')
```
- below we are comparing two diffrent cluster maps as we can see corr_rex has is much more neat and precise and better
- because we removed dependeces which will help us get  clear picture


looking at dark colours we can determine relation darker the colour stronger relation
 * Income is correlated to total purchases, total camps and amounts
 * webvists is related to dependents
 * Amount spent,number of purchases are weaker to dependents (least correlation) whereas purchasing is correlated with dependents
 
* Plot illustrating the effect of high income on spending(limiting income to < 200000 to remove outlier):
```python
print('Plot illustrating the effect of high income on spending:')
g = sns.lmplot(x='Income',y='Total_purchases',data=df[df['Income'] < 200000])
g.set_xticklabels(rotation=30)
```
*Plot illustrating effect of having dependents (kids & teens) on spending & number of deals purchased:
```python
fig, axs = plt.subplots(2)
plt.figure(figsize=(7,7))
sns.boxplot(x='Dependents', y='Total_mnts', data=df,ax = axs[0])
sns.boxplot(x='Dependents', y='NumDealsPurchases', data=df,ax = axs[1])
```

 * number of visits is negatively correlated with web purchases 
* but when compared to vists and deal purchase we found a positive correlation that suggests that deals are effective way to make ppl purchase more

```python
sns.lmplot(x='NumWebVisitsMonth',y='NumWebPurchases',data = df)
sns.lmplot(x='NumWebVisitsMonth',y='NumDealsPurchases',data = df)
```

 ## statistical analysis


# What factors are significantly related to the number of store purchases?

```python
plt.figure(figsize=(8,3))
sns.distplot(df['NumStorePurchases'], kde=True, hist=True, bins=12)
plt.title('NumStorePurchases distribution', size=16)
plt.ylabel('count');
```

-now we'll do some modeling

```python
type(df.Year_Birth[2])
```
```python
df.drop(columns=['ID', 'Dt_Customer','Year_Birth'], inplace=True)
```
```python
# one-hot encoding of categorical features
from sklearn.preprocessing import OneHotEncoder
# get categorical features and review number of unique values
cat=df.select_dtypes(exclude= np.number)
print("Number of unique values per categorical feature:\n", cat.nunique())
# use one hot encoder
enc = OneHotEncoder(sparse=False).fit(cat)
cat_encoded = pd.DataFrame(enc.transform(cat))
cat_encoded.columns = enc.get_feature_names(cat.columns)
# merge with numeric data
num = df.drop(columns =cat.columns)
df2 = pd.concat([cat_encoded, num], axis=1)
df2.head()
```
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# isolate X and y variables, and perform train-test split
X = df2.drop(columns='NumStorePurchases')
y = df2['NumStorePurchases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
preds = model.predict(X_test)

# evaluate model using RMSE
print("Linear regression model RMSE: ", np.sqrt(mean_squared_error(y_test, preds)))
print("Median value of target variable: ", y.median())
```

  * 'TotalPurchases', 'NumCatalogPurchases', 'NumWebPurchases', 'NumDealsPurchases' are significant in     weights compared to rest

```python
import eli5 
from eli5.sklearn import  PermutationImportance
perm =PermutationImportance(model,random_state=1).fit(X_train,y_train)
eli5.show_weights(perm,feature_names = X_test.columns.tolist(),top = 5)


 #### Inference 
* total purchase shows a positive trend compared to others 
* whereas other major factors are in negative trend

so we can conclude that no of store purchases are not due to websites,catlogs,deals .

```python
import shap

# calculate shap values 
ex = shap.Explainer(model, X_train)
shap_values = ex(X_test)
# plot
plt.title('SHAP summary for NumStorePurchases', size=16)
shap.plots.beeswarm(shap_values, max_display=5);
```

-Does US fare significantly better than the Rest of the World in terms of total purchases?

# %% [code]
country = df.groupby(['Country']).Total_purchases.sum().sort_values(ascending=False).plot(kind = 'bar')
plt.title('Country vs Purchases')
plt.ylabel("purchases")
country

# %% [markdown]
# Fish has Omega 3 fatty acids which are good for the brain. Accordingly, do "Married PhD candidates" have a significant relation with amount spent on fish? What other factors are significantly related to amount spent on fish?

# %% [code]
df2['Married_PhD'] = df2['Marital_Status_Married']+df2['Education_PhD']
df2['Married_PhD'] = df2['Married_PhD'].replace({2:'Married-PhD', 1:'Other', 0:'Other'})

# %% [code]
sns.boxplot(x='Married_PhD',y='MntFishProducts',data =df2)

# %% [code]
df2.drop(columns='Married_PhD', inplace=True)

# %% [markdown]
# What other factors are significantly related to amount spent on fish?

# %% [code]
X = df2.drop(columns = 'MntFishProducts')
y= df2['MntFishProducts']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
preds = model.predict(X_test)

# evaluate model using RMSE
print("Linear regression model RMSE: ", np.sqrt(mean_squared_error(y_test, preds)))
print("Median value of target variable: ", y.median())

# %% [markdown]
# * TotalMnt', 'MntWines', 'MntMeatProducts', 'MntGoldProds', 'MntSweetProducts', 'MntFruits' are significat in weights compared to rest.

# %% [code]
cnc = PermutationImportance(model,random_state =1).fit(X_test,y_test)
eli5.show_weights(cnc,feature_names = X_test.columns.tolist(), top=7)

# %% [code]
#shap
ex = shap.Explainer(model,X_train)
shap_values = ex(X_test)
plt.title('SHAP summary for MntFishProducts', size=16)
shap.plots.beeswarm(shap_values, max_display=7);

# %% [markdown]
# ### Infernce from above graph
# * total_mnts is positive for fish products 
# * where as fruits,meat,wines,sweets are in negative co-relation
# 
# #### to summarise
# * the ones who are buying fish are spending less on other above mentioned products

# %% [code]
# convert country codes to correct nomenclature for choropleth plot
# the dataset doesn't provide information about country codes
## ...so I'm taking my best guess about the largest nations that make sense given the codes provided
df['Country_code'] = df['Country'].replace({'SP': 'ESP', 'CA': 'CAN', 'US': 'USA', 'SA': 'ZAF', 'ME': 'MEX'})

# success of campaigns by country code
df_cam = df[['Country_code', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']].melt(
    id_vars='Country_code', var_name='Campaign', value_name='Accepted (%)')
df_cam = pd.DataFrame(df_cam.groupby(['Country_code', 'Campaign'])['Accepted (%)'].mean()*100).reset_index(drop=False)

# rename the campaign variables so they're easier to interpret
df_cam['Campaign'] = df_cam['Campaign'].replace({'AcceptedCmp1': '1',
                                                'AcceptedCmp2': '2',
                                                'AcceptedCmp3': '3',
                                                'AcceptedCmp4': '4',
                                                'AcceptedCmp5': '5',
                                                 'Response': 'Most recent'
                                                })

# choropleth plot
import plotly.express as px

fig = px.choropleth(df_cam, locationmode='ISO-3', color='Accepted (%)', facet_col='Campaign', facet_col_wrap=2,
                    facet_row_spacing=0.05, facet_col_spacing=0.01, width=700,
                    locations='Country_code', projection='natural earth', title='Advertising Campaign Success Rate by Country'
                   )
fig.show()

















