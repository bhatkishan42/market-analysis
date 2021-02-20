
 # Business Analysis with EDA & Statistics


 ### This project is documented using markdown and then uploaded on git hub
 ### Data-set for this project is taken from kaggle datasets


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
# %% [markdown]
# ### Exploring data


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


















