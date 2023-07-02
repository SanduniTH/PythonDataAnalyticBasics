#Step 1 - Importing Data and libraries
#Importing libraries
import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt 
import seaborn as sns #Helpful for EDA plots
plt.style.use('ggplot') #Using a stylesheet
#pd.set_option('max_columns', 200)

#Importing the data 
df = pd.read_excel('../input/arketing-campaign/marketing_campaign.xlsx')

#Step 2 - Understanding the data
df.shape  #The no. of columns and rows of the data
df.head(20) #First 20 rows of the data
df.columns #The name of columns 
df.describe() #Decsribes the columns (such as mean, medium)

#Step 3 - Preparing the data 
#Dropping irrelevant columns 
df = df[[#'ID', 
    'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome','Teenhome', 'Dt_Customer', 'Recency', 
     #'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
     'NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
     'AcceptedCmp2', 'Complain', 
      #'Z_CostContact', 'Z_Revenue', 'Response'
      ]].copy()

df.drop(['Recency'], axis=1) # to drop one column axis 1 - columns not rows

#Check the data type of each column 
df.dtypes

#Changing the data type from object to datetime in column Dt_customer
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# Rename our columns
df = df.rename(columns={'Dt_Customer': 'Date_joined'})

df.duplicated() #Check for duplicates
df.loc[df.duplicated()] #Shows duplicated values 
df.loc[df.duplicated(subset=['Coaster_Name'])].head(5) # Check for duplicates for a certain column
df.query('Coaster_Name == "Crystal Beach Cyclone"') # Checking an example duplicate
df = df.loc[~df.duplicated(subset=['Coaster_Name','Location','Opening_Date'])] \
    .reset_index(drop=True).copy()

#Functions to remove outliers
def remove_outliers_zscore(data, threshold=3):
    z_scores = (data - np.mean(data)) / np.std(data)
    filtered_data = data[abs(z_scores) < threshold]
    return filtered_data

def remove_outliers_iqr(data, threshold=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data

# Remove outliers using the z-score method
df[column_name] = remove_outliers_zscore(df[column_name])

# Remove outliers using the IQR method
df[column_name] = remove_outliers_iqr(df[column_name])

# Step 3: Feature Understanding (Univariate analysis)
# Plotting feature distributions (Histogram, Kde, Boxplots)

df['NumWebVisitsMonth'].value_counts() #counts the frequncy
#Plot a bar chart for frquency of web vists 
ax = df['NumWebVisitsMonth'].value_counts() \
    .head(10) \
    .plot(kind='bar', title='Frequncy of web visits')
ax.set_xlabel('Number of visits')
ax.set_ylabel('Frequncy')

#Plot a histogram for frquency of number of purchases
ax = df['NumWebPurchases'].plot(kind='hist',
                          bins=20,
                          title='Web Purchases')
ax.set_xlabel('Number of purchases')

#Plot KDE for frquency of number of purchases
ax = df['NumWebPurchases'].plot(kind='kde',
                          title='Web Purchases')
ax.set_xlabel('Number of purchases')

#Step 4: Feature Relationships
#Scatterplot, Heatmap Correlation, Pairplot, Groupby comparisons

df.plot(kind='scatter',
        x='NumWebVisitsMonth',
        y='Income',
        title='Income vs. NumWebVisitsMonth')
plt.show()

ax = sns.scatterplot(x='NumWebVisitsMonth',
                y='Income',
                hue='Year_Birth',
                data=df)
ax.set_title('Income vs. Num Store Purchases')
plt.show()

sns.pairplot(df,
             vars=['Income','NumStorePurchases', 
                   'NumWebPurchases','NumWebVisitsMonth'],
            hue='Year_Birth')
plt.show()

df_corr = df[['Income','NumStorePurchases', 
            'NumWebPurchases','NumWebVisitsMonth']].dropna().corr()
df_corr
sns.heatmap(df_corr, annot=True)
