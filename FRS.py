#!/usr/bin/env python
# coding: utf-8

# # Examine relationship between Disability and poverty.

# In[ ]:


# Importing libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_rows', None)


# In[ ]:


# Loading dataset.
df = pd.read_csv('HBAI-dataset-final-1.csv')
df.head()


# In[91]:


df.shape


# In[92]:


df.info()


# In[93]:


df.describe()


# In[94]:


df.head()


# In[95]:


df['Unnamed: 0'].unique()


# In[96]:


# Replacing column names.
df.columns = [
    "Financial year",
    "Region",
    "Median net household income(BHC)",
    "Median net household income(AHC)",
    "Disability mix within the family",
    "Economic status",
    "Gender",
    "Not disabled",
    "Extra1",
    "Extra2",
    "Disabled",
    "Extra3",
    "Extra4",
    "Extra5"
]
df = df.drop(columns=["Extra1", "Extra2", "Extra3", "Extra4", "Extra5"], \
             errors="ignore")

# Dropping the first row.
df = df.drop(index=0)

# Resetting the index.
df = df.reset_index(drop=True)


# In[97]:


df.head(50)


# In[98]:


df['Region'].unique()


# In[99]:


df['Median net household income(BHC)'].unique()


# In[100]:


df['Median net household income(AHC)'].unique()


# In[101]:


df['Disability mix within the family'].unique()


# In[102]:


df['Economic status'].unique()


# In[103]:


df['Gender'].unique()


# In[104]:


df.columns.to_list()[0:-3]


# In[105]:


# Filling missing values in the 'Region', 'Gender', 'Employment Status',
# 'Disability Status', 'Total Net Income'.
for col in enumerate(df.columns.to_list()[0:-3]):
    df[col[1]] = df[col[1]].ffill()

df.tail()


# In[106]:


df.info()


# In[107]:


df.describe()


# In[108]:


# Convert 'Not Disabled' and 'Disabled' column to numeric type.
df['Not disabled'] = pd.to_numeric(df['Not disabled'], errors='coerce')
df['Disabled'] = pd.to_numeric(df['Disabled'], errors='coerce')
df.head()


# In[109]:


df.shape


# In[110]:


# Checking for duplicates.
duplicates = df[df.duplicated()]
duplicates


# In[111]:


df.isnull().sum()


# In[112]:


df['Financial year'].unique().tolist()


# In[113]:


# Cleaning 'Financial Year' column.
df['Financial year'] = df['Financial year'].map({
    '2002-03 (cpi, r)' : '2002-03',
    '2003-04 (cpi, r)' : '2003-04',
    '2004-05 (cpi, r)' : '2004-05',
    '2005-06 (cpi, r)' : '2005-06',
    '2006-07 (cpi, r)' : '2006-07',
    '2007-08 (cpi, r)' : '2007-08',
    '2008-09 (cpi, r)' : '2008-09',
    '2009-10 (cpi, r)' : '2009-10',
    '2010-11 (cpi, r)' : '2010-11',
    '2011-12 (cpi, r)' : '2011-12',
    '2012-13 (cpi, r)' : '2012-13',
    '2013-14 (cpi, r)' : '2013-14',
    '2014-15 (r)' : '2014-15',
    '2015-16 (r)' : '2015-16',
    '2016-17 (r)' : '2016-17',
    '2017-18 (r)' : '2017-18',
    '2018-19 (r)' : '2018-19',
    '2019-20' : '2019-20',
    '2020-21 (covid2021)' : '2020-21',
    '2021-22 (covid2122, div)' : '2021-22',
    '2022-23 (colp, covid2223, div)' : '2022-23'
})

df.head()


# In[114]:


# Removing data in '2020-21' due to data quality concerns.
df =  df[df['Financial year'] != '2020-21']


# In[115]:


df.isnull().sum()


# In[116]:


# Filling missing values with zero because there are no data.
df = df.fillna(0)


# In[117]:


# Creating new column name "Total population".
df['Total population'] = df.apply(
    lambda x: x['Not disabled'] + x['Disabled'],axis=1
)

df.head()


# In[118]:


# Checking unique values in the 'Region' column.
df['Region'].unique()


# In[119]:


# Cleaning 'Region' column data.
df['Region'] = df['Region'].map({
    'North East (E12000001)' : 'North East',
    'North West (E12000002)' : 'North West',
    'Yorkshire and The Humber (E12000003)' : 'Yorkshire and The Humber',
    'East Midlands (E12000004)' : 'East Midlands',
    'West Midlands (E12000005)' : 'West Midlands',
    'East (E12000006)' : 'East',
    'London (E12000007) (Inner and Outer split not available before 1997/98)' \
        : 'London',
    'Inner London (E12000007)' : 'Inner London',
    'Outer London (E12000007)' : 'Outer London',
    'South East (E12000008)' : 'South East', 
    'South West (E12000009)' : 'South West',
    'Wales (W92000004)' : 'Wales',
    'Scotland (S92000003)' : 'Scotland',
    'Northern Ireland (N92000002)' : 'Northern Ireland'
})


# In[120]:


# Removing 'London' data while keeping 'London Split regions'.
df = df[df['Region'] != 'London']


# In[121]:


df.head()


# In[122]:


# Checking unique values in the 'Median net household income(BHC)' column.
df['Median net household income(BHC)'].unique().tolist()


# In[123]:


# Cleaning 'Median net household income(BHC)' column data.
df = df.copy()

df['Median net household income(BHC)'] = df[
    'Median net household income(BHC)'
].map({
    'Not in low income (at or above threshold)' : 'Not in low income',
    'In low income (below threshold)' : 'In low income'
})

df.head()


# In[124]:


# Checking unique values in the 'Median net household income(AHC)' column.
df['Median net household income(AHC)'].unique().tolist()


# In[125]:


# Cleaning 'Median net household income(AHC)' column data.
df['Median net household income(AHC)'] = df[
    'Median net household income(AHC)'
].map({
    'Not in low income (at or above threshold)' : 'Not in low income',
    'In low income (below threshold)' : 'In low income'
})

df.head()


# In[126]:


# Checking unique values in the 'Disability mix within the family' column.
df['Disability mix within the family'].unique().tolist()


# In[127]:


# Cleaning 'Employment status' column data.
df['Economic status'].unique()


# In[128]:


# Exploring Population Count Over Time by Disability Status.
disability_over_time = df.groupby('Financial year')[['Not disabled', 'Disabled']]\
    .sum()
disability_over_time


# In[129]:


# Visualising Population Count Over Time by Disability Status.
plt.figure(figsize=(10, 5))
plt.plot(
    disability_over_time.index, 
    disability_over_time['Disabled'], 
    label='Disabled', 
    marker='o'
)
plt.plot(
    disability_over_time.index, 
    disability_over_time['Not disabled'], 
    label='Not Disabled', 
    marker='o'
)
plt.title('Population Count by Disability Status Over Years')
plt.xlabel('Financial Year')
plt.ylabel('Population Count')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 1. The disabled population has shown a consistent upward trend from approximately 10 million in 2002-03 to about 15 million in 2022-23.
# 2. Most notable increase observed from 2013-14 onwards.
# 3. Growth rate appears to have accelerated in recent years (2018-2023).
# 4. The non-disabled population maintained relatively stable numbers around 47-50 million.
# 5. Slight fluctuations but no dramatic changes.
# 6. Minor plateau observed between 2005-06 and 2014-2018.
# 7. Consistent disparity between disabled and non-disabled populations.
# 8. Non-disabled population approximately 3-4 times larger than disabled population.
# 9. Gap has slightly decreased in recent years due to faster growth in disabled population.

# In[130]:


# Population Count Over Time in regions.
df_region = df.groupby(
    ['Financial year', 'Region']
)[['Not disabled', 'Disabled']].sum()
df_region.head(20)


# In[131]:


# Visualising Disabled population count by Financial year and Region.
# Pivot data for the heatmap.
disabled_heatmap = df_region['Disabled'].unstack(level='Region')
not_disabled_heatmap = df_region['Not disabled'].unstack(level='Region')

# Plotting heatmap for 'Disabled Population Count by Year and Region'.
plt.figure(figsize=(10, 5))
sns.heatmap(disabled_heatmap, cmap="Blues", annot=False)
plt.title('Disabled Population Count by Year and Region')
plt.xlabel('Region')
plt.ylabel('Financial Year')
plt.show()

# Plotting heatmap for 'Not Disabled Population Count by Year and Region'.
plt.figure(figsize=(10, 5))
sns.heatmap(not_disabled_heatmap, cmap="Oranges", annot=False)
plt.title('Not Disabled Population Count by Year and Region')
plt.xlabel('Region')
plt.ylabel('Financial Year')
plt.show()


# In[132]:


# Filtering the data for 2022-23.
df_2223 = df_region.xs('2022-23', level='Financial year')

# Plotting a grouped bar chart.
plt.figure(figsize=(10, 5))
df_2223.plot(kind='barh', color=['#66B2FF', '#FF9999'])
plt.title('Population Count by Disability Status in Each Region (2022-23)')
plt.xlabel('Population Count')
plt.ylabel('Region')
plt.legend(['Not Disabled', 'Disabled'])
plt.tight_layout()
plt.show()


# ### Findings
# - South East: Largest overall population with highest disability rates.
# - North West: Higher disability rates relative to population.
# - Yorkshire and The Humber: Showing increasing trends on heatmap.
# - Inner London: Lower disability rates.
# - Northern Ireland: Consistently lower disability and population .
# - North East: Moderate to low disability populations.
# 
# Time Period: 2015-2023
# 
# - General increase in disability rates across most regions.
# - Most significant growth in South East and North West.
# - Stable patterns in Northern Ireland and Inner London.
# 
# 
# 
# Urban-Rural
# 
# - Urban areas: Lower disability rates relative to population.
# - Rural areas: Higher proportion of disability.
# - London shows unique pattern with lower disability rates despite large population.

# In[133]:


# Population Count Over Time in Median net household income(BHC) Before 
# Housing Cost.
income_bhc = df.groupby(
    ['Financial year', 'Median net household income(BHC)']
)[['Not disabled', 'Disabled']].sum().unstack(level=1)
income_bhc


# In[134]:


income_bhc.index


# In[135]:


# Plotting stacked bar charts for Disabled and Not Disabled populations 
# by income level Before Housing Cost over time.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# Plotting for Disabled population.
income_bhc['Disabled'].plot(
    kind='bar', 
    stacked=True, 
    ax=ax1, 
    color=['#FF9999', '#FF6666']
)
ax1.set_title('Income Distribution for Disabled Population BHC')
ax1.set_xlabel('Financial Year')
ax1.set_ylabel('Population Count')
ax1.legend(['In Low Income', 'Not in Low Income'], title='Income Level')

# Plotting for Not Disabled population.
income_bhc['Not disabled'].plot(
    kind='bar', 
    stacked=True, 
    ax=ax2, 
    color=['#9999FF', '#6666FF']
)
ax2.set_title('Income Distribution for Not Disabled Population BHC')
ax2.set_xlabel('Financial Year')
ax2.legend(['In Low Income', 'Not in Low Income'], title='Income Level')

# Rotating x-axis labels for readability.
for ax in [ax1, ax2]:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.suptitle(
    'Income Distribution by Disability Status Over Time Before Housing Cost'
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[136]:


# Plotting line charts for each income level Before Housing Cost in both 
# Disabled and Not Disabled populations.
plt.figure(figsize=(14, 8))

# Plotting for Disabled population.
plt.plot(
   income_bhc.index, 
   income_bhc['Disabled', 'In low income'], 
   label='Disabled - In Low Income', 
   color='salmon', 
   linestyle='--', 
   marker='o'
)
plt.plot(
   income_bhc.index, 
   income_bhc['Disabled', 'Not in low income'], 
   label='Disabled - Not in Low Income', 
   color='red', 
   marker='o'
)

# Plotting for Not Disabled population.
plt.plot(
    income_bhc.index, 
    income_bhc['Not disabled', 'In low income'], 
    label='Not Disabled - In Low Income', 
    color='skyblue', 
    linestyle='--', 
    marker='o'
)
plt.plot(
   income_bhc.index, 
   income_bhc['Not disabled', 'Not in low income'],
   label='Not Disabled - Not in Low Income', 
   color='blue', 
   marker='o'
)

# Adding labels, title, and legend.
plt.xlabel('Financial Year')
plt.ylabel('Population Count')
plt.title(
   'Income Distribution by Disability Status Over Time Before Housing Cost'
)
plt.legend(title='Group & Income Level')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# ### Findings
# 1. The population of disabled individuals "Not in Low Income" has consistently been larger than those "In Low Income".
# 2. Both income groups show a slight upward trend over time, indicating an increase in the disabled population overall. However, the "Not in Low Income" category grows at a faster rate than the "In Low Income" category.
# 3. The majority of the non-disabled population is in the "Not in Low Income" group.
# 4. The line chart shows that the "Not Disabled - Not in Low Income" group has the highest population count throughout the years, with a gradual upward trend.
# 5. The "Disabled - Not in Low Income" group, while lower in count than non-disabled individuals, also shows a slight increase over time.
# 6. Disabled individuals are consistently more likely to be in low income compared to non-disabled individuals. The "Disabled - In Low Income" group has the smallest population count but still exhibits a slight upward trend.
# 7. The gap between "Not in Low Income" and "In Low Income" is more pronounced for non-disabled individuals, suggesting that disability status has a strong association with income level and economic challenges.

# In[137]:


# Calculating total population by year.
income_bhc['Total'] = income_bhc['Not disabled']['In low income'] \
    + income_bhc['Not disabled']['Not in low income'] \
        + income_bhc['Disabled']['In low income'] \
            + income_bhc['Disabled']['Not in low income']

# Calculating percentages.
income_bhc['% Not Disabled In Low Income'] = (
    income_bhc['Not disabled']['In low income'] / income_bhc['Total']
) * 100
income_bhc['% Not Disabled Not in Low Income'] = (
    income_bhc['Not disabled']['Not in low income'] / income_bhc['Total']
) * 100

income_bhc['% Disabled In Low Income'] = (
    income_bhc['Disabled']['In low income'] / income_bhc['Total']
) * 100
income_bhc['% Disabled Not in Low Income'] = (
    income_bhc['Disabled']['Not in low income'] / income_bhc['Total']
) * 100

# Displaying the result.
income_bhc[[
    '% Not Disabled In Low Income', 
    '% Not Disabled Not in Low Income', 
    '% Disabled In Low Income', 
    '% Disabled Not in Low Income'
]]


# In[138]:


# Preparing data for the stacked area chart.
stacked_data_disabled = income_bhc[[
    '% Disabled In Low Income', 
    '% Disabled Not in Low Income'
]]
stacked_data_not_disabled = income_bhc[[
    '% Not Disabled In Low Income', 
    '% Not Disabled Not in Low Income'
]]

# Plotting the stacked area chart.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plotting for Disabled population.
ax1.stackplot(
    income_bhc.index,
    stacked_data_disabled['% Disabled In Low Income'],
    stacked_data_disabled['% Disabled Not in Low Income'],
    labels=['In Low Income', 'Not in Low Income'],
    colors=['lightcoral', 'red'],
    alpha=0.8
)
ax1.set_title(
    'Income Distribution for Disabled Population (Percentage) BHC', 
    fontsize=14
)
ax1.set_xlabel('Financial Year', fontsize=12)
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.tick_params(axis='x',rotation=45)
ax1.legend(loc='upper left')

# Plotting for Not Disabled population.
ax2.stackplot(
    income_bhc.index,
    stacked_data_not_disabled['% Not Disabled In Low Income'],
    stacked_data_not_disabled['% Not Disabled Not in Low Income'],
    labels=['In Low Income', 'Not in Low Income'],
    colors=['lightblue', 'blue'],
    alpha=0.8
)
ax2.set_title(
    'Income Distribution for Not Disabled Population (Percentage) BHC', 
    fontsize=14
)
ax2.set_xlabel('Financial Year', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(loc='upper left')

# Displaying the plot.
plt.suptitle(
    'Income Distribution by Disability Status Over Time (Percentage) Before Housing Cost', 
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# ### Findings
# 1. Non-Disabled Population
# 
#     Approximately 60-70% are not in low income
#     Only 10-15% fall into low income category before housing cost
#     More stable distribution over time.
# 
# 
# 2. Disabled Population
# 
#     Only 10-20% are not in low income
#     Significantly (~5%)higher proportion in low income before housing cost
#     Shows more variation over time.

# In[139]:


df.head()


# In[140]:


# Population Count Over Time in Median net household income(AHC) After 
# Housing Cost.
income_ahc = df.groupby(
    ['Financial year', 'Median net household income(AHC)']
)[['Not disabled', 'Disabled']].sum().unstack(level=1)
income_ahc


# In[141]:


# Plotting stacked bar charts for Disabled and Not Disabled populations 
# by income level Aafter Housing Cost over time.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# Plotting for Disabled population.
income_ahc['Disabled'].plot(
    kind='bar', 
    stacked=True, 
    ax=ax1, 
    color=['#FF9999', '#FF6666']
)
ax1.set_title('Income Distribution for Disabled Population AHC')
ax1.set_xlabel('Financial Year')
ax1.set_ylabel('Population Count')
ax1.legend(['In Low Income', 'Not in Low Income'], title='Income Level')

# Plotting for Not Disabled population.
income_ahc['Not disabled'].plot(
    kind='bar', 
    stacked=True, 
    ax=ax2, 
    color=['#9999FF', '#6666FF']
)
ax2.set_title('Income Distribution for Not Disabled Population AHC')
ax2.set_xlabel('Financial Year')
ax2.legend(['In Low Income', 'Not in Low Income'], title='Income Level')

# Rotating x-axis labels for readability.
for ax in [ax1, ax2]:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.suptitle(
    'Income Distribution by Disability Status Over Time After Housing Cost'
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[ ]:





# In[142]:


# Plotting line charts for each income level after housing cost in both 
# Disabled and Not Disabled populations.
plt.figure(figsize=(14, 8))

# Plotting for Disabled population.
plt.plot(
   income_ahc.index, 
   income_ahc['Disabled', 'In low income'], 
   label='Disabled - In Low Income', 
   color='salmon', 
   linestyle='--', 
   marker='o'
)
plt.plot(
   income_ahc.index, 
   income_ahc['Disabled', 'Not in low income'], 
   label='Disabled - Not in Low Income', 
   color='red', 
   marker='o'
)

# Plotting for Not Disabled population.
plt.plot(
    income_ahc.index, 
    income_ahc['Not disabled', 'In low income'], 
    label='Not Disabled - In Low Income', 
    color='skyblue', 
    linestyle='--', 
    marker='o'
)
plt.plot(
   income_ahc.index, 
   income_ahc['Not disabled', 'Not in low income'],
   label='Not Disabled - Not in Low Income', 
   color='blue', 
   marker='o'
)

# Adding labels, title, and legend.
plt.xlabel('Financial Year')
plt.ylabel('Population Count')
plt.title(
   'Income Distribution by Disability Status Over Time After Housing Cost'
)
plt.legend(title='Group & Income Level')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# ### Findings
# 1. Disabled Population:
#     While there is some fluctuation in the "in low income" and "not in low income" categories for the disabled population, the gap has not significantly closed over the years.
#     An increasing trend in the "not in low income" category for disabled people is visible in the more recent years, though the overall percentage remains considerably lower compared to the non-disabled population.
# 2. Non-Disabled Population:
#     The majority of the non-disabled population is "not in low income," as indicated by the dominant blue lines in the chart.
#     There is a slight increase in the proportion of non-disabled people in the "not in low income" category, indicating some improvement over time.

# In[143]:


# Calculating total population by year.
income_ahc['Total'] = income_ahc['Not disabled']['In low income'] \
    + income_ahc['Not disabled']['Not in low income'] \
        + income_ahc['Disabled']['In low income'] \
            + income_ahc['Disabled']['Not in low income']

# Calculating percentages.
income_ahc['% Not Disabled In Low Income'] = (
    income_ahc['Not disabled']['In low income'] / income_ahc['Total']
) * 100
income_ahc['% Not Disabled Not in Low Income'] = (
    income_ahc['Not disabled']['Not in low income'] / income_ahc['Total']
) * 100

income_ahc['% Disabled In Low Income'] = (
    income_ahc['Disabled']['In low income'] / income_ahc['Total']
) * 100
income_ahc['% Disabled Not in Low Income'] = (
    income_ahc['Disabled']['Not in low income'] / income_ahc['Total']
) * 100

# Displaying the result.
income_ahc[[
    '% Not Disabled In Low Income', 
    '% Not Disabled Not in Low Income', 
    '% Disabled In Low Income', 
    '% Disabled Not in Low Income'
]]


# In[144]:


# Visualising Income Distribution by Disability Status Over Time 
# (Percentage) After Housing Cost.
# Preparing data for the stacked area chart.
stacked_data_disabled_ahc = income_ahc[[
    '% Disabled In Low Income', 
    '% Disabled Not in Low Income'
]]
stacked_data_not_disabled_ahc = income_ahc[[
    '% Not Disabled In Low Income', 
    '% Not Disabled Not in Low Income'
]]

# Plotting the stacked area chart.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plotting for Disabled population.
ax1.stackplot(
    income_bhc.index,
    stacked_data_disabled_ahc['% Disabled In Low Income'],
    stacked_data_disabled_ahc['% Disabled Not in Low Income'],
    labels=['In Low Income', 'Not in Low Income'],
    colors=['lightcoral', 'red'],
    alpha=0.8
)
ax1.set_title(
    'Income Distribution for Disabled Population (Percentage) AHC', 
    fontsize=14
)
ax1.set_xlabel('Financial Year', fontsize=12)
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.tick_params(axis='x',rotation=45)
ax1.legend(loc='upper left')

# Plotting for Not Disabled population.
ax2.stackplot(
    income_ahc.index,
    stacked_data_not_disabled_ahc['% Not Disabled In Low Income'],
    stacked_data_not_disabled_ahc['% Not Disabled Not in Low Income'],
    labels=['In Low Income', 'Not in Low Income'],
    colors=['lightblue', 'blue'],
    alpha=0.8
)
ax2.set_title(
    'Income Distribution for Not Disabled Population (Percentage) AHC', 
    fontsize=14
)
ax2.set_xlabel('Financial Year', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(loc='upper left')

# Displaying the plot.
plt.suptitle(
    'Income Distribution by Disability Status Over Time (Percentage) After Housing Cost', 
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# ### Findings
# 1. High Low-Income Rate Among Disabled Population:
#     Throughout the observed years, the disabled population consistently has a higher percentage of people in the "in low income" category compared to the non-disabled population.
#     This disparity has grown over time, with the most recent data showing an increasing percentage of disabled people in low income, reaching 5.6% in 2022-23.
# 2. Trends in the Non-Disabled Population:
#     There is a slight upward trend in the percentage of disabled individuals in the "in low income" category over the years, from around 4% in the early 2000s to approximately 5.6% in 2022-23.
#     The non-disabled population has a majority in the "not in low income" category, with percentages consistently above 60%.
#     However, there is a slight downward trend in the percentage of non-disabled individuals in the "not in low income" category over the years, from around 65% in the early 2000s to approximately 60% in 2022-23.

# In[145]:


df.head()


# In[146]:


# Population Count Over Time in Disability mix within the family.
disability_mix_in_family = df.groupby([
    'Financial year', 
    'Disability mix within the family'
])[[
    'Total population'
]].sum().unstack(level=1).stack(level=1, future_stack=True).reset_index()
disability_mix_in_family


# In[147]:


disability_mix_in_family.columns


# In[148]:


# Visualising Population Count by Disability Mix within the Family Over 
# Time (Disabled).
# Plotting.
fig, ax = plt.subplots(figsize=(12, 8))
categories = disability_mix_in_family[
    'Disability mix within the family'
].unique()
bottom = None  # for stacking bars.

for category in categories:
    category_data = disability_mix_in_family[
        disability_mix_in_family[
            'Disability mix within the family'
        ] == category
    ]
    if bottom is None:
        bottom = category_data['Total population'].values
        ax.bar(
            category_data['Financial year'], 
            category_data['Total population'], 
            label=category
        )
    else:
        ax.bar(
            category_data['Financial year'], 
            category_data['Total population'], 
            bottom=bottom, 
            label=category
        )
        bottom += category_data['Total population'].values

# Formatting.
ax.set_title(
    'Population by Disability Mix within the Family Over Time'
)
ax.set_xlabel('Financial Year')
ax.set_ylabel('Population Count')
ax.legend(title="Disability Mix within the Family")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[149]:


# Population Count Over Time by Disability Mix within the Family Using
# line chart.
# Creating the plot.
plt.figure(figsize=(14, 7))
plt.title(
    "Population Over Time by Disability Mix within the Family", 
    fontsize=16
)

# Looping through each unique 'Disability mix within the family' to 
# create a line for each.
for disability_mix in disability_mix_in_family[
    'Disability mix within the family'
].unique():
    subset = disability_mix_in_family[
        disability_mix_in_family[
            'Disability mix within the family'
        ] == disability_mix
    ]
    plt.plot(
        subset['Financial year'], 
        subset['Total population'], 
        label=disability_mix
    )

# Adding labels and legend.
plt.xlabel("Financial Year")
plt.ylabel("Total Population")
plt.xticks(rotation=45)
plt.legend(
    title="Disability Mix within the Family", 
    bbox_to_anchor=(1.05, 1), 
    loc='upper left')
plt.tight_layout()
plt.show()


# ### Findings
# 
# 
# - The majority of the population lives in families where no one is disabled (around 40-45 million).
# - The total disabled population has shown a gradual increase from about 15 million to 20 million over the period.
# 
# 
# Families with Disabled Adults Only:
# - Shows a steady upward trend from about ~15 million in 2002-03 to ~20 million in 2022-23.
# - Represents roughly 75-80% of all families with disabilities.
# 
# 
# Mixed Disability Families (Adults and Children):
# - Shows slight increase over time.
# 
# 
# Families with Disabled Children Only:
# - Relatively stable but showing slight growth.

# In[150]:


df.head()


# In[151]:


# Population Count Over Time in Economic status the family.
economic_status = df.groupby([
    'Financial year', 
    'Economic status'
])[[
    'Not disabled', 'Disabled'
]].sum().unstack(level=1).stack(level=1, future_stack=True).reset_index()


# In[152]:


economic_status['Economic status'].unique().tolist()


# In[153]:


# Mapping the Economic status column to broader categories.
status_mapping = {
    'Full-time Employee': 'Employee',
    'Full-time Self-Employed': 'Employee',
    'Part-time Employee': 'Employee',
    'Part-time Self-Employed': 'Employee',
    'Looking after family/home': 'Other',
    'Not applicable (individual is not an adult)': 'Other',
    'Other Inactive': 'Other',
    'Permanently sick/disabled': 'Unemployed',
    'Retired': 'Other',
    'Student': 'Other',
    'Temporarily sick/injured': 'Temporarily sick/injured',
    'Unemployed': 'Unemployed'
}

# Applying the mapping to create a new column.
economic_status['Economic status'] = economic_status[
    'Economic status'
].map(status_mapping)

# Filtering out rows categorized as 'Other'.
filtered_data = economic_status[economic_status['Economic status'].isin([
    'Employee', 'Unemployed', 'Permanently sick/disabled'
])]

# Grouping by year and the new Economic status categories, summing the 
# populations.
filtered_data = filtered_data.groupby([
    'Financial year', 
    'Economic status'
])[['Not disabled', 'Disabled']].sum().reset_index()

filtered_data.head()


# In[154]:


# Visualising Population Count Over Time by Economic Status and 
# Not Disabled.
plt.figure(figsize=(12, 8))

# Plotting for each Economic status with 'Not disabled'.
for status in filtered_data['Economic status'].unique():
    data_subset = filtered_data[filtered_data['Economic status'] == status]
    plt.plot(
        data_subset['Financial year'], 
        data_subset['Not disabled'], 
        marker='o', 
        label=f"{status}"
    )

# Adding titles and labels.
plt.title("Population Count Over Time by Economic Status and Not Disabled")
plt.xlabel("Financial Year")
plt.ylabel("Population Count")
plt.xticks(rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
plt.tight_layout()
plt.show()


# In[155]:


# Visualising Population Count Over Time by Economic Status and Disabled.
plt.figure(figsize=(12, 8))

# Plotting for each Economic status with 'Disabled'.
for status in filtered_data['Economic status'].unique():
    data_subset = filtered_data[filtered_data['Economic status'] == status]
    plt.plot(
        data_subset['Financial year'], 
        data_subset['Disabled'], 
        marker='o', 
        label=f"{status}"
    )

# Adding titles and labels.
plt.title("Population Count Over Time by Economic Status and Disabled")
plt.xlabel("Financial Year")
plt.ylabel("Population Count")
plt.xticks(rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
plt.tight_layout()
plt.show()


# In[156]:


# Visualising Population Count Over Time by Economic Status and 
# Disability.
# Getting the unique economic statuses and determining subplot grid size.
economic_statuses = filtered_data['Economic status'].unique()
n_statuses = len(economic_statuses)
n_cols = 2
n_rows = (n_statuses + n_cols - 1) // n_cols

fig, axes = plt.subplots(
    n_rows, 
    n_cols, 
    figsize=(16, 12), 
    sharex=True, 
    sharey=True
)
fig.suptitle(
    "Population Count Over Time by Economic Status and Disability", 
    fontsize=16
)

# Flattening axes array for easy indexing, in case of empty cells in grid.
axes = axes.flatten()

# Plotting for each economic status in its own subplot.
for i, status in enumerate(economic_statuses):
    ax = axes[i]
    data_subset = filtered_data[filtered_data['Economic status'] == status]
    
    # Plotting 'Not disabled' population.
    ax.plot(
        data_subset['Financial year'], 
        data_subset['Not disabled'], 
        marker='o', 
        label="Not disabled"
    )
    
    # Plotting 'Disabled' population.
    ax.plot(
        data_subset['Financial year'], 
        data_subset['Disabled'], 
        marker='x', 
        linestyle='--', 
        label="Disabled"
    )
    ax.set_title(status)
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Population Count")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[157]:


df.head()


# In[158]:


# Turning 'Not disabled' and 'Disabled' columns to rows.
df = pd.melt(
    df, 
    id_vars=[
        'Financial year', 
        'Region', 
        'Median net household income(BHC)',
        'Median net household income(AHC)',
        'Disability mix within the family',
        'Economic status',
        'Gender'
    ],
    value_vars=['Not disabled', 'Disabled'],
    var_name='Disability status',                  
    value_name='Population count'
)

df.head()


# In[159]:


df1 = df.copy()


# In[160]:


df1['Disability mix within the family'].unique()


# In[161]:


# Extracting the starting year as an integer for simpler analysis.
df1['Financial year'] = df1['Financial year'].str.split('-').str[0].astype(int)


# In[162]:


# Converting categorical columns to numeric codes for correlation 
# analysis.
categorical_cols = ['Region', 'Gender', 'Disability status']
df1[categorical_cols] = df1[
    categorical_cols
].apply(lambda x: x.astype('category').cat.codes)


# In[163]:


df1['Economic status'].unique().tolist()


# In[164]:


# Defining the mapping of Economic status to risk points.
economic_status_risk_mapping = {
    'Full-time Employee': 0,
    'Part-time Employee': 0.5,
    'Full-time Self-Employed': 0,
    'Part-time Self-Employed': 0.5,
    'Unemployed': 1,
    'Retired': 0,
    'Student': 0.5,
    'Looking after family/home': 0.75,
    'Permanently sick/disabled': 1,
    'Temporarily sick/injured': 0.75,
    'Other Inactive': 0,
    'Not applicable (individual is not an adult)': 0
}

# Applying the mapping to create a new column for Economic Status Risk.
df1['Economic Status Risk'] = df1[
    'Economic status'
].map(economic_status_risk_mapping)


# In[165]:


# Encoding  Median net household income(BHC) and Median net household 
# income(AHC) columns and renaming column names.
df1['Income Level (BHC)'] = df1[
    'Median net household income(BHC)'
].apply(lambda x: 1 if x == 'In low income' else 0)
df1['Income Level (AHC)'] = df1[
    'Median net household income(AHC)'
].apply(lambda x: 1 if x == 'In low income' else 0)


# In[166]:


# Simplifying Disability Mix column data.
df1['Disability Mix Group'] = df1[
    'Disability mix within the family'
].map({
    'In a family where no-one is disabled': 'No Disability',
    'In a family with disabled adult/s and child/ren': 'Mixed Disability',
    'In a family with disabled child/ren only': 'Child Disability',
    'In a family with disabled adult/s only': 'Adult Disability'
})
# Encoding the simplified disability mix.
df1['Disability Mix Group'] = df1[
    'Disability Mix Group'
].astype('category').cat.codes


# In[167]:


df1['Disability Mix Group'].unique()


# In[168]:


# Creating new features.
df1['Disability_Income_Interaction'] = df1[
    'Disability status'
] * df1['Income Level (AHC)']
df1['Disability_Region_Interaction'] = df1['Disability status'] * df1['Region']


# In[169]:


# Aggregating population by financial year, disability status, and 
# income level.
population_summary = df1.groupby([
    'Financial year', 
    'Disability status', 
    'Income Level (AHC)'
])['Population count'].sum().reset_index()
population_summary.head()


# In[170]:


# Calculating correlation matrix and visualising.
correlation_matrix = population_summary.corr()

# Plotting the heatmap.
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[171]:


df1.head()


# In[172]:


# Calculating the Poverty Risk score.
# Initializing Poverty Risk with 0.
df1['Poverty risk'] = 0

# Adding points based on Income Level (BHC and AHC).
df1['Poverty risk'] += df1['Income Level (BHC)']
df1['Poverty risk'] += df1['Income Level (AHC)']

# Adding points based on Disability Status.
df1['Poverty risk'] += df1['Disability status']

# Adding points based on Economic Status Risk.
df1['Poverty risk'] += df1['Economic Status Risk']

# Display the updated DataFrame.
df1.head()


# In[173]:


# Aggregating population by financial year, disability status, income 
# level and Poverty risk.
population_summary1 = df1.groupby([
    'Financial year', 
    'Disability status', 
    'Income Level (AHC)', 
    'Poverty risk'
])['Population count'].sum().reset_index()
population_summary1.head()


# In[174]:


# Calculating correlation matrix and visualising.
correlation_matrix1 = population_summary1.corr()

# Plotting the heatmap.
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[175]:


# Calculating correlation matrix and visualising.
correlation_matrix2 = df1.drop(
    columns=[
        'Median net household income(BHC)', 
        'Median net household income(AHC)', 
        'Disability mix within the family', 
        'Population count', 
        'Economic status'
    ], axis=1).corr()

# Plotting the heatmap.
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix2, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[176]:


df1['Poverty risk'].unique()


# In[ ]:




