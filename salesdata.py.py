#!/usr/bin/env python
# coding: utf-8

# ### Sales Analysis
# 
# Import Necessary Libraries

# In[1]:


import pandas as pd
import os


# ### Task #1: Merge 12 months of sales data into a single file

# In[2]:


df = pd.read_csv("/Users/Bruno/Desktop/data_science/Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/Sales_April_2019.csv")

files = [file for file in os.listdir('/Users/Bruno/Desktop/data_science/Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data')]

all_months_data = pd.DataFrame()

for file in files:
    df = pd.read_csv("/Users/Bruno/Desktop/data_science/Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/"+file)
    all_months_data = pd.concat([all_months_data, df])

all_months_data.to_csv("all_data1.csv", index=False)


# #### Read in updated dataframe

# In[3]:


all_data = pd.read_csv("all_data1.csv")
all_data.head()


# ## Clean up the Data!

# ### Drop rows of NAN

# In[4]:


nan_df = all_data[all_data.isna().any(axis=1)]
nan_df.head()

all_data = all_data.dropna(how='all')

all_data.head()


# ### Find 'Or' and delete it

# In[7]:


all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']


# ##### Convert Columns to the correct type

# In[8]:


all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered']) # make Int
all_data['Price Each'] = pd.to_numeric(all_data['Price Each']) # Make Float
all_data.head()


# ## Augment data with additional columns

# #### Task 2: Add Month Column

# In[9]:


all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')
all_data.head()


# ### Task 3: Add a sales column

# In[10]:


all_data['Total Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']
all_data.head()


# In[ ]:





# #### Question 1: What was the best month for sales? How much was earned that month?

# In[11]:


results = all_data.groupby('Month').sum()


# In[12]:


import matplotlib.pyplot as plt

months = range(1,13)

plt.bar(months, results['Total Sales'])
plt.xticks(months)
plt.ylabel('Sales in USD ($)')
plt.xlabel('Month number')
plt.show()


# ### Question #2: What city had the highest number of sales?

# ##### Task 1: Insert a City Column

# In[ ]:


# let's use .apply()


# In[13]:


def get_city(address):
    return address.split(',')[1]

def get_state(address):
    return address.split(',')[2].split(' ')[1]

all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")

all_data.head()


# ##### Question 2 cont'd: What city had the highest number of sales? 

# In[15]:


results = all_data.groupby('City').sum()
results


# In[29]:


import matplotlib.pyplot as plt


cities = [city for city, df in all_data.groupby('City')]

plt.bar(cities, results['Total Sales'])
plt.xticks(cities, rotation='vertical', size=8)
plt.ylabel('Sales in USD ($)')
plt.xlabel('City name')
plt.show()


# ### Question 3: What time should we display advertisements to maximize the likelihood of customer's buying product?

# In[20]:


all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])


# In[25]:


all_data['Hour'] = all_data['Order Date'].dt.hour
all_data['Minute'] = all_data['Order Date'].dt.minute
all_data.head()


# In[53]:


hours = [hour for hour, df in all_data.groupby('Hour')]

plt.plot(hours, all_data.groupby(['Hour']).count())
plt.xticks(hours)
plt
plt.grid()
plt.show()         

# My recommandation is around 11am (11) or 7pm (19)


# ### Question 4: What products are most often sold together?

# In[70]:


df = all_data[all_data['Order ID'].duplicated(keep=False)]

df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))

df = df[['Order ID', 'Grouped']].drop_duplicates()

df.head(5)


# In[72]:


from itertools import combinations
from collections import Counter

count = Counter()

for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))

for key, value in count.most_common(10):
    print(key, value)


# #### Question 5: What product sold the most? Why do you think it sold the most?

# In[89]:


product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum()['Quantity Ordered']

products = [product for product, df in product_group]

plt.bar(products, quantity_ordered)
plt.ylabel('Quantity Ordered')
plt.xlabel('Product')
plt.xticks(products, rotation='vertical', size=8)
plt.show()


# Consumers tend to use these products to power/charge a variety of electronic products


# In[95]:


prices = all_data.groupby('Product').mean()['Price Each']

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='g')
ax2.plot(products, prices, 'b-')
ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered', color='g')
ax2.set_ylabel('Price ($)', color='b')
ax1.set_xticklabels(products, rotation='vertical', size=8)

plt.show()



# In[ ]:





# In[ ]:




