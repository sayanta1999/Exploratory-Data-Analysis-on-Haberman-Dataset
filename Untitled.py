#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
df = pd.read_csv('haberman.csv')
print(df.head(10))
print(df.shape)
print(df.isnull().sum())
#print(np.sum(df.loc[df['status']==1]))
print(df['status'][df['status']==1].sum())
print(df['status'][df['status']==2].sum())
print(df.describe())


# Observations:
# 1)"status"=1 means patient has survived 5 years or more
#   "status"=2 means patient had died before 5 years.
# 2)"nodes" refers to number of positive axillary nodes
# 3)There are 306 data points, 4 features (including class,ie, "status")
# 4)There are 225 data points of status=1 and 162 data points of status=2
# 5)No Nan Values present in the dataset

# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.FacetGrid(df,hue='status',height=8).map(sns.distplot,'age').add_legend()
plt.show()
sns.FacetGrid(df,hue='status',height=6).map(sns.distplot,'year').add_legend()
plt.show()
sns.FacetGrid(df,hue='status',height=6).map(sns.distplot,'nodes').add_legend()
plt.show()


# In[56]:


count,bin_edges = np.histogram(df['age'],bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='PDF')
plt.plot(bin_edges[1:],cdf,label='CDF')
plt.legend()
plt.show()

count,bin_edges = np.histogram(df['year'],bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='PDF')
plt.plot(bin_edges[1:],cdf,label='CDF')
plt.legend()
plt.show()

count,bin_edges = np.histogram(df['nodes'],bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='PDF')
plt.plot(bin_edges[1:],cdf,label='CDF')
plt.legend()
plt.show()


# In[35]:


sns.boxplot(data=df,x='status',y='age')
plt.show()
sns.boxplot(data=df,x='status',y='year')
plt.show()
sns.boxplot(data=df,x='status',y='nodes')
plt.show()


# In[37]:


sns.violinplot(data=df,x='status',y='age')
plt.show()
sns.violinplot(data=df,x='status',y='year')
plt.show()
sns.violinplot(data=df,x='status',y='nodes')
plt.show()


# In[52]:


sns.set_style('whitegrid')
sns.pairplot(df,hue='status',height=6)
plt.legend()
plt.show()

