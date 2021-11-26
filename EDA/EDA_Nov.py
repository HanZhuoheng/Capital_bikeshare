#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import datetime as dt


# In[16]:


path='/Users/weiyut/Downloads/cleaned_bike_data_2019.csv'
bike = pd.read_csv(path)


# In[45]:


month = map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").month, bike['Start_day'])
bike['month'] = list(month)


# In[ ]:


bike['Average Bike Availability(%)'] = bike['percent_full']*100


# ### The function to plot the average bike availability of a particular station or overall

# In[19]:


def station_aval(station = False, plot = False):
    if station:
        temp = bike[bike['station_name'] == station][['start_hour','percent_full','weekend_flag']]
    else:
        temp = bike[['start_hour','percent_full','weekend_flag']]
    
    temp = temp.groupby(['start_hour'])['percent_full'].agg(['mean','count'])
    if plot == True and station: 
        sns.set(font_scale = 2)
        sns.set_style("whitegrid")
        plt.figure(figsize=(20,8))
        ax=sns.barplot(x=temp.index, y=100*temp['mean'], color= '#6FBAA7')
        plt.ylim(temp['mean'].min()-0.05,temp['mean'].max()+0.02)
        ax.set_ylabel('Average Bike Availability(%)')
        ax.set_xlabel('Hour of Day') 
        ax.set_title('Bike station - '+station)
    elif plot == True:
        sns.set(font_scale = 2)
        sns.set_style("whitegrid")
        plt.figure(figsize=(20,8))
        ax=sns.barplot(x=temp.index, y=100*temp['mean'], color= '#6FBAA7')
        plt.ylim(40,50)
        ax.set_ylabel('Average Bike Availability(%)')
        ax.set_xlabel('Hour of Day') 
        ax.set_title('The average bike availability decreases during the working hours')
    else:
        return temp


# # Overall average bike availability of 2019

# In[20]:


station_aval(plot = True)


# # Weekends vs weekdays in 2019

# In[21]:


sns.set_style("whitegrid")
palette ={1: "#A7BC6A", 0: "#6FAEBA"}
plt.figure(figsize=(20,8))
ax = sns.lineplot(y='Average Bike Availability(%)',x="start_hour", estimator = np.mean, data=bike, hue='weekend_flag',  palette=palette, ci = None, linewidth = 3, marker = 'o')
ax.set_ylabel('Average Bike Availability(%)')
ax.set_xlabel('Hour of Day') 
ax.set_ylim(42, 52)
plt.legend(title='', loc='upper right', labels=['Weekend', 'Weekday'])
plt.title('Bike availability on weekends and weekdays is comparable')
plt.show()


# # Location of bikeshare stations

# In[30]:


dc_img=mpimg.imread('https://github.com/Tristal25/Capital_bikeshare_36601/blob/master/data/DCmap.png?raw=true')
bike['Average Bike Availability(%)'] = bike['percent_full']*100
bike['Station Capacity'] = bike['total_capacity'] 
#a  = pd.DataFrame(bike.groupby(['station_name'])[['percent_full','longitude', 'latitude']].mean())
a  = pd.DataFrame(bike.groupby(['station_name'])[['Average Bike Availability(%)','longitude', 'latitude','Station Capacity']].mean())
a.plot(kind="scatter", x="longitude", y="latitude", label = 'station'
       ,c="Station Capacity", cmap=plt.get_cmap("Blues"),
        colorbar=True, vmax = 20, alpha=0.8, figsize=(25,15), s = 50
      )
plt.xlim(-77.369,-76.825)
plt.ylim(38.75,39.15)
plt.ylabel('')
plt.imshow(dc_img, alpha=0.8, extent=[-77.369, -76.825, 38.75, 39.15])
plt.title('There are less stations in the non-downtown areas')
plt.xticks([])
plt.yticks([])
plt.show()


# In[121]:


plt.figure(figsize=(15,15))
a  = pd.DataFrame(bike.groupby(['station_name'])[['Average Bike Availability(%)','longitude', 'latitude', 'total_capacity']].mean())
minsize = min(bike['total_capacity'])
maxsize = max(bike['total_capacity'])
sns.scatterplot(x="longitude", y="latitude", s = 100, data =a, label = 'station')
plt.xlim(-77.369,-76.825)
plt.ylim(38.75,39.15)
plt.ylabel('')
plt.xlabel('')
plt.imshow(dc_img, alpha=0.8, extent=[-77.369, -76.825, 38.75, 39.15])
plt.title('There are less stations in the non-downtown areas ')
plt.xticks([])
plt.yticks([])
#plt.legend(loc=1, prop={'size': 15},title="Maximum Capactiy", markerscale = 1)
plt.show()


# #### Summary statistics of bikeshare station capacity

# In[33]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
bike['total_capacity'].describe()


# #### Summary statistics of bike availability across all stations through out the entire 2019 (observations are by hour)

# In[63]:


bike['percent_full'].describe()['75%']


# ### Average bike availability by hours throughout the entire 2019 

# In[64]:


for i in range(0,24):
    a  = pd.DataFrame(bike[bike['start_hour'] == i].groupby(['station_name'])[['Average Bike Availability(%)','longitude', 'latitude']].mean())
    a.plot(kind="scatter", x="longitude", y="latitude", label = 'station',
    c="Average Bike Availability(%)", cmap=plt.get_cmap("jet"),
    colorbar=True, alpha=0.8, vmax = bike['percent_full'].describe()['75%']*100, figsize=(15,10), s = 50) 
    # We set the maximum of colorbar equals the 75% percentile, 
    # so once the bike availability exceeds this number, it shows the deepest color in the colorbar.
    # This makes us easier to visualize our bike availability, because the distribution of bike availability is right skewed.
    # If we use the maximum bike availabilty, we wouldn't be able to visualize the difference, since most of the stations would be closer to bluish side. 
    plt.title('Hour of Day - '+ str(i)+':00')
    plt.xlim(-77.369,-76.825)
    plt.ylim(38.75,39.15)
    plt.legend()
    plt.imshow(dc_img, alpha=0.8, extent=[-77.369, -76.825, 38.75, 39.15])
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('')
    plt.show()
    plt.close()
    
# 8am 6pm


# ### Average bike availability by month throughout the entire 2019 

# In[65]:


import calendar
for i in range(1,13):
    m  = pd.DataFrame(bike[bike['month'] == i].groupby(['station_name'])[['Average Bike Availability(%)','longitude', 'latitude']].mean())
    m.plot(kind="scatter", x="longitude", y="latitude", label = 'station',
    c="Average Bike Availability(%)", cmap=plt.get_cmap("jet"),
    colorbar=True, alpha=0.8, vmax = bike['percent_full'].describe()['75%']*100, figsize=(15,10), s = 50)
    # We set the maximum of colorbar equals the 75% percentile, 
    # so once the bike availability exceeds this number, it shows the deepest color in the colorbar.
    # This makes us easier to visualize our bike availability, because the distribution of bike availability is right skewed.
    # If we use the maximum bike availabilty, we wouldn't be able to visualize the difference, since most of the stations would be closer to bluish side. 
    plt.title(list(calendar.month_name)[i]+' 2019')
    plt.xlim(-77.369,-76.825)
    plt.ylim(38.75,39.15)
    plt.legend()
    plt.imshow(dc_img, alpha=0.8, extent=[-77.369, -76.825, 38.75, 39.15])
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('')
    plt.show()
    plt.close()
    


# # NOT USING

# In[ ]:


#top_n = bike[bike['percent_full']<=0.33].groupby('station_name')['percent_full'].count().sort_values(ascending = False).head(50)
(bike[bike['percent_full']>=0.71].groupby('station_name')['percent_full'].count()/bike.groupby('station_name')['percent_full'].count()).sort_values(ascending = False).head(5)


# In[ ]:


def top_n_low_avail_cons(n , p, fw = 15, fh = 15):
    top_n = (bike[bike['percent_full']<=p].groupby('station_name')['percent_full'].count()/bike.groupby('station_name')['percent_full'].count()).sort_values(ascending = False).head(n)
    top_n_lowest = bike[bike['station_name'].isin(list(top_n.index))].groupby('station_name')[['longitude','latitude']].mean()
    plt.rcParams["figure.figsize"] = (fw,fh)
    top_n_lowest.plot(kind="scatter", x="longitude", y="latitude", label = 'station', color = 'blue')
    plt.xlim(-77.369,-76.825)
    plt.ylim(38.75,39.15)
    plt.imshow(dc_img, alpha=0.8, extent=[-77.369, -76.825, 38.75, 39.15])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    #plt.show()


# In[ ]:


top_n_low_avail_cons(50, bike['percent_full'].quantile(0.25), 15, 15)
plt.title('Top 50 stations with consistent low bike availability')
plt.show()
## Suburb areas consistently have higher bike availablity. 


# In[ ]:


def top_n_high_avail_cons(n , p, fw = 15, fh = 15):
    top_n = (bike[bike['percent_full']>=p].groupby('station_name')['percent_full'].count()/bike.groupby('station_name')['percent_full'].count()).sort_values(ascending = False).head(n)
    top_n_lowest = bike[bike['station_name'].isin(list(top_n.index))].groupby('station_name')[['longitude','latitude']].mean()
    dc_img=mpimg.imread('DCmap.png')
    plt.rcParams["figure.figsize"] = (fw,fh)
    top_n_lowest.plot(kind="scatter", x="longitude", y="latitude", label = 'station', color = 'red')
    plt.xlim(-77.369,-76.825)
    plt.ylim(38.75,39.15)
    plt.imshow(dc_img, alpha=0.8, extent=[-77.369, -76.825, 38.75, 39.15])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    #plt.show()


# In[ ]:


top_n_high_avail_cons(50, bike['percent_full'].quantile(0.75), 15, 15)
plt.title('Top 50 stations with consistent high bike availability')
plt.show()


# In[116]:


plt.figure(figsize=(15,15))
a  = pd.DataFrame(bike.groupby(['station_name'])[['Average Bike Availability(%)','longitude', 'latitude', 'total_capacity']].mean())
minsize = min(bike['total_capacity'])
maxsize = max(bike['total_capacity'])
sns.scatterplot(x="longitude", y="latitude",
                    hue="total_capacity", size="total_capacity", sizes=(10*minsize, 20*maxsize), 
                    data=a, palette = "Blues")
plt.xlim(-77.369,-76.825)
plt.ylim(38.75,39.15)
plt.ylabel('')
plt.xlabel('')
plt.imshow(dc_img, alpha=0.8, extent=[-77.369, -76.825, 38.75, 39.15])
plt.title('There are less stations in the non-downtown areas ')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, prop={'size': 15},title="Maximum Capactiy", markerscale = 1)
plt.show()


# In[ ]:




