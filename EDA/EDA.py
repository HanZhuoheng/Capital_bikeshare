import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

# Data import
bike = pd.read_csv('data/cleaned_bike_data_june2019.csv')


bike.head(5)
bike['Average Bike Availability(%)'] = bike['percent_full']*100


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
        plt.ylim(45,50)
        ax.set_ylabel('Average Bike Availability(%)')
        ax.set_xlabel('Hour of Day') 
        ax.set_title('Average Bike Availablility over a day for every bike station in DC area')
    else:
        return temp


# # Overall



station_aval(plot = True)


# # Weekends vs weekdays




sns.set_style("whitegrid")
palette ={1: "#A7BC6A", 0: "#6FAEBA"}
plt.figure(figsize=(20,8))
ax = sns.barplot(y="Average Bike Availability(%)",x="start_hour", estimator = np.mean, data=bike, hue='weekend_flag',  palette=palette, ci = None)
ax.set_ylabel('Average Bike Availability(%)')
ax.set_xlabel('Hour of Day') 
ax.set_ylim(40, 55)
plt.legend(title='', loc='upper right', labels=['Weekend', 'Weekday'])
plt.title('Average bike availablility over a day by weekend versus weekday')
plt.show()


# # Location




dc_img=mpimg.imread('https://github.com/Tristal25/Capital_bikeshare_36601/blob/master/data/DCmap.png?raw=true')
bike['Average Bike Availability(%)'] = bike['percent_full']*100
#a  = pd.DataFrame(bike.groupby(['station_name'])[['percent_full','longitude', 'latitude']].mean())
a  = pd.DataFrame(bike.groupby(['station_name'])[['Average Bike Availability(%)','longitude', 'latitude']].mean())
a.plot(kind="scatter", x="longitude", y="latitude", label = 'station'
       ,c="Average Bike Availability(%)", cmap=plt.get_cmap("jet"),
        colorbar=True, vmax = 80, alpha=0.8, figsize=(25,15), s = 50
      )
plt.xlim(-77.369,-76.825)
plt.ylim(38.75,39.15)
plt.ylabel('')
plt.imshow(dc_img, alpha=0.8, extent=[-77.369, -76.825, 38.75, 39.15])
plt.title('Average bike availability over a day for every station in the DC area')
plt.xticks([])
plt.yticks([])
plt.show()





for i in range(0,24):
    a  = pd.DataFrame(bike[bike['start_hour'] == i].groupby(['station_name'])[['Average Bike Availability(%)','longitude', 'latitude']].mean())
    a.plot(kind="scatter", x="longitude", y="latitude", label = 'station',
    c="Average Bike Availability(%)", cmap=plt.get_cmap("jet"),
    colorbar=True, alpha=0.8, vmax = 80, figsize=(15,10), s = 50)
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


# # Some summary stats

# # Which stations consistently have problems with bike availability?




#top_n = bike[bike['percent_full']<=0.33].groupby('station_name')['percent_full'].count().sort_values(ascending = False).head(50)
(bike[bike['percent_full']>=0.71].groupby('station_name')['percent_full'].count()/bike.groupby('station_name')['percent_full'].count()).sort_values(ascending = False).head(5)





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





top_n_high_avail_cons(50, bike['percent_full'].quantile(0.75), 15, 15)
plt.title('Top 50 stations with consistent high bike availability')
plt.show()



