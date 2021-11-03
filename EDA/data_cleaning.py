# Package Import

import pandas as pd
import datetime 
import numpy as np
import csv


# Data Import

raw_bike_data = pd.read_csv('data/201906-capitalbikeshare-tripdata.csv')
bike_capacity = pd.read_csv('data/Capital_Bike_Share_Locations.csv')


# Datatime formatting

## Transform the data to pandas date time variables
raw_bike_data['Start date'] = pd.to_datetime(raw_bike_data['Start date'], format = '%Y-%m-%d %H:%M:%S')
raw_bike_data['End date'] = pd.to_datetime(raw_bike_data['End date'], format = '%Y-%m-%d %H:%M:%S')


# Extract the day info and time info separately

## Extract the start and end times and date into separate columns
raw_bike_data['Start_time'] = raw_bike_data['Start date'].dt.time
raw_bike_data['End_time'] = raw_bike_data['End date'].dt.time

raw_bike_data['Start_day'] = raw_bike_data['Start date'].dt.date
raw_bike_data['End_day'] = raw_bike_data['End date'].dt.date


# add week/weekend flag
# day_info = bike_data['Start date'].dt.weekday


## Define hour
raw_bike_data['start_hour'] = raw_bike_data['Start date'].dt.hour



# Partition data into two tables: departures and arrivals

## Make a ride id column so that we can merge two dfs together later
raw_bike_data['ride_id'] = np.arange(raw_bike_data.shape[0])

## Make two separate tables so we can easily calculate the number of arrivals/departures
arrivals = raw_bike_data
departures = raw_bike_data


# Arrivals Table

## Aggregate the data
arrivals_agg = arrivals.groupby(['Start_day', 'start_hour', 'Start station number']).count()

## We only need one column to get the counts
arrivals_df = arrivals_agg['Bike number']

## Move the start day, start hour, and station number variables to columns instead of indexes
arrivals_df = arrivals_df.reset_index(level = [0,1,2])

## Normalize the variable names so we can merge tables
arrivals_df.rename({'Start station number':'station_id', 'Bike number':'num_arrivals'}, axis = 1, inplace = True)

# Departures Table

## Aggregate the data
depart_agg = departures.groupby(['Start_day', 'start_hour', 'End station number']).count()

## We only need one column to get the counts
depart_df = depart_agg['Bike number']

## Move the start day, start hour, and station number variables to columns instead of indexes
depart_df = depart_df.reset_index(level = [0,1,2])

## Normalize the variable names so we can merge tables
depart_df.rename({'End station number':'station_id', 'Bike number':'num_depart'}, axis = 1, inplace = True)


# Merge departure and arrivals table together

final_bike_data = pd.merge(arrivals_df, depart_df, how = 'inner', on = ['Start_day', 'start_hour', 'station_id'])
final_bike_data.head()


# Calculate the difference in arrivals and departures

final_bike_data['diff'] = final_bike_data['num_arrivals'] - final_bike_data['num_depart']
final_bike_data.head()


# Bikeshare capacity data

## Now we need the capacity information from the other dataset so that we can set a limit to the cumulative difference for bike availability

filtered_capacity = bike_capacity[['NAME','NUM_DOCKS_AVAILABLE', 'LONGITUDE', "LATITUDE"]]
filtered_capacity.head()

## Get a dataframe with all the bike stations and their names
unique_stations = raw_bike_data.drop_duplicates(subset=['Start station number'])[['Start station number', 'Start station']]


## Add back the station names and the capacity to the dataset
full_data_stations = pd.merge(final_bike_data, unique_stations, how = 'left', left_on = 'station_id', right_on = 'Start station number')
full_bike_data = pd.merge(full_data_stations, filtered_capacity, how = 'left', left_on = 'Start station', right_on = 'NAME')

full_bike_data['NAME'].dropna()


## There's only 30 stations that do not have a capacity

full_bike_data.drop(columns = ['NAME', 'Start station number'], inplace = True)
full_bike_data.rename({'Start station': 'station_name', 
                       'NUM_DOCKS_AVAILABLE': 'num_docks', 
                       'LONGITUDE': 'longitude', 
                       'LATITUDE':'latitude'} , axis = 1, inplace = True)


# Calculate cumulative sum of diff for every station

## Sort the values so that we can view the stations individually one at a time in chronological order
full_bike_data.sort_values(['station_id','Start_day','start_hour'], inplace = True)

## Need to reset index so that our foor loop indices match up to the row number
full_bike_data.reset_index(inplace = True)

station_num = full_bike_data.loc[0,'station_id']
cumu_sum = 0
for i in range(1,len(full_bike_data)):
    # Check to see if the station number has changed
    if full_bike_data.loc[i,'station_id'] != station_num:
        # If it has, reset the sum to the new station
        station_num = full_bike_data.loc[i,'station_id']
        cumu_sum = 0
    # If the cumu sum has reached max capacity
    if cumu_sum + full_bike_data.loc[i, 'diff'] >= full_bike_data.loc[i,'num_docks']:
        full_bike_data.at[i,'cumu_sum'] = full_bike_data.loc[i,'num_docks']
        cumu_sum = full_bike_data.loc[i,'num_docks']
    # If the cumu sum has reached minimum capacity
    elif cumu_sum + full_bike_data.loc[i, 'diff'] < 0:
        full_bike_data.at[i,'cumu_sum'] = 0
        cumu_sum = 0
    else:
        # Add to the cumulative sum and add the data point to the table
        cumu_sum += full_bike_data.loc[i, 'diff'] 
        full_bike_data.at[i,'cumu_sum'] = cumu_sum

pd.set_option("display.max_rows", None)

full_bike_data.to_csv(r'data/cleaned_data.csv', index = False)



