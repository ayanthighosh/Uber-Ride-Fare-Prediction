#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#file_location = "DataSet (1).xlsx"

#dfs = pd.read_excel(file_location, sheet_name="Sheet1")
dfs = pd.read_excel(r'C:\Users\ayanthg\Documents\D\ds.xlsx')


# In[3]:


dfs


# In[4]:


import plotly.express as px
import plotly.graph_objects as go


# In[5]:


# timestamp vs trips...helps to analyse trips in a given timefrmae ....zoom-in to see points plot 
fig = px.scatter(dfs, x='timestamp', y='trip_id')
fig.show()


# In[6]:


# returnig customers at timpestamps
fig = px.scatter(dfs, x='customer_id', y='timestamp')
fig.show()


# In[7]:


# frequency of returnig customers with color scale demonstrating avreage time between 2 trips
import numpy as np
customers_grp = dfs.groupby(['customer_id']).groups
cus_grp = []
for k,v in customers_grp.items():
    if len(v)>=2:# atleast 2 trips
        trip_timestamps = [dfs["timestamp"][i] for i in v]
        avg_time_2_trips = ((np.max(trip_timestamps) - np.min(trip_timestamps))/len(v))/60000 # in minutes
    #     if avg_time_2_trips==0:
    #         avg_time_2_trips = 40000
        cus_grp.append([k,len(v),avg_time_2_trips])

z=pd.DataFrame(cus_grp)
z.columns=["customer", "trip_freq","avg_time_2_trips"]
fig = px.scatter(z, x="customer", y="trip_freq",color='avg_time_2_trips')
fig.show()


# In[ ]:


# Trips' geographic reagion of coverage

pickupLat_max = dfs['pick_lat'].max()
pickupLat_min = dfs['pick_lat'].min()
dropLat_max = dfs['drop_lat'].max()
dropLat_min = dfs['drop_lat'].min()
pickupLng_max = dfs['pick_lng'].max()
pickupLng_min = dfs['pick_lng'].min()
dropLng_max = dfs['drop_lng'].max()
dropLng_min = dfs['drop_lng'].min()
print ("Max pick-up reagio geographic spread latitude: "+ str(pickupLat_max)+", longitude: " + str(pickupLng_max))
print ("Min pick-up reagio geographic spread latitude: "+ str(pickupLat_min)+", longitude: " + str(pickupLng_min))
print ("Max dropping reagio geographic spread latitude: "+ str(dropLat_max)+", longitude: " + str(dropLng_max))
print ("Min dropping reagio geographic spread latitude: "+ str(dropLat_min)+", longitude: " + str(dropLng_min))

# range_pLat = (pickupLat_max - pickupLat_min)/10
# range_dLat = (dropLat_max - dropLat_min)/10
# range_pLng = (pickupLng_max - pickupLng_min)/10
# range_dLng = (dropLng_max - dropLng_min)/10


# In[ ]:


# geographic region spread of pickup loactions
fig = px.scatter(dfs, x='pick_lat', y='pick_lng')
fig.show()


# In[ ]:


# geographic region spread of droping loactions
fig = px.scatter(dfs, x='drop_lat', y='drop_lng')
fig.show()


# In[9]:


# 4-Dimensional representation of trips (used in deciding trip routes and also for sorting short and long trips)
import plotly.express as px
fig = px.scatter_3d(dfs, x='pick_lat', y='pick_lng', z='timestamp',
              color='travel_distance')
fig.show()


# In[ ]:


### PART 2  Top 5 pairs of hex (resolution=8) clusters where most of the trips happened? 
# You can refer to the library listed below to get hexid for a given latitude and longitude.


# In[ ]:


#!pip install h3


# In[10]:


import h3

given_grid_resolution = 8

def getHexID(location=[0, 0], resolution = given_grid_resolution):
    return h3.geo_to_h3(location[0], location[1], resolution)
    


# In[11]:


### calculating hexa pairs between pickup and drop

number_of_Customer = len(dfs)
trip_dict = {}
for i in range(number_of_Customer):
    # pickup location
    pickup_loc = [dfs["pick_lat"][i], dfs["pick_lng"][i]]
    # droping location
    drop_loc = [dfs["drop_lat"][i], dfs["drop_lng"][i]]
    # pickup hexa ID
    pickup_HexaId = getHexID(location = pickup_loc)
    # droping hexa ID
    drop_HexaId = getHexID(location = drop_loc)
    # dictionary to store hexa pairs and keep counter for number of trips
#     print(pickup_HexaId, drop_HexaId)
    trip_key = pickup_HexaId +"||"+drop_HexaId
    try:
        trip_dict[trip_key] += 1
    except:
        trip_dict[trip_key] = 1
    


# In[ ]:


# print(trip_dict)


# In[12]:


### printing top 5 pairs

# sorting on number of trips
sorted_trip_list = sorted(trip_dict.items(), key=lambda item: item[1])
sorted_trip_list.reverse()

sorted_trip_output = [[i+1,k.replace("||"," to "), v ] for i,(k,v) in enumerate(sorted_trip_list)]

df_sorted_trip_output = pd.DataFrame(sorted_trip_output)
df_sorted_trip_output.columns =['Rank', 'Hex pair (source_hexid, destination_hexid)','Totaltrips']

df_sorted_trip_output.head(5)


# In[13]:


### saving pairs into xlxs file
df_sorted_trip_output.to_excel("output_part_2.xlsx")


# In[ ]:


# Part 3. Metric calculation
# What is the average duration between the 1st trip and the 2nd trip of customers? 
# Note: Consider only the customers who have done 2 or more trips.


# In[14]:


import numpy as np
# grouping data by customer id in dictionary using pandas
customers_grp = dfs.groupby(['customer_id']).groups

# selecting customer who have done 2 or more trips.
list_cust_time_taken = []

for k,v in customers_grp.items():
    # k is customerID
    # v is list of index of rows of  customerID in file
    if len(v)>=2:
        dic = {}
        # retriving all timestamp and travel_time for the customerID
        for i in v:
            dic[dfs["timestamp"][i]] = dfs["travel_time"][i]
        # taking first 2 trips by sorting
        sort_li = sorted(dic.items())
        # calculating duration between the 1st trip and the 2nd trip
        time_taken = sort_li[1][0] - sort_li[0][0] #- sort_li[0][1]
#         print(sort_li)
#         print(k, time_taken)
        list_cust_time_taken.append(time_taken)
# finding average    
average = np.mean(list_cust_time_taken) 


print("Average duration between the 1st trip and the 2nd trip of customers", average)


# In[ ]:


# Part 4. Model building
# Build a model to predict trip_fare using travel_distance and travel_time. 
# Measure the accuracy of the model and use the model to 
# predict trip_fare for a trip with travel_distance of 3.5 kms and travel_time of 15 minutes.


# In[ ]:


import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[ ]:


X = dfs[["travel_distance", "travel_time"]]


# In[ ]:


Y = dfs["trip_fare"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
print(coeff_df)


# In[ ]:


# running preiction on test data
y_pred = regressor.predict(X_test)


# In[ ]:


# chekcing results from model
df_res = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_res.head(25))


# In[ ]:


# calculating predictions 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


regressor.score(X,Y)


# In[ ]:


query = [[3.5, 15]]
ans_query = regressor.predict(query)
print("predicted trip_fare for a trip with travel_distance of 3.5 kms and travel_time of 15 minutes:", ans_query[0])

