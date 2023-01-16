import pandas as pd
import requests
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn import linear_model
import statsmodels.api as sm

#aggregating rental prices for latest year
rental_rates = pd.read_csv("renting-out-of-flats.csv")
rental_rates = rental_rates.groupby(["town", "flat_type"]).agg({"monthly_rent": "mean"}).apply(lambda x: round(x))

#aggregrating rental prices by year 2021
other_rental = pd.read_csv("median-rent-by-town-and-flat-type.csv")
other_rental["median_rent"] = other_rental["median_rent"].apply(lambda x: x.strip())
other_rental = other_rental[(other_rental["median_rent"] != 'na') & (other_rental["median_rent"] != '-')]
other_rental["quarter"] = other_rental["quarter"].apply(lambda x: x[:4])
other_rental = other_rental.groupby(["quarter", "town", "flat_type"]).first()

#aggregating number of lease years left and age of estate by town

columns_df = ['town', 'lease_commence_date']
columns_age_of_estate = ['town', 'established_in']
df = pd.read_csv("Downloads/resale-flat-prices/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv", usecols=columns_df)
age_of_estate = pd.read_csv("Downloads/age_of_estate.csv", usecols=columns_age_of_estate)

age_of_estate['town'].str.upper()
age_of_estate['age_of_town'] = 2023 - age_of_estate['established_in']
#print(age_of_estate)
df['lease_left'] = 2023 - df['lease_commence_date']
df['max_lease_left'] = df.loc[:, 'lease_left']
df['min_lease_left'] = df.loc[:, 'lease_left']
df['mean_lease_left'] = df.loc[:, 'lease_left']
df['median_lease_left'] = df.loc[:, 'lease_left']
df = df.groupby(['town']).agg({'min_lease_left':'min', 'max_lease_left' :'max', 'mean_lease_left':'mean', 'median_lease_left':'median' }).apply(lambda x: round(x))
df = pd.merge(df, age_of_estate, on='town')

location_df = pd.read_csv('/location  (2).csv')

location_df = location_df.drop(location_df[location_df['Town'] == 'nil'].index)
location_df['cords'] = None
location_df['restaurant'] = None
location_df['primary_school'] = None
location_df['shopping_mall'] = None
location_df['bus_station'] = None
location_df['subway_station'] = None


#gathering data for number of amenities aggregated by town
def get_coordinates(place):
  url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={place}%20MRT&inputtype=textquery&fields=geometry&key=AIzaSyAzfD_KB89k-qZX4MMPdHM6ZV2fELbniQY"
  payload={}
  headers = {}
  response = requests.request("GET", url, headers=headers, data=payload)
  results = response.json()
  results_cords = results['candidates'][0]['geometry']['location']
  return f"{results_cords['lat']}%2C{results_cords['lng']}"

def get_amenities(cords, amenity):
  if amenity == "restaurant":
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={cords}&radius=1000&type={amenity}&minprice=2&key=AIzaSyAzfD_KB89k-qZX4MMPdHM6ZV2fELbniQY"
  elif amenity == "subway_station":
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={cords}&radius=2000&type={amenity}&key=AIzaSyAzfD_KB89k-qZX4MMPdHM6ZV2fELbniQY"
  else: 
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={cords}&radius=1000&type={amenity}&key=AIzaSyAzfD_KB89k-qZX4MMPdHM6ZV2fELbniQY"
  payload={}
  headers = {}
  response = requests.request("GET", url, headers=headers, data=payload)
  results = response.json()
  a = len(results["results"])
  if "next_page_token" in results:
    time.sleep(3)
    next_page_token = results['next_page_token']
    nxt_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={cords}&radius=1000&type={amenity}&pagetoken={next_page_token}&key=AIzaSyAzfD_KB89k-qZX4MMPdHM6ZV2fELbniQY"
    response = requests.request("GET", nxt_url, headers=headers, data=payload)
    results = response.json()
    b = len(response.json()["results"])
    if "next_page_token" in results:
      time.sleep(3)
      next_page_token = results['next_page_token']
      nxt_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={cords}&radius=1000&type={amenity}&pagetoken={next_page_token}&key=AIzaSyAzfD_KB89k-qZX4MMPdHM6ZV2fELbniQY"
      nxt_response = requests.request("GET", nxt_url, headers=headers, data=payload)
      c = len(nxt_response.json()["results"])
      return a + b + c
    else:
      return a + b
  else:
    return a

for town in location_df["Town"]:
  if (town == "Central") :
    cords = get_coordinates("Tanjong pagar")
  else:
    cords = get_coordinates(town)
  location_df.loc[location_df['Town'] == town, "cords"] = cords
  location_df.loc[location_df['Town'] == town, "restaurant"] = get_amenities(cords, "restaurant")
  location_df.loc[location_df['Town'] == town, "primary_school"] = get_amenities(cords, "primary_school")
  location_df.loc[location_df['Town'] == town, "shopping_mall"] = get_amenities(cords, "shopping_mall")
  location_df.loc[location_df['Town'] == town, "bus_station"] = get_amenities(cords, "bus_station")
  location_df.loc[location_df['Town'] == town, "subway_station"] = get_amenities(cords, "subway_station")


  
#pre-processing of data
ameneties_df = pd.read_csv("ameneties.csv")
age_df = pd.read_csv("age_data.csv")
rental_df = pd.read_csv("rental2byyear.csv")
location_df = pd.read_csv("location.csv")

age_df["town"] = age_df["town"].apply(lambda x: x.lower())
rental_df["town"] = rental_df["town"].apply(lambda x: x.lower())
ameneties_df["Town"] = ameneties_df["Town"].apply(lambda x: x.lower())
location_df["Town"] = location_df["Town"].apply(lambda x: x.lower())

merge_1 = ameneties_df.merge(age_df, how = "inner", left_on = "Town", right_on = "town")
merge_2 = merge_1.merge(rental_df, left_on = "Town", right_on = "town", how = "inner")
merge_3 = merge_2.merge(location_df, on = "Town", how = "inner")

#converts 1-rm to 1
def change(x): 
  first = x[0]
  if (first == "1"):
    return 1
  elif (first == "2"):
    return 2
  elif (first == "3"):
    return 3
  elif first == "4": 
    return 4
  elif first == "5":
    return 5
  else:
    return 6

merge_3["flat_type"] = merge_3["flat_type"].apply(lambda x: change(x))

#training of linear regression model
train, test = train_test_split(merge_3, test_size = 0.3)
train_y = train["monthly_rent"]
test_y = test["monthly_rent"]
train_x = train[["restaurant", "primary_school", "shopping_mall", "bus_station", "subway_station", 
                 "mean_lease_left", "flat_type"]]
test_x = test[["restaurant", "primary_school", "shopping_mall", "bus_station", "subway_station", 
                 "mean_lease_left", "flat_type"]]

train_y = train_y.values.reshape(-1,1)
test_y = test_y.values.reshape(-1,1)

linear_model = LinearRegression()

linear_model.fit(train_x, train_y)

linear_model.coef_

y_pred = linear_model.predict(test_x)

#Using OLS regression
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


model = sm.OLS(train_y, train_x).fit()
predictions = model.predict(test_x) 
 
print_model = model.summary()
print(print_model)

#finding error values
MAE = mean_absolute_error(test_y, y_pred)
MAE
MAPE = mean_absolute_percentage_error(test_y, y_pred)
MAPE
MSE = mean_squared_error(test_y, y_pred)
MSE

#demo of model
def predict_prices(town, rooms):
  restaurant = ameneties_df.loc[ameneties_df['Town'] == town, 'restaurant']
  primary_school = ameneties_df.loc[ameneties_df['Town'] == town, 'primary_school']
  shopping_mall = ameneties_df.loc[ameneties_df['Town'] == town, 'shopping_mall']
  bus_station = ameneties_df.loc[ameneties_df['Town'] == town, 'bus_station']
  subway_station = ameneties_df.loc[ameneties_df['Town'] == town, 'subway_station']
  mean_lease_left = age_df.loc[age_df['town'] == town, 'mean_lease_left']

  return 12.3163 * float(restaurant) + 45.6489 * float(primary_school) + -19.8443 * float(shopping_mall) + -2.9791 * float(bus_station) + 14.9765 * float(subway_station) + 31.2560 * float(mean_lease_left) + 267.3964 * float(rooms)

print(predict_prices('ang mo kio', 4))
