# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:12:56 2024

@author: juliu
"""
import pandas as pd
from geopy.geocoders import Nominatim # to transform country codes
import pycountry

import os


#change directory
path = r"D:\Data\Dropbox\LifeAfter\Datascientest\Climate"
os.chdir(path)

# load data
df_default = pd.read_csv("Data/data_clean/merge.csv")

# drop unnamed index 
df_default = df_default.drop("Unnamed: 0", axis = 1)

# okay so we have a couple of years for a couple of countries missing. For now we ignore that
a = df_default.groupby(["country"]).year.count()

geolocator = Nominatim(user_agent="http")
# transform countries to longitude and latitude (taken from https://dev.to/leul12/geo-coding-country-names-in-python-12ef)
def get_coordinates(country):
    print(country)
    try:
        country_obj = pycountry.countries.get(name=country)
        #geolocator = Nominatim(user_agent="http")
        location = geolocator.geocode({'country':country_obj.name})
        return location.latitude, location.longitude
    except AttributeError:
        try:
            country_obj = pycountry.countries.lookup(country)
            if country in weird_countries: # Taiwan is referenced by ISO standard as republic of China in pycountry, but no geopy
                location = geolocator.geocode({'country':country_obj.common_name})
            else:
            #geolocator = Nominatim(user_agent="http")
                location = geolocator.geocode({'country':country_obj.name})
            return location.latitude, location.longitude
        except LookupError:
            try:
                country_obj = pycountry.countries.search_fuzzy(country)
                #geolocator = Nominatim(user_agent="http")
                location = geolocator.geocode({'country':country_obj.name})
                return location.latitude, location.longitude
            except AttributeError:
                    country_obj = pycountry.countries.get(alpha_2=miss_countries[country])
                    #geolocator = Nominatim(user_agent="http")
                    location = geolocator.geocode({'country':country_obj.name})
                    return location.latitude, location.longitude
            except LookupError:
                    country_obj = pycountry.countries.get(alpha_2=miss_countries[country])
                    #geolocator = Nominatim(user_agent="http")
                    if country == "Democratic Republic of Congo": # Congo is strange
                        location = geolocator.geocode(country_obj.name)
                    else:
                        location = geolocator.geocode({'country':country_obj.name})
                    return location.latitude, location.longitude


# countries where that don't work here we use alpha-2 codes manually looked up
miss_countries ={'Bolivia': 'BO', 'Brunei': 'BN', 'Cape Verde':'CV', 
                 "Cote d'Ivoire": 'CI', "Democratic Republic of Congo": 'CD',
                 "East Timor": 'TL', "Russia": 'RU',  "Taiwan": 'TW', "Turkey":"TR"}

# countries where the offical name is not listed in geopy but the common name
weird_countries = ['South Korea', 'Taiwan', 'Bolivia', "Iran", "Moldova", "North Korea", 'Tanzania', 'Venezuela']

# get countrie coordinates
countries = pd.DataFrame()
countries['country'] = df_default['country'].unique()
countries[['latitude', 'longitude']] = countries['country'].apply(get_coordinates).apply(pd.Series)

# merge latitdue and longitude with original data set
df_new = pd.merge(df_default, countries, on = 'country')

# save new data set
df_new.to_csv("Data/data_clean/merge_loc.csv")