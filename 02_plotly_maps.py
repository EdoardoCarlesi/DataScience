import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cufflinks as cf
import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
from plotly import __version__

init_notebook_mode(connected = True)
data_path = '/home/edoardo/Udemy/DataScience/Dispense/09-Geographical-Plotting/'

agri = '2011_US_AGRI_Exports'
gdp = '2014_World_GDP'
election = '2012_Election_Data'
power = '2014_World_Power_Consumption'

print('\n\n\n\n')

df = pd.read_csv(data_path + election)
print(df.head(0))

data = dict(type='choropleth',
        colorscale='Viridis', reversescale = True,
        locations=df['State Abv'],
        locationmode='USA-states',
        z=df['Voting-Age Population (VAP)'],
        text=df['State'],
        colorbar={'title':'Voting Age Population'},
        marker=dict(line=dict(color = 'rgb(25, 25, 25)', width=3))
        )

layout = dict(title = '2012 VAP',  
            geo = dict(scope='usa', 
            showlakes = True,
            lakecolor='rgb(85, 173, 240)'))

choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)
choromap.write_html('US_VotingAge.html')

'''


df = pd.read_csv(data_path + power)
print(df.head(0))

data = dict(type = 'choropleth',
            locations = df['Country'],
            locationmode = "country names",
            z = df['Power Consumption KWH'],
            text = df['Text'], 
            colorbar = {'title':'Power Consumption KWH'})

print(df['Power Consumption KWH'])

geo = dict(showframe = False,
        projection = {'type':'mercator'})

layout = dict(title = '2014 Power Consumption', geo = geo)

choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)
choromap.write_html('World_power_consumption.html')

'''


'''
    WORLD MAPS


df = pd.read_csv(data_path + gdp)
print(df.head())

data = dict(type = 'choropleth',
            locations = df['CODE'],
            z = df['GDP (BILLIONS)'],
            text = df['COUNTRY'], 
            colorbar = {'title':'GDP Billion USD'})

geo = dict(showframe = False,
        projection = {'type':'mercator'})

layout = dict(title = '2014 Global GDP', geo = geo)

choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)
choromap.write_html('World_test.html')

'''

'''

    US DATA

df = pd.read_csv(data_path + agri)
print(df.head(0))
data = dict(type='choropleth',
        colorscale='Portland',
        locations=df['code'],
        locationmode='USA-states',
        z=df['total exports'],
        text=df['text'],
        colorbar={'title':'Millions USD'},
        marker=dict(line=dict(color = 'rgb(25, 25, 25)', width=5))
        )

layout = dict(title = '2011 US Agro Export',  
            geo = dict(scope='usa', 
            showlakes = True,
            lakecolor='rgb(85, 173, 240)'))

choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)
choromap.write_html('USA_test.html')
'''


'''
data = dict(type = 'choropleth',
        locations = ['AZ', 'CA', 'NY'], 
        locationmode = 'USA-states',
        colorscale = 'Portland',
        text = ['Arizona', 'Cali', 'New York'],
        z = [1.0, 2.0, 3.0], 
        colorbar = {'title': 'ColorBar Title'})

layout = dict(geo = {'scope': 'usa'})

choromap = go.Figure(data = [data], layout = layout)

iplot(choromap)
choromap.write_html('USA_test.html')
'''
