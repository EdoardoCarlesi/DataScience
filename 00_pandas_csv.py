import pandas as pd
from sqlalchemy import create_engine


path = '/home/edoardo/Udemy/DataScience/CovidData/World/csse_covid_19_data/csse_covid_19_time_series/'
name = 'time_series_covid19_confirmed_global.csv'

file_csv = path + name

df = pd.read_csv(path + name) #.to_csv(index = False)


#country = 'Italy'
#print(df.loc[df['Country/Region'] == country])
#print(df.to_csv(index = False))
#html_path = 'https://www.fdic.gov/bank/individual/failed/banklist.html'
#df2 = pd.read_html(html_path)

engine = create_engine('sqlite:///:memory:')
df.to_sql('my_table', engine)

sqldf = pd.read_sql('my_table', con=engine)

print(sqldf)
#print(df2[0].head())

