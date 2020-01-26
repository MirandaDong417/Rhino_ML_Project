import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")



def clean_train_data(trainset_fname):
	# trainset_fname is a csv file

	print(f"LOADING TRAINING DATA FROM FILE {trainset_fname}")

	# load the dataset for training 
	data = pd.read_csv(trainset_fname,sep = None)

	# make sure column names have no space in the front or end
	data.columns = data.columns.str.strip()

	# sex index as ID 
	data = data.set_index('ID')

	# Since our target variable is column "is booking", we need to drop the rows where "is booking" is null.
	data = data.dropna(subset = ['is booking'])

	# Only keep the columns that are available before lead score prediction, plus the target variable "is booking"
	df = data[['Language',
	 'Website',
	 'created',
	 'Client email domain',
	 'Enquiry type',
	 'Enquiry status',
	 'Currency',
	 'Client budget',
	 'Arrival date',
	 'Arrival date text',
	 'Departure date',
	 'Num nights',
	 'Stay duration',
	 'Adults',
	 'Children',
	 'Flights booked',
	 'Residential country code',
	 'Detected country code',
	 'Detected city',
	 'Budget value',
	 'Click path',
	 'User agent',
	 'User repeat',
	 'User referral',
	 'GA source',
	 'GA medium',
	 'Device',
	 'GA keyword',
	 'GA campaign',
	 'GA language',
	 'GA country',
	 'Session duration',
	 'is booking',
	 'Sessions',
	 'Avg. session length (sec)',
	 'Avg. pageviews per session',
	 'Pageviews',
	 'Hits']]

	# column "Click path", remove the strings that have no semantic value, such as 'url','www','<li>'
	# convert all text to lowercase
	df['Click path'] = df['Click path'].str.replace('url',' ').str.replace('http',' ').str.replace(
	    'www',' ').str.replace('com',' ').str.replace('<ol>',' ').str.replace('</ol>',' ').str.replace(
	    '<li>',' ').str.replace('</li>',' ').str.replace('s://',' ').str.strip().str.lower()


	# column "created", change the data type to datetime64[ns]
	# then create a new column specifying month of enquiry submission
	df['created'] = pd.to_datetime(df['created'], errors = 'coerce')
	df['Created month'] = df['created'].dt.month_name()


	# column "Arrival date", "Departure date", change data type to datetime64[ns]
	df['Arrival date'] = pd.to_datetime(df['Arrival date'], dayfirst = True,errors = 'coerce')
	df['Departure date'] = pd.to_datetime(df['Departure date'], dayfirst = True,errors = 'coerce')


	# Create a column "Created to arrival" that contains the time difference between "Created" and "Arrival date"
	df["Created to arrival"] = df['Arrival date'].subtract(df['created'])/np.timedelta64(1,'D')

	# There are some rows that have no "Arrival date" value but have "Arrival date text" and "created" values
	# Now we can try calculate more values for "Created to arrival" by subtracting "created" from a cleaned version of "Arrival date text"

	# Since most of the values in "Arrival date text" are in qualified "month-abbrev two-digit-year" format
	# We will drop all of the rest disqualified values and change this column into a datetime 64 datatype
	df['Arrival date text'] = pd.to_datetime(df['Arrival date text'], format='%b %y',errors = 'coerce')
	useful_adt_rows = (df['Created to arrival'].isnull()) & (df['Arrival date text'].notnull()) & (df['created'].notnull())


	# There are some rows that dont have "Created to arrival" yet have "Arrival date text" and "created"
	# So we'll calculate the time difference for these rows and fill in the "Created to arrival" column
	df.loc[useful_adt_rows,'Created to arrival'
	      ] = df['Arrival date text'].subtract(df['created'])/np.timedelta64(1,'D')

	# some time differences are negative numbers, which make the 'Arrival date','Departure date' data invalid 
	# so we need to drop 'Arrival date','Departure date','Created to arrival' data for rows with negative time difference
	df.loc[df['Created to arrival'] < 0,['Arrival date','Departure date','Created to arrival']] = np.nan

	# now we can drop the column "created"
	df = df.drop(columns = 'created')

	# we can also drop the columns "Arrival date","Arrival date text", and "Departure date"
	df = df.drop(columns= ['Arrival date','Arrival date text','Departure date'])


	# column "Stay duration" may help us fill in more null values for "Num nights"
	# A lot of rows of "Stay duration" are numbers, so we can change these to float type and fill them into "Num nights" column correspondingly
	isnumber = df['Stay duration'].str.isnumeric() == True
	df.loc[isnumber, 'Num nights'] = pd.to_numeric(df.loc[isnumber,'Stay duration'])

	# For "Stay duration", we can fill in nan values for the rows that 
	# have "Num nights" value and focus on the rest of the non-empty rows
	df.loc[(df['Num nights'].notnull()), 'Stay duration'] = np.nan

	# A lof of cells contain strings ending with "nights" or "day" or "days day", so we can strip these strings off
	df['Stay duration'] = df['Stay duration'].str.rstrip(' nights').str.rstrip(' day').str.rstrip(' days day')

	# many cells contains "-" to connect the range of two numbers
	# for these cells, we split the cell based on "-", for those that are split into two strings, we assume they represent the range of the stay duration
	# and calculate the mean value for this range by first convert them into float

	dash = df['Stay duration'].str.contains('-',na=False) == True
	df['Stay duration'] = df.loc[dash,'Stay duration'].str.split('-')

	def avehelper(x):
	    if type(x) == list:
	        if len(x) ==2:
	            if x[0].strip().isdigit() and x[1].strip().isdigit():
	                return 1
	        return x

	qualified_helper = df['Stay duration'].apply(avehelper) == 1
	df.loc[qualified_helper,'Num nights'] = df.loc[qualified_helper]['Stay duration'].apply(
	    lambda x: (int(x[1].strip()) - int(x[0].strip()))/2)

	# Since the rest of the rows of "Stay duration" are in very messy format, we ignore these values and drop the column "Stay duration"
	df = df.drop(columns = 'Stay duration')


	# column "Budget value", current data type is object
	# some rows contain numbers but are in string format, we need to change them to float
	# there are also  some rows that are not numbers, we need to change these values to NaN
	def is_number(s):
		try:
			float(s)
			return True
		except ValueError:
			return False

	isnum = ((df['Budget value'].apply(type) == str) & (df['Budget value'].apply(is_number)== True))
	notnum = ((df['Budget value'].apply(type) == str) & (df['Budget value'].apply(is_number)== False))
	df.loc[isnum ,'Budget value'] = pd.to_numeric(df.loc[isnum,'Budget value'])
	df.loc[notnum ,'Budget value'] = np.NaN

	# There are some rows that have budget value of 0. We assume that it's non applicable and set them to NaN
	df.loc[df['Budget value'] == 0, 'Budget value'] = np.nan

	# there are also some rows that have budget value but no currency specified. We need to change these budget values to NaN
	df.loc[(df['Budget value'].notnull()) & (df['Currency'].isnull()), 'Budget value'] = np.nan


	# unify 'Currency' as all upper cases
	df['Currency'] = df['Currency'].str.upper()

	# We are using an API, Alpha Vantage, to get realtime currency exchange rates, and create a mapping dictionray of exchange rates
	# Since it only supports 5 requests every one minute, we do it in two parts and wait 61 seconds in between
	import os
	import pandas_datareader.data as web
	import time 

	currency_part1 = web.DataReader(["USD/EUR","USD/GBP","USD/ZAR"], "av-forex", 
	                    access_key= 'A8MBHNJSQGBPPBEX')
	print('wait 61 sec for Alpha Vantange API to get realtime currency exchange rates')
	time.sleep(61)
	print('finish waiting; continue running')

	currency_part2 = web.DataReader(["USD/BRL","USD/AUD","USD/CAD","USD/CHF"], "av-forex", 
	                    access_key= 'A8MBHNJSQGBPPBEX')
	currency_dataframe = pd.concat([currency_part1,currency_part2],axis = 1)
	currency_dataframe.loc['Exchange Rate'] = pd.to_numeric(currency_dataframe.loc['Exchange Rate'])
	currency_dataframe.columns = currency_dataframe.columns.str.lstrip('USD/')
	currency_dict = currency_dataframe.loc['Exchange Rate'].to_dict()
	currency_dict['USD'] = 1
	df['Currency'] = df['Currency'].map(currency_dict)

	# Calculate the "Budget value" in US Dollars by dividing its original value by the excahnge rate in "Currency" column
	df.loc[(df['Currency'].notnull()) & (df['Currency'] != 1),
	       'Budget value'] = df['Budget value'] / df['Currency']
	df = df.round({'Budget value':1})

	# now we can drop the "Currency" column
	df = df.drop(columns = 'Currency')


	# column "Budget value" use variable grouping group them into 3 categories "Luxury","Standard", and "Value" 
	# based on the distribution of "Luxury","Standard", and "Value" in column "Client budget"
	# then we cobime columns "Client budget" with "Budget value" 

	# distribution of "Luxury","Standard", and "Value" in column "Client budget"
	budget_dis = (df['Client budget'].value_counts())/(df[df['Client budget'].notnull()].shape)[0]
	budget_cumu_dis = [0,budget_dis['Value'],budget_dis['Value'] + budget_dis['Standard'],1]

	df.loc[df['Client budget'].isnull(),'Client budget'] = pd.qcut(
	    df['Budget value'], budget_cumu_dis, labels = ['Value','Standard','Luxury'])

	# now we can drop the column "Budget value"
	df = df.drop(columns = 'Budget value')


	# There are a few rows whose Detected country code is different from Residential country code
	# We assume that the Residential country code is incorrect (for example, for those whose detected city are in California.
	# the residential code is CA, which makes the residential code less credible)
	# Therefore, we combine the two columns and prefer the "Detected country code" when the two column values conflict
	df.loc[(df['Detected country code'].isnull()) & (df['Residential country code'].notnull()),
	       'Detected country code'] = df['Residential country code']

	# now we can drop "Residential country code" column and rename "Detected country code" as "Country code"
	df = df.drop(columns = 'Residential country code')
	df = df.rename({'Detected country code': 'Country code'}, axis=1)


	# "GA keyword" column has "(not set)" and "(not provided)" values, 
	# which should be replaced by None
	df.loc[df['GA keyword'] == "(not set)",'GA keyword'] = None
	df.loc[df['GA keyword'] == "(not provided)",'GA keyword'] = None

	# convert all text to lower case
	df['GA keyword'] = df['GA keyword'].str.lower()
	df['GA keyword'] = df['GA keyword'].fillna('None').astype('U').values


	# 'GA source' convert text data to lower case
	# strip words like 'com', 'net' that have no semantic value
	df['GA source'] = df['GA source'].str.lower()


	# Since "GA language" has the same meaning as "Language", and "Language" column fewer empty rows
	# We will drop "GA language" column
	df = df.drop(columns = 'GA language')


	# Since "GA country" has the same meaning as "Country code", and "Country code" column fewer empty rows
	# We will drop "GA country" column
	df = df.drop(columns = 'GA country')


	# The model building process shows that dropping the column "Detected city", "Client email domain", and "GA campaign" will lead to a higher metric
	# so we drop the column "Detected city", "Client email domain", and "GA campaign" 
	df = df.drop(columns = ['Detected city','Client email domain', 'GA campaign'])


	# Finally, delete the rows where target variable "is booking" is neither 1 nor 0
	df = df.drop(df[(df['is booking'] != 0)&(df['is booking'] != 1)].index)
	df['is booking'] = df['is booking'].astype('category')


	X = df.drop('is booking',axis = 1)
	y = df['is booking']
	return X, y











def clean_predict_data(input_data):
	# the input_data is a pandas dataframe

	input_data.columns = input_data.columns.str.strip()
	input_data = input_data.set_index('ID')

	# Only keep the columns that are available before lead score prediction, and those that have been used in model training 
	df = input_data[['Language',
	 'Website',
	 'created',
	 'Enquiry type',
	 'Enquiry status',
	 'Currency',
	 'Client budget',
	 'Arrival date',
	 'Arrival date text',
	 'Departure date',
	 'Num nights',
	 'Stay duration',
	 'Adults',
	 'Children',
	 'Flights booked',
	 'Residential country code',
	 'Detected country code',
	 'Budget value',
	 'Click path',
	 'User agent',
	 'User repeat',
	 'User referral',
	 'GA source',
	 'GA medium',
	 'Device',
	 'GA keyword',
	 'Session duration',
	 'Sessions',
	 'Avg. session length (sec)',
	 'Avg. pageviews per session',
	 'Pageviews',
	 'Hits']]


	# cleaning data, similar to the cleaning process in model training above,
	# but since we don't have the variable "is booking" when we want to predict lead score
	# we delete everything related to "is booking", and keep the rest the same.

	df['Click path'] = df['Click path'].str.replace('url',' ').str.replace('http',' ').str.replace(
	    'www',' ').str.replace('com',' ').str.replace('<ol>',' ').str.replace('</ol>',' ').str.replace(
	    '<li>',' ').str.replace('</li>',' ').str.replace('s://',' ').str.strip().str.lower()


	df['created'] = pd.to_datetime(df['created'], errors = 'coerce')
	df['Created month'] = df['created'].dt.month_name()

	df['Arrival date'] = pd.to_datetime(df['Arrival date'], dayfirst = True,errors = 'coerce')
	df['Departure date'] = pd.to_datetime(df['Departure date'], dayfirst = True,errors = 'coerce')

	df["Created to arrival"] = df['Arrival date'].subtract(df['created'])/np.timedelta64(1,'D')

	df['Arrival date text'] = pd.to_datetime(df['Arrival date text'], format='%b %y',errors = 'coerce')
	useful_adt_rows = (df['Created to arrival'].isnull()) & (df['Arrival date text'].notnull()) & (df['created'].notnull())

	df.loc[useful_adt_rows,'Created to arrival'
	      ] = df['Arrival date text'].subtract(df['created'])/np.timedelta64(1,'D')

	df.loc[df['Created to arrival'] < 0,['Arrival date','Departure date','Created to arrival']] = np.nan

	df = df.drop(columns = 'created')
	df = df.drop(columns= ['Arrival date','Arrival date text','Departure date'])




	isnumber = df['Stay duration'].str.isnumeric() == True
	df.loc[isnumber, 'Num nights'] = pd.to_numeric(df.loc[isnumber,'Stay duration'])

	df.loc[(df['Num nights'].notnull()), 'Stay duration'] = np.nan
	
	df['Stay duration'] = df['Stay duration'].str.rstrip(' nights').str.rstrip(' day').str.rstrip(' days day')

	dash = df['Stay duration'].str.contains('-',na=False) == True
	df['Stay duration'] = df.loc[dash,'Stay duration'].str.split('-')

	def avehelper(x):
	    if type(x) == list:
	        if len(x) ==2:
	            if x[0].strip().isdigit() and x[1].strip().isdigit():
	                return 1
	        return x

	qualified_helper = df['Stay duration'].apply(avehelper) == 1
	df.loc[qualified_helper,'Num nights'] = df.loc[qualified_helper]['Stay duration'].apply(
	    lambda x: (int(x[1].strip()) - int(x[0].strip()))/2)

	df = df.drop(columns = 'Stay duration')

	
	def is_number(s):
		try:
			float(s)
			return True
		except ValueError:
			return False

	isnum = ((df['Budget value'].apply(type) == str) & (df['Budget value'].apply(is_number)== True))
	notnum = ((df['Budget value'].apply(type) == str) & (df['Budget value'].apply(is_number)== False))
	df.loc[isnum ,'Budget value'] = pd.to_numeric(df.loc[isnum,'Budget value'])
	df.loc[notnum ,'Budget value'] = np.NaN

	df.loc[df['Budget value'] == 0, 'Budget value'] = np.nan
	df.loc[(df['Budget value'].notnull()) & (df['Currency'].isnull()), 'Budget value'] = np.nan

    

	df['Currency'] = df['Currency'].str.upper()

	import os
	import pandas_datareader.data as web
	import time 
	currency_part1 = web.DataReader(["USD/EUR","USD/GBP","USD/ZAR"], "av-forex", 
	                    access_key= 'A8MBHNJSQGBPPBEX')
	print('wait 61 sec for Alpha Vantange API to get realtime currency exchange rates')
	time.sleep(61)
	print('finish waiting; continue running')
	currency_part2 = web.DataReader(["USD/BRL","USD/AUD","USD/CAD","USD/CHF"], "av-forex", 
	                    access_key= 'A8MBHNJSQGBPPBEX')
	currency_dataframe = pd.concat([currency_part1,currency_part2],axis = 1)
	currency_dataframe.loc['Exchange Rate'] = pd.to_numeric(currency_dataframe.loc['Exchange Rate'])
	currency_dataframe.columns = currency_dataframe.columns.str.lstrip('USD/')
	currency_dict = currency_dataframe.loc['Exchange Rate'].to_dict()
	currency_dict['USD'] = 1
	df['Currency'] = df['Currency'].map(currency_dict)

	df.loc[(df['Currency'].notnull()) & (df['Currency'] != 1),
	       'Budget value'] = df['Budget value'] / df['Currency']
	df = df.round({'Budget value':1})
	df = df.drop(columns = 'Currency')


	budget_dis = (df['Client budget'].value_counts())/(df[df['Client budget'].notnull()].shape)[0]
	budget_cumu_dis = [0,budget_dis['Value'],budget_dis['Value'] + budget_dis['Standard'],1]

	df.loc[df['Client budget'].isnull(),'Client budget'] = pd.qcut(
	    df['Budget value'], budget_cumu_dis, labels = ['Value','Standard','Luxury'])
	df = df.drop(columns = 'Budget value')


	df.loc[(df['Detected country code'].isnull()) & (df['Residential country code'].notnull()),
	       'Detected country code'] = df['Residential country code']
	df = df.drop(columns = 'Residential country code')
	df = df.rename({'Detected country code': 'Country code'}, axis=1)


	df.loc[df['GA keyword'] == "(not set)",'GA keyword'] = None
	df.loc[df['GA keyword'] == "(not provided)",'GA keyword'] = None
	df['GA keyword'] = df['GA keyword'].str.lower()
	df['GA keyword'] = df['GA keyword'].fillna('None').astype('U').values


	df['GA source'] = df['GA source'].str.lower()

	return df






def save_pipeline(pipeline):
	"""exports a pipeline with a timestamp appended"""
	print("SAVING TRAINED PIPELINE")
	## below is another method that I've tried, but neither of them worked in terminal
	## The joblib method can work in jupyter notebook 
	# pickle.dump(pipeline,open('pipeline.pkl','wb'))
	joblib.dump(pipeline, "pipeline.pkl")


def load_pickle(pickle_file):
	# return pickle.loads(pickle_file)
	return joblib.load(pickle_file)











