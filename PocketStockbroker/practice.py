import csv

# Simply opens the file and prints the data
with open('AAPL.csv') as RawStockData:
    applePrices = csv.reader(RawStockData)
    for row in applePrices:
        print(row)


"""
import pandas as pd
from pandas_datareader import data as web  # Package and modules for importing data; this code may change depending on pandas version
import datetime

# We will look at stock prices over the past year, starting at January 1, 2016
start = datetime.datetime(2016,1,1)
end = datetime.date.today()

# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
apple = web.DataReader("AAPL", "google", start, end)

type(apple)

apple.head()




"""

"""
# Simple web scraper that stores stock prices in CSV file

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

# Evaluates if webpage response is valid
def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


# Prints errors to the console; TODO: Timestamp, write to errors.txt file
def log_error(e):
    # Prints the error to the console
    print(e)


# Returns contents of webpage if valid format (HTML, XML, etc.)
def simple_get_url(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


# Attempts to scrape MarketWatch for stock prices
raw_html = simple_get_url('https://www.marketwatch.com/investing/future/crude%20oil%20-%20electronic')
html = BeautifulSoup(raw_html, 'html.parser')
# Prints the raw HTML to the screen
for h in html.select('h3'):

"""
