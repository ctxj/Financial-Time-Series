#Imports
from aiohttp import ClientSession
from itertools import chain
import pandas as pd
import asyncio

#Stock ticker and dates for data required
#IEX api key
start = '2019/04/01' #earliest date available on non-premium IEX accounts
end = '2020/11/27'
key = 'IEX API KEY' #enter api key from IEX
ticker = 'FB'

#Convert dates into api url format
daterange = pd.date_range(start=start, end=end).strftime('%Y%m%d')

#Function for asynchronous api request
async def fetch(session, url):
    async with session.get(url) as response:
        data = await response.json()
        return data

async def main(daterange):
    async with ClientSession() as session:
        tasks = []
        for date in daterange:
            tasks.append(
                asyncio.create_task(
                    fetch(session, f'https://cloud.iexapis.com/stable/stock/{ticker}/chart/date/{date}?token={key}&chartIEXOnly=true',)))
        content = await asyncio.gather(*tasks, return_exceptions=True)
        return content

#Unable to make more than 30 request at a time
#Function to breakout dates into batches
def batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

batch_dates = batch(list(daterange), 30) #List of dates in batches of 30

df = pd.DataFrame()

#Loop through dates in batches
for dates in list(batch_dates):
    results = asyncio.get_event_loop().run_until_complete(main(dates))
    df2 = pd.DataFrame(list(chain.from_iterable(results)))
    df = df.append(df2)

df.to_csv(f'{ticker}.csv', index=False)

