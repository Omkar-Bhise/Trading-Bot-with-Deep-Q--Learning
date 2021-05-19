"""
Created on Tue May 17 15:36:47 2021

@author: shekh

Here we extract data from twitter using the snscrape  python wrapper.
Since it is a scraping library, we divide each company's data into three time ranges 
in order to avoid we scraping issues.
"""
import snscrape.modules.twitter as sntwitter
import pandas as pd

dtelist = pd.date_range(start ='04-24-2020', 
         end ='04-24-2021', freq ='1D')

tweets_list2 = []
prev = str(dtelist[0])[0:10]
for dte in dtelist:
    start_date = str(dte)[0:10]
    search_query = '#apple since:'+prev+ ' until:'+start_date
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
        if i>100:
            break
        tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    prev = start_date
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
tweets_df2.to_csv("Twitter_dataApple2020.csv", index=False)