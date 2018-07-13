"""
Marc Simpson
07/13/2018
Datalogue Open Source Twitter Project

Takes in tweets from designated user and decides whether the
tweets are Positive/Negative/Neutral using Sentiment Analysis.
It also collects the hashtags from the tweets collected and
creates a word cloud of the hashtags.

Make sure to have updated python, Anaconda/miniconda and install all
necessary libraries before starting. (pip install OR conda install)

Update secrets.py and setup a Twitter application (see README from original project)

To run:
"python senitment_analysis.py -n <user_name> -l <desired number of tweets>"

(1,000 tweets is default)
"""

#Imported libraries some new and most from tweets_analyzer.py
import tweepy
import json
import pandas as pd
import numpy as np
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib as mpl
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import operator
from textblob import TextBlob
import re
import sys
import warnings

#This ignores all warnings of issues that will be fixed in future versions
if not sys.warnoptions:
    warnings.simplefilter("ignore")

#This next chunk is taken from tweets_analyzer.py
__version__ = '0.2-dev'

from secrets import consumer_key, consumer_secret, access_token, access_token_secret


parser = argparse.ArgumentParser(description=
"Simple Twitter Profile Analyzer (https://github.com/x0rz/tweets_analyzer) version %s" % __version__,
 usage='%(prog)s -n <screen_name> [options]')
parser.add_argument('-l', '--limit', metavar='N', type=int, default=1000, help=
'limit the number of tweets to retreive (default=1000)')
parser.add_argument('-n', '--name', required=True, metavar="screen_name", help='target screen_name')

args = parser.parse_args()

#Sets up the Twitter API with our keys
def Setup_Twitter():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    return api

#Create an extractor object to extract and make a list of the tweets
extract_tweet = Setup_Twitter()
tweets = extract_tweet.user_timeline(screen_name=args.name, total=args.limit)

#Creates the dataframe to store the tweet data
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

#Create the columns to store the tweet data in our dataframe
data['len']  = np.array([len(tweet.text) for tweet in tweets])
data['ID']   = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])


#Cleans the tweet text by getting rid of characters/links
def Clean_Tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

#Sentiment Analysis function on the tweets
def Analyze_Sentiment(tweet):
    analysis = TextBlob(Clean_Tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

#New column in our dataframe to store the tweet sentiments
data['SA'] = np.array([ Analyze_Sentiment(tweet) for tweet in data['Tweets'] ])

#Labels the tweets according to their sentiment
positive_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neutral_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
negative_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]


"""Show we have done everything up to this point correctly by printing the percentages
of the sentiments of the tweets"""

print("Percentage of positive tweets: {}%".format(len(positive_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neutral_tweets)*100/len(data['Tweets'])))
print("Percentage of negative tweets: {}%".format(len(negative_tweets)*100/len(data['Tweets'])))

#Print the tweets categorized by the sentiment
print("\n\nPositive tweets:")
for tweet in positive_tweets[:1]:
    print(positive_tweets)

print("\n\nNegative tweets:")
for tweet in negative_tweets[:1]:
    print(negative_tweets)

print("\n\nNeutral tweets:")
for tweet in neutral_tweets[:1]:
    print(neutral_tweets)

"""At this point I wanted to do some data visualization, but for the sentiment
analysis above it would just be a chart of some sort...

So I decided to collect the hashtags from our collected tweets and create
a Word Cloud of these tweets. """

#New column in our dataframe to store the hashtags of the tweets we collected
data["HT"] = [tweet.entities.get('hashtags') for tweet in tweets]

#Extracting the text from the hashtag column of the dataframe
HT_dataframe = pd.DataFrame()
a = 0

for tweet in range(0,len(tweets)):
    hashtag = tweets[tweet].entities.get('hashtags')
    for i in range(0,len(hashtag)):
        Htag = hashtag[i]['text']
        HT_dataframe.set_value(a,'Hashtag',Htag)
        a = a+1

#Join all the text from the hashtags
Combined_HT = " ".join(HT_dataframe['Hashtag'].values.astype(str))

#Removes unnecessary items from the text (URLs, RTs, and twitter handles)
New_WC = " ".join([word for word in Combined_HT.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])

"""Grab the .png of the image you want to use for the word cloud from the
CORRECT path where it is saved (I chose an airplane)"""
HT_Image = imread("C:\\Users\\Marc\\Documents\\Git\\Twitter_project\\tweets_analyzer\\plane_mask.png", flatten=True)

#Create the Word Cloud and save it to designated area
wc = WordCloud(background_color="white", stopwords=STOPWORDS, mask = HT_Image)
wc.generate(New_WC)
plt.imshow(wc)
plt.axis("off")
plt.savefig("C:\\Users\\Marc\\Documents\\Git\\Twitter_project\\tweets_analyzer\\HT_WCloud.png", dpi=300)
plt.show()
