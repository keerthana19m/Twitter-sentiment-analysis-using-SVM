import re
import tweepy
import pickle
import sys
import time
import nltk
import csv
from tweepy import OAuthHandler
from textblob import TextBlob
from preprocessing import *
from polarity import *
from svmdem import *
from sklearn import svm
from tabulate import tabulate

#class twitter data contains methods for twitter authentication, cleaning the tweet and get tweet sentiment
class TwitterData(object):
  
    def __init__(self):
      
        # keys and tokens for authentication
        consumer_key = 'OkahnyrW1x41rMTflVxPXqScu'
        consumer_secret = '5Eyyt3yCqWNe6UCj90RxaFR3UvdmkqdEJQWeQHmfxODQUEfYy1'
        access_token = '262111773-vcR4O5lG0Udyu9bKOvpWowWJtiVCmlvHZBn9iLbt'
        access_token_secret = 'YS3yu9uhqY05Gwxw3CfABRFElPvMlLnDKyGIuhXM8fme6'
 
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")


    #remove links and special characters from tweet using regex
    def cleaning(self, tweet):
    
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
 
    # set sentiment using textblobs sentiment method
    def get_tweet_sentiment(self, tweet):
        
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.cleaning(tweet))
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    #function to fetch the tweets and parse them
    def get_tweets(self, query, count ):
    
        # empty list to store parsed tweets
        tweets = []
 
        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q = query, count = count)
 
            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}
 
                # saving text of tweet
                parsed_tweet['text'] = self.cleaning(tweet.text)
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
 
                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
 
            # return parsed tweets
            return tweets
 
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
    


#main method which prompts user to enter the value for hashtags and performs calls to clean, preprocess,generate feature vector and to determine
#sentiment and polarity of each tweet in the tweets file
#Also makes call to svm function 
 
def main():
    print("\n*****************************************************\n")
    print("Twitter Sentiment Analysis")
    print("\n*****************************************************\n")
    # creating object of TwitterData Class
    api = TwitterData()
    # calling function to get tweets
    hashtag = raw_input('Enter a hashtag for which tweets are to be obtained: ')
    print("Collecting live tweets from twitter...")
    tweets = api.get_tweets(query = hashtag, count = 200)
    time.sleep(5)
    print("READY")
    time.sleep(5)
    keys = tweets[0].keys()
    with open('feeds.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(tweets)
    inpTweets = csv.reader(open('feeds.csv', 'rb'), delimiter=',')
    new_tweets = []
    st = open('stopwords.txt', 'r')
    stopWords = getStopWordList('stopwords.txt')
    next(inpTweets) #skip the header
    print("Preprocessing Tweets and Obtaining Feature Vectors...")
    i = 0
    for row in inpTweets:
        sentiment = row[1]
        tweet = row[0]
        print("\n*****************************************************\n")
        print("Original Tweet(Cleaned) : " + tweet)
        processedTweet = processTweet(tweet)
        print("\n\nPre-Processed Tweet#%d: " %i)
        i = i + 1
        print(processedTweet)
        print("\n")
        featureVector = getFeatureVector(processedTweet,stopWords)
        print ("Extracted Feature Vector:")
        print(featureVector)
        print("\n")
        polarityscore = calculatescores(featureVector)
        print ("Polarity Score:" )
        print(polarityscore)
        print("\n\n")
        new_tweets.append((featureVector, sentiment, polarityscore));
        lis_tweets = [list(elem) for elem in new_tweets]


    print("\n*****************************************************\n")
    print("Result of feature vector and polarity calculation: ")
    print("\n")
    print(tabulate(lis_tweets))
    print("\n*****************************************************\n")
    print("Training SVM Classifier....\n")
    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # percentage of positive tweets
    print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # percentage of negative tweets
    print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
    # percentage of neutral tweets
    print("Neutral tweets percentage: {} %  ".format(100*(len(tweets)-len(ntweets)-len(ptweets))/len(tweets)))
    trainsvm(lis_tweets)

   


if __name__ == "__main__":
    # calling main function
    main()