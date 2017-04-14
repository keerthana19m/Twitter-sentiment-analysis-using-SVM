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
from sklearn import svm
from gettwitterinput import *
from tabulate import tabulate
from collections import Counter

#using the sentiment and polarity of each tweeet, determine the emotion and train the svm. also test the classifier by prompting user to enter any tweet and see the prediction
def trainsvm(new_tweets):

	#training data
	labels = []
	samples = []
	samples = new_tweets

	for tweet in new_tweets:
		senti = tweet[1] 
		polar = tweet[2] 
		if senti == "neutral" and polar in [0,1,-1]:
			labels.append('disturbed')
		elif senti == "neutral" and polar == 2:
			labels.append('glad')
		elif senti == "neutral" and polar == -2:
			labels.append('frown')
		elif senti == "neutral" and polar == 3:
			labels.append('chill')
		elif senti == "neutral" and polar == -3:
			labels.append('disappointed')
		elif senti == "positive" and polar in [0,1,-1]:
			labels.append('cheerful')
		elif senti == "positive" and polar == 2:
			labels.append('happy')
		elif senti == "positive" and polar == -2:
			labels.append('frown')
		elif senti == "positive" and polar == -3:
			labels.append('disappointed')
		elif senti == "positive" and polar == 3:
			labels.append('Excited')
		elif senti == "negative" and polar in [0,1,-1]:
			labels.append('anxious')
		elif senti == "negative" and polar == 2:
			labels.append('frown')
		elif senti == "negative" and polar == -2:
			labels.append('sad')
		elif senti == "negative" and polar == 3:
			labels.append('disappointed')
		elif senti == "negative" and polar == -3:
			labels.append('deeply depressed')
		elif senti == "negative" and polar < -3:
			labels.append('Needs help!')
		elif senti == "neutral" and polar < -3:
			labels.append('Needs help!')
		elif senti == "neutral" and polar > 3:
			labels.append('over the clouds')
		elif senti == "positive" and polar > 3:
			labels.append('over the clouds')
		else:
			labels.append('neutral')

	#print(labels)
	c=Counter(labels)
	#print(c)
	print("Percentage of emotions detected in the training set : ")
	print("------------------------------------------------------")
	print("Disturbed tweets percentage: {} %".format(100*c['disturbed']/len(labels)))
	print("cheerful tweets percentage: {} %".format(100*c['cheerful']/len(labels)))
	print("anxious tweets percentage: {} %".format(100*c['anxious']/len(labels)))
	print("over the clouds tweets percentage: {} %".format(100*c['over the clouds']/len(labels)))
	print("Needs help tweets percentage: {} %".format(100*c['Needs help!']/len(labels)))
	print("Excited tweets percentage: {} %".format(100*c['Excited']/len(labels)))
	print("happy tweets percentage: {} %".format(100*c['happy']/len(labels)))
	print("frown tweets percentage: {} %".format(100*c['frown']/len(labels)))
	print("sad tweets percentage: {} %".format(100*c['sad']/len(labels)))
	print("deeply depressed tweets percentage: {} %".format(100*c['deeply depressed']/len(labels)))
	print("Glad tweets percentage: {} %".format(100*c['glad']/len(labels)))
	print("Frown tweets percentage: {} %".format(100*c['frown']/len(labels)))
	print("Chill tweets percentage: {} %".format(100*c['chill']/len(labels)))
	print("Disappointed tweets percentage: {} %".format(100*c['disappointed']/len(labels)))
	print("Neutral tweets percentage: {} %".format(100*c['neutral']/len(labels)))


	pol =[]
	for tweet in new_tweets:
		sen = tweet[1]
		if sen == "positive":
			val = 1
			pol.append((val,tweet[2]))
			feat_tweets = [list(elem) for elem in pol]

		elif sen == "neutral":
			val = 0
			pol.append((val,tweet[2]))
			feat_tweets = [list(elem) for elem in pol]
		else:
			val = 2
			pol.append((val,tweet[2]))
			feat_tweets = [list(elem) for elem in pol]


	#svm classification and training
	clf = svm.SVC(gamma=0.001, C=100.)
	clf.fit(feat_tweets,labels)
	print("Training done.\n")
	
	#for testing the code
	test_tweets =[]
	p_tweet = {} # empty dictionary to comprise of test inputs text and polarity 
	test_input = [] #push dict into a list
	st = open('stopwords.txt', 'r')
	stopWords = getStopWordList('stopwords.txt')
	testtweet = raw_input('Enter a tweet message to find its sentiment polarity: ') # get user input tweet for testing svm
	cleanedtest = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", testtweet).split()) # clean to remove symbols,emoticons
	print("Calculating Polarity of your tweet....")
	analysis = TextBlob(cleanedtest) #sentiment calculation
	p_tweet['text'] = cleanedtest	
	
	# determine the sentiment of the test tweet
	if analysis.sentiment.polarity > 0:
		p_tweet['sentiment'] = 'positive'

	elif analysis.sentiment.polarity == 0:
		p_tweet['sentiment'] = 'neutral'
	else:
		p_tweet['sentiment'] = 'negative'

	#preprocessing and polarity calc of test tweet
	procTweet = processTweet(p_tweet['text'])
	featVector = getFeatureVector(procTweet,stopWords)
	polscore = calculatescores(featVector)
	test_tweets.append((featVector, p_tweet['sentiment'], polscore));
	listed_tweets = [list(elem) for elem in test_tweets]


	#fetch the polarity and svm of user inputted tweet to feed into svm.svm only accepts numerical values and not categorical

	sen_user = listed_tweets[0][1] 
	pol_user = listed_tweets[0][2] 
	lis_svm = []

	if sen_user == "positive":

		val_user = 1
		lis_svm.append((val_user,pol_user))

	elif sen_user == "neutral":

	    val_user = 0
	    lis_svm.append((val_user,pol_user))
	else:

	    val_user = 2
	    lis_svm.append((val_user,pol_user))

	#create a list of polarity and sentiment of the user inputted tweet and predict the emotion
	send_svm = [list(elem) for elem in lis_svm]
	print("Emotion Analysed: ")
	print(clf.predict(send_svm))





  


