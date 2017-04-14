import re
import csv
import pprint
import nltk.classify

def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    patt = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return patt.sub(r"\1\1", s)

#process_tweet function converts tweets to lower case, replace links with URL and user with AT_USER, removes additional white spaces and removes #
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet


#getStopWordList function uses the stopwords.txt file to remove the words which do not display any emotion and are rendered trivial for the analysis.
#rt and url are appended to remove retweets and urls too
def getStopWordList(fname):
    #read the stopwords
    stopWords = []
    stopWords.append('rt')
    stopWords.append('url')
    stopWords.append('at_user')

    fp = open(fname, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords


#feature vector generation - split the tweet into words and if there are two or more occurences of a word like yay yay or okayyyyy - replace it with yay and okay
#remove punctuations if any
#remove stop words from the tweet
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector    

