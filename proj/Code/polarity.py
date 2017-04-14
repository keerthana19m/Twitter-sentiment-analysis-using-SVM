import sys
import json
from pprint import pprint
    
#determine the polarity of each tweet. from the deature vector generated for each tweet, compare if the elements of the feature vector exists in the corpus
#and if they do, sum the impact value for the particular element. this way sum for all the elements in each feature vector

def calculatescores(featureVector):
	  
    afinnfile = open('AFINN-111.txt', 'r')

    scores = {} # initialize an empty dictionary
    for line in afinnfile:
        term, score  = line.split("\t")  
        scores[term] = int(score)  
    #print scores.items()

    newList=[]
    sum=0

    for word in featureVector:
        if word in scores:
            sum=sum+scores.get(word)
            newList.append(word)
    return sum
    
    

	
