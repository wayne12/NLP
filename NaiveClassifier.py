import os
import collections
import string
import math
import pickle
import NgramMod

vocaName = 'storage/mailVocab.voc'
nonSpamName = 'storage/nonSpam.dic'
spamName = 'storage/spam.dic'

def LoadCorpus(directory):
	data = {}#define a dictionary
	for entry in os.listdir(directory):
		entryPath = os.path.join(directory,entry)
		if os.path.isfile(entryPath):
			with open(entryPath,'r') as f:
				data[entry] = f.read()
				f.close()
	return data

# extract features from documents of a catagory
def wordsTokenize(data):
	words = []
	#triple = []
	#index = 1
	for value in data.values():
		fileTks = NgramMod.WordTokenize(value)
		#counter = collections.Counter(fileTks)
		words.append(fileTks)
	return words
	
def GetVocab(nonSpamWords,spamWords)
	words = nonSpamWords + spamWords
	vocb = list(set(tks))
    vocb.insert(0,"<v>")
    vocb.append("</v>")
    return vocb
    
def wordsCount(tokens):
     my_dict = dict(collections.Counter(tokens))
     return my_dict
     
def writeToFile(data,path,protocal):
	with open(path,'wb+') as f:
    	pickle.dump(data,f,protocal)
   		f.close()
   	return

def trainingData(trainPath):
	#load files
    nonSpamDat = LoadCorpus(argv[1]+'/nonspam-train')
    spamDat = LoadCorpus(argv[1]+'/spam-train')
	#read files to tokens
    nonSpamTks = wordsTokenize(nonSpamDat)
    spamTks = wordsTokenize(spamDat)
	#generate dictionaries for spam and nonspam documents
    nonSpamDic = wordsCount(nonSpamTks)
    spamDic = wordsCount(spamTks)

   	voca = GetVocab(nonSpamTks,spamTks)
    	
   	writeToFile(nonSpamDic,nonSpamName,pickle.HIGHEST_PROTOCOL)
   	writeToFile(spamDic,spamName,pickle.HIGHEST_PROTOCOL)    	
   	writeToFile(voca,vocaName,0)
	return

		
		
