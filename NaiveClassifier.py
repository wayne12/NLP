import os
import collections
import string
import math
import pickle
import NgramMod

vocaName = 'storage/mailVocab.voc'
nonSpamName = 'storage/nonSpam.dic'
spamName = 'storage/spam.dic'

# information about files frequency of each category
categFreqName = 'storage/labelDat.dic'

cateFreqKeys = ('spamDoc','nonSpamDoc')

def LoadFiles(directory):
	data = {}#define a dictionary
	for entry in os.listdir(directory):
		entryPath = os.path.join(directory,entry)
		if os.path.isfile(entryPath):
			with open(entryPath,'r') as f:
				data[entry] = f.read()
				f.close()
	return data

def getCategFreq(nonSpamWords,spamWords):
    return {'spamDoc':len(spamWords),'nonSpamDoc':len(nonSpamWords)}

# extract features from documents of a catagory
def wordsTokenize(data):
	words = []
	#triple = []
	#index = 1
	for value in data.values():
		fileTks = NgramMod.WordTokenize(value)
		#counter = collections.Counter(fileTks)
		words.extend(fileTks)
	return words
	
def GetVocab(nonSpamWords,spamWords):
    words = nonSpamWords + spamWords
    vocb = list(set(words))
    vocb.insert(0,"<v>")
    vocb.append("</v>")
    return vocb
    
def wordsCount(tokens):
    return dict(collections.Counter(tokens))
     
def writeToFile(data,path,protocal):
    with open(path,'wb+') as f:
        pickle.dump(data,f,protocal)
        f.close()
    return None

def readFromFile(name):
    with open(name,'rb') as f:
        dat = pickle.load(f)
        f.close()
    return dat

def trainingData(trainPath):
    #load files
    nonSpamDat = LoadFiles(os.path.join(trainPath ,'nonspam-train'))
    spamDat = LoadFiles(os.path.join(trainPath,'spam-train'))

    ## count documents of each type
    categFreqDict = getCategFreq(nonSpamDat,spamDat)

    #tokenize words in files
    nonSpamTks = wordsTokenize(nonSpamDat)
    spamTks = wordsTokenize(spamDat)
    #generate dictionaries for spam and nonspam documents
    nonSpamDic = wordsCount(nonSpamTks)
    spamDic = wordsCount(spamTks)

    voca = GetVocab(nonSpamTks,spamTks)

    writeToFile(nonSpamDic,nonSpamName,pickle.HIGHEST_PROTOCOL)
    writeToFile(spamDic,spamName,pickle.HIGHEST_PROTOCOL)
    writeToFile(voca,vocaName,0)
    writeToFile(categFreqDict,categFreqName,pickle.HIGHEST_PROTOCOL)
    return None

if __name__ == '__main__':
    from sys import argv
    if len(argv)>1:
        trainingData(argv[1])
