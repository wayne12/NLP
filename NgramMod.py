import re
import collections
import string
import nltk
import math
import random

vocSize = 0

def WordTokenize(str):
    return nltk.word_tokenize(str)
    
def SentTokenize(str):
    return filter(None, re.split("[\r\n]+",str))   #nltk.sent_tokenize(str)
   
def ExtractVocab(tks):
    #tks = nltk.word_tokenize(str)
    vacb = list(set(tks)) #remove repeated words
    vacb.insert(0,"<v>")
    vacb.append("</v>")
    vocSize = len(vacb)
    return vacb
    
def senTokenizeToWords(sentks): #actually word tokenize with <s> and </s> symbols
    subtks = []
    for sent in sentks:
        subtks.append("<s>")
        subtks.extend(nltk.word_tokenize(sent))
        subtks.append("</s>")
    return subtks

#check if data saved in a common storage (file) 
def fileCheck(name):
    if type(name) == type(str()):
        f = open(name, 'r')
        str = f.read()
        f.close()
        tks = re.split(r'$&$',  str) #items in text must be separated by $&$
        return tks
    else:
        return name

#count 2 or more consecutive words
def FreqPairCnt(tokens,len = 2,stopwFlag=False,path=''):#exclude punctuation and stopwords
    if (stopwFlag == True)and(path!=''):
       f = open(path,'r')
       stopws = f.read().split()
       f.close()
    punct = list(string.punctuation)
    savetoken = tokens[:]#don't want to modify tokens[]
    for indx, str in enumerate(savetoken):
       if str in punct:
          savetoken.pop(indx)
       elif (stopwFlag == True)and(str in stopws):
          savetoken.pop(indx)
    pairs = [savetoken[x:x+len] for x in xrange(0, len(savetoken)-len+1)]
    tupPairs = [tuple(sublist) for sublist in pairs]
    counter = collections.Counter(tupPairs)
    return counter    

def FreqCount(tokens):
    counter = collections.Counter(tokens)
    return counter

#implement bigram language model using Add-one smoothing
#return: probability of sequence using log space
#parameters: 
    #seq:input sequence, 
    #cunter: counter for single words
    #bicunter: counter for 2 consecutive words
def BigramMod(seq, cunter, bicunter): 
    #caculate probability of sequence
    proba = 0.0
    tks = SentTokenize(seq)
    tks.insert(0, "<s>")
    tks.append("</s>")
    size = len(tks)-1
    for x in range(0, size):
        proba += math.log((bicunter[tuple([tks[x], tks[x+1]])]+1)/(cunter[tks[x]]+vocSize))
    return proba
    
#implement trigram language model using Add-one smoothing
#return: probability of sequence using log space
#parameters: 
    #seq:input sequence, 
    #cunter: counter for single words
    #bicunter: counter for 2 consecutive words    
    #tricunter: counter for 3 consecutive words
def TrigramMod(seq, cunter, bicunter, tricunter): 
    #caculate probability of sequence
    proba = 0.0
    tks = SentTokenize(seq)
    tks.insert(0, "<s>")
    tks.append("</s>")
    #size = len(tks)-1
    #for start word
    proba += math.log((bicunter[tuple([tks[0], tks[1]])]+1)/(cunter[tks[0]]+vocSize))
    #for other words
    for x in range(0, len(tks)-2):
        proba += math.log((tricunter[tuple([tks[x], tks[x+1], tks[x+2]])]+1) \
        /(bicunter[tuple([tks[x], tks[x+1]])]+vocSize))
    return proba    
    
#paramteters:
#     preSeq: one or two words (one for bigram, two for trigram)
def GetMostLikly(preSeq, cunter, bicunter, tricunter=None, ngram=2):
    tks = SentTokenize(preSeq)#if preSeq is begin, it should include '<s>'
    maxfreq=0
    proba = 0.0
    predict = ''
    len = len(tks)
    if len == 1 and tks[0] == '<s>' and ngram == 3:
        ngram = 2 #for start word, using bigram implement trigram
    if ngram==2:
        sortItems = sorted(bicunter.items(), key=lambda item:item[0][0]==tks[len-1], reverse=True)
        for item in sortItems:
            if item[0][0] != tks[len-1]: break
            if item[1] > max:
                maxfreq = item[1]
                predict = item[0][1]
        proba = math.log((maxfreq+1)/(cunter[tks[len-1]]+vocSize))
    elif ngram == 3:
        sortItems = sorted(tricunter.items(), key=lambda item:item[0][0]==tks[len-2] \
        and item[0][1]==tks[len-1], reverse=True)
        for item in sortItems:
            if item[0][0] != tks[len-2] or item[0][1] != tks[len-1]: break
            if item[1] > max:
                maxfreq = item[1]
                predict = item[0][2]
        proba = math.log((maxfreq+1)/(bicunter[tuple([tks[len-2], tks[len-1]])]+vocSize))
    return (predict, proba)

def RandomSents(num,cunter,bicunter,tricunter=None,ngram=2):
    result = []
    if ngram == 2:
        for i in range(0, num-1):
            stop = random.randint(3,30)
            sentence = '<s>'
            proba = 0.0
            for j in range(0,stop):
                word = GetMostLikly(sentence, cunter, bicunter)
                sentence += word[0]
                proba += word[1]
                if word == '</s>' or j == stop:
                    break
                sentence += " "
            end = len(sentence)
            #remove <s> and </s>
            if sentence[end-4:end] == '</s>':end = end-4
            result.append((sentence[3:end],proba))
    elif ngram == 3:
        for i in range(0, num-1):
            stop = random.randint(3,30)
            sentence = '<s>'
            proba = 0.0
            for j in range(0,stop):
                word = GetMostLikly(sentence, cunter, bicunter,tricunter,ngram)
                sentence += word[0]
                proba += word[1]
                if word == '</s>' or j == stop:
                    break
                sentence += " "
            end = len(sentence)
            #remove <s> and </s>
            if sentence[end-4:end] == '</s>':end = end-4
            result.append((sentence[3:end],proba))
    return result
