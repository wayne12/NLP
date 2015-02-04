import re
import collections
import string
import nltk
import math

def WordTokenize(str):
    return nltk.word_tokenize(str)
    
def SentTokenize(str):
    return filter(None, re.split(r'[\r\n]+',str))   #nltk.sent_tokenize(str)
   
def ExtractVocab(tks):
    #tks = nltk.word_tokenize(str)
    vacb = list(set(tks))
    vacb.insert(0,"<v>")
    vacb.append("</v>")
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
    Proba = 0.0
    tks = SentTokenize(seq)
    tks.insert(0, "<s>")
    tks.append("</s>")
    size = len(tks)-1
    for x in range(0, size):
        proba += math.log((bicunter[tuple([tks[x], tks[x+1]])]+1)/(cunter[tks[x]]+size))
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
    Proba = 0.0
    tks = SentTokenize(seq)
    tks.insert(0, "<s>")
    tks.append("</s>")
    size = len(tks)-1
    #for start word
    proba += math.log((bicunter[tuple([tks[0], tks[1]])]+1)/(cunter[tks[0]]+size))
    #for other words
    for x in range(0, len(tks)-2):
        proba += math.log((tricunter[tuple([tks[x], tks[x+1], tks[x+2]])]+1) \
        /(bicunter[tuple([tks[x], tks[x+1]])]+size))
    return proba    
