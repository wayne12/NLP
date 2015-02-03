import re
import collections
import string
import nltk

def Tokenize(str):
    tks = nltk.word_tokenize(str)
    return tks
    
def ExtractVocab(tks):
    #tks = nltk.word_tokenize(str)
    vacb = list(set(tks))
    vacb.insert(0,"<s>")
    vacb.append("</s>")
    return vacb    
    
def FreqPairCnt(tokens,len = 2, n=-1,stopwFlag=False,path=''):#exclude punctuation and stopwords
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
    if n!= -1:
       return counter.most_common(n)
    else:
       return counter.most_common()    

def FreqCount(tokens, n=-1):
    counter = collections.Counter(tokens)
    if n!= -1:
       return counter.most_common(n)
    else:
       return counter.most_common()

def NgramMod(seq, freq, nfreq, ngram=2): #using log space
    #caculate probability of sequence
    
    
