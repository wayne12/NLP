import re
import collections
import string
import nltk
import math
import random
import pickle

vocSize = 0
vocName = "storage/vocab.voc"
uniDic = "storage/uniDict.dic"
biDic = "storage/biDict.dic"
triDic = "storage/triDict.dic"

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
    with open(vocName,'wb+') as f:
        pickle.dump(vacb,f)
        f.close()
    return vacb
    
def senTokenizeToWords(sentks,ngram): #actually word tokenize with <s> and </s> symbols
    subtks = []
    if ngram == 2:
        for sent in sentks:
            subtks.append("<s>")
            subtks.extend(nltk.word_tokenize(sent))
            subtks.append("</s>")
    elif ngram == 3:
        for sent in sentks:
            subtks.extend(['<s>','<s>'])
            subtks.extend(nltk.word_tokenize(sent))
            subtks.append("</s>")
    return subtks

#check if data saved in a common storage (file) 
def fileCheck(name):
    if type(name) == type(str()):
        with open(name,'rb') as f:
            tks = pickle.load(f)
            f.close()
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
    dict = dict(collections.Counter(tupPairs))
    if len == 2:
        with open(biDic,'wb+') as f:
            pickle.dump(dict,f,pickle.HIGHEST_PROTOCOL)
    if len == 3:
        with open(triDic,'wb+') as f:
            pickle.dump(dict,f,pickle.HIGHEST_PROTOCOL)
    return dict

def FreqCount(tokens):
     dict = dict(collections.Counter(tokens))
     with open(uniDic,'wb+') as f:
            pickle.dump(dict,f,pickle.HIGHEST_PROTOCOL)
     return dict

#implement bigram language model using Add-one smoothing
#return: probability of sequence using log space
#parameters: 
    #seq:input sequence, 
    #cunter: counter for single words
    #bicunter: counter for 2 consecutive words
def BigramProb(tks, unidict, bidict):
    #caculate probability of sequence
    proba = 0.0
   # tks = WordTokenize(seq)
    tks.insert(0, "<s>")
    tks.append("</s>")
    size = len(tks)-1
    for x in range(0, size):
        proba *= (bidict[tuple([tks[x], tks[x+1]])]+1)/(unidict[tks[x]]+vocSize)
    return proba
    
#implement trigram language model using Add-one smoothing
#return: probability of sequence using log space
#parameters: 
    #seq:input sequence, 
    #cunter: counter for single words
    #bicunter: counter for 2 consecutive words    
    #tricunter: counter for 3 consecutive words
def TrigramProb(tks, bidict, tridict):
    #caculate probability of sequence
    proba = 0.0
    #tks = WordTokenize(seq)
    tks.insert(0, "<s>")
    tks.insert(0, "<s>")
    tks.append("</s>")

    #size = len(tks)-1
    #for start word
    #proba += math.log((bicunter[tuple([tks[0], tks[1]])]+1)/(cunter[tks[0]]+vocSize))

    #for other words
    for x in range(0, len(tks)-2):
        proba *= (tridict[tuple([tks[x], tks[x+1], tks[x+2]])]+1) \
        /(bidict[tuple([tks[x], tks[x+1]])]+vocSize)
    return proba    
    
#paramteters:
#     preSeq: one or two words (one for bigram, two for trigram)
#     order: 0(pre sequence); 1(following sequence); 2(around sequence)
def GetMostLikly(Seq, unidict, bidict, tridict=None, ngram=2):
    tks = WordTokenize(Seq)#if preSeq is begin, it should include '<s>'
    maxfreq=0
    proba = 0.0
    predict = ''
    len = len(tks)
    if len < 1:
        return
    if ngram==2:
        sortItems = sorted(bidict.items(), key=lambda item:item[0][0]==tks[len-1], reverse=True)
        for item in sortItems:
            if item[0][0] != tks[len-1]: break
            if item[1] > max:
                maxfreq = item[1]
                predict = item[0][1]
        proba = math.log((maxfreq+1)/(unidict[tks[len-1]]+vocSize))
    elif ngram == 3:
        sortItems = sorted(tridict.items(), key=lambda item:item[0][0]==tks[len-2] \
        and item[0][1]==tks[len-1], reverse=True)
        for item in sortItems:
            if item[0][0] != tks[len-2] or item[0][1] != tks[len-1]: break
            if item[1] > max:
                maxfreq = item[1]
                predict = item[0][2]
        proba = math.log((maxfreq+1)/(bidict[tuple([tks[len-2], tks[len-1]])]+vocSize))
    return (predict, proba)

def RandomSents(num,unidict,bidict,tridict=None,ngram=2):
    result = []
    unidict = fileCheck(unidict)
    bidict = fileCheck(bidict)
    if ngram == 2:
        for i in range(0, num-1):
            stop = random.randint(3,30)
            sentence = '<s>'
            proba = 0.0
            for j in range(0,stop):
                word = GetMostLikly(sentence, unidict, bidict)
                sentence += word[0]
                proba += word[1]
                if word == '</s>' or j == stop-1:
                    break
                sentence += " "
            end = len(sentence)
            #remove <s> and </s>
            if sentence[end-4:end] == '</s>':end = end-4
            result.append((sentence[3:end],proba))
    elif ngram == 3:
        tridict = fileCheck(tridict)
        for i in range(0, num-1):
            stop = random.randint(3,30)
            sentence = '<s> <s>'
            proba = 0.0
            for j in range(0,stop):
                word = GetMostLikly(sentence, unidict, bidict,tridict,ngram)
                sentence += word[0]
                proba += word[1]
                if word == '</s>' or j == stop-1:
                    break
                sentence += " "
            end = len(sentence)
            #remove <s> and </s>
            if sentence[end-4:end] == '</s>':end = end-4
            result.append((sentence[7:end],proba))
    return result
def getPerplexity(seq, unidict, bidict,tridict=None,ngram=2):
    unidict = fileCheck(unidict)
    bidict = fileCheck(bidict)
    tks = WordTokenize(seq)
    if ngram == 2:
        proba = BigramProb(tks,unidict,bidict)
    elif ngram == 3:
        tridict = fileCheck(tridict)
        proba = TrigramProb(tks,bidict,tridict)
    perpl = math.pow(proba,(-1/len(tks)))
    return perpl
def evalModels(seq, unidict, bidict,tridict=None,ngram=2):
    unidict = fileCheck(unidict)
    bidict = fileCheck(bidict)
    senTks = SentTokenize(seq)
    pickSents = []
    saveSents = []
    #pick sentences and delete one word
    for i in range(0,40):
        index = random.randint(1,len(senTks))
        tks = WordTokenize(senTks[index-1])
        saveSents.append(tks)
        sent = tks[:]
        #delete one word at place (index%len(tks))-1
        sent[(index%len(tks))-1,(index%len(tks))] = []
        pickSents.append((sent,(index%len(tks))-1))
    if ngram == 2:
        proba = GetMostLikly(tks,unidict,bidict)
    elif ngram == 3:
        tridict = fileCheck(tridict)
        proba = GetMostLikly(tks,bidict,tridict)
    return proba
if __name__ == '__main__':
    from sys import argv
    if len(argv)>1:
        f = open(argv[1],'r')
        traintxt = f.read()
        f.close()
        words = WordTokenize(traintxt)
        sentes = SentTokenize(traintxt)
        voca = ExtractVocab(words)
        bigramTks = senTokenizeToWords(sentes,2)
        trigramTks = senTokenizeToWords(sentes,3)
        uniDict = FreqCount(words)
        biDict = FreqPairCnt(bigramTks)
        triDict = FreqPairCnt(trigramTks,3)
        result2 = RandomSents(50,uniDict,biDict)
        result3 = RandomSents(50,uniDict,biDict,triDict,3)