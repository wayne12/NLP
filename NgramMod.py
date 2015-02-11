import re
import collections
import string
import nltk
import math
import random
import pickle
import os

vocSize = 0
vocName = "storage/a2q1/vocab.voc"
uniDicName = "storage/a2q1/uniDict.dic"
biDicName = "storage/a2q1/biDict.dic"
triDicName = "storage/a2q1/triDict.dic"

def WordTokenize(str):
    return nltk.word_tokenize(str)
    
def SentTokenize(str):
    return filter(None, re.split("[\r\n]+",str))   #nltk.sent_tokenize(str)
   
def ExtractVocab(tks):
    #tks = nltk.word_tokenize(str)
    vocb = list(set(tks)) #remove repeated words
    vocb.insert(0,"<s>")
    vocb.append("</s>")
    return vocb
    
def GetVocSize(vocb):
    return len(vocb)-2 #exclude '<s>'and '</s>'


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

def FreqCount(tokens):
     return dict(collections.Counter(tokens))

#count 2 or more consecutive words
def FreqPairCnt(tokens,size = 2,stopwFlag=False,path=''):#exclude punctuation and stopwords
    if (stopwFlag == True)and(path!=''):
       f = open(path,'r')
       stopws = f.read().split()
       f.close()
    punct = list(string.punctuation)
    savetoken = tokens[:]#don't want to modify tokens[]
    for indx, chars in enumerate(savetoken):
       if chars in punct:
          savetoken.pop(indx)
       elif (stopwFlag == True)and(chars in stopws):
          savetoken.pop(indx)
    pairs = [savetoken[x:x+size] for x in xrange(0, len(savetoken)-size+1)]
    tupPairs = [tuple(sublist) for sublist in pairs]
    return dict(collections.Counter(tupPairs))

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
        proba *= (bidict.get(tuple([tks[x], tks[x+1]]),0)+1.0)/(unidict.get(tks[x],0)+vocSize)
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
        proba *= (tridict.get(tuple([tks[x], tks[x+1], tks[x+2]]),0)+1.0) \
        /(bidict.get(tuple([tks[x], tks[x+1]]),0)+vocSize)
    return proba

def predictWord(tks, ndict, ngram=2):
    #tks = WordTokenize(Seq)#if preSeq is begin, it should include '<s>'
    maxfreq = 0
    baseline = 30
    proba = []
    predict = []
    leng = len(tks)
    total = 0
    if leng < 1:
        return (None,0)
    searchResult = []

    if ngram==2:
        for key,val in ndict.iteritems():
            if key[0] == tks[leng-1]:
                total += val
                searchResult.append((key,val))
        sortItems = sorted(searchResult,key = lambda item:item[1],reverse=True)
        size = len(sortItems)
        size = size/3
        if size != 0:
            stop = random.randint(0,size)
        else: stop = 0
        proba = math.log((sortItems[stop][1]+1.0)/(total+vocSize))
        predict = sortItems[stop][0][1]

    elif ngram == 3:
        for key,val in ndict.iteritems():
            if key[0] == tks[leng-2] and key[1] == tks[leng-1]:
                total += val
                searchResult.append((key,val))
        sortItems = sorted(searchResult,key = lambda item:item[1],reverse=True)
        size = len(sortItems)
        size = size/3
        if size != 0:
            stop = random.randint(0,size)
        else: stop = 0
        proba = math.log((sortItems[stop][1]+1.0)/(total+vocSize))
        predict = sortItems[stop][0][2]
    if False:'''
    if ngram==2:
        sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[leng-1], reverse=True)
        for item in sortItems:
            if item[0][0] != tks[leng-1]: break
            val = item[1]
            total += val
            if val > maxfreq:
            if val > baseline:
                proba.append(float(math.log((val+1.0)/(total+vocSize))))
                predict.append(item[0][1])
    elif ngram == 3:
        sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[leng-2] \
                                                          and item[0][1]==tks[leng-1], reverse=True)
        for item in sortItems:
            if item[0][0]!=tks[leng-2] or item[0][1]!=tks[leng-1]: break
            total += item[1]
            if item[1] > maxfreq:
                proba.append(float(math.log((item[1]+1.0)/(total+vocSize))))
                predict.append(item[0][2])
                '''
    return(predict,proba)
#paramteters:
#     preSeq: one or two words (one for bigram, two for trigram)
#     order: 0(pre sequence); 1(following sequence); 2(around sequence)
def GetMostlikly(tks, ndict, ngram=2,order=0,):
    #tks = WordTokenize(Seq)#if preSeq is begin, it should include '<s>'
    maxfreq=0
    proba = 0.0
    predict = ''
    leng = len(tks)
    total = 0
    if leng < 1:
        return (None,0)
    y = 0
    z = leng -2
    if ngram==2:
        if order == 0:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[leng-1], reverse=True)
            for item in sortItems:
                if item[0][0] != tks[leng-1]: break
                total += item[1]
                if item[1] > maxfreq:
                    maxfreq = item[1]
                    predict = item[0][1]
            proba = math.log((maxfreq+1.0)/(total+vocSize))
        elif order == 1:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][1]==tks[0], reverse=True)
            for item in sortItems:
                if item[0][1] != tks[0]: break
                total += item[1]
                if item[1] > maxfreq:
                    maxfreq = item[1]
                    predict = item[0][0]
            proba = math.log((maxfreq+1.0)/(total+vocSize))
    elif ngram == 3:
        if order == 0:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[leng-2] \
            and item[0][1]==tks[leng-1], reverse=True)
            for item in sortItems:
                if item[0][0]!=tks[leng-2] or item[0][1]!=tks[leng-1]: break
                total += item[1]
                if item[1] > maxfreq:
                    maxfreq = item[1]
                    predict = item[0][2]
        elif order == 1:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][1]==tks[0] \
            and item[0][2]==tks[1], reverse=True)
            for item in sortItems:
                if item[0][1]!=tks[0] or item[0][2]!=tks[1]: break
                total += item[1]
                if item[1] > maxfreq:
                    maxfreq = item[1]
                    predict = item[0][0]
        elif order == 2:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[leng-2] \
            and item[0][2]==tks[leng-1], reverse=True)
            for item in sortItems:
                if item[0][0]!=tks[leng-2] or item[0][2]!=tks[leng-1]: break
                total += item[1]
                if item[1] > maxfreq:
                    maxfreq = item[1]
                    predict = item[0][1]
        proba = math.log((maxfreq+1.0)/(total+vocSize))
    return (predict, proba)

def RandomSents(num,ndict,ngram=2):
    result = []
    if ngram == 2:
        #bidict = fileCheck(ndict)
        for i in range(0, num-1):
            stop = random.randint(4,35)
            sentence = ['<s>']
            proba = 0.0
            for j in range(0,stop):
                word = predictWord(sentence, ndict)
                sentence.append(word[0])
                proba += word[1]
                if word[0] == '</s>' or j == stop-1:
                    break
            end = len(sentence)
            #remove <s> and </s>
            if sentence[end-1] == '</s>':end = end-1
            result.append((sentence[1:end],proba))
    elif ngram == 3:
        #tridict = fileCheck(ndict)
        for i in range(0, num-1):
            stop = random.randint(3,30)
            sentence = ['<s>', '<s>']
            proba = 0.0
            for j in range(0,stop):
                word = predictWord(sentence,ndict,ngram)
                sentence.append(word[0])
                proba += word[1]
                if word[0] == '</s>' or j == stop-1:
                    break
            end = len(sentence)
            #remove <s> and </s>
            if sentence[end-1] == '</s>':end = end-1
            result.append((sentence[2:end],proba))
    return result
def getPerplexity(seq, bidict,ndict,ngram=2):
    ndict = fileCheck(ndict)
    bidict = fileCheck(bidict)
    tks = WordTokenize(seq)
    if ngram == 2:
        proba = BigramProb(tks,ndict,bidict)
    elif ngram == 3:
        proba = TrigramProb(tks,bidict,ndict)
    perpl = math.pow(proba,(-1/len(tks)))
    return perpl
# evaluate bigram model and trigram model by predict word_tokenize
# a word from 40 sentences.   
def evalModels(seq, bidict,tridict=None,ngram=2):
    bidict = fileCheck(bidict)
    senTks = SentTokenize(seq)
    biCnt = 0
    triCnt = 0
    #pick sentences and delete one word
    for i in range(0,40):
        index = random.randint(1,len(senTks))
        tks = WordTokenize(senTks[index-1])
        delPos = index%len(tks)-1
        ################### for bigram
        if delPos == 0:
            prePar = ['<s>']
            afterPar = [tks[delPos+1]]
        elif delPos == len(tks)-1:
            prePar = [tks[delPos-1]]
            afterPar = ['</s>']
        else:
            prePar = [tks[delPos-1]]
            afterPar = [tks[delPos+1]]
        # using previous word to predic
        preResult = GetMostlikly(prePar,bidict)
        # using following word to predic
        afterResult = GetMostlikly(afterPar,bidict)
        # pick the prediction that has maxmum probability
        if preResult[1]>afterResult[1]:
            biWord = preResult[0]
        else:
            biWord = afterResult[0]
        ################### for trigram
        tridict = fileCheck(tridict)
        if delPos == 0:
            prePar = ['<s>','<s>']
            midPar = ['<s>',tks[1]]
            afterPar = [tks[1],tks[2]]
        if delPos == 1:
            prePar = ['<s>',tks[0]]
            midPar = [tks[0],tks[2]]
            afterPar = [tks[2],tks[3]]
        elif delPos == len(tks)-2:
            prePar = [tks[delPos-2],tks[delPos-1]]
            midPar = [tks[delPos-1],tks[delPos+1]]
            afterPar = [tks[delPos+1],'</s>']
        elif delPos == len(tks)-1:
            prePar = [tks[delPos-2],tks[delPos-1]]
            midPar = [tks[delPos-1],'</s>']
            afterPar = []#don't calculate it
        else:
            prePar = [tks[delPos-2],tks[delPos-1]]
            midPar = [tks[delPos-1],tks[delPos+1]]
            afterPar = [tks[delPos+1],tks[delPos+2]]
        # using previous 2 words to predic
        preResult = GetMostlikly(prePar,tridict)
        # using a previous word and a following word to predic
        midResult = GetMostlikly(midPar,tridict)
        # using following 2 words to predic
        afterResult = GetMostlikly(afterPar,tridict)
        # pick the prediction that has maxmum probability
        maxpro = 0
        triWord = ''
        if preResult[1] > maxpro:
            maxpro = preResult[1]
            triWord = preResult[0]
        if midResult[1] > maxpro:
            maxpro = midResult[1]
            triWord = midResult[0]
        if afterResult[1] > maxpro:
            maxpro = afterResult[1]
            triWord = afterResult[0]
        ################## evaluate according to result
        print ("**********************************")
        print ("The original sentence is: "+' '.join(tks))
        print ("Result of Bigram is: "+' '.join(tks[:delPos])+biWord+\
        ' '.join(tks[delPos+1:]))
        print ("Result of Trigram is: "+' '.join(tks[:delPos])+triWord+\
        ' '.join(tks[delPos+1:]))
        if biWord == tks[delPos]:
            biCnt += 1
            print ("Bigram model match! ")
        if triWord == tks[delPos]:
            triCnt	+= 1
            print ("Trigram model match! ")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Total sentens number: 40")
    print("Bigram matched: " + str(biCnt))
    print("Trigram matched: " + str(triCnt))
    if biCnt > triCnt:
        print("Bigram has better performance!")
    elif biCnt < triCnt:
        print("Trigram has better performance!")
    else:
        print("Bigram and Trigram have the same performance")
    return

def writeToFile(data,path,protocal=0):
    with open(path,'wb+') as f:
        pickle.dump(data,f,protocal)
        f.close()
    return None

def readFromFile(name):
    with open(name,'rb') as f:
        dat = pickle.load(f)
        f.close()
    return dat

def trainModel(traintxt):
    words = WordTokenize(traintxt)
    sentes = SentTokenize(traintxt)
    voca = ExtractVocab(words)

    bigramTks = senTokenizeToWords(sentes,2)
    trigramTks = senTokenizeToWords(sentes,3)
    uniDict = FreqCount(words)
    biDict = FreqPairCnt(bigramTks)
    triDict = FreqPairCnt(trigramTks,3)

    writeToFile(voca,vocName)
    writeToFile(uniDict,uniDicName,pickle.HIGHEST_PROTOCOL)
    writeToFile(biDict,biDicName,pickle.HIGHEST_PROTOCOL)
    writeToFile(triDict,triDicName,pickle.HIGHEST_PROTOCOL)
    return None

def testModel(str1):
    voca = readFromFile(vocName)
    uniDict = readFromFile(uniDicName)
    biDict = readFromFile(biDicName)
    triDict = readFromFile(triDicName)

    global vocSize
    vocSize = GetVocSize(voca)

    result2 = RandomSents(50,biDict)
    result3 = RandomSents(50,triDict,3)

    print("Sentences generate by Bigram and their Probabilities :")
    for item in result2:
        print (' '.join(item[0])+" : "+str(item[1]))

    print("Sentences generate by Trigram and their Probabilities :")
    for item in result3:
        print (' '.join(item[0])+" : "+str(item[1]))

    PPL2 = getPerplexity(str1,biDict,uniDict)
    PPL3 = getPerplexity(str1,biDict,triDict)
    print("Perplexity of Bigram is: " + str(PPL2))
    print("Perplexity of Trigram is: " + str(PPL3))

    evalModels(str1)

    return None

if __name__ == '__main__':
    from sys import argv
    if len(argv)>1:
        print (argv[1])
        # with open(os.path.join(argv[1],'train.txt'),'r') as f:
        #     traintxt = f.read()
        #     f.close()
        #     trainModel(traintxt)

        with open(os.path.join(argv[1],'test.txt'),'r') as f:
            str1 = f.read()
            f.close()
            testModel(str1)


