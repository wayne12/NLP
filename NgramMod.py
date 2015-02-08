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
def GetMostLikly(tks, ndict, ngram=2,order=0,):
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
            proba = math.log((maxfreq+1)/(total+vocSize))
        elif order == 1:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][1]==tks[0], reverse=True)
            for item in sortItems:
                if item[0][1] != tks[0]: break
		total += item[1]
                if item[1] > maxfreq:
                    maxfreq = item[1]
                    predict = item[0][0]
            proba = math.log((maxfreq+1)/(total+vocSize))
    elif ngram == 3:
        if order == 0:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[leng-2] \
            and item[0][1]==tks[leng-1], reverse=True)
            for item in sortItems:
            	if item:item[0][0]!=tks[leng-2] or item[0][1]!=tks[leng-1]: break
	    	total += item[1]
            	if item[1] > maxfreq:
                	maxfreq = item[1]
                	predict = item[0][2]
        elif order == 1:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][1]==tks[0] \
            and item[0][2]==tks[1], reverse=True)
            for item in sortItems:
            	if item:item[0][1]!=tks[0] or item[0][2]!=tks[1]: break
	    	total += item[1]
            	if item[1] > maxfreq:
                	maxfreq = item[1]
                	predict = item[0][0]
        elif order == 2:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[leng-2] \
            and item[0][2]==tks[leng-1], reverse=True)
            for item in sortItems:
            	if item:item[0][0]!=tks[leng-2] or item[0][2]!=tks[leng-1]: break
	    	total += item[1]
            	if item[1] > maxfreq:
                	maxfreq = item[1]
                	predict = item[0][1]
        proba = math.log((maxfreq+1)/(total+vocSize))
    return (predict, proba)

def RandomSents(num,ndict,ngram=2):
    result = []
    if ngram == 2:
    	bidict = fileCheck(ndict)
        for i in range(0, num-1):
            stop = random.randint(3,30)
            sentence = ['<s>']
            proba = 0.0
            for j in range(0,stop):
                word = GetMostLikly(sentence, bidict)
                sentence.append(word[0])
                proba += word[1]
                if word[0] == '</s>' or j == stop-1:
                    break
            end = len(sentence)
            #remove <s> and </s>
            if sentence[end-1] == '</s>':end = end-1
            result.append((sentence[1:end],proba))
    elif ngram == 3:
        tridict = fileCheck(ndict)
        for i in range(0, num-1):
            stop = random.randint(3,30)
            sentence = ['<s>', '<s>']
            proba = 0.0
            for j in range(0,stop):
                word = GetMostLikly(sentence,tridict,ngram)
                sentence.append(word[0])
                proba += word[1]
                if word[0] == '</s>' or j == stop-1:
                    break
            end = len(sentence)
            #remove <s> and </s>
            if sentence[end-1] == '</s>':end = end-1
            result.append((sentence[2:end],proba))
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
      	preResult = GetMostLikly(prePar,bidict)
      	# using following word to predic
        afterResult = GetMostLikly(afterPar,bidict)
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
        preResult = GetMostLikly(prePar,tridict)
        # using a previous word and a following word to predic
        midResult = GetMostLikly(midPar,tridict)
        # using following 2 words to predic
        afterResult = GetMostLikly(afterPar,tridict)
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
        print ("Result of Bigram is: "+' '.join(tks[:delPos]+biWord+\
        ' '.join(tks[delPos+1:]))
        print ("Result of Trigram is: "+' '.join(tks[:delPos]+triWord+\
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
        result2 = RandomSents(50,biDict)
        result3 = RandomSents(50,triDict,3)
        f = open(argv[2],'r')
        str1 = f.read()
        f.close()
        evalModels(str1)
