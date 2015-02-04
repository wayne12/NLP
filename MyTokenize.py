import re
import collections
import string

def MyWordTokenize(str):
    tks = filter(None,re.split(r'[ ,;:?!\r\n\.]+',str))
    savetks = tks[:]
    offset = 0
    for idx,  str in enumerate(savetks):
       ind = str.find('\'')
       if ind != -1:
           tks.pop(idx+offset)
           tks.insert(idx+offset, str[:ind])
           tks.insert(idx+offset+1, str[ind:])
           offset += 1
    return tks
def MySentTokenize(str):
   return filter(None, re.split(r'[\r\n]+',str))   
def ContainsNum(tokens):
   result = []
   for str in tokens:
        if any(c.isdigit() for c in str):
            result.append(str)
   return result
def ContainsPunct(tokens):
    #punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    result = []
    for str in tokens:
            if  any(c in string.punctuation for c in str):
                result.append(str)
    return result 
def ContainsCharAndNum(tokens):
    result = []
    numflag = False
    charflag = False
    for str in tokens:
        for c in str:
            if c.isdigit()and(numflag==False): 
                numflag = True
            elif c.isalpha()and(charflag==False):
                charflag = True
            elif (numflag==True) and (charflag==True):
                result.append(str)
                break
    return result
def FreqCount(tokens, n=-1):
    punct = list(string.punctuation)
    counter = collections.Counter(tokens)
    if n!= -1:
       return counter.most_common(n)
    else:
       return counter.most_common()
def FreqCountNoPunct(tokens,n=-1,stopwFlag=False,path=''):
    if (stopwFlag == True)and(path!=''):
       f = open(path,'r')
       stopws = f.read().split()
       f.close()
    punct = list(string.punctuation)
    savetoken = tokens[:]#don't want to modify tokens[]
    for indx, str in enumerate(savetoken):
       if str in punct:
          savetoken.pop(indx)
       elif(stopwFlag == True)and(str in stopws):
          savetoken.pop(indx)
    counter = collections.Counter(savetoken)
    if n!= -1:
       return counter.most_common(n)
    else:
       return counter.most_common()
def FreqPairCnt(tokens,n=-1,stopwFlag=False,path=''):#exclude punctuation and stopwords
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
    pairs = [savetoken[x:x+2] for x in xrange(0, len(savetoken)-1)]
    tupPairs = [tuple(sublist) for sublist in pairs]
    counter = collections.Counter(tupPairs)
    if n!= -1:
       return counter.most_common(n)
    else:
       return counter.most_common()
def DisplyStatistic(str):
    wTokens = MyWordTokenize(str)
    sTokens = MySentTokenize(str)
    print '\nFirst 20s tokens in token list:'
    print wTokens[0:20]
    print 'First 10 sentences in sentence list:'
    print sTokens[0:10]
    print 'Corpus size:'
    print '   number of sentence:', len(sTokens)
    print '   number of words:', len(wTokens)
    print 'Vocabulary size:', len(set(wTokens))
    dEntries = ContainsNum(wTokens)
    print 'Number of vocabulary entries contain digits:', len(dEntries)
    pEntries = ContainsPunct(wTokens)
    print 'Number of vocabulary entries contain punctuation:', len(pEntries)
    mixEntries =  ContainsCharAndNum(wTokens)
    print 'Number of vocabulary entries contain both letters and digits:', len(mixEntries)
    print 'Top 100 Most frequent words and its frequency (include stopwords):'
    print FreqCountNoPunct(wTokens,100)
    print 'Top 100 Most frequent words and its frequency (exclude stopwords):'
    print FreqCountNoPunct(wTokens,100,True,'stopwords.txt')
    print 'Top 100 Most frequent pairs and its frequency (exclude stopwords):'
    print  FreqPairCnt(wTokens,100,True,'stopwords.txt')
    return
if __name__=="__main__":
    from sys import argv
    if len(argv) > 1:
        dic = open(argv[1],'r')
        sstr = dic.read()
        dic.close()
        DisplyStatistic(sstr)
