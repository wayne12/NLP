import NaiveClassifier
import os
import collections
import string
import math
import pickle
import NgramMod

vocaName = 'storage/mailVocab.voc'
nonSpamName = 'storage/nonSpam.dic'
spamName = 'storage/spam.dic'
# information about files frequency of each type
categFreqName = 'storage/labelDat.dic'
cateFreqKeys = ('spamDoc','nonSpamDoc')

vocabSize = 0
spamSize = 0
nonSpamSize = 0
spamDocProb = 0.0
nonSpamDocProb = 0.0

def ReadTestData(data):
    dataDic = {}
    index = 1

    #obtain testing data
    for value in data.itervalues():
        fileTks = NgramMod.WordTokenize(value)
        dataDic[index] = list(set(fileTks))
        index += 1
    return dataDic

#start to test
def testData(dataDic,spamDic,nonSpamDic):
    spamProba = 0.0
    nonSpamProba = 0.0
    spamCount = 0
    nonSpamCount = 0
    sameCount = 0

    for value in dataDic.itervalues():#each value represents a file
        #calculate argmax
        for word in value:#value represent word list in test file
            # here assume all words in test data belong to Vocabulary
            spamProba += math.log(float(spamDic.get(word,0)+1.0)/(spamSize+vocabSize))
            nonSpamProba += math.log(float(nonSpamDic.get(word,0)+1.0)/(nonSpamSize+vocabSize))
        spamProba += math.log(spamDocProb)
        nonSpamProba += math.log(nonSpamDocProb)
        if spamProba > nonSpamProba:
            spamCount += 1
        elif spamProba < nonSpamProba:
            nonSpamCount += 1
        else:
            sameCount += 1
    return {'spam':spamCount,'nonSpam':nonSpamCount,'same':sameCount}



def getCategFreq(categDict,name):
    total = 0.0
    for value in categDict.itervalues():
        total += value
    return float(categFreq.get(name,0)/total)

def getDicSize(dataDic):
    size = 0
    for value in dataDic.itervalues():
        size += value
    return size

if __name__ == '__main__':
    from sys import argv
    if len(argv)>1:
        vocab = NaiveClassifier.readFromFile(vocaName)
        spamDic = NaiveClassifier.readFromFile(spamName)
        nonSpamDic = NaiveClassifier.readFromFile(nonSpamName)
        categFreq = NaiveClassifier.readFromFile(categFreqName)

        spamDocProb = getCategFreq(categFreq,cateFreqKeys[0])
        nonSpamDocProb = getCategFreq(categFreq,cateFreqKeys[1])

        spamTestDat = NaiveClassifier.LoadFiles(os.path.join(argv[1],'spam-test'))
        nonSpamTestDat = NaiveClassifier.LoadFiles(os.path.join(argv[1],'nonspam-test'))

# transform test data to tokens
        sapmTestWords = ReadTestData(spamTestDat)
        nonSpamTestWords = ReadTestData(nonSpamTestDat)

        vocabSize = NgramMod.GetVocSize(vocab)
        spamSize = getDicSize(spamDic)
        nonSpamSize = getDicSize(nonSpamDic)

        #test spam documents
        result1 = testData(sapmTestWords,spamDic,nonSpamDic)
        #test nonspam documents
        result2 = testData(nonSpamTestWords,spamDic,nonSpamDic)

        print("test for span,"+' spam: '+str(result1['spam'])+\
              ' nonspam: '+ str(result1['nonSpam'])+' same: '+ str(result1['same']) )
        print("test for nonspan,"+' spam: '+str(result2['spam'])+\
              ' nonspam: '+ str(result2['nonSpam'])+' same: '+ str(result2['same']) )
