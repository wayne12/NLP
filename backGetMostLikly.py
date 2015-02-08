#paramteters:
#     preSeq: one or two words (one for bigram, two for trigram)
#     order: 0(pre sequence); 1(following sequence); 2(around sequence)
def GetMostLikly(Seq, ndict, ngram=2,order=0,):
    tks = WordTokenize(Seq)#if preSeq is begin, it should include '<s>'
    maxfreq=0
    proba = 0.0
    predict = ''
    len = len(tks)
    total = 0
    if len < 1:
        return
    y = 0
    z = len -2
    if ngram==2:
        if order == 0:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[len-1], reverse=True)
        elif order == 1:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][1]==tks[0], reverse=True)
            for item in sortItems:
                if item[0][0] != tks[len-1]: break
		total += item[1]
                if item[1] > maxfreq:
                    maxfreq = item[1]
                    predict = item[0][1]
        proba = math.log((maxfreq+1)/(total+vocSize))
        if order == 1:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][1]==tks[len-1], reverse=True)
    elif ngram == 3:
        if order == 0:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[len-2] \
            and item[0][1]==tks[len-1], reverse=True)
        elif order == 1:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][1]==tks[0] \
            and item[0][2]==tks[1], reverse=True)
            y = 1
            z = 0
        elif order == 2:
            sortItems = sorted(ndict.items(), key=lambda item:item[0][0]==tks[len-2] \
            and item[0][2]==tks[len-1], reverse=True)
            y = 2
        for item in sortItems:
            if item[0][y%2] != tks[z] or item[0][(y+1)/2+1] != tks[z+1]: break
	    total += item[1]
            if item[1] > maxfreq:
                maxfreq = item[1]
                predict = item[0][(y+2)%3]
        proba = math.log((maxfreq+1)/(total+vocSize))
    return (predict, proba)
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
