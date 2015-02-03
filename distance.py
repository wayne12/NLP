
def distance(target, source, insertcost, deletecost, replacecost):
    n = len(target)+1
    m = len(source)+1
    # set up dist and initialize values
    dist = [ [0 for j in range(m)] for i in range(n) ]
    trace = [ [[0 for k in range(3)]for j in range(m)] for i in range(n) ]
    for i in range(1,n):
        dist[i][0] = dist[i-1][0] + insertcost
        trace[i][0][0] = 1
    for j in range(1,m):
        dist[0][j] = dist[0][j-1] + deletecost
        trace[0][j][1] = 1
    # align source and target strings
    for j in range(1,m):
        for i in range(1,n):
            inscost = insertcost + dist[i-1][j]
            delcost = deletecost + dist[i][j-1]
            if (source[j-1] == target[i-1]): add = 0
            else: add = replacecost
            substcost = add + dist[i-1][j-1]
            dist[i][j] = min(inscost, delcost, substcost)
            if dist[i][j] == inscost:
            	trace[i][j][0] = 1
            if (dist[i][j] == delcost):
                #dist[i][j] = delcost
                trace[i][j][1] = 1
            if(dist[i][j] == substcost):
                trace[i][j][2] = 1
    # show min edit distance
    #print dist
    print "levenshtein distance =",  dist[n-1][m-1]
    return trace
globalCount = 0
def tracePath(target, source, showStr, i,j, trace):
        global globalCount
        if globalCount == 0:
           return
        showStr1 = showStr[:] #copy value of showStr to a new one
        showStr2 = showStr[:] #copy value of showStr to a new one
        tempStr1 = showStr #store reference to showStr
        tempStr2 = showStr #store reference to showStr
        #saveStr = showStr[:]
        if (i==0)and(j==0):
            print globalCount, ":"
            print  showStr[0]
            print  showStr[1]
            print  showStr[2]
            print '-------'
            #showStr= ['', '','']
            globalCount -= 1
            return
        if trace[i][j][0] == 1:
            tempStr1 = showStr1 #store reference to showStr1
            tempStr2 = showStr2 #store reference to showStr2
            showStr[2] = '_' + showStr[2] 
            showStr[0] = target[i-1]+ showStr[0]
            showStr[1] = ' ' + showStr[1] 
            i = i-1
            tracePath(target, source, showStr, i,j, trace)		
            i = i+1
           # showStr = saveStr[:]
        if trace[i][j][1] == 1:
            if tempStr2 != showStr2: tempStr2 = showStr2
            tempStr1[2] = source[j-1] + tempStr1[2]
            tempStr1[0] = '_' + tempStr1[0] 
            tempStr1[1] = ' ' + tempStr1[1] 
            j = j-1
            tracePath(target, source, tempStr1, i,j, trace)
            j = j+1
            #tempStr1 = saveStr[:]
        if trace[i][j][2] == 1:
           tempStr2[2] = source[j-1]+ tempStr2[2]
           tempStr2[0] = target[i-1] + tempStr2[0] 
           if source[j-1] == target[i-1]:
               tempStr2[1] = '|' + tempStr2[1] 
           else:
               tempStr2[1] = ' ' + tempStr2[1]
           i = i -1
           j = j-1
           tracePath(target, source, tempStr2, i,j, trace)
           i = i+1
           j = j+1
          # tempStr1 = saveStr[:]
def disp_distance(target, source, trace, ndisply=1):
    n = len(target)+1
    m = len(source)+1
    size = max(m, n)
    #showSource = ''
    #showTarget = ''
    #showMid = ''
    showStr = ['','','']
    i=n-1 #target 9
    j=m-1 #source 9
   # print trace
  #  while (i != 0)or(j != 0):
    global globalCount
    globalCount = int(ndisply)
#  def tracePath():
    tracePath(target, source, showStr, i,j, trace)

    #print  showTarget
    #print  showMid
    #print  showSource
    #return
if __name__=="__main__":
    from sys import argv
    if len(argv) > 2:
       trace = distance(argv[1], argv[2], 1, 1, 2)
       if len(argv) >= 4:
            disp_distance(argv[1], argv[2], trace, argv[3])
       else:
            disp_distance(argv[1], argv[2], trace)
