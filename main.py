import numpy as np
import matplotlib.pyplot as plt
import random
import math
from multiprocessing import Process

# The following global parameters' values are set during each
# iteration of the simulation (i.e., per graph):
# lambda_new - incoming new customers rate 
lambda_new = 0
# mu_s - service rate first station (G/G/1 queue)
mu_s = 0
# mu_r - service rate of retrail station
mu_r = 0
# pLeave - probability of rejected customer to leave the system
pLeave = 0.0
# c_1, c_2 - costs of queue lenght and rejected count resp.
c_1 = 0
c_2 = 0

# Should be the same for all tests:
MAX_MU_R = 100
DELTA_MU_R =  2
MAX_B_SIZE = 300

# Single cost calculation parameters:
B_COST_ITERATIONS = 300
DELTA_TIME = 0.1
MAX_TIME = 10 + DELTA_TIME
TIME_SLOTS = 100 # MAX_TIME/DELTA_TIME

# An array of B_COST_ITERATIONS arrays, 
# each of size MAX_TIME/DELTA_TIME
newCustomers = []
sServiceTime = []
def calcCostAtT(iteration, state, t, mu_r, b, tIndex):
    global pLeave
    global c_1
    global c_2
    global newCustomers
    global sServiceTime
    bufferredInCurIteration = \
        newCustomers[iteration][tIndex] + \
        state['retrailsByTime'][tIndex]   + \
        state['inBuffer']   
    rejectedInIteration = 0
    if bufferredInCurIteration > b:
        rejectedInIteration = bufferredInCurIteration - b
        state['rejectedUntilNow'] += rejectedInIteration
        bufferredInCurIteration = b

    servedInIteration = min(sServiceTime[iteration][tIndex],\
                        bufferredInCurIteration)

    state['inBuffer'] = bufferredInCurIteration  - servedInIteration 
    
    state['debug_timeToBuffer'][tIndex]=state['inBuffer']
    state['debug_timeToRejections'][tIndex]=rejectedInIteration
    state['debug_rejectedUntilNow'][tIndex]=state['rejectedUntilNow']
    
    if rejectedInIteration > 0 and mu_r > 0:
        rejectedProbs =\
            np.random.uniform(0.0, 1.0,rejectedInIteration)                 
        for i in range(0, rejectedInIteration):
            if rejectedProbs[i] > pLeave:
                retryTime = math.ceil(random.expovariate(mu_r*DELTA_TIME)) + t
                if retryTime < MAX_TIME - DELTA_TIME:
                    state['retrailsByTime'][int(retryTime/DELTA_TIME)] += 1

    return np.exp(-1.0*t)*(c_1*state['inBuffer'] +\
                         c_2*state['rejectedUntilNow'])*DELTA_TIME

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    from errno import EEXIST
    from os import makedirs,path
    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
        
def debug_drawSysByTime(timeAxis, systemState, iteration, b):
    if iteration > 0:
        return
    global newCustomers
    global sServiceTime
    global pLeave
    plt.subplot(511)
    plt.plot(timeAxis, systemState['debug_timeToBuffer'], label='inBuffer')
    plt.ylabel("")
    plt.xlabel("time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    
    plt.subplot(512)
    plt.plot(timeAxis, systemState['debug_timeToRejections'], label='Rejections')
    plt.ylabel("")
    plt.xlabel("time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    
    plt.subplot(513)    
    plt.plot(timeAxis, systemState['debug_rejectedUntilNow'], label='Rejected until now')
    plt.ylabel("")
    plt.xlabel("time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    
    plt.subplot(514)    
    plt.plot(timeAxis, newCustomers[iteration], label='newCustomers')
    plt.ylabel("")
    plt.xlabel("time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 

    plt.subplot(515)    
    plt.plot(timeAxis, sServiceTime[iteration], label='sServiceTime')
    plt.ylabel("")
    plt.xlabel("time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
    
    mkdir_p('sysByTime')
    plt.savefig('sysByTime/'+\
    'c1'+str(c_1)+\
    '_c2'+str(c_2)+\
    '_p'+str(pLeave)+\
    '_lm'+str(lambda_new)+\
    '_b'+str(b)+\
    '.png', bbox_inches='tight')
    plt.clf()
                    
def calcCostOfB(iteration, mu_r,b):
    global lambda_new
    global mu_s
    systemState = {
    'debug_timeToBuffer': np.zeros(TIME_SLOTS, dtype=int),
    'debug_timeToRejections' : np.zeros(TIME_SLOTS, dtype=int),
    'debug_rejectedUntilNow' : np.zeros(TIME_SLOTS, dtype=int),
    'bufferredOfPrevIteration' : 0,
    'rejectedUntilNow' : 0,
    'inBuffer': 0,
    'retyUntilNow':0,
    'retrailsByTime': np.zeros(TIME_SLOTS, dtype=int)}
    
    totalCost = 0.0    
    currCost = 0.0
    timeAxis = []
    tIndex = 0
    for t in np.arange(DELTA_TIME, MAX_TIME, DELTA_TIME):
        currCost = calcCostAtT(iteration, systemState, t, mu_r, b, tIndex)
        totalCost += currCost
        timeAxis += [t]
        tIndex += 1
    debug_drawSysByTime(timeAxis, systemState, iteration, b)
    return totalCost

def calcAvgCostOfB(mu_r,b):
    costs = np.zeros(B_COST_ITERATIONS, dtype=float)
    for i in range(0, B_COST_ITERATIONS, 1):
        costs[i] = calcCostOfB(i, mu_r,b)
    return np.average(costs)    
    
def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a) 
                      if smallest == element]

def drawCostOfB(costsX, costsY, mu_r):
    global lambda_new
    global mu_s
    global pLeave
    global c_1 
    global c_2
    plt.plot(costsX, costsY)
    plt.ylabel("cost(b)")
    plt.xlabel("b")
    mkdir_p('costGraphs')
    plt.savefig('costGraphs/costOfB'+\
    '_c1'+str(c_1)+\
    '_c2'+str(c_2)+\
    '_lm'+str(lambda_new)+\
    '_p'+str(pLeave)+\
    '_mu_r'+str(mu_r)+\
    '.png', bbox_inches='tight')
    plt.clf()

    
def calculateBOfMu(mu_r):
    bToCost = np.zeros(MAX_B_SIZE+1, dtype=float)
    #for i in range(0, len(bToCost), 1):
        #bToCost[i] = 1000000
    bToCost[0] = 1000000
    for b in range(1, MAX_B_SIZE+1, 1):
        bToCost[b] = calcAvgCostOfB(mu_r,b)
    costsX = []
    costsY = []    
    for i in range(1, len(bToCost), 1):
        costsX += [str(i)]
        costsY += [str(bToCost[i])]
    drawCostOfB(costsX, costsY,mu_r)    
    minVal, bArray = locate_min(bToCost)
    if len(bArray) > 1:
        print "Found more than one opt b, with value " + str(minVal) + ":"
        print bArray
    return bArray[0]

def drawMuToBGraph(muToB):
    global lambda_new
    global mu_s
    global pLeave
    global c_1 
    global c_2
    plt.plot(range(0, MAX_MU_R, DELTA_MU_R), muToB)
    plt.ylabel("b")
    plt.xlabel("mu_r")
    mkdir_p('simGraphs')
    plt.savefig('simGraphs/simulation'+\
    '_c1'+str(c_1)+\
    '_c2'+str(c_2)+\
    '_lm'+str(lambda_new)+\
    '_p'+str(pLeave)+\
    '.png', bbox_inches='tight')
    plt.clf()

def simulate():
    # Iterate over different mu_r values
    muToB = np.zeros(int(MAX_MU_R/DELTA_MU_R), dtype=float)
    for mu_r in range(0, MAX_MU_R, DELTA_MU_R):
        muToB[int(mu_r/DELTA_MU_R)] = calculateBOfMu(mu_r)
        #print "For mu_r = " + str(mu_r) + " got b = "\
        #+ str(2**int(muToB[mu_r* int(1/DELTA_MU_R)]))
    drawMuToBGraph(muToB)
    
def main(c1,c2,l,p):
    global lambda_new
    global mu_s
    global pLeave
    global c_1 
    global c_2
    global newCustomers
    global sServiceTime
    c_1 = c1
    c_2 = c2
    lambda_new = l   
    mu_s = l+1
    pLeave = p
    for i in range(B_COST_ITERATIONS):
        newCustomers += \
        [np.random.poisson(\
        lambda_new*DELTA_TIME,\
        TIME_SLOTS)]
        sServiceTime += \
        [np.random.poisson(\
        mu_s*DELTA_TIME, \
        TIME_SLOTS)]
    simulate()
        
def runMultiProc():
    processes = []
    
    for c1, c2 in [(1,1),(1,2),(2,1),(1.5,1),(1,1.5),(10,1),(1,10)]:
        for l in [1000,500]:
            for pL in [0.5, 0.25, 0.75]:
                p = Process(target=main, args=(c1,c2,l,pL))
                p.start()
                processes.append(p)
    
    for p in processes:
        p.join()
    
    print "Done"

runMultiProc()    