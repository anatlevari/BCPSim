import numpy as np

# The following global parameters' values are set during each
# iteration of the simulation (i.e., per graph):
# lambda_new - incoming new customers rate 
lambda_new = 90
# mu_s - service rate 
mu_s = 91
# c_1 - cost of queue lenght
c_1 = 10

# Should be the same for all tests:
MAX_B_SIZE_FACTOR = 100 # |b| = 2^x, from 2^0 to 2^MAX_B_SIZE_FACTOR

# Single cost calculation parameters:
B_COST_ITERATIONS = 200
MAX_TIME = 10
DELTA_TIME = 0.1

def calcCostAtT(state, t, b):
    global c_1
    bufferredInCurIteration = \
        state['newCustomers'][t/DELTA_TIME] + \
        state['inBuffer']   

    rejectedInIteration = 0
    if bufferredInCurIteration > b:
        rejectedInIteration = bufferredInCurIteration - b
        state['rejectedUntilNow'] += rejectedInIteration
        bufferredInCurIteration = b
    
    servedInIteration = min(state['sServiceTime'][t/DELTA_TIME],\
                        bufferredInCurIteration)

    state['inBuffer'] = bufferredInCurIteration  - servedInIteration 
    return np.exp(-1.0*t)*(1*bufferredInCurIteration+ \
			1*state['rejectedUntilNow'])*DELTA_TIME
                         
def calcCostOfB(b):
    global lambda_new
    global mu_s
    systemState = {
    'bufferredOfPrevIteration' : 0,
    'rejectedUntilNow' : 0,
    'inBuffer': 0,
    'newCustomers': np.random.poisson(lambda_new*DELTA_TIME, MAX_TIME/DELTA_TIME),
    'sServiceTime': np.random.poisson(mu_s*DELTA_TIME, MAX_TIME/DELTA_TIME) }
    
    totalCost = 0.0    
    currCost = 0.0
    for t in np.arange(DELTA_TIME, MAX_TIME, DELTA_TIME):
        currCost = calcCostAtT(systemState, t, b)
        totalCost += currCost
    return totalCost

def calcAvgCostOfB(b):
    costs = np.zeros(B_COST_ITERATIONS, dtype=float)
    for i in range(0, B_COST_ITERATIONS):
        costs[i] = calcCostOfB(b)
    return np.average(costs)    
    
def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a) 
                      if smallest == element]

def calculateB():
    bToCost = np.zeros(MAX_B_SIZE_FACTOR+1, dtype=float)
    for b in np.arange(0, MAX_B_SIZE_FACTOR+1, 1):
        bToCost[b] = calcAvgCostOfB(b)
    print "b:cost"
    for i in range(len(bToCost)):
        print str(i) + ":" + str(bToCost[i])
    minVal, bArray = locate_min(bToCost)
    if len(bArray) > 1:
        print "Found more than one opt b, with value " + str(minVal) + ":"
        print bArray
    return bArray[0]
  
def main():
    global lambda_new
    global mu_s
    global c_1
    bOpt = calculateB()
    print "Found optimal buffer size: " + str(bOpt)
        
main()