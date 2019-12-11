from sys import path
path.append('./simelements')
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from multiprocessing import Process

from myLine import MyLine
from cumrandstrm    import CumulRandomStream
from genrandstrm    import GeneralRandomStream

# The following global parameters' values are set during each
# iteration of the simulation (i.e., per graph):
# lambda_new - incoming new customers rate, per minute
lambda_new = 0
# mu_s - service rate first station (G/G/1 queue), per minute
mu_s = 0.0
service_time_bound  =  0.0

# mu_r - service rate of retrail station
mu_r = 0.0
# pLeave - probability of rejected customer to leave the system
pLeave = 1.0
# c_1, c_2 - costs of queue lenght and rejected count resp.
c_1 = 0.0
c_2 = 0.0

alpha = 8
# In minutes
closingtime    = 1.0

b_size = 0
rejected_till_now = 0 

queue =  MyLine()

# -- Simulation clock time in minutes --

# Should be the same for all tests:
START_MU_R =  4999.9
DELTA_MU_R = 0.004
MAX_MU_R = 5000.004

MAX_B_SIZE = 44
FIRST_B = 38
DELTA_B = 1



# Tests per one buffer
nrealizations  =  2000

#Org
#seed = 201703271005+ 2**26 - 270317
seed = 201703271005 
seed2 = 201703271005+ 2**26 - 270317

server_service_time   = GeneralRandomStream(seed)
retrail_service_time =  GeneralRandomStream(seed2)

targetfunction_time_vector = []
targetfunction_val_vector = []
debug_print_targetfuncval = False

plt.figure(figsize=(10,5))
c1 = 40.0
c2 = 2.0
l = 10000.0
mus = 10000.01
pL = 0.1
print("c1,c2,l,mus,pL = "+str(c1)+","\
+str(c2)+","\
+str(l)+","\
+str(mus)+","\
+str(pL))

# ----------------------------------------------------------------------

#@profile
def new_customer(tim, mu_r,isReturned):
    global pLeave
    global rejected_till_now
    global c_1
    global c_2
    global queue
    global mu_s
    global targetfunction_time_vector
    global targetfunction_val_vector
    global b_size
    global server_service_time
    global retrail_service_time
    global alpha
    
    targetfunction_time_vector.append(tim)
    if queue.ninline >= b_size:
        rejected_till_now += 1 
   #     targetfunction_val_vector.append(1)
        targetfunction_val_vector.append(\
            np.exp(-1.0*alpha*tim)*(c_1*(queue.ninline+(1-queue.nfreeserv)) +\
            c_2*rejected_till_now))
        if random.random() <= pLeave: 
            return
        if mu_r > 0.0:
            retrytime = retrail_service_time.rexpo(1.0/float(mu_r))
            #print("Put event of returned customer at", (tim + retrytime))
            newEventTime = tim + retrytime
            if newEventTime <= closingtime:
                queue.put_event("Returned customer", tim + retrytime)
        return
        
    queue.place_last_in_line()

  #  targetfunction_val_vector.append(1)
    targetfunction_val_vector.append(\
            np.exp(-1.0*alpha*tim)*(c_1*(queue.ninline+(1-queue.nfreeserv)) +\
            c_2*rejected_till_now))
            
    if queue.nfreeserv > 0:
        queue.call_next_in_line()
        servtime = server_service_time.rexpo(1.0/mu_s) 
        #print("Put event of customer served at", (tim + servtime))
        newEventTime = tim + servtime
        if newEventTime <= closingtime:
            queue.put_event("Customer served", tim + servtime)

#@profile
def customer_served(tim):
    global queue
    global rejected_till_now
    global mu_s
    global targetfunction_time_vector
    global targetfunction_val_vector
    global server_service_time
    global alpha

    queue.server_freed_up()
    targetfunction_time_vector.append(tim)
    targetfunction_val_vector.append(\
        np.exp(-1.0*alpha*tim)*(c_1*(queue.ninline+(1-queue.nfreeserv)) +\
        c_2*rejected_till_now))
            
    if queue.ninline > 0:
        queue.call_next_in_line()
        servtime = server_service_time.rexpo(1.0/mu_s)        
        newEventTime = tim + servtime
        if newEventTime <= closingtime:
            queue.put_event("Customer served", tim + servtime)
        #print("Put event of customer served at", (tim + servtime))

def print_targetfunction(iteration, b, mu_r, targetfunction_val_vector, targetfunction_time_vector):
    global newCustomers
    global sServiceTime
    global pLeave
    global alpha
    
    plt.figure(figsize=(20,10))
    plt.plot(targetfunction_time_vector, targetfunction_val_vector , label='event time to target function value')
    plt.ylabel("")
    plt.xlabel("time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    mkdir_p('targetrFunction')
    plt.savefig('targetrFunction/'+\
    'i'+str(iteration)+\
    '_a'+str(alpha)+\
    '_ct'+str(closingtime)+\
    'mu_r'+str(mu_r)+\
    '_c1'+str(c_1)+\
    '_c2'+str(c_2)+\
    '_p'+str(pLeave)+\
    '_lm'+str(lambda_new)+\
    '_b'+str(b)+\
    '.png', bbox_inches='tight')
    plt.clf()
    plt.close()
    
#@profile    
def calcCostOfB(mu_r, b, iteration):
    global queue
    global lambda_new
    global rejected_till_now
    global targetfunction_time_vector
    global targetfunction_val_vector
    global b_size
    global server_service_time

    b_size = b
    
    queue = MyLine()
    rstrmarriv = CumulRandomStream(iteration*1984+1)  
    newCustomerTime = rstrmarriv.rexpo_cum(lambda_new)     
    if newCustomerTime <= closingtime:
        queue.put_event("New customer", newCustomerTime)
    #print("Put event of new  customer at", newCustomerTime)
    targetfunction_time_vector = []
    targetfunction_val_vector = []
    rejected_till_now = 0
    # Actual simulation:
    while True:
        event, clocktime = queue.get_next_event()
        if event ==  None or clocktime > closingtime:
            break;
        if event == "New customer":     
            new_customer(clocktime, mu_r, False)
            new_customer_time = rstrmarriv.rexpo_cum(lambda_new)
            if new_customer_time < closingtime:
                queue.put_event("New customer", new_customer_time)                
                #print("Put event of new  customer at", new_customer_time)
        elif event == "Returned customer":
            new_customer(clocktime, mu_r, True)            
        elif event == "Customer served":  
            customer_served(clocktime)


    # Calc BCPsim integral
    #print_targetfunction(iteration, b, mu_r, targetfunction_val_vector, targetfunction_time_vector)

    cost = targetfunction_time_vector[0] * targetfunction_val_vector[0]/2.0
    for i in range(1,len(targetfunction_time_vector)):
        dx = targetfunction_time_vector[i] - targetfunction_time_vector[i-1]    
        cost += dx * (targetfunction_val_vector[i] + targetfunction_val_vector[i-1])/2.
    del queue

    return cost

#@profile
def calcAvgCostOfB(mu_r,b):
    #with open('simGraphs/avgPerIteration.log', 'a') as avglogFile:
#        print("mu_r = " ,mu_r, file=avglogFile)
    costs = []
    global server_service_time
    global retrail_service_time
    server_service_time  = GeneralRandomStream(seed)
    retrail_service_time = GeneralRandomStream(seed2)
    for i in range(0, nrealizations):
        if i % 100 == 0:
            print("calcAvgCostOfB b = " + str(b) + " nrealizations = " + str(i))
        costs += [calcCostOfB(mu_r,b, i)]
 #       if i % 10 == 0:
 #           print("calcAvgCostOfB b = " , str(b) , " nrealizations = " ,str(i), " avg till now = ",np.mean(costs))
 #           print("calcAvgCostOfB b = " , str(b) , " nrealizations = " ,str(i), " avg till now = ",np.mean(costs), file=avglogFile)
    mean_res = np.mean(costs)    
    #print(str(b)+" costs: " + str(costs))
    print("Mean of "+str(b)+" cost: "+str(mean_res))
    return mean_res

#@profile    
def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a) 
                      if smallest == element]                          
 
def mkdir_p(mypath):
    from errno import EEXIST
    from os import makedirs,path
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
  
#@profile         
def calculateValOfMu(mu_r):    
    global c_1 
    global c_2    
    with open('simGraphs/avgPerBufferMu'+str(mu_r)+'_pl'+str(pL)+'_c1'+str(c_1)+'_c2'+str(c_2)+'.log', 'a') as avglogFile:
        bToCost = np.zeros((MAX_B_SIZE-FIRST_B)/DELTA_B + 1, dtype=float)
        xLabelsOfval = []
        for b in range(FIRST_B, MAX_B_SIZE+1, DELTA_B):
            xLabelsOfval += [str(b)]        
            mean_res = calcAvgCostOfB(mu_r,b)
            print("Mean of\t"+str(b)+"\tcost:\t"+str(mean_res), file=avglogFile)
            bToCost[(b-FIRST_B)/DELTA_B] = mean_res
        minVal, bArray = locate_min(bToCost)
        
        if len(bArray) > 1:
            print("Found more than one opt b, with value ", str(minVal), ":", bArray)
    return minVal, bArray[0]

def drawMuToBGraph(muToB, graphName):
    global lambda_new
    global mu_s
    global pLeave
    global c_1 
    global c_2    
    global nrealizations
    plt.plot(np.arange(START_MU_R, MAX_MU_R, DELTA_MU_R), muToB)
    plt.ylabel("b")
    plt.xlabel("mu_r")
    plt.savefig('simGraphs/bGraph'+graphName+'.png', bbox_inches='tight')
    plt.clf()  

def drawMuToValGraph(muToVal, graphName):
    global lambda_new
    global mu_s
    global pLeave
    global c_1 
    global c_2
    global nrealizations
    global MAX_MU_R
    plt.plot(np.arange(START_MU_R, MAX_MU_R, DELTA_MU_R), muToVal)
    plt.ylabel("Val. func.")
    plt.xlabel("mu_r")
    plt.savefig('simGraphs/valGraph'+graphName+'.png', bbox_inches='tight')
    plt.clf()

#@profile  
def simulate():
    mkdir_p('simGraphs')
    graphName = '_ct'+str(closingtime)+\
    '_nr'+str(nrealizations)+\
    '_mmur'+str(MAX_MU_R)+\
    '_dmur'+str(DELTA_MU_R)+\
    '_mb'+str(MAX_B_SIZE)+\
    '_fb'+str(FIRST_B)+\
    '_db'+str(DELTA_B)+\
    '_c1'+str(c_1)+\
    '_c2'+str(c_2)+\
    '_lm'+str(lambda_new)+\
    '_p'+str(pLeave)
    # Iterate over different mu_r values
    with open('simGraphs/bVXY'+graphName+'.txt', 'a') as b_outfile:
        with open('simGraphs/bGXY'+graphName+'.txt', 'a') as v_outfile:
            print("mu_r\tb", file=b_outfile)
            print("mu_r\tval", file=v_outfile)

            print("Simulation start")
            muToB =  np.zeros(int(math.ceil((MAX_MU_R-START_MU_R)/DELTA_MU_R)), dtype=float)
            muToVal = np.zeros(int(math.ceil((MAX_MU_R-START_MU_R)/DELTA_MU_R)), dtype=float)
            for mu_r in np.arange(START_MU_R, MAX_MU_R, DELTA_MU_R):
                indexMur = int((mu_r-START_MU_R)/DELTA_MU_R)
                muToVal[indexMur], muToB[indexMur] = calculateValOfMu(mu_r)
                muToB[indexMur] = (muToB[indexMur]* DELTA_B) + FIRST_B
                print("{}\t{}".format(mu_r,muToB[indexMur]), file=b_outfile)
                print("{}\t{}".format(mu_r,muToVal[indexMur]), file=v_outfile)
                b_outfile.flush()
                v_outfile.flush()
            drawMuToValGraph(muToVal,graphName)
            drawMuToBGraph(muToB,graphName)
#@profile   
def main(c1,c2,l,mus,p):
    global lambda_new
    global mu_s
    global service_time_bound
    global pLeave
    global c_1 
    global c_2
    c_1 = c1
    c_2 = c2
    lambda_new = l   
    mu_s = mus
    service_time_bound = mus*100
    pLeave = p
    simulate()
    print("Done")

main(c1,c2,l,mus, pL)