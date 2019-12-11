from heap  import Heap
from queue import Queue

MINFLOAT = 0.5**1074
TWOMACHEPS  = 2.0*(0.5**52)  

class MyLine():
    def __init__(self):        
        self.ninline   = 0     # Present number waiting in line
        self.nfreeserv = 1              # Present number of free servers
        self.line = Queue()
        self.eventsdict = {}
        self.timeheap = Heap([])
        self.newCustomerTime = None
        self.customerServedTime = None
        
    #@profile
    def place_last_in_line(self):
        self.ninline += 1
        self.line.put(1)

    #@profile
    def call_next_in_line(self):      
        self.line.get()
        self.nfreeserv = 0
        self.ninline -= 1

    def server_freed_up(self):
        self.nfreeserv = 1

    def put_event(self, eventtype, eventtime):
        eventtim  = eventtime
        if eventtype == "New customer":     
            self.newCustomerTime = eventtim
        elif eventtype == "Returned customer":
            delta     = TWOMACHEPS*abs(eventtim) + MINFLOAT
            while True:   # Take care of ties!
                if (eventtim in self.eventsdict):
                    eventtim += delta
                else:
                    break
            self.eventsdict[self.timeheap.push(eventtim)] = eventtype        
        elif eventtype == "Customer served":  
            self.customerServedTime = eventtim    

    def retry_is_next_event(self):
        try:
            nexttime = self.timeheap.shift()
        except IndexError:
            nexttime = None
        try:
            nextevent = self.eventsdict[nexttime]
            del self.eventsdict[nexttime]
        except KeyError:
            nextevent = None
        return nextevent, nexttime
                
    def get_next_event(self):
        if self.customerServedTime == None and self.newCustomerTime == None:
            return self.retry_is_next_event()
            
        if self.customerServedTime != None and self.newCustomerTime != None:   
            if  len(self.timeheap) > 0 and \
                self.customerServedTime > self.timeheap[0] and \
                 self.newCustomerTime > self.timeheap[0]  :
                return self.retry_is_next_event()
            elif  self.customerServedTime >  self.newCustomerTime:
                resTime = self.newCustomerTime
                self.newCustomerTime = None
                return "New customer", resTime
            else:
                resTime = self.customerServedTime
                self.customerServedTime = None
                return "Customer served", resTime 
                
        if self.customerServedTime == None:
            if  len(self.timeheap) > 0 and \
                self.newCustomerTime > self.timeheap[0]:
                return self.retry_is_next_event()
            else:
                resTime = self.newCustomerTime
                self.newCustomerTime = None
                return "New customer", resTime
                
        if self.newCustomerTime == None:
            if len(self.timeheap) > 0 and \
                self.customerServedTime > self.timeheap[0]:
                return self.retry_is_next_event()
            else:
                resTime = self.customerServedTime
                self.customerServedTime = None
                return "Customer served", resTime 