import simpy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats
import copy
import time
plt.style.use("seaborn")

N_PROCESSORS = 5

UTILIZATION = 0.8

N_TYPES = 5

TYPE_PROB = np.ones(N_TYPES)/N_TYPES

PROCESS_T = [36,20,25,42,31]

ALLOWANCE_FACTOR = 2.5

ARRIVAL_T = np.mean(PROCESS_T)/(UTILIZATION*N_PROCESSORS)

#determine setup_time by job_type
SET_UP_TIME = np.array([[0,4,2,3,9],
                        [6,0,3,4,6],
                        [5,1,0,8,5],
                        [5,8,6,0,7],
                        [8,2,4,6,0]])

WEIGHTS = np.ones(N_TYPES)

QUEUE_MAX_CONTENT = float('inf')

TARGET_OUTPUT = 200

ACTION_SPACES = 7  #[SPT,EDD,MST,ST,CR,WSPT,#WMDD]

class Order:
    def __init__(self, ID, j_type, arrival_time, process_time, due_dates):
        self.id = ID
        self.type = int(j_type)
        self.arrival_time = arrival_time
        self.process_time = process_time
        self.due_dates = due_dates
        self.finish_time = None
        self.is_delay = None
        
class Source:
    def __init__(self, name, factory):
        self.name = name
        self.fac = factory
        self.env = factory.env
        #output: the number of jobs arrival
        self.output = 0
        self.inter_arrival = None
        
    def set_port(self, output_port):
        self.queue = output_port
        self.process = self.env.process(self.generate_order())
             
    def generate_order(self):
        id = 0
        while True:
            self.inter_arrival = np.random.exponential(ARRIVAL_T)
            yield self.env.timeout(self.inter_arrival)

            order_type = np.random.choice(5, 100, p=TYPE_PROB)[0]+1
            due_date = self.env.now + ALLOWANCE_FACTOR*PROCESS_T[order_type - 1]
            order = Order(id, order_type, self.env.now, PROCESS_T[order_type - 1], due_date)

            self.output += 1
            #yield self.env.timeout(0)  
            if self.queue.is_queue_full() == True:
                #print("{} : order {} ,type{} arrive".format(self.env.now, order.id, order.type))
                self.queue.order_arrival(order)
                self.on_exit()
                self.queue.send_to_port()
            id+=1

    def on_exit(self):
        self.fac.L.change(self.env.now, 1)
        
        
class Queue:
    def __init__(self, factory, max_content, name):
        self.name = name
        self.fac = factory
        self.env = factory.env
        self.space = []    #queue
        self.max_content = max_content
        
    def set_port(self, output_port):
        self.processors = output_port
        
    def is_queue_full(self):
        assert len(self.space) <= self.max_content
        if len(self.space) == self.max_content:
            return False
        elif len(self.space) < self.max_content:
            return True 
        
    def send_to_port(self):
        if len(self.space) == 1:
            for i in range(len(self.processors)):
                if self.processors[i].status == True :
                    self.release_order(self.fac.dispatcher.action, i)
                    break
        if len(self.space) > 1:
            for i in range(len(self.processors)):
                if self.processors[i].status == True and len(self.space) > 0:
                    self.fac.dispatcher.dispatch_for(i)
    
    def release_order(self, rule_for_sorting, processor_id):
        self.sort_queue(rule_for_sorting, processor_id)
        order = self.space[0]
        self.on_exit(order)
        self.space.pop(0)
        self.processors[processor_id].receive_order(order)
    
    def sort_queue(self, rule_for_sorting, processor_id):
        p_type = self.processors[processor_id].previous_type
        
        if rule_for_sorting == 0:  #SPT
            self.space.sort(key = lambda entity : entity.process_time)
        elif rule_for_sorting == 1: #EDD
            self.space.sort(key = lambda entity : entity.due_dates)
        elif rule_for_sorting == 2: #MST 
            if p_type != 0:
                self.space.sort(key = lambda entity : SET_UP_TIME[p_type - 1,entity.type - 1])
            else:
                self.space.sort(key = lambda entity : entity.arrival_time)
        elif rule_for_sorting == 3: #ST
            self.space.sort(key = lambda entity : entity.due_dates - entity.process_time)
        elif rule_for_sorting == 4: #CR
            self.space.sort(key = lambda entity : entity.due_dates / entity.process_time)
        elif rule_for_sorting == 5:  #WSPT
            if p_type != 0:
                self.space.sort(key = lambda entity : (SET_UP_TIME[p_type - 1,entity.type - 1]+entity.process_time) / WEIGHTS[entity.type -1])
            else:
                self.space.sort(key = lambda entity : entity.process_time / WEIGHTS[entity.type -1])
        elif rule_for_sorting == 6:  #FIFO
            self.space.sort(key = lambda entity : entity.arrival_time)
        
        elif rule_for_sorting == 7:  #WMDD
            if p_type != 0:
                self.space.sort(key = lambda entity : 
                    max(entity.due_dates - entity.process_time,entity.process_time) / WEIGHTS[entity.type -1]
                    )
            else:
                self.space.sort(key = lambda entity : 
                    max(entity.due_dates -self.env.now - entity.process_time,SET_UP_TIME[p_type - 1,entity.type - 1]+entity.process_time) / WEIGHTS[entity.type -1]
                    )
        
        #print([order.id for order in self.space])

    def order_arrival(self, order):
        self.on_entry(order)
        self.space.append(order)
    
    def on_entry(self, order):
        order.entry_queue_t = self.env.now
        
    def on_exit(self, order):
        order.exit_queue_t = self.env.now
        self.fac.W.calculate_mean(order.exit_queue_t - order.entry_queue_t)
            
class Processor:
    def __init__(self, factory, Processor_id, name):
        self.id = Processor_id
        self.name = name
        self.fac = factory
        self.status = True    #processor is free or not
        self.env = factory.env
        self.previous_type = 0
        self.utilization = L_calculator()

    def set_port(self,input_port, output_port):
        self.queue = input_port
        self.output = output_port
        
    def receive_order(self, order):
        self.status = False
        self.on_entry()
        #print("{} : order {} ,type{} start treating at processor{}".format(self.env.now, order.id, order.type, self.id))
        self.env.process(self.setup_processor(order))
    
    def setup_processor(self, order):
        setup_time = SET_UP_TIME[self.previous_type - 1][order.type - 1] if self.previous_type != 0 else 0
        yield self.env.timeout(setup_time)
        self.env.process(self.process_order(order))
        
    def process_order(self, order):
        process_time = min(ALLOWANCE_FACTOR*order.process_time,np.random.exponential(order.process_time))
        
        yield self.env.timeout(process_time)
        self.status = True
        #print("{} : order {} ,type{} finish treating at processor{}".format(self.env.now, order.id, order.type, self.id))   
        
        self.on_exit(order)
        if self.output == self.fac.sink:
            self.output.complete_order(order)
        else:
            self.output.order_arrival(order)
            
        self.previous_type = order.type
        
        self.queue.send_to_port()
    
    def on_entry(self):
        self.utilization.change(self.env.now, 1)
        
    def on_exit(self, order):
        if order.due_dates < self.env.now:
            order.is_delay = True
        else:
            order.is_delay = False
        self.utilization.change(self.env.now, -1)
        
        
class Sink:
    def __init__(self, factory):
        self.env = factory.env
        self.input = 0
        self.warehouse = []
        self.fac = factory
        self.number_of_late = np.zeros(N_TYPES)
          
    def complete_order(self, order):
        self.input += 1 
        self.on_entry()
        order.finish_time = self.env.now
        if order.is_delay == True:
            self.number_of_late[order.type - 1] += 1
        
        if self.input >= TARGET_OUTPUT:
            self.fac.terminal.succeed()
            
        self.warehouse.append(order)
    
    def on_entry(self):
        self.fac.L.change(self.env.now, -1)
        
        
class Dispatcher:
    def __init__(self, factory):
        self.env = factory.env
        self.action = 0
        self.fac = factory

    def dispatch_for(self, processor_id):
        self.on_decision_epoch()
        self.fac.queue_1.release_order(self.action, processor_id)

    def on_decision_epoch(self):
        pass


#Statistics
class L_calculator:
    def __init__(self):
        self.cumulative = 0
        self.time_lower = 0
        self.time_upper = 0
        self.L_now = 0
        self.L = 0
        
    def change(self, time_now, change):
        self.time_upper = time_now
        self.cumulative += (self.time_upper - self.time_lower) * self.L_now
        self.L_now += change
        self.time_lower = time_now

    def reset(self,time_now, L_now):
        self.cumulative = 0
        self.time_upper = 0
        self.L = 0
        self.time_lower = time_now
        self.L_now = L_now

    def caculate_mean(self):
        self.L = self.cumulative / self.time_upper
        return self.L

class W_calculator:
    def __init__(self):
        self.output = 0
        self.mean_waiting_time = 0
    
    def calculate_mean(self, waiting_t):
        self.output += 1
        self.mean_waiting_time += (waiting_t - self.mean_waiting_time)/self.output
        

class Factory:
    def build(self):  
        self.env = simpy.Environment()
        self.n_processor_1 = N_PROCESSORS
        self.queue_1 = Queue(self, QUEUE_MAX_CONTENT, 'queue_1')
        self.processors_1 = [] 
        self.source = Source('source_1', self)
        self.sink = Sink(self)
        self.dispatcher = Dispatcher(self)
        
        self.source.set_port(self.queue_1)
        self.queue_1.set_port(self.processors_1)
        self.append_processor(
            self.processors_1, self.n_processor_1, 'processor_1', self.queue_1, self.sink
        )
        
        #terminal event
        self.terminal   = self.env.event()
        
        #statistics
        self.L = L_calculator()
        self.W = W_calculator()
        
    def append_processor(self, processors, num, name, input_port, output_port):
        for i in range(num):
            processor = Processor(self, i, name)
            processor.set_port(input_port, output_port)
            processors.append(processor)
    
    def warm_up(self, warm_up_time):
        self.dispatcher.action = 6 #FIFO
        self.env.run(until = warm_up_time)
        self.terminal = self.env.event()
        wip = self.L.L_now
        self.L.reset(self.env.now,wip)
        self.W = W_calculator()
        self.sink.intput = 0
        self.sink.warehouse = []
        self.sink.number_of_late = np.zeros(N_TYPES)
        for p in self.processors_1:
            if p.status == False:
                p.utilization.reset(self.env.now,1)
            else:
                p.utilization.reset(self.env.now,0)


UPM = Factory()
performance = np.zeros((30,7))
for i in range(30):
    for j in range(ACTION_SPACES):
        UPM.build()
        UPM.warm_up(600)
        UPM.env.run(until = UPM.terminal)
        performance[i,j] = np.sum(UPM.sink.number_of_late)/UPM.sink.input

pd.DataFrame(performance,columns=['SPT','EDD','MST','ST','CR','WSPT','FIFO']).to_csv('dp_rule.csv')
"""
UPM = Factory()
UPM.build()
UPM.warm_up(600)
UPM.dispatcher.action = 6
UPM.env.run(until = UPM.terminal)
print('--------------end simulation--------------')
UPM.L.change(UPM.env.now, 0)
print('Tardy jobs:', UPM.sink.number_of_late)
print('Tardy jobs percentage:',np.sum(UPM.sink.number_of_late)/UPM.sink.input)
print('WIP:', round(UPM.L.caculate_mean(),2))
print('Mean Waiting Time:', round(UPM.W.mean_waiting_time,2))
for p in UPM.processors_1:
    p.utilization.change(UPM.env.now, 0)
    print('Processor:{} utilization:{}'.format(p.id, round(p.utilization.caculate_mean(),2)))
"""