import simpy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display
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

QUEUE_MAX_CONTENT = 40

TARGET_OUTPUT = 200

ACTION_SPACES = 7  #[SPT,EDD,MST,ST,CR,WSPT,#WMDD]

DELTA_L = 5

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
        self.space_state = np.zeros(max_content)
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
                    self.release_order(i)
                    break
        if len(self.space) > 1:
            for i in range(len(self.processors)):
                if self.processors[i].status == True and len(self.space) > 0:
                    self.fac.dispatcher.dispatch_for(i)
    
    def release_order(self, processor_id, order_id = 0):
        order = self.space[order_id]
        self.on_exit(order)
        del self.space[order_id]
        self.processors[processor_id].receive_order(order)

    def order_arrival(self, order):
        self.on_entry(order)
        self.space.append(order)
    
    def on_entry(self, order):
        self.space_state[len(self.space)] = 1
        order.entry_queue_t = self.env.now
        
    def on_exit(self, order):
        self.space_state[len(self.space)-1] = 0
        order.exit_queue_t = self.env.now
        self.fac.W.calculate_mean(order.exit_queue_t - order.entry_queue_t)
            
class Processor:
    def __init__(self, factory, Processor_id, name):
        self.id = Processor_id
        self.name = name
        self.fac = factory
        self.status = True    #processor is free or not
        self.env = factory.env
        self.type_now = 0
        self.previous_type = 0
        self.available_t = 0
        self.utilization = L_calculator()

    def set_port(self,input_port, output_port):
        self.queue = input_port
        self.output = output_port
        
    def receive_order(self, order):
        self.status = False
        self.on_entry()
        #print("{} : order {} ,type{} start treating at processor{}".format(self.env.now, order.id, order.type, self.id))
        self.type_now = order.type
        self.setup_time = SET_UP_TIME[self.previous_type - 1][order.type - 1] if self.previous_type != 0 else 0
        self.process_time = min(ALLOWANCE_FACTOR*order.process_time,np.random.exponential(order.process_time))
        self.available_t = self.env.now + self.setup_time + self.process_time

        self.fac.compute_reward(self.env.now, self.setup_time+self.process_time, order)
        self.env.process(self.setup_processor(order))
    
    def setup_processor(self, order):
        yield self.env.timeout(self.setup_time)
        self.env.process(self.process_order(order))
        
    def process_order(self, order):
        
        yield self.env.timeout(self.process_time)
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
            self.fac.decision_epoch = True
            
        self.warehouse.append(order)
    
    def on_entry(self):
        self.fac.L.change(self.env.now, -1)
        
        
class Dispatcher:
    def __init__(self, factory):
        self.env = factory.env
        self.fac = factory
        self.dispatch_time = 0
        self.listeners = []

    def dispatch_for(self, processor_id):
        self.dispatch_time += 1
        self.on_decision_epoch(processor_id)

    def on_decision_epoch(self, processor_id):
        if  self.fac.is_warm == True:
            self.fac.queue_1.release_order(processor_id)
        else:
            self.fac.decision_epoch = True
            self.listeners.append(self.fac.processors_1[processor_id])

    def execute_decision(self, action):
        for listener in self.listeners:
            listener.queue.release_order(listener.id, action)
        
        self.listeners = []


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

        # time to make a decision
        self.decision_epoch = False
        
        # next event algo --> simpy.env.step
        self.next_event = self.env.step

        #reward
        self.reward = 0
        
        #statistics
        self.L = L_calculator()
        self.W = W_calculator()
        
    def append_processor(self, processors, num, name, input_port, output_port):
        for i in range(num):
            processor = Processor(self, i, name)
            processor.set_port(input_port, output_port)
            processors.append(processor)
    
    def warm_up(self, warm_up_time):
        self.is_warm = True
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

    def reset(self):
        self.build()
        self.warm_up(600)
        self.is_warm = False
        while not self.decision_epoch:
            self.next_event()
        self.decision_epoch = False
        state = self.get_state()
        return state, self.queue_1.space_state

    def _pass_action(self, action):
        ## execute the action ##
        assert self.queue_1.space_state[action] == 1
        self.dispatcher.execute_decision(action)

    def step(self, action):
        self._pass_action(action)
        while not self.decision_epoch:
            self.next_event()

        self.decision_epoch = False
        state = self.get_state()
        reward = self.get_reward()
        done = self.terminal.triggered
        inf = None
        if done:
            inf = np.sum(self.sink.number_of_late)/self.sink.input
        return state, reward, done, self.queue_1.space_state, inf

    def get_state(self):
        #state
        state = np.zeros((QUEUE_MAX_CONTENT,4+N_PROCESSORS))
        for i,order in enumerate(self.queue_1.space):
            setup_time = SET_UP_TIME[self.dispatcher.listeners[0].previous_type - 1][order.type - 1]
            state[i,0] = order.type
            state[i,1] = order.arrival_time
            state[i,2] = order.process_time
            state[i,3] = order.due_dates - self.env.now
            state[i,4] = order.due_dates - order.process_time - setup_time - self.env.now
            j = 1
            for processor in self.processors_1:
                if processor.status == False:
                    setup_time = SET_UP_TIME[processor.type_now - 1][order.type - 1]
                    state[i,4+j] = order.due_dates - order.process_time - setup_time\
                        - processor.available_t

                    j+=1
        return state

    #reward method
    def compute_reward(self, start_process_t, process_t, order):
        weights = np.array(WEIGHTS, dtype = np.float32)
        weights = weights / np.sum(weights)
        Latest_start_process_t =  order.due_dates - process_t
        max_delay = Latest_start_process_t - order.arrival_time
        reward = (Latest_start_process_t - start_process_t)/max_delay if max_delay > 0 else -100
        weighted_reward = weights[order.type - 1] * reward if reward >= 0 else weights[order.type - 1] * -100

        self.reward += weighted_reward

        return self.reward

    def get_reward(self):
        reward = self.reward
        self.reward = 0

        return reward

"""
UPM = Factory()
state, legal = UPM.reset()
while True:
    action = 0
    print('------------------------------------------------')
    print(legal)
    print('state:')
    display(pd.DataFrame(state))
    print('action:',action)
    next_state, reward, done, legal = UPM.step(action)
    print('reward:', reward)
    state = next_state
    if done:
        print('------------------------------------------------')
        print('makespan', UPM.env.now)
        print('terminal state:')
        display(pd.DataFrame(state))
        break
"""