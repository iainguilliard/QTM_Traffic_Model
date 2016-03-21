from __future__ import division
import math
import random
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import qtm_plot as qtm
import argparse
import json as json
import pandas as  pd
from scipy.stats import gaussian_kde
from scipy.optimize import leastsq

DEBUG_PLOT = False

def random():
    return np.random.normal()

class signal():

    def __init__(self,scedule,position,amber_time):
        scedule_state = [0]
        scedule_time = [0]
        first_i = 0
        if scedule[0] == 0:
            scedule_state[0] = 1
            first_i = 1
        for t in scedule[first_i:]:
            if scedule_state[-1] == 0:
                scedule_state.append(1)
            else:
                scedule_state.append(0)
            scedule_time.append(scedule_time[-1] + t)
        scedule_state[-1] = 0
        #print scedule_time
        #print scedule_state
        self.scedule = interp.interp1d(scedule_time,range(len(scedule_state)))
        self.scedule_time = scedule_time
        self.scedule_state = scedule_state
        self.scedule_state_f = interp.interp1d(scedule_time,scedule_state,kind='zero')
        self.position = position
        self.amber_time = amber_time
        sced = interp.interp1d(scedule_time,self.scedule_state,kind='zero')
        sced_t = np.linspace(self.scedule_time[0],self.scedule_time[-1],10000)
        if DEBUG_PLOT:
            plt.figure(figsize=(20,2))
            plt.plot(sced_t,sced(sced_t))

    def red(self,t):
        return self.state(t) > 0 or (t+self.amber_time < self.scedule_time[-1] and self.state(t+self.amber_time) > 0)

    def next_red_time(self,t):
        i = int(self.scedule(t))
        if self.scedule_state[i] > 0:
            return t
        else:
            if i < len(self.scedule_time):
                return self.scedule_time[i+1]
            else:
                return 1e6

    def state(self,t):
        if self.scedule_time[-1] >= t:
            return self.scedule_state[int(self.scedule(t))]
        else:
            return 0

    def plot(self,plt):
        # print self.scedule_time
        # print self.scedule(self.scedule_time)
        sced_t = np.linspace(self.scedule_time[0],self.scedule_time[-1],10000)
        plt.plot(sced_t,self.scedule_state_f(sced_t),c='b')

class link():
    def __init__(self, index, signal = None):
        self.q_in = [0]
        self.q_out= [0]
        self.q_in_t = [0]
        self.q_out_t= [0]
        self.index = index
        self.signal = signal

    delta = 1e-6

    def __add_car(self, q, q_t, t):
        q_t.append(t)
        q.append(q[-1] + 1)

    def car_in(self, t):
        self.__add_car(self.q_in, self.q_in_t, t)

    def car_out(self,t):
        self.__add_car(self.q_out, self.q_out_t, t)

    def __get(self, q, q_t, time):
        x = q_t
        y = q
        if time[0] < q_t[0]:
            x = [time[0]] + x
            y = [q[0]] + y
        if time[-1] > q_t[-1]:
            x = x + [time[-1]]
            y = y + [q[-1]]

        q_f = interp.interp1d(x,y,kind='zero')
        return q_f(time)

    def arrivals(self,time):
        return self.__get(self.q_in,self.q_in_t,time)

    def departures(self,time):
        return self.__get(self.q_out,self.q_out_t,time)


class path():

    def __init__(self,length,free_flow_speed,signals,inflow,links,link_offsets,indexes):
        self.length = length
        self.free_flow_speed = free_flow_speed
        self.cars = []
        self.cars_travelling = []
        self.signals = signals
        self.inflow = inflow
        self.indexes = indexes
        self.free_flow_travel_time = length / free_flow_speed
        self.links = links
        self.link_offsets = link_offsets
        self.link_f = self.link_f = interp.interp1d(link_offsets,range(len(links) + 1),kind='zero')
        self.car_xf = None
        self.car_tf = None

    def add(self,car):
        self.cars.append(car)
        self.cars_travelling.append(car)

    def remove(self,car):
        self.cars_travelling.remove(car)

    def get_link_index(self,x):
        if 0 <= x < self.length:
            return self.indexes[int(self.link_f(x))]
        else:
            return None

    def get_link(self,x):
        if 0 <= x <= self.length:
            return self.links[int(self.link_f(x))]
        else:
            return None

    def get_link_offset(self,x):
        if self.get_link_index(x) is None: print x,self.length
        return self.link_offsets[int(self.link_f(x))]

    def nextCarDistance(self,car):
        i = self.cars_travelling.index(car)
        #print 'i=',i
        if i > 0:
            #print 'dist=',self.cars[i-1].position - self.cars[i].position
            #print 'i-1:',self.cars_travelling[i-1].position
            #print 'i-1:',(self.cars_travelling[i-1].position - self.cars_travelling[i-1].length)
            #print 'i  :',self.cars_travelling[i].position
            return (self.cars_travelling[i-1].position - self.cars_travelling[i-1].length) - self.cars_travelling[i].position
            #return self.cars_travelling[i-1].position - self.cars_travelling[i].position
        else:
            return 1e9

    def nextCarSpeed(self,car):
        i = self.cars_travelling.index(car)
        #print 'i=',i
        if i > 0:
            #print 'dist=',self.cars[i-1].position - self.cars[i].position
            return self.cars_travelling[i-1].speed
        else:
            return 0

    def distanceToStopLine(self,car,t):
        sig = None

        for s in reversed(self.signals):
            if s.red(t):
                if car.position < s.position: sig = s
        if sig is None:
            return 1e9
        else:
            stop_line_distance = sig.position - car.position
            time_to_next_red = sig.next_red_time(t) - t
            if car.speed * time_to_next_red > stop_line_distance:
                return 1e9
            else:
                return stop_line_distance

    def time_to_next_red(self,car,t):
        sig = None

        for s in reversed(self.signals):
            if s.red(t):
                if car.position < s.position: sig = s
        if sig is None:
            return 1e9
        else:
            time_to_next_red = t - sig.next_red_time(t)
            return time_to_next_red - t

    def get_stops(self,threshhold):
        total_stops = []
        for car in self.cars:
            total_stops.append(car.get_stops(threshhold))
        return total_stops

    def get_flow_density(self,t,dt,x,dx,plt=None,theta=0,kwargs={}):
        sum_q = 0
        sum_k = 0
        j=0
        for car in self.cars:
            #print car.time_log
            #print car.position_log
            t0,t1,x0,x1 = None,None,None,None
            i0,i1 = None,None
            if car.time_log[0] < t + dt and car.time_log[-1] > t:
                for i,car_t in enumerate(car.time_log):
                    if t <= car_t <= t + dt:
                        car_x =  car.position_log[i]
                        if x <= car_x <= x + dx:
                            if x0 is None:
                                if i > 0:
                                    alpha_x = (t - car.time_log[i - 1]) / (car_t - car.time_log[i - 1])
                                    alpha_t = (x - car.position_log[i - 1]) / (car_x - car.position_log[i - 1])
                                    if car.time_log[i - 1] < t and car.position_log[i - 1] >= x:
                                        t0 = t

                                        x0 = car.position_log[i - 1] + alpha_x * (car_x - car.position_log[i - 1])
                                    elif car.time_log[i - 1] >= t and car.position_log[i - 1] < x:
                                        x0 = x

                                        t0 = car.time_log[i - 1] + alpha_t * (car_t - car.time_log[i - 1])
                                    else:
                                        x0 = x # car.position_log[i - 1] + alpha_x * (car_x - car.position_log[i - 1])
                                        alpha_t = (x0 - car.position_log[i - 1]) / (car_x - car.position_log[i - 1])
                                        t0 = car.time_log[i - 1] + alpha_t * (car_t - car.time_log[i - 1])
                                        if t0 < t:
                                            t0 = t
                                            alpha_x = (t0 - car.time_log[i - 1]) / (car_t - car.time_log[i - 1])
                                            x0 = car.position_log[i - 1] + alpha_x * (car_x - car.position_log[i - 1])

                                else:
                                    x0 = car_x
                                    t0 = car_t
                                i0 = i
                            else:
                                x1 = car_x
                                t1 = car_t
                                i1 = i
                        elif car_x > x + dx:
                            break
                    elif car_t > t + dt:
                        break
            if None not in (t0,t1,x0,x1): #if t0 is not None and x0 is not None and t1 is not None and x1 is not None:
                if i1 < len(car.time_log) - 1:
                    if car.time_log[i1 + 1] > t + dt and car.position_log[i1 + 1] <= x + dx:
                        t1 = t + dt
                        alpha_x = (t + dt - car.time_log[i1]) / (car.time_log[i1 + 1] - car.time_log[i1])
                        x1 = car.position_log[i1] + alpha_x * (car.position_log[i1 + 1] - car.position_log[i1])

                    elif car.time_log[i1 + 1] <= t + dt and car.position_log[i1 + 1] > x + dx:
                        x1 = x + dx
                        alpha_t = (x + dx - car.position_log[i1]) / (car.position_log[i1 + 1] - car.position_log[i1])
                        t1 = car.time_log[i1] + alpha_t * (car.time_log[i1 + 1]- car.time_log[i1])
                    elif car.time_log[i1 + 1] <= t + dt and car.position_log[i1 + 1] <= x + dx:
                        t1 = car.time_log[i1+1]
                        x1 = car.position_log[i1+1]
                    else:
                        x1 = x + dx
                        alpha_t = (x1 - car.position_log[i1]) / (car.position_log[i1 + 1] - car.position_log[i1])
                        t1 = car.time_log[i1] + alpha_t * (car.time_log[i1 + 1]- car.time_log[i1])
                        if t1 > t + dt:
                            t1 = t+ dt
                            alpha_x = (t1 - car.time_log[i1]) / (car.time_log[i1 + 1] - car.time_log[i1])
                            x1 = car.position_log[i1] + alpha_x * (car.position_log[i1 + 1] - car.position_log[i1])

                else:

                    x1 = car.position_log[i1] # x + dx
                    t1 = car.time_log[i1] # t + dt
                l_i = x1 - x0
                t_i = t1 - t0
                sum_q += l_i
                sum_k += t_i
            #print j,t0,t1,x0,x1
                if plt is not None:
                    #plt.plot(car.time_log[i0-1:i1+2], car.position_log[i0-1:i1+2],'k-',marker='.',hold=True)
                    #plt.plot([t0],[x0],'rx',hold=True)
                    #plt.plot([t1],[x1],'rx')
                    plt.plot([t0] + car.time_log[i0:i1+1] + [t1],[x0] + car.position_log[i0:i1+1] + [x1],**kwargs)
                j += 1
        if j < theta:
            return 0,0
        else:
            sum_q /= (dt * dx)
            sum_k /= (dt * dx)
            return sum_q, sum_k

    def get_flow_density2(self,t,dt,x,dx):
        sum_q = 0
        sum_k = 0
        j=0
        for car in self.cars:
            #print car.time_log
            #print car.position_log
            t0 = None
            t1 = None
            x0 = None
            x1 = None
            for i,car_t in enumerate(car.time_log):
                if t <= car_t <= t + dt:
                    if t0 is None:
                        t0 = t0
                        t1 = t0
                    else:
                        t1 = car_t
                    car_x =  car.position_log[i]
                    if x <= car_x <= x + dx:
                        if x0 is None:
                            x0 = x
                            x1 = x
                        else:
                            x1 = car_x
            if None not in [t0,t1,x0,x1]:
                l_i = x1 - x0
                t_i = t1 - t0
                sum_q += l_i
                sum_k += t_i
            #print j,t0,t1,x0,x1
            j += 1
        sum_q /= (dt * dx)
        sum_k /= (dt * dx)
        return sum_q, sum_k

    def get_qk(self,t,dt,x,dx,plt,xr,tr):

        if self.car_xf is None or self.car_tf is None:
            self.car_xf = []
            self.car_tf = []
            for car in self.cars:
                self.car_xf.append(interp.interp1d([-1e9,0] + car.time_log, [0,0] + car.position_log, bounds_error=False, fill_value=car.position_log[-1]))
                self.car_tf.append(interp.interp1d([-1e9,0] + car.position_log,[0,0] + car.time_log, bounds_error=False, fill_value=car.time_log[-1]))

        sum_q = 0
        sum_k = 0
        print 'square=',t,dt,x,dx
        for i,car in enumerate(self.cars):
            #print self.car_xf[i](t + dt),self.car_xf[i](t)
            l_i = self.car_xf[i](t + dt) - self.car_xf[i](t)
            t_i = self.car_tf[i](x + dx) - self.car_tf[i](x)
            sum_q += l_i
            sum_k += t_i
            if l_i > 0  and t_i > 0:
                plt.plot(self.car_tf[i](xr),self.car_xf[i](tr))
            #print i,l_i,t_i
        q = sum_q / (dt * dx)
        k = sum_k / (dt * dx)
        return q, k



class car():

    def __init__(self, path, t, init_speed,maxspeed, params=None):


        self.position = 0
        self.speed = init_speed
        self.width = 1.7
        self.length = 3             #4.667  # 3 + 2 * random()
        self.maxSpeed = maxspeed    # IDM param: v0, desired speed when driving on a free road (m/s)
        self.s0 = 2                 # IDM param: s0, minimum bumper-to-bumper distance to the front vehicle (m)
        self.timeHeadway = 1.5      #1.5# IDM param: T, desired safety time headway when following other vehicles (s)
        self.maxAcceleration = 2    #1 # IDM param: a, acceleration in everyday traffic (m/s^2)
        self.maxDeceleration = 3    # IDM param: b, "comfortable" braking deceleration in everyday traffic (m/s^2)
        if params is not None:
            if 'random' in params and params['random'] is not None:
                randomf = random()
            else:
                randomf = 0
            self.width = params['width']
            self.length = params['length'] + randomf
            if 'maxSpeed' in params:
                self.maxSpeed = params['maxspeed']
            self.timeHeadway = params['timeHeadway']
            self.s0 = params['s0']
            self.maxAcceleration = params['maxAcceleration']
            self.maxDeceleration = params['maxDeceleration']
        self.distanceToStopLine = 1e6
        self.path = path
        self.path.add(self)
        self.link = None
        self.t = t
        self.start_t = t
        self.travel_time = 0
        self.delay = 0

        self.position_log = [0]
        self.acceleration_log = [0]
        self.speed_log = [init_speed]
        self.time_log = [t]
        self.stopLineDistance_log = [0]
        self.nextRedTime_log = [0]
        self.link_log = [-1]
        self.link_indexes = None
        self.links = None
        self.link_position_log = [0]

    def nextCarDistance(self):
        return self.path.nextCarDistance(self)

    def nextCarSpeed(self):
        return self.path.nextCarSpeed(self)

    def nextStopLineDistance(self,t):
        return self.path.distanceToStopLine(self,t)

    def nextRedTime(self,t):
        return self.path.time_to_next_red(self,t)


    def getAcceleration(self,t):
        nextCarDistance = self.nextCarDistance() #self.trajectory.nextCarDistance;
        nextCarSpeed = self.nextCarSpeed()
        distanceToNextCar = max(nextCarDistance, 1e-6)
        distanceToStopLine = self.nextStopLineDistance(t) #self.path.length - self.position
        #print 'd',distanceToNextCar, distanceToStopLine
        a = self.maxAcceleration
        b = self.maxDeceleration
        deltaSpeed = self.speed - nextCarSpeed
        freeRoadCoeff = (self.speed / self.maxSpeed) ** 4
        distanceGap = self.s0
        timeGap = self.speed * self.timeHeadway
        breakGap = self.speed * deltaSpeed / (2 * math.sqrt(a * b))
        safeDistance = distanceGap + timeGap + breakGap
        busyRoadCoeff = (safeDistance / distanceToNextCar) ** 2
        safeIntersectionDistance = 1 + timeGap + (self.speed ** 2) / (2 * b)
        intersectionCoeff = (safeIntersectionDistance / distanceToStopLine) ** 2
        coeff = 1 - freeRoadCoeff - busyRoadCoeff - intersectionCoeff
        return self.maxAcceleration * coeff



    def move(self,t,DT):
        if self.position < self.path.length:
            #print t <= self.t < t + DT,t,self.t,t+DT
            if t <= self.t < t + DT:
                prev_pos = self.position
                delta = t + DT - self.t
                acceleration = self.getAcceleration(self.t)
                if self.speed + acceleration * delta < 0:
                    self.position += (-0.5 * self.speed ** 2 )/ acceleration
                    self.speed = 0
                else:
                    self.speed += acceleration * delta
                    step = self.speed * delta + 0.5 * acceleration * delta ** 2
                    self.position += step

                if self.position > self.path.length:
                    error = (self.path.length - prev_pos) / (self.position - prev_pos)
                    self.position = self.path.length
                    self.path.remove(self)
                    self.t += DT * error
                    self.travel_time = self.t - self.start_t
                    #print self.travel_time
                    self.delay = max(0,self.travel_time - self.path.free_flow_travel_time)
                    #print self.path.indexes,self.start_t,self.delay
                    self.link.car_out(self.t)
                    self.link = None
                else:
                    self.t += delta #self.time_log[-1]
                    if self.link is not None:
                        prev_link = self.link
                        self.link = self.path.get_link(self.position)
                        if self.link != prev_link:
                            #print 'going from %d to %d' %(prev_link.index,self.link.index)
                            if prev_link is not None:
                                prev_link.car_out(self.t)
                            if self.link is not None:
                                self.link.car_in(self.t)
                                self.links.append(self.link)
                                self.link_indexes.append(self.link.index)
                    else:
                        self.link = self.path.links[0]
                        self.links = [self.link]
                        self.link.car_in(t)
                        self.link_indexes = [self.path.indexes[0]]
                self.position_log.append(self.position)
                self.speed_log.append(self.speed)
                self.acceleration_log.append(acceleration)
                self.time_log.append(self.t)
                self.stopLineDistance_log.append(self.nextStopLineDistance(self.t))
                self.nextRedTime_log.append(self.nextRedTime(self.t))
                i = self.path.get_link_index(self.position)
                if self.link is None or i is None:
                    self.link_log.append(-1)
                    self.link_position_log.append(0)
                else:
                    self.link_log.append(i)
                    self.link_position_log.append(self.position - self.path.get_link_offset(self.position))
            else:
                self.link_log.append(-1)
                self.link_position_log.append(0)
        else:
            self.link_log.append(-1)
            self.link_position_log.append(0)

    def get_stops(self,threshold):
        stopped = np.array(self.speed_log) < threshold
        #print stopped
        #print stopped[:-1] & ~stopped[1:]
        #return (stopped[:-1] & ~stopped[1:]).nonzero()[0]
        return np.count_nonzero(~stopped[:-1] & stopped[1:])

    def plot(self,plt):
        plt.plot(self.time_log,self.speed_log)
        plt.title('Car speed')
        plt.xlabel('time (s)')
        plt.ylabel('speed (m/s)')



def find_paths(data):
    paths = []
    for flow in  data["Flows"].keys():
        f = flow.split('_')
        q1 = int(f[0])
        q2 = int(f[1])
        found_q1 = False
        found_q2 = False
        p1 = None
        p2 = None
        for p in paths:
            if q1 in p:
                found_q1 = True
                p1 = p

            if q2 in p:
                found_q2 = True
                p2 = p

        if found_q1 and found_q2:
            if p1 == p2:
                print "Error! queue %d->%d already added!" % (q1,q2)
            else:
                paths.remove(p2)
                p1 += p2
        elif found_q1:
            p1.insert(p1.index(q1)+1,q2)
        elif found_q2:
            p2.insert(p2.index(q2),q1)
        else:
            paths.append([q1,q2])
        #print paths
    return paths



class microsim:

    def __init__(self, data, free_flow_speed, time_factor=1, amber_time=5, road_width=0, lost_time = 0,
                 no_lost_time = False, params=None, slt_weight=0):
        self.cars = []
        self.time = []
        self.signals = []
        self.data_signals = []
        self.data_time = []
        self.amber_time = amber_time
        self.free_flow_speed = free_flow_speed
        self.time_factor = time_factor
        self.load_file(data, free_flow_speed, time_factor=time_factor, road_width=road_width, lost_time=lost_time,
                       no_lost_time=no_lost_time, slt_weight=slt_weight)
        self.params = idm_params
        self.slt_weight = slt_weight


    def load_file(self, data, free_flow_speed, time_factor = 1, road_width=0, lost_time = 0, no_lost_time = False, slt_weight = 0.5):
        data_DT = [x * time_factor for x in data['Out']['DT']]
        self.data_time = [x * time_factor for x in data['Out']['t']]
        q_paths = find_paths(data)
        self.paths = []
        self.links = [None for i in range(len(data['Queues']))]
        if 'lost_time' in data['Out'] and no_lost_time == False:
            lost_time = data['Out']['lost_time']  * time_factor
        print 'lost_time :', lost_time
        self.lost_time = lost_time
        print 'amber_time:', self.amber_time
        for q_path in q_paths:

            distance = 0
            signals = []
            links = []
            link_offsets = []
            for i in q_path:
                queue = data['Queues'][i]
                q_length = free_flow_speed * queue['Q_DELAY'] * time_factor
                link_offsets.append(distance)
                distance += q_length
                #print i,q_length
                sig = None
                if queue['Q_P'] is not None:
                    l = queue['Q_P'][0]
                    ph = queue['Q_P'][1]
                    #print (l,ph),data['Lights'][l]['P_MIN'][ph],data['Lights'][l]['P_MAX'][ph]
                    p_state = [1 if x < 0.5 else 0 for x in data['Out']['p_{%d,%d}' % (l,ph)]]

                    if p_state[0] == 0:
                        state = 0
                    else:
                        state = 1
                    schedule = []
                    t = 0
                    if state == 1:
                        schedule.append(0)
                    active = False
                    # startup_lost_time_weight = 0.5
                    for j,x in enumerate(p_state[1:]):
                        t += data_DT[j]
                        if x != state:

                            if active == False:
                                if x == 0: # Green
                                    t -= lost_time * slt_weight
                                active = True
                            elif x == 0: # Green
                                t -= lost_time * slt_weight
                            else:
                                t += lost_time * slt_weight
                            schedule.append(t)
                            t=0

                        state = x
                    if np.sum(schedule) < self.data_time[-1]:
                        schedule.append(t)
                    if np.sum(schedule) < self.data_time[-1]:
                        schedule.append(self.data_time[-1] - np.sum(schedule))
                    #print self.data_time[-1]
                    #if schedule[-1] < self.data_time[-1]:
                    #    schedule.append(self.data_time[-1])
                    sig = signal(schedule,distance - road_width,self.amber_time)
                    sced_p = interp.interp1d(self.data_time,p_state,kind='zero')
                    sced_t = np.linspace(self.data_time[0],self.data_time[-1],10000)
                    if DEBUG_PLOT:
                        plt.plot(sced_t,sced_p(sced_t))
                        plt.ylim(-0.1,1.1)
                        plt.grid(True)
                        plt.xlim(0,100)
                        plt.show()
                    signals.append(sig)
                    self.data_signals.append(p_state)
                    self.signals.append(sig)
                l = link(i,signal)
                links.append(l)
                self.links[i] = l
            link_offsets.append(distance)
            #print [x/data_DT[k] for k,x in enumerate(data['Out']['q_{%d,in}' % q_path[0]]) ]
            inflow = interp.interp1d(self.data_time,[x/data_DT[k] for k,x in enumerate(data['Out']['q_{%d,in}' % q_path[0]]) ],kind='zero')
            self.paths.append(path(distance,free_flow_speed,signals,inflow,links,link_offsets,q_path))


    def indexes_to_paths(self,indexes):
        paths = []
        for j,path in enumerate(self.paths):
            append = False
            for i in indexes:
                if i in path.indexes:
                    append = True
            if append:
                paths.append(path)
            else:
                paths.append(None)
        return paths

    def indexes_to_path_indexes(self,indexes):
        paths = []
        for j,path in enumerate(self.paths):
            append = False
            for i in indexes:
                if i in path.indexes:
                    append = True
            if append:
                paths.append(j)
            else:
                paths.append(None)
        return paths


    def simulate(self,DT,indexes = None):
        self.DT = DT
        if indexes is not None:
            paths = self.indexes_to_paths(indexes)
        else:
            paths = self.paths
        for path in paths:
            if path is not None:
                self.time = np.arange(0,self.data_time[-1],DT)
                self.cars.append(self.microsim_road(path,self.time,DT))
            else:
                self.cars.append([])



    def microsim_road(self,road, time, DT):
        cars = []
        EPSILON = 1e-6
        cars_in = 0
        t = 0
        while t < time[-1] and road.inflow(t + 1.0/road.inflow(t)) > EPSILON:
            cars.append(car(road,t,road.free_flow_speed,road.free_flow_speed,params=self.params))
            #print 'adding car at t=%f' % t,'(%f)' % t
            t += 1.0/road.inflow(t)
            #if t % (1.0/road.inflow(t)) < EPSILON:
            #    cars.append(car(road,t,road.free_flow_speed,road.free_flow_speed))

            cars_in += 1
        #print 'cars_in',cars_in
        for t in time:
            #print 't=',t % in_flow_rate,
            for c in cars:
                #print '%2.2f' % c.position,
                c.move(t,DT)
        return cars

    def average_delay(self):
        total_delay = 0
        total_cars = 0
        for path in self.cars:
            total_cars += len(path)
            for car in path:
                total_delay += car.delay
        return total_delay / total_cars

    def delay(self):
        delay = []
        for path in self.cars:
            for car in path:
                delay.append(car.delay)
        return delay

    def total_travel_time(self):
        total_travel_time = 0
        for path in self.cars:
            for car in path:
                total_travel_time += car.travel_time
        return total_travel_time

    def total_traffic_in(self):
        total_cars = 0
        for path in self.cars:
            total_cars += len(path)
        return total_cars


    def plot(self,plt,index,traces=1.0,green_bands=None,title=' '):

        road = self.paths[index]

        if traces is not None and traces != 'off':
            trace_alpha = float(traces)
            for i,c in enumerate(self.cars[index]):
                plt.plot(c.time_log,c.position_log,label='position %d' % i,alpha = trace_alpha)

        for l in road.signals:
            for i,x in enumerate(l.scedule_state):
                if x > 0:
                    if i < len(l.scedule_state) - 1:
                        plt.plot([l.scedule_time[i],l.scedule_time[i+1]],[l.position,l.position],'k',lw=2,solid_capstyle='round')
                    else:
                        plt.plot([l.scedule_time[i],l.scedule_time[-1]],[l.position,l.position],'k',lw=2,solid_capstyle='round')

        if green_bands is not None:
            if len(green_bands) > 0:
                alpha = green_bands[0]
            else:
                alpha  = 0.5
            slt_time = self.slt_weight * self.lost_time
            for j,l in enumerate(road.signals):
                if j < len(road.signals) - 1:
                    link_length = road.signals[j+1].position - l.position
                else:
                    link_length = road.length - l.position
                for i,x in enumerate(l.scedule_state[:-1]):
                    if x > 0 and i+2 < len(l.scedule_state):
                            t0 = l.scedule_time[i+1] + slt_time
                            t1 = l.scedule_time[i+1] + link_length / self.free_flow_speed + slt_time
                            t2 = l.scedule_time[i+2] + link_length / self.free_flow_speed
                            t3 = l.scedule_time[i+2]
                            plt.fill_betweenx([l.position,l.position + link_length],[t0,t1],[t3,t2],edgecolor='none',facecolor='g',alpha=alpha,lw=0)
                            #plt.plot([t0,t1,t2],[l.position,l.position + link_length,l.position + link_length],'y',alpha = 0.5)
                            #plt.plot([t0,t3,t2],[l.position,l.position,l.position + link_length],'c',alpha = 0.5)

        if title ==' ': plt.title(road.indexes)
        plt.ylim(0,road.length)
        plt.xlabel('time (s)')
        plt.ylabel('distance (m)')

    def plot_signal(self,plt,index,xlim=None):
        plt.figure(figsize=(12,2))
        sced_t = np.linspace(self.data_time[0],self.data_time[-1],10000)
        sced_f = interp.interp1d(self.data_time,self.data_signals[index],kind='zero')
        plt.plot(sced_t,sced_f(sced_t),c='r',alpha=0.1)
        plt.ylim(-0.1,1.1)
        if xlim is not None:
            plt.xlim(xlim[0],xlim[1])
        self.signals[index].plot(plt)

    def plot_nfd(self,plt,dt=20,dx=20):
        nfd_data = [(0,0)]
        nfd_data_tx = [(None,0,0)]
        for path in sim.paths:
            for t in np.arange(self.data_time[0],self.data_time[-1] + dt,dt):
            #for t in np.arange(100,350,dt):
                # for x in np.arange(350,400,dx):
                for x in np.arange(0,path.length + dx,dx):
                    #q0,k0 = path_0.get_flow_density2(t,10,x,dx)
                    q1,k1 = path.get_flow_density(t,dt,x,dx)
                    if not (q1 == 0.0 and k1 == 0.0):
                        nfd_data.append((q1,k1))
                        nfd_data_tx.append((path,t,x))
            #print fd_data
            nfd_x = np.array([x[1] for x in nfd_data])
            nfd_y = np.array([x[0] for x in nfd_data])

        xy = np.vstack([nfd_x,nfd_y])
        z = gaussian_kde(xy)(xy)

        idx = z.argsort()
        nfd_x, nfd_y, z = nfd_x[idx], nfd_y[idx], z[idx]
        plt.xlim(0,np.max(nfd_x)*1.1)
        plt.ylim(0,np.max(nfd_y)*1.1)
        plt.scatter(nfd_x,nfd_y,c=z,edgecolors='',s=50) # edgecolors='g',facecolors='none'
        plt.xlabel('Density (Vehicles/m)')
        plt.ylabel('Flow (Vehicles/s)')
        plt.title('Fundamential Diagram')
        return {'k':nfd_x,'q':nfd_y}

    def write(self,data):
        data['Out']['average_delay'] = self.average_delay()
        data['Out']['delay'] = self.delay()
        data['Out']['total_travel_time'] = self.total_travel_time()
        data['Out']['total_traffic_in'] = self.total_traffic_in()
        data['Out']['Microsim'] = {'cars': [],
                                   'time': list(self.time),
                                   'DT' : self.DT,
                                   'time_factor': self.time_factor,
                                   'free_flow_speed':  self.free_flow_speed,
                                   'amber_time': self.amber_time
                                   }
        for car_path in self.cars:
            for car in car_path:
                data['Out']['Microsim']['cars'].append({'position': list(car.link_position_log), 'link': list(car.link_log)})


def plot_annotations(plt,annotations):
    if annotations['arrows'] is not None:
        #print args.annotation_arrow
        for annotation in annotations['arrows']:
            kwargs = json.loads(annotation[1])
            plt.annotate(annotation[0],**kwargs)
    if annotations['text'] is not None:
        #print args.annotation_text
        for annotation in annotations['text']:
            kwargs = json.loads(annotation[3])
            x = annotation[0]
            y = annotation[1]
            text = annotation[2]
            plt.text(x,y,text,**kwargs)


def plot_fig(plt,figsize):
    if figsize is not None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    else:
        plt.figure()

# def test():
#     light1 = signal([30,30,30,10], 200, amber_time=5)
#     light2 = signal([0,30,30,30,10], 500, amber_time=5)
#     signals = [light1,light2]
#
#     free_flow_speed = 48280.3/3600.0
#     DT = 0.5
#     time = np.arange(0,200,DT) #np.arange(0,data_time[-1],DT)
#     inflow_limit = 40
#     inflow = interp.interp1d([0,inflow_limit,time[-1]],[0.5,0.5,0])
#     road = path(1000,free_flow_speed,signals,inflow) #paths[index]
#     cars = microsim_road(road,time,DT)
#     return cars,road

def nfd_model(nfd_data, model, plt=None):

    nfd_x = nfd_data['k']
    nfd_y = nfd_data['q']

    print
    print 'Model: ',model
    print

    if model == 'cubic' or model == 'quadratic':
        # nbins = 50
        # bins = np.linspace(0,0.2,nbins)
        # binplace = np.digitize(nfd_x, bins)
        # nfd_median=[]
        # nfd_median_x=[]
        # for i in range(1,nbins):
        #     if len(np.where(binplace == i)) > 0:
        #         nfd_median.append(np.median(nfd_y[np.where(binplace == i)]))
        #         nfd_median_x.append(bins[i - 1])
        # plt.scatter(nfd_median_x, nfd_median, edgecolors='r', facecolors='none') # edgecolors='g'
        if model == 'cubic':
            n = 3
        else:
            n = 2

        nfd_z = np.polyfit(nfd_x, nfd_y, n)
        p = np.poly1d(nfd_z)
        print 'p = ',p
        roots = np.sort(np.roots(p))
        print 'roots = ',roots
        p_derivative = np.polyder(p)
        k_jam = roots[1].real
        k_0 = roots[0].real
        vf = p_derivative(k_0)
        vb = -p_derivative(k_jam)
        W = vb/vf
        xp = np.linspace(k_0, k_jam, 100)
        yp = p(xp)
        max_flow = np.max(yp)
        if plt is not None:
            plt.plot(xp, yp,c='k',label=model,ls=':')

    elif model == 'leastsq_1':

        def residuals(p, y, x, w):
            a,b,b2,c = p
            x0 = np.where(x < b/2)
            x1 = np.where(x > b/2)
            if b2 > b/2:
                x2 = np.where(x > b2)
            else:
                x2 = np.where(x > b/2)
            #w = y / np.max(y)
            err = w * (y - (a * x**2 - a * b * x))
            err[x1] = w[x1] * (y[x1] + a * b**2 / 4)
            err[x2] =  np.abs( ( ((a * b**2) / (4 * c - 4 * b2)) * x[x2] - ((a * b**2 * c) / (4 * c - 4 * b2)) ) - y[x2]) / np.sqrt(((a * b**2) / (4 * c - 4 * b2))**2 + 1)
            return err

        def peval(x, p):
            a,b,b2,c = p
            x0 = np.where(x < b/2)
            x1 = np.where(x > b/2)
            x2 = np.where(x > b2)
            y = a * x**2 - a * b * x
            y[x1] = - a * b**2 / 4
            y[x2] = ((a * b**2) / (4 * c - 4 * b2)) * x[x2] - ((a * b**2 * c) / (4 * c - 4 * b2))
            return y

        b0 = 2 * nfd_x[np.argmax(nfd_y)]
        a0 = (-4 * np.max(nfd_y)) / b0**2
        c0 = np.max(nfd_x)
        v0 = -a0 * b0
        b20 = b0 / 2 # -max(nfd_y) / (v0/2) + c0 #b0/2 + (c0 - b0/2) / 2
        plt.scatter([b0/2,c0,b20],[-a0*b0**2/4,0,-a0*b0**2/4],c='y')
        p0 = [a0,b0,b20,c0]
        x_lsq = np.linspace(0, c0, 100)

        w = nfd_y * 1.0 / np.max(nfd_y)
        plsq = leastsq(residuals, p0, args=(nfd_y, nfd_x, w))

        a,b,b2,c = plsq[0]

        print ' a =',a
        print ' b =',b
        print 'b/2=',b/2
        print 'b2 =',b2
        print ' c =',c
        k_jam = c
        max_flow = -(a*b**2)/4
        vf = -a*b
        vb = -((a*b**2)/4) / (c - b2)
        W = vb / vf
        if plt is not None:
            plt.plot(x_lsq, peval(x_lsq,p0),c='c',label='leastsq_1: p0')
            x_lsq = np.linspace(0, c, 100)
            # y_lsq = a * np.square(x_lsq) - a * b * x_lsq
            plt.plot(x_lsq, peval(x_lsq,plsq[0]),c='k',label='leastsq_1')

    elif model == 'CTM':

        k_m = 0.03728227153
        max_flow = q_m = 0.5
        vf = q_m / k_m
        k_jam = 0.15
        W= 0.5
        vb = vf * W
        k_m2 = k_jam - (k_m / W)
        plt.plot([0,k_m,k_m2,k_jam],[0,q_m,q_m,0],c='r',label='CTM model')

    print 'Max flow   :',max_flow
    print 'Jam density:',k_jam
    print 'Free flow speed     :', vf
    print 'Backwards wave speed:', vb
    print 'W ratio             :',W



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="the model file to simulate", nargs='+')
    parser.add_argument("--plot", help="Index of queue to plot (plots path queue is on)",nargs='*', type=int)
    parser.add_argument("-i","--index", help="Indexes of queues to simulate",nargs='*', type=int)
    parser.add_argument("--figsize", help="Set size of figure", nargs = 2, type=float)
    parser.add_argument("--free_flow_speed", help="Set the free flow speed", type=float, default = 48280.3/3600.0) # in m/s (48.2803 km/h = 30 mph)
    parser.add_argument("--DT", help="Set the time step", type=float, default = 0.5)
    parser.add_argument("--time_factor", help="Set the time factor", type=float,default = 1.0)
    parser.add_argument("-o","--out", help="write results to file sim")
    parser.add_argument("--plot_car", help="plot speed of car n in path m", action='append', nargs = 2, type=int)
    parser.add_argument("--save_fig", help="Save the fig to file SAVE_FIG")
    parser.add_argument("--save_csv", help="Save the data to csv file SAVE_CSV")
    parser.add_argument("--dpi", help="DPI to save plots in", type=int, default = 300)
    parser.add_argument("--plot_depart", help="plot cumulative departure curve of queue n ", nargs='*', type=int)
    parser.add_argument("--plot_arrival", help="plot cumulative arrival curve of queue n ", nargs='*', type=int)
    parser.add_argument("--plot_stops", help="count the number of stops in path ", action='append', nargs='+', type=int)
    parser.add_argument("--plot_nfd", help="Plot the network fundimental diagram with optional parameters ", action='append', nargs='*', type=int)
    parser.add_argument("--plot_green_bands",help="plot green bands with alpha", nargs='*', type=float)
    parser.add_argument("--plot_traces",help="plot car traces with alpha or turn off with 'off'",default=1.0)
    parser.add_argument("--stop_threshold", help="velocity threshold below which a car is considered stopped ", type=float, default  = 1.0)
    parser.add_argument("--plot_signal", help="plot signal n ", nargs='*', type=int)
    parser.add_argument("--max_stops", help="number of stops to count for histogram ", type=int)
    parser.add_argument("--ylim", help="range of y axis", nargs = 2, type=float)
    parser.add_argument("--xlim", help="range of x axis", nargs = 2, type=float)
    parser.add_argument("--title", help="array of titles for the plots", nargs = '*')
    parser.add_argument("--road_width", help="set road_width", type=float, default = 0) # default = 10
    parser.add_argument("--no_lost_time", help="do not add lost time", action='store_true', default = False)
    parser.add_argument("--lost_time", help="Add lost time to phase time", type = float, default = 0)
    parser.add_argument("--slt_weight", help="fraction of lost to time to use as start up lost time", type = float, default = 0.5)
    parser.add_argument("--amber_time", help="Amber light duration", type = float, default = 2.5)
    parser.add_argument("--annotation_arrow", help="optional text followed by json dictionary of matplotlib annotation parameters to plot an arrow",nargs=2,action='append')
    parser.add_argument("--annotation_text", help="x y text followed by json dictionary of matplotlib text parameters",nargs=4,action='append')
    parser.add_argument("--idm_params", help="Override defaults with JSON formated dictionary of IDM car parameters: {'width','length','maxspeed','s0','timeHeadway', 'maxAcceleration','maxDeceleration'}")
    parser.add_argument("--random", help="randomize the microsim",action='store_true',default = None)

    args = parser.parse_args()

    free_flow_speed = args.free_flow_speed
    #print 'Free flow speed:',free_flow_speed,'m/s'
    time_factor = args.time_factor
    DT = args.DT

    if args.idm_params is not None:
        idm_params = json.loads(args.idm_params)
        idm_params['random'] = args.random
    else:
        idm_params = None
    print 'idm_params:',idm_params

    file_sets,loaded_file_sets = qtm.read_files(args.files,return_file_sets=True)
    plot_flag = False
    for k,files in enumerate(file_sets):
        for l,data in enumerate(files):
            #data = file[0][0]

            sim = microsim(data, free_flow_speed, time_factor=time_factor,
                           road_width=args.road_width, lost_time = args.lost_time,
                           no_lost_time=args.no_lost_time, amber_time=args.amber_time, params=idm_params,
                           slt_weight=args.slt_weight)
            sim.simulate(DT,indexes = args.index)
            average_delay = sim.average_delay()
            delay = sim.delay()
            total_travel_time = sim.total_travel_time()
            total_traffic_in = sim.total_traffic_in()

            print 'Total traffic in: %s \tAverage delay: %s \tTotal travel time %s' \
                  % (total_traffic_in,average_delay,total_travel_time)

            if args.out is not None:
                sim.write(data)
                out_file = loaded_file_sets[k][l].split('.')[0] + '_' + args.out + '.' + loaded_file_sets[k][l].split('.')[1]
                print out_file
                f = open(out_file,'w')
                json.dump(data,f)
                f.close()

            if args.plot is not None:
                plot_flag = True
                if len(args.plot) == 0:
                    plot_paths = [i for i,p in enumerate(sim.paths)]
                else:
                    plot_paths = sim.indexes_to_path_indexes(args.plot)
                k=0
                for i in plot_paths:
                    if i is not None:
                        plot_fig(plt,args.figsize)
                        sim.plot(plt,i,traces = args.plot_traces,
                                 green_bands = args.plot_green_bands,
                                 title = args.title)
                        if args.title is not None:
                            plt.title(args.title[k])
                        if args.ylim is not None:
                            plt.ylim(args.ylim[0],args.ylim[1])
                        if args.xlim is not None:
                            plt.xlim(args.xlim[0],args.xlim[1])
                        plot_annotations(plt,dict(arrows=args.annotation_arrow,text=args.annotation_text))
                        if args.save_fig is not None:
                            plt.savefig(args.save_fig,bbox_inches='tight',dpi=args.dpi)
                        k += 1

            if args.plot_car is not None:
                plot_flag = True
                plot_fig(plt,args.figsize)
                for car_plot in args.plot_car:
                    paths = sim.indexes_to_paths([car_plot[0]])
                    for path in paths:
                        if path is not None:
                            car = path.cars[car_plot[1]]
                            car.plot(plt)

            if args.plot_depart is not None or args.plot_arrival is not None:
                plot_flag = True
                plot_fig(plt,args.figsize)
                t_range = np.arange(sim.data_time[0],sim.data_time[-1],DT)
                if args.plot_arrival is not None:
                    for queue in args.plot_arrival:
                        plt.plot(t_range,sim.links[queue].arrivals(t_range))
                if args.plot_depart is not None:
                    for queue in args.plot_depart:
                        plt.plot(t_range,sim.links[queue].departures(t_range))

            if args.plot_stops is not None:
                plot_flag = True
                for plot_i,plot in enumerate(args.plot_stops):
                    plot_fig(plt,args.figsize)
                    paths = sim.indexes_to_paths(plot)
                    stops = []
                    plot_data = {}
                    for path in paths:

                        if path is not None:
                            print path.indexes
                            stops += path.get_stops(args.stop_threshold)
                    if args.max_stops is not None:
                        bins = np.arange(-0.5,args.max_stops+1,1.0)
                        ticks = range(args.max_stops+1)
                    else:
                        bins = np.arange(-0.5,max(stops)+1,1.0)
                        ticks = range(max(stops)+1)
                    #print 'stops:',stops
                    print 'hist:',plt.hist(stops, bins = bins)
                    plot_data['stops'] = stops
                    plt.xticks(ticks)
                    plt.xlabel('Number of stops')
                    plt.ylabel('Number of vehicles')
                    plot_annotations(plt,dict(arrows=args.annotation_arrow,text=args.annotation_text))
                    if args.title is not None:
                        plt.title(args.title[plot_i])
                    if args.ylim is not None:
                        plt.ylim(args.ylim[0],args.ylim[1])
                    if args.save_fig is not None:
                        filename = args.save_fig.split('.')[0]
                        type = args.save_fig.split('.')[1]
                        plt.savefig(filename + '_%d' % plot_i + '.' + type,bbox_inches='tight',dpi=args.dpi)
                    if args.save_csv is not None:
                        filename = args.save_csv.split('.')[0]
                        type = args.save_csv.split('.')[1]
                        frame = pd.DataFrame(plot_data)
                        frame.to_csv(filename + '_%d' % plot_i + '.' + type)

            if args.plot_signal is not None:
                plot_flag = True
                for sig in args.plot_signal:
                    sim.plot_signal(plt,sig,xlim=args.xlim)

            if args.plot_nfd is not None:
                plot_flag = True
                plot_fig(plt,args.figsize)
                nfd_data = sim.plot_nfd(plt)
                nfd_model(nfd_data,'CTM',plt=plt)
                nfd_model(nfd_data,'quadratic',plt=plt)
                nfd_model(nfd_data,'cubic',plt=plt)
                nfd_model(nfd_data,'leastsq_1',plt=plt)
                if args.save_csv is not None:
                    #filename = args.save_csv.split('.')[0]
                    #type = args.save_csv.split('.')[1]
                    frame = pd.DataFrame(nfd_data)
                    frame.to_csv(args.save_csv)

    if plot_flag:
        plt.show();

    path_0 = sim.paths[2]

    plt.figure(1,figsize = (14,5))
    plt.subplot(121)

    # x=359 # 353.2
    # dx=50
    # t=38.4
    # dt=10.6

    # x=357 # 353.2
    # dx=50
    # t=59.7
    # dt=10.6

    # x=396
    # dx=.25
    # t=79.7
    # dt=1.6

    x=340
    dx=70
    t=30
    dt=70

    path_0.get_flow_density(t,dt,x,dx,plt,kwargs=dict(c='k',ls=':',lw=0.5))
    # plt.axhline(x,ls=':')
    # plt.axhline(x+dx,ls=':')
    # plt.axvline(t,ls=':')
    # plt.axvline(t+dt,ls=':')

    #plt.show();

    # x=350 # 353.2
    # dx=10
    # t=40
    # dt=10

    x=350
    dx=50
    t=40
    dt=50

    path_0.get_flow_density(t,dt,x,dx,plt,kwargs=dict(c='k',ls='-',lw=2.0))
    plt.axhline(x,ls='-',color='r')
    plt.axhline(x+dx,ls='-',color='r')
    plt.axvline(t,ls='-',color='r')
    plt.axvline(t+dt,ls='-',color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Space-Time Window')
    #plt.show();

    # nfd_data = []
    # dt = 20
    # dx = 20
    # for t in np.arange(sim.data_time[0],sim.data_time[-1],dt):
    # #for t in np.arange(100,350,dt):
    #     # for x in np.arange(350,400,dx):
    #     for x in np.arange(0,900,dx):
    #         #q0,k0 = path_0.get_flow_density2(t,10,x,dx)
    #         q1,k1 = path_0.get_flow_density(t,dt,x,dx)
    #         nfd_data.append((q1,k1))
    # #print fd_data
    # plt.scatter([x[1] for x in nfd_data],[x[0] for x in nfd_data],edgecolors='b',facecolors='none')

    plt.subplot(122)

    # nfd_data = [(0,0)]
    # nfd_data_tx = [(None,0,0)]
    # dt = 20
    # dx = 20
    # for path in sim.paths:
    #     for t in np.arange(sim.data_time[0],sim.data_time[-1],dt):
    #     #for t in np.arange(100,350,dt):
    #         # for x in np.arange(350,400,dx):
    #         for x in np.arange(0,1000,dx):
    #             #q0,k0 = path_0.get_flow_density2(t,10,x,dx)
    #             q1,k1 = path.get_flow_density(t,dt,x,dx)
    #             if not (q1 == 0.0 and k1 == 0.0):
    #                 nfd_data.append((q1,k1))
    #                 nfd_data_tx.append((path,t,x))
    #     #print fd_data
    #     nfd_x = np.array([x[1] for x in nfd_data])
    #     nfd_y = np.array([x[0] for x in nfd_data])
    #
    # xy = np.vstack([nfd_x,nfd_y])
    # z = gaussian_kde(xy)(xy)
    #
    # idx = z.argsort()
    # nfd_x, nfd_y, z = nfd_x[idx], nfd_y[idx], z[idx]
    #
    nfd_x = nfd_data['k']
    nfd_y = nfd_data['q']
    plt.scatter(nfd_x,nfd_y,edgecolors='g',facecolors='none',s=50) #
    # nfd_data = sim.plot_nfd(plt,20,20)

    for n_test in range(len(nfd_data)):
        if nfd_x[n_test] > 0.18:
            n = n_test
            break
    else:
        n = int(np.random.uniform(0,len(nfd_data) - 1))
    #for i,qk in enumerate(nfd_data):
    #   plt.text(qk[1],qk[0],'%d' % i)
    plt.text(nfd_data['k'][n],nfd_data['q'][n],'%d' % n)
    nbins = 50
    bins = np.linspace(0,0.2,nbins)
    binplace = np.digitize(nfd_x, bins)
    # nfd_median=[]
    # nfd_median_x=[]
    # for i in range(1,nbins):
    #     if len(np.where(binplace == i)) > 0:
    #         nfd_median.append(np.median(nfd_y[np.where(binplace == i)]))
    #         nfd_median_x.append(bins[i - 1])
    # plt.scatter(nfd_median_x, nfd_median, edgecolors='r', facecolors='none') # edgecolors='g'

    nfd_z = np.polyfit(nfd_x, nfd_y, 3)
    nfd_p = np.poly1d(nfd_z)
    xp = np.linspace(0, 0.2, 100)
    plt.plot(xp, nfd_p(xp),c='k',label='Parabola fit',ls=':')

    def residuals(p, y, x, w):
        a,b,b2,c = p
        x0 = np.where(x < b/2)
        x1 = np.where(x > b/2)
        if b2 > b/2:
            x2 = np.where(x > b2)
        else:
            x2 = np.where(x > b/2)
        #w = y / np.max(y)
        err = w * (y - (a * x**2 - a * b * x))
        err[x1] = w[x1] * (y[x1] + a * b**2 / 4)
        err[x2] =  np.abs( ( ((a * b**2) / (4 * c - 4 * b2)) * x[x2] - ((a * b**2 * c) / (4 * c - 4 * b2)) ) - y[x2]) / np.sqrt(((a * b**2) / (4 * c - 4 * b2))**2 + 1)
        return err

    def peval(x, p):
        a,b,b2,c = p
        x0 = np.where(x < b/2)
        x1 = np.where(x > b/2)
        x2 = np.where(x > b2)
        y = a * x**2 - a * b * x
        y[x1] = - a * b**2 / 4
        y[x2] = ((a * b**2) / (4 * c - 4 * b2)) * x[x2] - ((a * b**2 * c) / (4 * c - 4 * b2))
        return y

    b0 = 2 * nfd_x[np.argmax(nfd_y)]
    a0 = (-4 * np.max(nfd_y)) / b0**2
    c0 = np.max(nfd_x)
    v0 = -a0 * b0
    b20 = b0 / 2 # -max(nfd_y) / (v0/2) + c0 #b0/2 + (c0 - b0/2) / 2
    plt.scatter([b0/2,c0,b20],[-a0*b0**2/4,0,-a0*b0**2/4],c='y')
    p0 = [a0,b0,b20,c0]
    x_lsq = np.linspace(0, c0, 100)
    plt.plot(x_lsq, peval(x_lsq,p0),c='c',label='lsq fit: p0')
    w = nfd_y * 1.0 / np.max(nfd_y)
    plsq = leastsq(residuals, p0, args=(nfd_y, nfd_x, w))

    a,b,b2,c = plsq[0]

    print 'Least squares fit:'
    print ' a =',a
    print ' b =',b
    print 'b/2=',b/2
    print 'b2 =',b2
    print ' c =',c
    k_jam = c
    max_flow = -(a*b**2)/4
    vf = -a*b
    vb = -((a*b**2)/4) / (c - b2)
    W = vb / vf
    print 'Max flow   :',max_flow
    print 'Jam density:',k_jam
    print 'Free flow speed     :', vf
    print 'Backwards wave speed:', vb
    print 'W ratio             :',W
    x_lsq = np.linspace(0, c, 100)
    # y_lsq = a * np.square(x_lsq) - a * b * x_lsq
    plt.plot(x_lsq, peval(x_lsq,plsq[0]),c='k',label='lsq fit')

    plt.xlabel('Density (Vehicles/m)')
    plt.ylabel('Flow (Vehicles/s)')
    plt.title('Fundamential Diagram')
    plt.xlim(0,0.25)
    plt.ylim(0,0.7)
    k_m = 0.03728227153
    q_m = 0.5
    k_j = 0.15
    W= 0.5
    k_m2 = k_j - (k_m / W)
    plt.plot([0,k_m,k_m2,k_j],[0,q_m,q_m,0],c='r',label='CTM model')
    plt.legend(loc='best')
    plt.show();



    # plt.figure(2,figsize = (14,5))
    # plt.subplot(1,2,1)

    # print nfd_data_tx[n]
    # t=nfd_data_tx[n][1]
    # x=nfd_data_tx[n][2]
    # path=nfd_data_tx[n][0]
    # path.get_flow_density(t,dt,x,dx,plt,kwargs=dict(c='k',ls='-',lw=2.0))
    # plt.title('Fundamential Diagram @ %d path=%s' % (n,path.indexes))
    # plt.xlabel('Time (s)')
    # plt.ylabel('Distance (m)')
    # plt.show();
    #plt.plot(sim.data_time,path_0.car_xf[0](sim.data_time))
    #plt.plot(sim.data_time,path_0.car_xf[10](sim.data_time))

    # print 'x@t=0',path_0.car_xf[0](0)
    # print 't@x=0',path_0.car_tf[0](0)
    # print 'path_0.cars[0].time_log     ',path_0.cars[0].time_log
    # print 'path_0.cars[0].poitition_log',path_0.cars[0].position_log
    # print 'path_0.cars[10].time_log     ',path_0.cars[10].time_log
    # print 'path_0.cars[10].poitition_log',path_0.cars[10].position_log
    # road = np.arange(0,1000,10)
    # plt.plot(road,path_0.car_tf[0](road))
    # plt.plot(road,path_0.car_tf[10](road))
    # plt.show();
    # d = np.array([2,3,2,0,0,2,3,3,0,0,0,0])
    # print d
    # pos = d < 1
    # print pos
    # print pos[:-1]
    # print pos[1:]
    # print ~pos[:-1] & pos[1:]
    # print np.count_nonzero(~pos[:-1] & pos[1:])


