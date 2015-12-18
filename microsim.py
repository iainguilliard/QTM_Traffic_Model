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

def random():
    return 0 #np.random.uniform()

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
        self.position = position
        self.amber_time = amber_time

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







class car():

    def __init__(self,path,t,init_speed,maxspeed):
        self.position = 0
        self.speed = init_speed
        self.width = 1.7
        self.length = 3 #4.667  # 3 + 2 * random()
        self.maxSpeed = maxspeed    # IDM param: v0, desired speed when driving on a free road (m/s)
        self.s0 = 2                 # IDM param: s0, minimum bumper-to-bumper distance to the front vehicle (m)
        self.timeHeadway = 1.5      #1.5# IDM param: T, desired safety time headway when following other vehicles (s)
        self.distanceToStopLine = 1e6
        self.maxAcceleration = 2    #1 # IDM param: a, acceleration in everyday traffic (m/s^2)
        self.maxDeceleration = 3    # IDM param: b, "comfortable" braking deceleration in everyday traffic (m/s^2)
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

    def __init__(self,data,free_flow_speed,time_factor=1,amber_time=5,road_width=0):
        self.cars = []
        self.time = []
        self.amber_time = amber_time
        self.free_flow_speed = free_flow_speed
        self.time_factor = time_factor
        self.load_file(data,free_flow_speed,time_factor=time_factor,road_width=road_width)


    def load_file(self,data,free_flow_speed,time_factor = 1,road_width=0):
        data_DT = [x * time_factor for x in data['Out']['DT']]
        self.data_time = [x * time_factor for x in data['Out']['t']]
        q_paths = find_paths(data)
        self.paths = []
        self.links = [None for i in range(len(data['Queues']))]
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
                    p_state = [1 if x < 0.5 else 0 for x in data['Out']['p_{%d,%d}' % (l,ph)]]
                    t = 0
                    if p_state[0] == 0:
                        state = 0
                    else:
                        state = 1
                    schedule = []
                    if state == 1:
                        schedule.append(0)
                    for j,x in enumerate(p_state[1:]):
                        t += data_DT[j-1]
                        if x != state:
                            schedule.append(t)
                            t = 0
                        state = x
                    #print schedule
                    sig = signal(schedule,distance - road_width,self.amber_time)
                    signals.append(sig)
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
            cars.append(car(road,t,road.free_flow_speed,road.free_flow_speed))
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


    def plot(self,plt,index):
        road = self.paths[index]
        for i,c in enumerate(self.cars[index]):
            plt.plot(c.time_log,c.position_log,label='position %d' % i)
            for l in road.signals:
                for i,x in enumerate(l.scedule_state[:-1]):
                    if x > 0:
                        plt.plot([l.scedule_time[i],l.scedule_time[i+1]],[l.position,l.position],'k',lw=2)
            if args.title ==' ': plt.title(road.indexes)
            plt.xlabel('time (s)')
            plt.ylabel('distance (m)')

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
    parser.add_argument("--stop_threshold", help="velocity threshold below which a car is considered stopped ", type=float, default  = 1.0)
    parser.add_argument("--max_stops", help="number of stops to count for histogram ", type=int)
    parser.add_argument("--ylim", help="range of y axis", nargs = 2, type=float)
    parser.add_argument("--xlim", help="range of x axis", nargs = 2, type=float)
    parser.add_argument("--title", help="array of titles for the plots", nargs = '*')
    parser.add_argument("--road_width", help="set road_width", type=float, default = 0) # default = 10

    args = parser.parse_args()

    free_flow_speed = args.free_flow_speed
    #print 'Free flow speed:',free_flow_speed,'m/s'
    time_factor = args.time_factor
    DT = args.DT

    file_sets,loaded_file_sets = qtm.read_files(args.files,return_file_sets=True)
    for k,files in enumerate(file_sets):
        for l,data in enumerate(files):
            #data = file[0][0]

            sim = microsim(data,free_flow_speed,time_factor=time_factor,road_width=args.road_width)
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

                if len(args.plot) == 0:
                    plot_paths = [i for i,p in enumerate(sim.paths)]
                else:
                    plot_paths = sim.indexes_to_path_indexes(args.plot)
                k=0
                for i in plot_paths:
                    if i is not None:
                        plot_fig(plt,args.figsize)
                        sim.plot(plt,i)
                        if args.title is not None:
                            plt.title(args.title[k])
                        if args.ylim is not None:
                            plt.ylim(args.ylim[0],args.ylim[1])
                        if args.xlim is not None:
                            plt.xlim(args.xlim[0],args.xlim[1])
                        if args.save_fig is not None:
                            plt.savefig(args.save_fig,bbox_inches='tight',dpi=args.dpi)
                        k += 1

            if args.plot_car is not None:
                plot_fig(plt,args.figsize)
                for car_plot in args.plot_car:
                    paths = sim.indexes_to_paths([car_plot[0]])
                    for path in paths:
                        if path is not None:
                            car = path.cars[car_plot[1]]
                            car.plot(plt)

            if args.plot_depart is not None or args.plot_arrival is not None:
                plot_fig(plt,args.figsize)
                t_range = np.arange(sim.data_time[0],sim.data_time[-1],DT)
                if args.plot_arrival is not None:
                    for queue in args.plot_arrival:
                        plt.plot(t_range,sim.links[queue].arrivals(t_range))
                if args.plot_depart is not None:
                    for queue in args.plot_depart:
                        plt.plot(t_range,sim.links[queue].departures(t_range))

            if args.plot_stops is not None:

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


    if args.plot is not None or args.plot_car or args.plot_depart or args.plot_stops is not None:
        plt.show();



    # d = np.array([2,3,2,0,0,2,3,3,0,0,0,0])
    # print d
    # pos = d < 1
    # print pos
    # print pos[:-1]
    # print pos[1:]
    # print ~pos[:-1] & pos[1:]
    # print np.count_nonzero(~pos[:-1] & pos[1:])


