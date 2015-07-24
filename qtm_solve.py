#! /usr/bin/python
from __future__ import division
from gurobipy import *
import pprint
import json
import sys
import argparse
import time as clock_time
import os, platform, subprocess, re
import copy
import random
import zipfile
import zlib
#import plot_model as pl

def get_processor_name():
    if platform.system() == "Windows":
        family = platform.processor()
        name = subprocess.check_output(["wmic","cpu","get", "name"]).strip().split("\n")[1]
        return ' '.join([name, family])
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command, shell=True).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""

def milp_solver(data,t0,t1,n_samples,dt0=None,dt1=None,DT_file=False,DT_vari=None,minor_frame=0,major_frame=0,blend_frame=None,nthreads=0,fixed_plan=None,
           timelimit=0,accuracy=0,verbose=False,label=None,step=None,run=None,obj='NEW',alpha=1.0,beta=0.001,
           DT_offset=0,seed=0,tune=False,tune_file='',param=None):
    start_time = clock_time.time()
    try:

        t_period = (t1-t0)
        print 't0=',t0,'t1=',t1,'t_period=',t_period
        t=t0
        DT=[]
        DT_factor=1
        if 'DT_factor' in data:
            DT_factor=data['DT_factor']
        print 'DT_vari',DT_vari
        print 'DT_factor',DT_factor
        if DT_vari != None:
            if DT_offset>0:
                DT.append(DT_offset)
            if 'len' in DT_vari[0]:
                for DT_step in DT_vari:
                    DT_len = DT_step['len']
                    n=0
                    while n< DT_len  and t<=t1:
                        DT.append(DT_step['DT'])
                        t+=DT[-1]
                        n+=1
            elif len(DT_vari)==2:
                while t<t0+minor_frame:
                    DT.append(DT_vari[0]['DT']*DT_factor)
                    t+=DT[-1]
                if blend_frame == None:
                    blend_frame=major_frame-minor_frame
                while t<t0+minor_frame+blend_frame:
                    alpha = (t-minor_frame-t0)/(major_frame - minor_frame)
                    DT.append(DT_vari[0]['DT']*DT_factor*(1-alpha) + DT_vari[1]['DT']*DT_factor*alpha)
                    t+=DT[-1]
                while t<t0+major_frame:
                        DT.append(DT_vari[1]['DT']*DT_factor)
                        t+=DT[-1]
            else:
                for DT_step in DT_vari:
                    DT_start = t0 + DT_step['start']*t_period
                    DT_stop = t0 + DT_step['stop']*t_period

                    while t>= DT_start and t<DT_stop and t<=t1:

                        DT.append(DT_step['DT']*DT_factor)
                        t+=DT[-1]
            DT.append(DT[-1])
            DT.append(DT[-1])
            N=len(DT)
            n_samples = N-2
            print DT

        elif "DT" in data and DT_file == True:
            DT = data["DT"]
            N = len(DT)
            n_samples = len(DT)

        else:
            N=n_samples+2
            if DT_offset>0:
                if DT_offset < (t_period)/n_samples:
                    DT = [DT_offset] + [(t_period)/n_samples for n in range(N-2)] + [(t_period)/n_samples - DT_offset]
                else:
                    n_offset = int(DT_offset / ((t_period)/n_samples))
                    residual = DT_offset - (n_offset)*( (t_period)/n_samples )
                    if residual > 0:
                        DT = [DT_offset] + [(t_period)/n_samples for n in range(N-1-n_offset)] + [(t_period)/n_samples - residual]
                    else:
                        DT = [DT_offset] + [(t_period)/n_samples for n in range(N-n_offset)]
                        N = 1 + N-n_offset
            elif dt0 != None and dt1 != None:
                DT = [dt0]
                t = t0 + dt0
                while t<=t1:
                    alpha = (t-t0)/(t1 - t0)
                    DT.append(dt0*(1-alpha) + dt1*alpha)
                    t += DT[-1]
                DT.append(dt1)
                DT.append(dt1)
                N = len(DT)
            else:
                DT = [(t_period)/n_samples for n in range(N)]
        print 'DT_offset=',DT_offset
        if len(DT) < 10:
            print 'DT=',DT
        else:
            print 'DT=',DT[0:10],'...'
        DT_s={}
        for dt in DT:
            if str(dt) not in DT_s:
                DT_s[str(dt)]=1
            else:
                DT_s[str(dt)]+=1
        if len(DT_s) > 1:
            print 'DT = ',DT_s
        else:
            print 'DT = ',DT[0]
        print 'N  = ',N
        t=t0
        time = []
        for _dt in DT:
            time.append(t)
            t+=_dt
        if 'Flow_Weights' in data:
            flow_weights = data['Flow_Weights']
            for n,t in enumerate(time):
                for f in flow_weights:

                    if n==0:
                        flow_weights[f]['wt']=[0]*N
                    if t > flow_weights[f]['start'] and t <= flow_weights[f]['end']:
                        flow_weights[f]['wt'][n]=flow_weights[f]['weight']
                    else:
                        flow_weights[f]['wt'][n]=0
                    if n>1:
                        if t > flow_weights[f]['start'] and time[n-1] < flow_weights[f]['start']:
                            flow_alpha = (flow_weights[f]['start'] - time[n-1]) / (t - time[n-1])
                            flow_weights[f]['wt'][n] *= flow_alpha
                        if t > flow_weights[f]['end'] and time[n-1] < flow_weights[f]['end']:
                            flow_alpha = (flow_weights[f]['end'] - time[n-1]) / (t - time[n-1])
                            flow_weights[f]['wt'][n-1] *= flow_alpha


        else:
            flow_weights=None

        # Define Constants
        T_MAX = time[N-1]+1


        epsilon = 0.1

        Q = len(data['Queues'])
        L = len(data['Lights'])

        Q_IN  = [ q['Q_IN']   for q in data['Queues'] ]
        Q_in  = [ [0]   for q in data['Queues'] ]
        Q_OUT = [ q['Q_OUT']  for q in data['Queues'] ]
        if flow_weights != None:
            Q_IN_weight = [ [0]*N for q in data['Queues'] ]
            for i,q in enumerate(data['Queues']):
                if 'weights' in q:
                    for w in q['weights']:
                        for n in range(N):
                            Q_IN_weight[i][n]+=flow_weights[w]['wt'][n]
                else:
                    Q_IN_weight[i]=[1]*N
        Q_p   = [ q['Q_P']    for q in data['Queues'] ]
        Q_MAX = [ q['Q_MAX']  for q in data['Queues'] ]
        Q_DELAY = [ q['Q_DELAY']  for q in data['Queues'] ]

        P_MAX = [ [ p for p in light['P_MAX']] for light in data['Lights'] ]
        P_MIN = [ [ p for p in light['P_MIN']] for light in data['Lights'] ]
        C_MAX = [ light['C_MAX'] for light in data['Lights'] ]
        C_MIN = [ light['C_MIN'] if 'C_MIN' in light.keys() else 0 for light in data['Lights'] ]
        l = [ [ p for p in light['P_MAX']] for light in data['Lights'] ]

        # initial conditions
        q0 =    [ q['q0']     for q in data['Queues'] ]
        q0_in = [ q['q0_in']  for q in data['Queues'] ]
        q0_out =[ q['q0_out'] for q in data['Queues'] ]
        p0 = [ [ p for p in light['p0']] for light in data['Lights'] ]
        d0 = [ [ d for d in light['d0']] for light in data['Lights'] ]
        f0 = [[[0 for n in range(N)] for j in range(Q)] for i in range(Q)]
        for i in range(Q):
            for j in range(Q):
                f_ij = '%d_%d' % (i,j)
                if f_ij in data['Flows'] and i != j:
                    f0[i][j] = data['Flows'][f_ij]['f0']
        # initial conditions (using previous results if found in file)
        if fixed_plan == None and step != None and step > 0 and 'Out' in data:
            out=data['Out']
            if run != None:
                out = out['Run'][run]
            if step != None and step > 0:
                out = out['Step'][step-1]
            n0 = 0
            for n,tn in enumerate(out['t']):
                if tn == t0:
                    n0 = n
                    break
                if tn > t0:
                    n0 = n-1
                    break
            n1=n0+1
            if n1>= len(out['t']): n1=n0
            dt_ratio = DT[0] / out['DT'][n0]
            td = t0-out['t'][n0]
            alpha = 1-(td / out['DT'][n0]) #(out['t'][n1] - out['t'][n0]))
            q0 =    [ alpha * out['q_%d' % i][n0] + (1-alpha) * out['q_%d' % i][n1]     for i,q in enumerate(data['Queues']) ]
            q0_in = [ (alpha * out['q_{%d,in}' % i][n0] + (1-alpha) * out['q_{%d,in}' % i][n1]) * dt_ratio  for i,q in enumerate(data['Queues']) ]
            q_in_prev = [ out['q_{%d,in}' % i] for i,q in enumerate(data['Queues']) ]
            t_prev = out['t']
            DT_prev = out['DT']
            q0_out =[ (alpha * out['q_{%d,out}' % i][n0] + (1-alpha) * out['q_{%d,out}' % i][n1]) * dt_ratio for i,q in enumerate(data['Queues']) ]
            p0 = [ [ out['p_{%d,%d}' %(i,j)][n0] for j,p in enumerate(light['p0'])] for i,light in enumerate(data['Lights']) ]
            d0 = [ [ (alpha * out['d_{%d,%d}' %(i,j)][n0] + (1-alpha) * out['d_{%d,%d}' %(i,j)][n1]) for j,d in enumerate(light['p0'])] for i,light in enumerate(data['Lights']) ]
            for i in range(Q):
                for j in range(Q):
                    f_ij = '%d_%d' % (i,j)
                    if f_ij in data['Flows'] and i != j:
                        f0[i][j] = (alpha * out['f_{%d,%d}' % (i,j)][n0] + (1-alpha) * out['f_{%d,%d}' % (i,j)][n1]) * dt_ratio
                    else:
                        f0[i][j] = 0


        # Create a new model
        m = Model("flow")

        # Create variables
        p = [[[m.addVar(vtype=GRB.BINARY, name="p%d_%d_%d" % (i,j,n))  for n in range(N)] for j in range(len(l[i]))] for i in range(L)]
        d = [[[m.addVar(name="d%d_%d_%d" % (i,j,n))  for n in range(N)] for j in range(len(l[i]))] for i in range(L)]
        q = [[m.addVar(lb=0, name="q%d_%d" % (i,n)) for n in range(N)] for i in range(Q)]
        q_in = [[m.addVar(lb=0, name="q%d_in_%d" % (i,n)) for n in range(N)] for i in range(Q)]
        q_stop = [[m.addVar(lb=0, name="q%d_stop_%d" % (i,n)) for n in range(N)] for i in range(Q)]
        q_out = [[m.addVar(lb=0, name="q%d_out_%d" % (i,n)) for n in range(N)] for i in range(Q)]
        d_q_out = [[m.addVar(lb=0, name="d_q%d_out_%d" % (i,n)) for n in range(N)] for i in range(Q)]
        d_q_in = [[m.addVar(lb=0, name="d_q%d_in_%d" % (i,n)) for n in range(N)] for i in range(Q)]

        in_q = [[m.addVar(lb=0, ub=Q_IN[i], name="in_q%d_%d" % (i,n)) for n in range(N)] for i in range(Q)]
        out_q = [[m.addVar(lb=0, ub=Q_OUT[i], name="out_q%d_%d" % (i,n)) for n in range(N)] for i in range(Q)]
        f = [[[None for n in range(N)] for j in range(Q)] for i in range(Q)]

        for i in range(Q):
            for j in range(Q):
                for n in range(N):
                    f_ij = '%d_%d' % (i,j)
                    if f_ij in data['Flows']:
                        f[i][j][n] = m.addVar(lb=0, ub=data['Flows'][f_ij]['F_MAX'], name="f_%d,%d_%d" % (i,j,n))

        total_travel_time = m.addVar(lb=0, name="total_travel_time")

        # Integrate new variables
        m.update()

        if fixed_plan != None:
            p_fixed = [[[ 0 for n in range(N) ] for j in range(len(l[i]))] for i in range(L)]
            d_fixed = [[[ d0[i][j] if n==0 else 0 for n in range(N) ] for j in range(len(l[i]))] for i in range(L)]
            t_fixed=fixed_plan['t']
            #print 'len(p_fixed) =',len(p_fixed[0][0])
            #print 'len(fixed_plan[p_{0,0}]) =',len(fixed_plan['p_{%d,%d}' % (0,0)])
            for i in range(L):
                for j in range(len(l[i])):
                    k=0
                    for n in range(n_samples):
                        #if i==0 and j==0 and k+1<len(t_fixed):
                        #    print time[n], '>=', t_fixed[k+1],time[n] >= t_fixed[k+1],
                        if k+1<len(t_fixed) and time[n] >= t_fixed[k+1]-1e-6:
                            k+=1
                        p_fixed[i][j][n]=fixed_plan['p_{%d,%d}' % (i,j)][k]
                        #if i==0 and j==0:
                        #    print 't_fixed[k]=%f'%t_fixed[k],'time[n]=%f'%time[n],'n=',n,'k=',k,'p_{%d,%d}=' % (i,j),p_fixed[i][j][n],
                        #if n<n_samples and k+1 < len(t_fixed) and time[n]>t_fixed[k+1]:

                        if n>0:
                            if p_fixed[i][j][n] == 1 and p_fixed[i][j][n-1] == 0:
                                d_fixed[i][j][n]=0
                            elif p_fixed[i][j][n] == 0 and p_fixed[i][j][n-1] == 0:
                                d_fixed[i][j][n]=d_fixed[i][j][n-1]
                            else:
                                d_fixed[i][j][n]=d_fixed[i][j][n-1] + DT[n-1]*p_fixed[i][j][n-1]
                        #if i==0 and j==0:
                        #    print 'd=',d_fixed[i][j][n]


        delay = quicksum([quicksum([time[n] * q_out[i][n] if Q_OUT[i]>0 else 0 for n in range(N)]) for i in range(Q) ])
        vehicle_holding = quicksum([quicksum([time[n] * q_out[i][n] if Q_OUT[i]==0 else 0 for n in range(N)]) for i in range(Q) ])
        in_flow = quicksum([quicksum([(T_MAX-time[n]) * in_q[i][n] if Q_IN[i]>0 else 0 for n in range(N)]) for i in range(Q) ])
        out_flow = quicksum([quicksum([(T_MAX-time[n]) * out_q[i][n] if Q_OUT[i]>0 else 0 for n in range(N)]) for i in range(Q) ])
        #m.setObjective(delay - beta*in_flow - beta*out_flow, GRB.MINIMIZE)
        #m.setObjective(delay + beta* vehicle_holding, GRB.MINIMIZE)

        sum_in =  quicksum([quicksum([time[n] * q_in[i][n] if Q_IN[i]>0 else 0 for n in range(n_samples)]) for i in range(Q) ])
        sum_out = quicksum([quicksum([time[n] * q_out[i][n] if Q_OUT[i]>0 else 0 for n in range(n_samples)]) for i in range(Q) ])
        m.addConstr(total_travel_time == sum_out - sum_in)

        if obj == 'MAX_OUT':
            out_flow = quicksum([ quicksum([ (T_MAX-time[n]) * out_q[i][n] * DT[n-1] for n in range(1,N)]) for i in range(Q) ])
            in_flow = quicksum( [ quicksum([ (T_MAX-time[n]) * in_q[i][n] * DT[n-1] for i in range(Q) ]) for n in range(1,N)])

            m.setObjective(alpha * out_flow  + beta*in_flow, GRB.MAXIMIZE)
        elif obj == 'MAX_QOUT':
            flow = quicksum([ quicksum([ (T_MAX-time[n]) * q_out[i][n] for n in range(N)]) if Q_OUT[i]>0 else 0 for i in range(Q) ])
            in_flow = quicksum( [ quicksum([ (T_MAX-time[n]) * in_q[i][n] * DT[n-1] for i in range(Q) ]) for n in range(1,N)])
            stops = 0.5 * quicksum( [ quicksum([ d_q_in[i][n] + d_q_out[i][n] for i in range(Q) ]) for n in range(1,N)])
            m.setObjective(alpha * flow  -  (1-alpha)*stops + beta*in_flow, GRB.MAXIMIZE)
            #m.setObjective(flow + beta*in_flow, GRB.MAXIMIZE)
        elif obj == 'MAX_ALL_QOUT':
            m.setObjective(quicksum([(T_MAX-time[n]) * q_out[i][n] + beta * ((T_MAX-time[n]) * in_q[i][n])  for i in range(Q) for n in range(1,N)]), GRB.MAXIMIZE)
            # m.setObjective(quicksum([(T_MAX-time[n]) * q_out[i][n] + 1*((T_MAX-time[n]) * in_q[i][n]) for i in range(Q) for n in range(N)]), GRB.MAXIMIZE)
        else:
            print 'No objective: %s' % obj
            return None
        # initial conditions at n = 0
        for i in range(L):
            for j in range(len(l[i])):
                m.addConstr(p[i][j][0] == p0[i][j])
                m.addConstr(d[i][j][0] == d0[i][j])

        for i in range(Q):
            m.addConstr(q[i][0] == q0[i])
            m.addConstr(q_in[i][0] == q0_in[i])
            m.addConstr(q_stop[i][0] == 0)
            m.addConstr(q_out[i][0] == q0_out[i])
            for j in range(Q):
                f_ij = '%d_%d' % (i,j)
                if f_ij in data['Flows'] and i != j:
                    m.addConstr(f[i][j][0] == f0[i][j])

        # feasrelax contraint and variable lists and penalties
        relax_constr=[]
        relax_vars=[]
        lbpen=[]
        ubpen=[]
        rhspen=[]

        cars_in =0
        cars_in_1 =0


        for n in range(1,N):

            for i in range(Q):


                if 'In_Flow_limit' in data and flow_weights == None:
                    if time[n-1] < data['In_Flow_limit'] and data['In_Flow_limit'] <= time[n]:
                        flow_alpha = (data['In_Flow_limit'] - time[n-1]) / (time[n]-time[n-1])
                        if i==1: print time[n],'flow_alpha=',flow_alpha
                        m.addConstr(in_q[i][n] <= flow_alpha*Q_IN[i])
                        cars_in+=flow_alpha*Q_IN[i]*DT[n-1]
                        Q_in[i].append(flow_alpha*Q_IN[i])
                        if i==1:
                            cars_in_1+=flow_alpha*Q_IN[i]*DT[n-1]
                            #print "cars_in_1 =",cars_in_1, ' @',time[n], ' alpha =',flow_alpha ,' limit =',data['In_Flow_limit'],time[n] < data['In_Flow_limit'] and data['In_Flow_limit'] < time[n+1]
                    elif time[n-1] < data['In_Flow_limit']:
                        m.addConstr(in_q[i][n] <= Q_IN[i])
                        cars_in+=Q_IN[i]*DT[n-1]
                        Q_in[i].append(Q_IN[i])
                        if i==1:
                            cars_in_1+=Q_IN[i]*DT[n-1]
                            #print "cars_in_1 =",cars_in_1, ' @',time[n]
                    else:
                        Q_in[i].append(0)
                        m.addConstr(in_q[i][n] <= 0)

                elif flow_weights != None:
                    m.addConstr(in_q[i][n] <= Q_IN[i]*Q_IN_weight[i][n]) #m.addConstr(in_q[i][n] <= Q_IN[i]*Q_IN_weight[i][n])
                    cars_in+=Q_IN[i]*DT[n-1]
                else:
                    m.addConstr(in_q[i][n] <= Q_IN[i]) #m.addConstr(in_q[i][n] <= Q_IN[i])
                    cars_in+=Q_IN[i]*DT[n-1]
                m.addConstr(q_in[i][n] == in_q[i][n] * DT[n-1] + quicksum([f[j][i][n]*DT[n-1] if '%d_%d' % (j,i) in data['Flows'] and i != j else 0 for j in range(Q)]))

                m.addConstr(out_q[i][n] <= Q_OUT[i])

                m.addConstr(q_out[i][n] == out_q[i][n] * DT[n] + quicksum([f[i][j][n]*DT[n] if '%d_%d' % (i,j) in data['Flows'] and i != j else 0 for j in range(Q)]))

                m.addConstr( q_out[i][n] <= q[i][n] + q_stop[i][n])


                m.addConstr(q[i][n] <= Q_MAX[i] ) #+ q_sl[i][n])

                t_q_in = time[n]-Q_DELAY[i]

                if t_q_in >= time[0]:
                    n1=1
                    t_n1=time[1]
                    while t_n1 <= t_q_in:
                        n1+=1
                        t_n1=time[n1]
                    n0=n1-1
                    alpha = (t_q_in-time[n0])/(time[n1]-time[n0])
                    #if i==1: print '@',time[n],'alpha=',alpha,'t_q_in=',t_q_in,'t_n0=',time[n0],'t_n1=',time[n1],'1/DT_n0-1=',(1.0/DT[n0-1]),'t_n-Q_DELAY-t_n0=',(time[n] - Q_DELAY[i] - time[n0])
                    #m.addConstr(q[i][n] == q[i][n-1] - q_out[i][n-1] + (1 - alpha) * q_in[i][n0] + alpha * q_in[i][n1])
                    m.addConstr(q[i][n] == q[i][n-1] - q_out[i][n-1]  + q_in[i][n0] * (DT[n-1]/DT[n0-1])  )
                    m.addConstr(q_stop[i][n] == q_in[i][n0] * (DT[n-1]/DT[n0-1])  )

                    #m.addConstr((1 - alpha) * q_in[i][n0] + alpha * q_in[i][n1] + quicksum([q_in[i][k] for k in range(n1+1,n)]) <= Q_MAX[i] - q[i][n-1])
                    m.addConstr(  q_in[i][n0] * (DT[n-1]/DT[n0-1]) + quicksum([q_in[i][k] for k in range(n0+1,n)]) <= Q_MAX[i] - q[i][n-1])
                    #m.addConstr( q_out[i][n-1] <= q[i][n-1] + q_in[i][n0] * (DT[n-1]/DT[n0-1]) )

                elif fixed_plan == None and step != None and step > 0 and 'Out' in data:
                    n1=1
                    t_n1=t_prev[1]
                    while t_n1 < t_q_in:
                        n1+=1
                        t_n1=t_prev[n1]
                    n0=n1-1
                    alpha = (t_q_in-t_prev[n0])/(t_prev[n1]-t_prev[n0])
                    #if i==1: print  '_',time[n],'alpha=',alpha,t_q_in,t_prev[n0],t_prev[n1]
                    #m.addConstr(q[i][n] == q[i][n-1] - q_out[i][n-1] + (1 - alpha) * q_in_prev[i][n0] + alpha * q_in_prev[i][n1])
                    m.addConstr(q[i][n] == q[i][n-1] - q_out[i][n] + (0) * q_in_prev[i][n0] + alpha * q_in_prev[i][n1])
                    n2 = n1
                    #print 'n2=',n2,'len(t_prev)=',len(t_prev),'len(time)',len(time)
                    #print 'time=',t_prev
                    #print 'time=',time
                    while t_prev[n2] < time[0]: n2+=1

                    m.addConstr((1 - alpha) * q_in[i][n0] + alpha * q_in[i][n1] + quicksum([q_in_prev[i][k] for  k in range(n1+1,n2)]+[q_in[i][k] for k in range(0,n)]) <= Q_MAX[i] - q[i][n-1])

                else:
                    m.addConstr(q[i][n] == q[i][n-1] - q_out[i][n-1])
                    m.addConstr(q_stop[i][n] == 0)
                    #m.addConstr( q_out[i][n] <= q[i][n] )

                m.addConstr( q[i][n] - q[i][n-1] == d_q_out[i][n] - d_q_in[i][n]  )


                if fixed_plan == None:
                    for j in range(Q):
                        f_ij = '%d_%d' % (i,j)
                        if f_ij in data['Flows'] and i != j:
                            if Q_p[i] != None:
                                m.addConstr( f[i][j][n] <= data['Flows'][f_ij]['F_MAX'] * quicksum([ p[Q_p[i][0]][q_p][n] for q_p in Q_p[i][1:] ]) )

                            if 'Pr' in data['Flows'][f_ij]:
                                Pr_ij = data['Flows'][f_ij]['Pr']
                                m.addConstr( f[i][j][n] <= Pr_ij * ( quicksum([f[i][k][n] if '%d_%d' % (i,k) in data['Flows'] and i != k else 0 for k in range(Q)]) ) )
                else:
                    for j in range(Q):
                        f_ij = '%d_%d' % (i,j)
                        if f_ij in data['Flows'] and i != j:
                            if Q_p[i] != None:
                                m.addConstr( f[i][j][n] <= data['Flows'][f_ij]['F_MAX'] * sum([ p_fixed[Q_p[i][0]][q_p][n] for q_p in Q_p[i][1:] ]) )

                            if 'Pr' in data['Flows'][f_ij]:
                                Pr_ij = data['Flows'][f_ij]['Pr']
                                m.addConstr( f[i][j][n] <= Pr_ij * ( quicksum([f[i][k][n] if '%d_%d' % (i,k) in data['Flows'] and i != k else 0 for k in range(Q)]) ) )
            if fixed_plan == None:
                for i in range(L):
                    m.addConstr(quicksum([p[i][j][n] for j in range(len(l[i]))]) == 1)
                    np=len(l[i])
                    for j in range(np):
                        m.addConstr(p[i][j][n-1] <= p[i][j][n] + p[i][(j+1) % np][n])
                        m.addConstr(p[i][j][n] + p[i][(j+1) % np][n] <= 1)


                        m.addConstr(d[i][j][n] >= d[i][j][n-1] + p[i][j][n-1] * DT[n-1] - 10*P_MAX[i][j] * (1 - p[i][j][n-1]) )
                        m.addConstr(d[i][j][n] <= d[i][j][n-1] + p[i][j][n-1] * DT[n-1] + 10*P_MAX[i][j] * (1 - p[i][j][n-1]) )

                        m.addConstr(d[i][j][n] >= d[i][j][n-1] - 10*P_MAX[i][j] * p[i][j][n])
                        m.addConstr(d[i][j][n] <= d[i][j][n-1] + 10*P_MAX[i][j] * p[i][j][n-1])

                        relax_constr.append(m.addConstr(d[i][j][n] >= P_MIN[i][j] * (1 - p[i][j][n])))
                        rhspen.append(2)

                        relax_constr.append(m.addConstr(d[i][j][n] <= P_MAX[i][j] * (1 - p[i][j][n] + p[i][j][n-1])))
                        rhspen.append(2)

                        relax_constr.append(m.addConstr(d[i][j][n] <= P_MAX[i][j] ) )
                        rhspen.append(2)

                        relax_vars.append(d[i][j][n])
                        ubpen.append(1)
                        lbpen.append(1)

                    relax_constr.append(m.addConstr(d[i][0][n-1] + quicksum([d[i][j][n] for j in range(1,np)]) <= C_MAX[i]))
                    rhspen.append(2)
                    relax_constr.append(m.addConstr(d[i][0][n-1] + quicksum([d[i][j][n] for j in range(1,np)]) >= C_MIN[i] * (p[i][0][n] - p[i][0][n-1])))
                    rhspen.append(2)

        print 'cars_in =',cars_in

        # Switch off console output from Gurobi
        if not verbose:
            m.setParam(GRB.Param.LogToConsole,0)
        if nthreads>0:
            m.setParam(GRB.Param.Threads,nthreads)
        if accuracy>0:
            m.setParam(GRB.Param.MIPGap,accuracy/100)
        if timelimit>0:
            m.setParam(GRB.Param.TimeLimit,timelimit)
        if seed != 0:
            m.setParam(GRB.Param.Seed,seed)

        if tune==True:
            m.tune()
            for i in range(m.tuneResultCount):
                m.getTuneResult(i)
                m.write(tune_file+str(i)+'.prm')
            m.getTuneResult(0)
        # rub the Gurobi optimizer
        if param != None:
            m.read(param)
        m.optimize()
        solve_time = clock_time.time() - start_time

        status_codes = [ ['NONE', 0,'Not a status code'],
            ['LOADED', 1, 'Model is loaded, but no solution information is available.'],
            ['OPTIMAL', 2,
             'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'],
            ['INFEASIBLE', 3, 'Model was proven to be infeasible.'],
            ['INF_OR_UNBD', 4,
             'Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.'],
            ['UNBOUNDED', 5,
             '''Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit.
             It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize.'''],
            ['CUTOFF', 6,
             'Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available.'],
            ['ITERATION_LIMIT', 7,
             '''Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter,
             or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter.'''],
            ['NODE_LIMIT', 8,
             'Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.'],
            ['TIME_LIMIT', 9,
             'Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.'],
            ['SOLUTION_LIMIT', 10,
             'Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.'],
            ['INTERRUPTED', 11, 'Optimization was terminated by the user.'],
            ['NUMERIC', 12, 'Optimization was terminated due to unrecoverable numerical difficulties.'],
            ['SUBOPTIMAL', 13, 'Unable to satisfy optimality tolerances; a sub-optimal solution is available.'],
            ['INPROGRESS', 14,
             'An asynchronous optimization call was made, but the associated optimization run is not yet complete.'],
        ]

        if m.status == GRB.INFEASIBLE or m.status == GRB.INF_OR_UNBD:
            #m.computeIIS()
            #m.write('model_iis.ilp')
            m.feasRelax(1,True,None,None,None,relax_constr,rhspen)
            #m.feasRelax(1,True,relax_vars,lbpen,ubpen,relax_constr,rhspen)
            m.optimize()
            solve_time = clock_time.time() - start_time

        if m.status == GRB.INFEASIBLE or m.status == GRB.INF_OR_UNBD:
            print "Solver status: ", status_codes[m.status][0]
            print status_codes[m.status][2]
            print "Model Infeasible, calulating IIS: ",
            m.computeIIS()
            m.write('model_iis_error.ilp')
            print 'Done.'
            return None
        else:
            print "Solver status: ", status_codes[m.status][0]
            print status_codes[m.status][2]
            print 'Obj:', m.objVal
            print 'Total Travel Time:', total_travel_time.x
            print 'Solve time: %s seconds' % solve_time

            #
            t=t0
            time = []
            for _dt in DT[:-1]:
                time.append(t)
                t+=_dt
            
            if 'Out' not in data:
                data['Out'] = dict()
            if run != None:
                if 'Run' not in data['Out']:
                    data['Out']['Run'] = []
                if run >=len(data['Out']['Run']):
                    data['Out']['Run'].append(dict())
                data_out = data['Out']['Run'][run]
            else:
                data_out = data['Out']
            if step != None:
                if 'Step' not in data_out:
                    data_out['Step'] = []
                data_out['Step'].append(dict())
                data_out = data_out['Step'][step]

            data_out['N'] = n_samples
            data_out['t'] = time
            data_out['DT'] = DT
            data_out['DT_offset'] = DT_offset
            data_out['solve_time'] = solve_time
            data_out['solver_runtime'] = m.Runtime
            data_out['objval'] = m.objVal
            data_out['obj'] = obj
            data_out['status']=status_codes[m.status][0]
            data_out['alpha']=alpha
            data_out['beta']=beta
            data_out['total_travel_time'] = total_travel_time.x
            data_out['seed'] = seed
            data_out['cpu'] = get_processor_name()
            data_out['num_vars'] = m.NumVars
            data_out['num_binvars'] = m.NumBinVars
            data_out['num_constrs'] = m.NumConstrs
            if timelimit<1e30:
                data_out['time_limit'] = timelimit
            else:
                data_out['time_limit'] = timelimit
            data_out['accuracy'] = accuracy
            if label != None:
                data_out['label'] = label
            else:
                label = (data['out_filename'].split('.')[0]).replace('_',' ')
                data_out['label'] = label
            for i,q_i in enumerate(data['Queues']):
                # q_i['q0']  =  q[i][n].x
                # q_i['q0_in'] = q_in[i][n].x
                # q_i['q0_out'] = q_out[i][n].x
                data_out[r'q_%d' % i]  =  [q[i][n].x for n in range(N-1)]
                data_out[r'q_{%d,in}' % i] = [q_in[i][n].x for n in range(N-1)]
                data_out[r'q_{%d,stop}' % i] = [q_stop[i][n].x for n in range(N-1)]
                data_out[r'total_q_{%d,in}' % i] = sum(data_out[r'q_{%d,in}' % i])
                if major_frame==0: print 'total q_in,%d=' % i, sum(data_out[r'q_{%d,in}' % i])
                data_out[r'q_{%d,out}' % i] = [q_out[i][n].x for n in range(N-1)]
                data_out[r'total_q_{%d,out}' % i] = sum(data_out[r'q_{%d,out}' % i])
                if major_frame==0: print 'total q_out,%d=' % i, sum(data_out[r'q_{%d,out}' % i])
                data_out[r'dq_{%d,out}' % i] = [d_q_out[i][n].x for n in range(N-1)]
                data_out[r'dq_{%d,in}' % i] = [d_q_in[i][n].x for n in range(N-1)]
                data_out[r'dq_%d' % i] = [d_q_in[i][n].x + d_q_out[i][n].x for n in range(N-1)]
                data_out[r'in_%d' % i] = [in_q[i][n].x for n in range(N-1)]
                data_out[r'out_%d' % i] = [out_q[i][n].x for n in range(N-1)]

                data_out[r'q_{%d,abs}' % i] = [ abs( q[i][n+1].x - q[i][n].x ) for n in range(N-1)]
            data_out[r'total_q_in'] = sum([sum(data_out[r'q_{%d,in}' % i]) for i,q_i in enumerate(data['Queues'])])
            data_out[r'total_q_out'] = sum([sum(data_out[r'q_{%d,out}' % i]) for i,q_i in enumerate(data['Queues'])])
            if major_frame==0:
                if data_out[r'total_q_in'] > data_out[r'total_q_out']:
                    print 'WARNING: total q_out < total_ q_in. Total q_in:',data_out[r'total_q_in'],'Total q_out:',data_out[r'total_q_out']
                else:
                    print 'Total q_in:',data_out[r'total_q_in'],'Total q_out:',data_out[r'total_q_out']
            for i,l_i in enumerate(data['Lights']):
                for j, _ in enumerate(l_i['p0']):
                    # l_i['p0'][j] = p[i][j][n].x
                    # l_i['c0'][j] = c[i][j][n].x
                    # l_i['d0'][j] = d[i][j][n].x
                    if fixed_plan == None:
                        data_out[r'p_{%d,%d}' % (i,j)] = [p[i][j][n].x for n in range(N-1)]
                        #data_out[r'c_{%d,%d}' % (i,j)] = [c[i][j][n].x for n in range(N-1)]
                        data_out[r'd_{%d,%d}' % (i,j)] = [d[i][j][n].x for n in range(N-1)]
                    else:
                        data_out[r'p_{%d,%d}' % (i,j)] = [p_fixed[i][j][n] for n in range(N-1)]
                        #data_out[r'c_{%d,%d}' % (i,j)] = [c[i][j][n].x for n in range(N-1)]
                        data_out[r'd_{%d,%d}' % (i,j)] = [d_fixed[i][j][n] for n in range(N-1)]

            for f_ij,f_i in data['Flows'].iteritems():
                (i,j) = f_ij.split('_')
                i = int(i); j = int(j)
                # f_i['f0'] = f[i][j][n].x
                data_out[r'f_{%d,%d}' % (i,j)] = [f[i][j][n].x for n in range(N-1)]

    except GurobiError,e:
        print 'Error reported: ',e.message

    return data

# def print_table(v):
#     #vars = [p[0][0],p[0][1],p[0][2],p[0][3],d[0][0],d[0][2],p[1][0],p[1][1],p[1][2],p[1][3],d[1][0],d[1][2],q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],f[0][2],f[1][3],q_in[0],q_in[1],q_in[2],q_out[0],q_out[1],q_out[2],q_out[3],q_out[4],q_out[5],q_out[6],q_out[7]]
#     vars = [p[0][0],p[0][1],p[0][2],p[0][3],d[0][0],d[0][2],d_sl[0][0],d_sl[0][1],d_sl[0][2],d_sl[0][3],re[0],rc[0],rs[0],q_sl[0],q_sl[1],q[0],q[1],q[2],f[0][2],f[1][2],q_in[0],q_in[1],q_in[2],q_out[0],q_out[1],q_out[2]]
#
#
#     for v in vars:
#         for n in range(N):
#             if n==0:
#                 print "%6s" % (v[0].varName),
#
#             if v[n].vType == 'B':
#                 if abs(v[n].x) < 0.1:
#                     print "  0",
#                 else:
#                     print " %2.0f" % (v[n].x),
#             else:
#                 print " %2.0f" % (v[n].x),
#         print

def rm_vars(data):
    for key in data.keys():
        if key[0:2] in ['q_','d_','p_','f_','dq'] or key in['t','DT','Fixed_Plan']:
            del data[key]

def plan_solver(data,args):
    if ('Plans' in data and args.plan in data['Plans']) or args.vplan or args.fplan:
        if args.seed:
            random.seed(args.seed)
        else:
            random.seed(0)
        start_time = clock_time.time()
        av_travel_time=0
        av_solve_time=0
        total_steps=0
        for run in range(args.average):

            if args.vplan:
                minor_frame = args.vplan[0]
                major_frame = args.vplan[1]
                horizon = args.vplan[2]
                blend_frame = args.vplan[3]
                DT_start = args.vplan[4]
                DT_stop = args.vplan[5]
                plan =  {
                    "label": "$\\Delta t=%f,%f$, $\\Pi=%f$, $\\pi=%f$" % (DT_start,DT_stop,major_frame,minor_frame),
                    "minor_frame": minor_frame,
                    "major_frame": major_frame,
                    "horizon" : horizon,
                    "blend_frame" : blend_frame,
                    "DT_vari": [{ "DT": DT_start, "start": 0.0, "stop": 0.5},
                            { "DT": DT_stop, "start": 0.5, "stop": 1.0}]

                }
            elif args.fplan:
                minor_frame = args.fplan[0]
                major_frame = args.fplan[1]
                horizon = args.fplan[2]
                DT_fixed = args.fplan[3]
                plan =  {
                    "label": "$\\Delta t=%f, $\\Pi=%f$, $\\pi=%f$" % (DT_fixed,major_frame,minor_frame),
                    "minor_frame": minor_frame,
                    "major_frame": major_frame,
                    "horizon" : horizon,
                    "DT_fixed": DT_fixed

                }
            else:
                plan = data['Plans'][args.plan]

            minor_frame = plan['minor_frame']
            major_frame = int(plan['major_frame'])
            horizon = plan['horizon']
            label = plan['label']
            blend_frame=None
            print 'Plan:' ,plan
            if 'DT_fixed' in plan:
                DT = plan['DT_fixed']*args.DT_factor
                n_samples = int(major_frame / DT)
                DT_vari=None
            else:
                blend_frame=plan['blend_frame']
                DT_vari = plan['DT_vari']
                n_samples=None
                DT= DT_vari[0]['DT']*args.DT_factor
                data['DT_factor']=args.DT_factor

            prev_step=None
            t0 = 0
            K=int(horizon/minor_frame)
            epsilon = 1e-6
            for k in range(K):
                t1 = t0+major_frame
                if args.tune != None and k==int(args.tune[0][1]):

                    tune = True
                    tune_file = args.tune[0][0]
                else:
                    tune = False
                    tune_file=''
                if args.average and args.average==1:
                    run_index=None
                else:
                    run_index=run
                if milp_solver(data,t0=t0,t1=t1,n_samples=n_samples, DT_vari=DT_vari,minor_frame=minor_frame,major_frame=major_frame,blend_frame=blend_frame,fixed_plan=None,nthreads=args.threads,timelimit=args.timelimit,accuracy=args.accuracy,verbose=args.verbose,
                       label=label+' step %d' % k,
                       step=k,run=run_index,obj=args.obj,alpha=args.alpha,beta=args.beta,DT_offset=args.DT_offset,seed=random.randint(0,1e9),tune=tune,tune_file=tune_file,param=args.param) is None:
                    return None
                t0+=minor_frame
                if run_index != None:
                    data_step = data['Out']['Run'][run]['Step']
                else:
                    data_step = data['Out']['Step']

                av_solve_time+=data_step[k]['solver_runtime']
                total_q_in = data_step[k]['total_q_in']
                total_q_out = data_step[k]['total_q_out']
                if total_q_in < epsilon and total_q_out < epsilon:
                    del data_step[k]
                    break
            total_steps += k

            K_horizon = k*minor_frame
            K_steps = k

            print 'Completed in %d steps' % k
            print 'Generating final solution from steps'


            if run_index != None:
                fixed_plan = data['Out']['Run'][run]['Fixed_Plan'] = dict()
            else:
                fixed_plan = data['Out']['Fixed_Plan'] = dict()
            fixed_plan['t'] = []
            fixed_plan['DT'] = []

            t0 = 0
            for k in range(K_steps):

                t1 = t0+minor_frame
                if run_index != None:
                    step_data = data['Out']['Run'][run]['Step'][k]
                else:
                    step_data = data['Out']['Step'][k]
                step_time = step_data['t']

                n=0
                while n< len(step_time) and t0>step_time[n]: n+=1
                while n< len(step_time) and step_time[n]<t1:
                    ratio = int(step_data['DT'][n]/DT)

                    for dt_i in range(ratio):
                        for i,l_i in enumerate(data['Lights']):
                            for j, _ in enumerate(l_i['p0']):
                                if r'p_{%d,%d}' % (i,j) not in fixed_plan:
                                    fixed_plan[r'p_{%d,%d}' % (i,j)]=[]
                                    fixed_plan[r'd_{%d,%d}' % (i,j)]=[]
                                fixed_plan[r'p_{%d,%d}' % (i,j)].append(step_data[r'p_{%d,%d}' % (i,j)][n])
                                fixed_plan[r'd_{%d,%d}' % (i,j)].append(step_data[r'd_{%d,%d}' % (i,j)][n])

                        fixed_plan['DT'].append(DT)
                        if n>0:
                            fixed_plan['t'].append(fixed_plan['t'][-1]+DT)
                        else:
                            fixed_plan['t'].append(step_time[0])
                    n+=1
                t0+=minor_frame
            #print r'd_{%d,%d}=' % (0,0),fixed_plan[r'd_{%d,%d}' % (0,0)]
            if args.DT_final != None:
                DT_final = args.DT_final
            else:
                DT_final = fixed_plan['DT'][0]
            if milp_solver(data,t0=0,t1=K_horizon,n_samples=int(K_horizon/DT_final), fixed_plan=fixed_plan,nthreads=args.threads,
                   timelimit=args.timelimit,accuracy=args.accuracy,verbose=args.verbose,
                   label=label,step=None,run=run_index,obj=args.obj,alpha=args.alpha,beta=args.beta,param=args.param) is None:
                return None

            if run_index != None:
                print data['Out']['Run'][run].keys()
                print data['Out'].keys()
                av_travel_time += data['Out']['Run'][run]['total_travel_time']
            else:
                av_travel_time += data['Out']['total_travel_time']

            solve_time = clock_time.time() - start_time
            if run_index != None:
                data['Out']['Run'][run]['solve_time'] = solve_time
            if args.nostepvars:
                if 'Run' in data['Out']:
                    for run in data['Out']['Run']:
                        for step in run['Step']:
                            rm_vars(step)
                if 'Step' in data['Out']:
                    for step in data['Out']['Step']:
                        rm_vars(step)
            #if args.average>1:
            #    if 'run' not in data: data['run']=[]
            #    data['run'].append(copy.deepcopy(data['results']))


        solve_time = clock_time.time() - start_time
        print 'Total Plan Solve time: %s seconds' % solve_time
        data['Out']['solve_time'] = solve_time
        data['Out']['plan'] = plan
        data['Out']['DT_offset'] = args.DT_offset
        data['Out']['av_travel_time'] = av_travel_time / args.average
        data['Out']['av_solve_time'] = av_solve_time / total_steps
        if args.novars:
            if 'Run' in data['Out']:
                for run in data['Out']['Run']:
                    for step in run['Step']:
                        rm_vars(step)
                    rm_vars(run)
            if 'Step' in data['Out']:
                for step in data['Out']['Step']:
                    rm_vars(step)
            rm_vars(data['Out'])
        print
        print 'Done.'
        print
        return data
    else:
        print
        print "Plan %s not found in data file %s" % (args.plan, args.file)
        print
        return None


def DT_final_solver(data,args):
    start_time = clock_time.time()
    data=milp_solver(data,t0=args.t0,t1=args.t1,n_samples=args.nsamples, fixed_plan=fixed,nthreads=args.threads,timelimit=args.timelimit,accuracy=args.accuracy,
                    verbose=args.verbose,
                    step=0,
                    obj=args.obj,alpha=args.alpha,beta=args.beta,DT_offset=args.DT_offset,seed=args.seed)

    DT = args.DT_final
    fixed_plan = data['Out']['Fixed_Plan'] = dict()
    fixed_plan['t'] = []
    fixed_plan['DT'] = []


    k=0
    t0=args.t0
    t1=args.t1



    step_data = data['Out']['Step'][k]
    step_time = step_data['t']

    n=0
    while n< len(step_time) and t0>step_time[n]: n+=1
    while n< len(step_time) and step_time[n]<t1:
        ratio = int(step_data['DT'][n]/DT)
        #print 'r=',ratio,
        for dt_i in range(ratio):
            for i,l_i in enumerate(data['Lights']):
                for j, _ in enumerate(l_i['p0']):
                    if r'p_{%d,%d}' % (i,j) not in fixed_plan:
                        fixed_plan[r'p_{%d,%d}' % (i,j)]=[]
                        fixed_plan[r'd_{%d,%d}' % (i,j)]=[]
                    fixed_plan[r'p_{%d,%d}' % (i,j)].append(step_data[r'p_{%d,%d}' % (i,j)][n])
                    fixed_plan[r'd_{%d,%d}' % (i,j)].append(step_data[r'd_{%d,%d}' % (i,j)][n])

            fixed_plan['DT'].append(DT)
            if n>0:
                fixed_plan['t'].append(fixed_plan['t'][-1]+DT)
            else:
                fixed_plan['t'].append(step_time[0])
        n+=1

    if milp_solver(data,t0=t0,t1=t1,n_samples=int((t1-t0)/DT), fixed_plan=data['Out']['Fixed_Plan'],nthreads=args.threads,
           timelimit=args.timelimit,accuracy=args.accuracy,verbose=args.verbose,
           step=None,obj=args.obj,alpha=args.alpha,beta=args.beta) is None:
        return None


    solve_time = clock_time.time() - start_time
    print 'Total Plan Solve time: %s seconds' % solve_time
    data['Out']['solve_time'] = solve_time
    print (data['Out']).keys()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="the model file to solve")
    parser.add_argument("--t0", help="start time",type=int)
    parser.add_argument("--t1", help="end time",type=int)
    parser.add_argument("--dt0", help="start time step",type=float)
    parser.add_argument("--dt1", help="end time step",type=float)
    parser.add_argument("-t", "--threads", help="number of threads",type=int,default=0)
    parser.add_argument("-n", "--nsamples", help="number of sample points",type=int)
    parser.add_argument("-o", "--out", help="save the solution as OUT")
    parser.add_argument("--verbose", help="log output to console", action="store_true", default=False)
    parser.add_argument("--timelimit", help="time limit to run solver in seconds",type=int,default=0)
    parser.add_argument("--accuracy", help="percentage accuracy of objective value to stop solver",type=float, default=0)
    parser.add_argument("--fixed", help="file to use for a fixed signal plan")
    parser.add_argument("--plan", help="Plan to use in FILE for a k-step solution")
    parser.add_argument("--vplan", help="Variable plan MINOR MAJOR HORIZON BLEND times, DT start end ,for a k-step solution",nargs=6,type=float)
    parser.add_argument("--fplan", help="Fixed Plan with MINOR MAJOR HORIZON times, DT for a k-step solution",nargs=4,type=float)
    parser.add_argument("--average", help="Average over N runs",type=int,default=1)
    parser.add_argument("--DT_factor", help="multiplier to modify the DT used in the plan",type=float,default=1.0)
    parser.add_argument("--DT_offset", help="initial offset for DT",type=float,default=0.0)
    parser.add_argument("--DT_final", help=" DT to use in the final step of plan",type=float)
    parser.add_argument("--DT_file", help="Use DT values defined in file", action="store_true", default=False)
    parser.add_argument("--alpha", help="Obj function alpha weight 1.0 = min delay, 0 = min stops ",type=float,default=1.0)
    parser.add_argument("--beta", help="Obj function beta weight for traffic holding term",type=float,default=0.001)
    parser.add_argument("--obj", help="Obj function to use: OLD, NEW",default='MAX_OUT')
    parser.add_argument("--seed", help="Set the random seed",type=int,default=0)
    parser.add_argument("--tune", help="Run grbtune on the model at step TUNE",action='append',nargs='*',type=str)
    parser.add_argument("--param", help="Load a Gurobi parameter prm file")
    parser.add_argument("--nostepvars", help="dont store each steps variables in the results",action="store_true", default=False)
    parser.add_argument("--novars", help="dont store any variables in the results",action="store_true", default=False)
    parser.add_argument("--zip", help="Compress the results file",action="store_true", default=False)

    args = parser.parse_args()

    f = open(args.file,'r')
    data = json.load(f)
    f.close()

    if args.out:
        data['out_filename']= args.out
    else:
        data['out_filename']= args.file

    fixed=None
    if args.fixed:
        f = open(args.fixed,'r')
        fixed = json.load(f)
        f.close()

        if 'Out' not in fixed:
            print 'File %s does not containt a solution to use as a fixed plan' % args.fixed
            fixed=None
        else:
            fixed = fixed['Out']
    if args.plan or args.vplan or args.fplan:
        data=plan_solver(data,args)
    elif args.DT_final and not args.plan:
        DT_final_solver(data,args)
    else:
        if args.tune != None:
            tune=True
            tune_file = args.tune[0][0]
        else:
            tune=False
            tune_file = ''
        data=milp_solver(data,t0=args.t0,t1=args.t1,dt0=args.dt0,dt1=args.dt1,n_samples=args.nsamples,DT_file=args.DT_file, fixed_plan=fixed,nthreads=args.threads,
                    timelimit=args.timelimit,accuracy=args.accuracy,
                    verbose=args.verbose,obj=args.obj,alpha=args.alpha,
                    beta=args.beta,DT_offset=args.DT_offset,seed = args.seed,
                    tune = tune,tune_file=tune_file,param=args.param)

    if data != None:
        out_file = 'out_'+args.file
        if args.out:
            out_file=args.out
        if not args.novars:
            if args.zip:
                out_file_zip = os.path.splitext(out_file)[0]+".zip"
                zf = zipfile.ZipFile(out_file_zip, mode='w',compression=zipfile.ZIP_DEFLATED)
                zf.writestr(out_file, json.dumps(data))
                zf.close()
            else:
                f = open(out_file,'w')
                json.dump(data,f)
                f.close()
        if 'Run' in data['Out']:
            for run in data['Out']['Run']:
                for step in run['Step']:
                    rm_vars(step)
                rm_vars(run)
        if 'Step' in data['Out']:
            for step in data['Out']['Step']:
                rm_vars(step)
        rm_vars(data['Out'])
        f = open(out_file+'.meta','w')
        json.dump(data,f)
        f.close()

