from __future__ import division
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.image as mpimg
from matplotlib import animation
from matplotlib.patches import Arc,Arrow,Circle
from matplotlib.colors import LinearSegmentedColormap
import math as math
import json
import sys
import argparse
import time as clock_time
import numpy as np
import scipy.interpolate as interp
import copy

# xy_data=[]
# cars = []
# car_polys = []
# car_color = None
# transits = []

def calc_traffic(data,time_factor):
    sim = data['Out']['Microsim']
    Queues = data['Queues']
    Q = len(Queues)
    results = data['Out']
    time=results['t']
    DT=results['DT']
    q_in_max = [0 for qk in data['Queues']]
    q_out_max = [0 for qk in data['Queues']]
    q_flow_max = [0 for qk in data['Queues']]
    for i in range(len(data['Queues'])):
        out_flow  = False
        in_flow = False
        for j in range(len(data['Queues'])):
            f_ij = '%d_%d' % (i,j)
            f_ji = '%d_%d' % (j,i)
            if f_ji in data['Flows']:
                q_in_max[i] += data['Flows'][f_ji]['F_MAX']
                in_flow = True
            if f_ij in data['Flows']:
                q_out_max[i] += data['Flows'][f_ij]['F_MAX']
                out_flow = True

        q_flow_max[i] = q_in_max[i]
        if q_in_max[i] == 0:
            q_flow_max[i] = q_out_max[i]
    Q_DELAY = [qk['Q_DELAY'] * time_factor for qk in data['Queues']]
    q_max=[qk['Q_MAX'] if qk['Q_MAX'] < 60 else 60 for qk in data['Queues']]
    q=[[results['q_%d' %i ][j]/q_max[i] for j in range(len(results['q_%d' %i ]))] for i in range(Q)]
    q_in=[[results['q_{%d,in}' %i ][j]/(q_flow_max[i] * DT[j]) for j in range(len(results['q_%d' %i ]))] for i in range(Q)]
    print q_in[5]
    q_out=[[results['q_{%d,out}' %i ][j]/q_max[i] for j in range(len(results['q_%d' %i ]))] for i in range(Q)]
    q_sig=[[] for qk in data['Queues']]
    for i in range(Q):
        q_p = Queues[i]['Q_P']
        if q_p != None:
            l = q_p[0]
            q_sig[i]=[0 for j in range(len(results['q_%d' %i ]))]
            for k in q_p[1:]:
                for j in range(len(results['q_%d' %i ])):
                    if results['p_{%d,%d}' % (l,k)][j] > 0.5:
                        q_sig[i][j] = 1
    return q,q_in,q_out,q_sig,Q_DELAY


def norm(rx0,ry0,rx1,ry1):
    rx = rx0 - rx1
    ry = ry0 - ry1
    lth = math.sqrt(rx * rx + ry * ry)
    return rx / lth, ry / lth


def test_point(x0, x1,px):

    dx1 = x1 - x0

    if dx1 > 0:
        return x0 <= px <= x1
    else:
        return x1 <= px <= x0



def plot_network(data,ax,cars,road_color='0.8',transit_safe_time = 5, q_delay = None, qtm_data=None, cmap = None):

    radius_light=15 # radius of intersection nodes
    r_s=4 # radius of light signals
    dist_para_edges=5 # distance to space two edges that share a pair of nodes
    width = 2 # width of bar
    line_width = 1
    track_width = 2 # width of track
    track_spacing = 5 # spacing between sleepers along track
    line_color = 'k'

    tail_width = 0
    head_width = 5
    light_color = 'w'
    text_color = 'k'
    ext = [-200,200,-110,110]
    if 'Plot' in data:
        if 'bg_image' in data['Plot']:
            if data['Plot']['bg_image'] != None:
                img = mpimg.imread(data['Plot']['bg_image'])
                bg_alpha = 1.0
                if 'bg_alpha' in data['Plot']:
                    if data['Plot']['bg_alpha'] != None:
                        bg_alpha = data['Plot']['bg_alpha']
                ax.imshow(img,extent=ext,alpha=bg_alpha)
        if 'extent' in data['Plot']:
            ext = data['Plot']['extent']

        if 'line_color' in data['Plot']:
                line_color = data['Plot']['line_color']
        if 'line_width' in data['Plot']:
                line_width = data['Plot']['line_width']
        if 'track_width' in data['Plot']:
                track_width = data['Plot']['track_width']
        if 'track_spacing' in data['Plot']:
            track_spacing = data['Plot']['track_spacing']

    nodes = []
    edges = {}
    xy_data=[]
    car_polys = []
    transits = []
    qtm_q_in_polys = []
    qtm_q_polys = []

    time = data['Out']['t']

    x_min=data['Nodes'][0]['p'][0]
    y_min=data['Nodes'][0]['p'][1]
    x_max=x_min
    y_max=y_min

    for i,n in enumerate(data['Nodes']):
        nodes.append({'n':n,'e':0,'l':None,'p':n['p'],'queues':[]})
        x=n['p'][0]
        y=n['p'][1]
        x_min=min(x_min,x)
        y_min=min(x_min,y)
        x_max=max(x_max,x)
        y_max=max(x_max,y)


    for i,q in enumerate(data['Queues']):
        n0 = q['edge'][0]
        n1 = q['edge'][1]
        q['n0'] = nodes[n0]
        q['n1'] = nodes[n1]
        pair = (n0, n1)
        if n1 < n0 : pair = (n1, n0)
        if pair in edges:
            edges[pair] += 1
        else:
            edges[pair] = 1
        q['pair'] = pair
        nodes[n0]['queues'].append(i)
        nodes[n1]['queues'].append(i)
        q['rx0'] = nodes[n0]['p'][0]
        q['ry0'] = nodes[n0]['p'][1]
        q['rx1'] = nodes[n1]['p'][0]
        q['ry1'] = nodes[n1]['p'][1]
        q['rx'],q['ry'] = norm(q['rx0'],q['ry0'],q['rx1'],q['ry1'])

    transit_nodes = copy.deepcopy(nodes)
    # for i,n in enumerate(nodes):
    #     print i,n['queues']

    plot_transits = False
    if 'Transits' in data:
        transit_id0 = data['Transits'][0]['id']
        for key in data['Out'].keys():
            if key[:len(transit_id0)] == transit_id0:
                plot_transits = True

    print 'plot_transits:', plot_transits

    if plot_transits:
        print 'Transits'
        for transit in data['Transits']:
            print transit['id'],transit['links']
            rx=0
            ry=0
            transit['queues'] = []
            for i in range(1,len(transit['links'])):
                n0 = transit['links'][i-1]
                n1 = transit['links'][i]
                pair = (n0, n1)
                if n1 < n0 : pair = (n1, n0)
                if pair in edges:
                    edges[pair] += 1
                else:
                    edges[pair] = 1
                if i==1:
                    rx0=nodes[n0]['p'][0]
                    ry0=nodes[n0]['p'][1]
                    rx1=nodes[n1]['p'][0]
                    ry1=nodes[n1]['p'][1]
                    rx,ry = norm(rx0,ry0,rx1,ry1)
                    rx0-=ry * dist_para_edges; ry0+=rx * dist_para_edges
                    rx1-=ry * dist_para_edges; ry1+=rx * dist_para_edges
                    nodes[n0]['p'][0]=rx0
                    nodes[n0]['p'][1]=ry0
                    nodes[n1]['p'][0]=rx1
                    nodes[n1]['p'][1]=ry1
                else:
                    rx1=nodes[n1]['p'][0]
                    ry1=nodes[n1]['p'][1]
                    rx1-=ry * dist_para_edges; ry1+=rx * dist_para_edges
                    nodes[n1]['p'][0]=rx1
                    nodes[n1]['p'][1]=ry1
                data['Queues'].append({'transit': True, 'edge': [n0,n1], 'pair': pair, 'p':n['p'],
                                       'n0': transit_nodes[n0], 'n1': transit_nodes[n1],
                                       'rx': rx, 'ry': ry})
                transit['queues'].append(data['Queues'][-1])


        for transit in data['Transits']:
            #print transit['id'],transit['links']
            for transit_link in transit['queues']:
                n0 = transit_link['n0']
                n1 = transit_link['n1']
                rx0=n0['p'][0]
                ry0=n0['p'][1]
                rx1=n1['p'][0]
                ry1=n1['p'][1]
                rx = rx1 - rx0
                ry = ry1 - ry0
                #print
                #print n1['queues']

                for k in n1['queues']:
                    q = data['Queues'][k]
                    test_q = -1
                    qrx0 = nodes[q['edge'][0]]['p'][0]
                    qry0 = nodes[q['edge'][0]]['p'][1]
                    qrx1 = nodes[q['edge'][1]]['p'][0]
                    qry1 = nodes[q['edge'][1]]['p'][1]
                    if k== test_q:
                        print 'queue:',k,'(',q['edge'][0],'->',q['edge'][1],')'
                        print 'x',qrx0,qrx1,rx1,rx,test_point(qrx0,qrx1,rx1)
                        print 'y',qry0,qry1,ry1,ry,test_point(qry0,qry1,ry1)
                    if q['edge'][0] not in transit['links'] and q['edge'][1] in transit['links']:
                        if k== test_q:
                            print 'stop at transit'
                        if abs(ry) > abs(rx) and test_point(qrx0,qrx1,rx1):
                            q['stop_transit'] = data['Queues'][-1]
                            if k== test_q:
                                print k,'NS stop_transit'
                        elif abs(rx) > abs(ry) and test_point(qry0,qry1,ry1):
                            q['stop_transit'] = data['Queues'][-1]
                            if k== test_q:
                                print k,'WE stop_transit'
                    elif q['edge'][0] in transit['links'] and q['edge'][1] not in transit['links']:
                        if k== test_q:
                            print 'start at transit'
                        if abs(ry) > abs(rx) and not test_point(qrx0,qrx1,rx1):
                            q['start_transit'] = data['Queues'][-1]
                            if k== test_q:
                                print k,'NS start_transit'
                        elif abs(rx) > abs(ry) and not test_point(qry0,qry1,ry1):
                            q['start_transit'] = data['Queues'][-1]
                            if k== test_q:
                                print k,'WE start_transit'
                    #else:
                    #    print 'skipped:',q['edge'][0],q['edge'][1]




    for i,l in enumerate(data['Lights']):
        n=nodes[l['node']]
        nodes[l['node']]['l']=i
        n['light']=i
    #     x=n['p'][0]
    #     y=n['p'][1]
    #     p = Circle((x,y), radius_light, fc="w",lw=1)
    #     ax.add_patch(p)
    #     ax.text(x-3,y-3,r'$l_{%d}$' % i,fontsize=16)

    sim = data['Out']['Microsim']
    sim_time=sim['time']
    sim_free_flow_speed = sim['free_flow_speed']
    sim_time_factor = sim['time_factor']
    sim_duration = sim_time[-1] - sim_time[0]

    for i,q in enumerate(data['Queues']):
        pair = edges[q['pair']] > 1
        n0= q['n0']
        n1= q['n1']
        rx0=n0['p'][0]
        ry0=n0['p'][1]
        rx1=n1['p'][0]
        ry1=n1['p'][1]

        rx = rx0-rx1
        ry = ry0-ry1
        lth = math.sqrt(rx*rx+ry*ry)
        rx/=lth
        ry/=lth
        trx0=rx0
        try0=ry0

        # if 'light' in n0:
        #     if pair and 'transit' not in q:
        #         theta = -math.asin(dist_para_edges/radius_light)
        #         trx = rx * math.cos(theta) - ry * math.sin(theta);
        #         ry = rx * math.sin(theta) + ry * math.cos(theta);
        #         rx=trx
        #     trx0-=rx * radius_light; try0-=ry * radius_light
        # elif pair and 'transit' not in q:
        #     trx0-=ry * dist_para_edges; try0+=rx * dist_para_edges
        #if pair and 'transit' not in q:
        #    trx0-=ry * dist_para_edges; try0+=rx * dist_para_edges
        rx = rx1-rx0
        ry = ry1-ry0
        lth = math.sqrt(rx*rx+ry*ry)
        rx/=lth
        ry/=lth
        # if 'light' in n1:
        #     if pair and 'transit' not in q:
        #         theta = math.asin(dist_para_edges/radius_light)
        #         trx = rx * math.cos(theta) - ry * math.sin(theta);
        #         ry = rx * math.sin(theta) + ry * math.cos(theta);
        #         rx=trx
        #     rx1-=rx * radius_light; ry1-=ry * radius_light
        # elif pair and 'transit' not in q:
        #     rx1+=ry * dist_para_edges; ry1-=rx * dist_para_edges
        #if pair and 'transit' not in q:
        #    rx1+=ry * dist_para_edges; ry1-=rx * dist_para_edges
        rx0=trx0
        ry0=try0
        n0['p'][0]=rx0
        n0['p'][1]=ry0
        n1['p'][0]=rx1
        n1['p'][1]=ry1
        rx = rx1-rx0
        ry = ry1-ry0
        lth = math.sqrt(rx*rx+ry*ry)
        tx=rx/lth * radius_light; ty=ry/lth * radius_light
        rx = rx0+(rx1-rx0)/2
        ry = ry0+(ry1-ry0)/2
        if not pair or (pair and 'transit' not in q):
            ax.text(rx+(ty-7),ry-(tx),r'$q_{%d}$' % i,fontsize=16)
        #plot([rx,rx+ty],[ry,ry-tx])
        qline_width = line_width
        qline_color = line_color
        qtrack_width = track_width
        qtrack_spacing = track_spacing
        if 'line_width' in q:
            qline_width = q['line_width']
        if 'line_color' in q:
            qline_color = q['line_color']
        if 'track_width' in q:
            qtrack_width = q['track_width']
        if 'track_spacing' in q:
            qtrack_spacing = q['track_spacing']
        if 'transit' in q:
            rx = rx1-rx0
            ry = ry1-ry0
            tx=(rx/lth) * qtrack_width; ty=(ry/lth) * qtrack_width
            N = int(lth / qtrack_spacing)
            dt = 1.0 / N
            t=0
            for j in range(N):
                qrx0 = rx0 + t * rx
                qry0 = ry0 + t * ry
                qrx1 = rx0 + (t+dt) * rx
                qry1 = ry0 + (t+dt) * ry
                ax.plot([qrx0+ty,qrx0-ty],[qry0-tx,qry0+tx], lw=qline_width,color=qline_color)
                t += dt
            ax.plot([rx0,rx1],[ry0,ry1], lw=qline_width*2,color=qline_color)
        else:
            rx = rx1-rx0
            ry = ry1-ry0
            tx=(rx/lth) * width; ty=(ry/lth) * width
            if 'start_transit' in q:
                qtrack_width = track_width
                if 'track_width' in q['start_transit']:
                    qtrack_width = q['start_transit']['track_width']
                #print i,'start_transit',rx * 10,ry * 10
                rx0 += q['rx'] * 2 * qtrack_width
                ry0 += q['ry'] * 2 * qtrack_width
            if 'stop_transit' in q:
                qtrack_width = track_width
                if 'track_width' in q['stop_transit']:
                    qtrack_width = q['stop_transit']['track_width']
                #print i,'stop_transit',rx * 10,ry * 10
                rx1 += q['rx'] * 2 * qtrack_width
                ry1 += q['ry'] * 2 * qtrack_width
            if n1['l'] is not None:
                rx1 += q['rx'] * 2 * width
                ry1 += q['ry'] * 2 * width
            if n0['l'] is not None:
                rx0 += q['rx'] * 2 * width
                ry0 += q['ry'] * 2 * width
            rx = rx1-rx0
            ry = ry1-ry0
            xy_data.append([[rx0,ry0],[rx1,ry1],[rx,ry],[tx,ty]])
            #print [[rx0,ry0],[rx1,ry1],[rx,ry]]
            #pl.plot([rx0,rx1],[ry0,ry1],'.',c='g')
            ax.add_patch( plt.Polygon([[rx0-ty,ry0+tx],[rx1-ty,ry1+tx],[rx1+ty,ry1-tx],[rx0+ty,ry0-tx]],
                                     closed=True,fill='y',color=road_color,ec=road_color,lw=1))

    for i,q in enumerate(data['Queues']):
        if 'transit' not in q:
            ((rx0,ry0),(rx1,ry1),(rx,ry),(tx,ty)) = xy_data[i]
            if q_delay is not None:
                bins = 50 #len(q_delay[i])
                for j in range(bins):
                    t0 = j / bins
                    t1 = (j + 1) / bins
                    qrx0 = rx0 + rx * t0
                    qry0 = ry0 + ry * t0
                    qrx1 = rx0 + rx * t1
                    qry1 = ry0 + ry * t1
                    delay_col = cmap.to_rgba(q_delay_f[i](t0))
                    ax.add_patch( plt.Polygon([[qrx0-ty,qry0+tx],[qrx1-ty,qry1+tx],[qrx1+ty,qry1-tx],[qrx0+ty,qry0-tx]],
                                     closed=True,fill='y',color=delay_col,ec=delay_col,lw=1))

            if qtm_data is not None:
                bins = qtm_data['bins'] #len(q_delay[i])
                qtm_q_in_polys.append([])
                for j in range(bins[i]):
                    t0 = j / bins[i]
                    t1 = (j + 1) / bins[i]
                    qrx0 = rx0 + rx * t0
                    qry0 = ry0 + ry * t0
                    qrx1 = rx0 + rx * t1
                    qry1 = ry0 + ry * t1
                    p = plt.Polygon([[qrx0-ty,qry0+tx],[qrx1-ty,qry1+tx],[qrx1+ty,qry1-tx],[qrx0+ty,qry0-tx]],
                                     closed=True,fill='y',color='none',ec='none',lw=1)
                    qtm_q_in_polys[i].append(p)
                    ax.add_patch(p)
                qrx0 = rx0 + rx * 0.5
                qry0 = ry0 + ry * 0.5
                qrx1 = rx0 + rx * 1
                qry1 = ry0 + ry * 1
                p = plt.Polygon([[qrx0-ty,qry0+tx],[qrx1-ty,qry1+tx],[qrx1+ty,qry1-tx],[qrx0+ty,qry0-tx]],
                                     closed=True,fill='y',color='none',ec='none',lw=1)
                ax.add_patch(p)
                qtm_q_polys.append(p)



    #for n in nodes:
    #    pl.plot([n['p'][0]],[n['p'][1]],'.',c='g')
        # if xy[0][0] != xy[1][0]:
        #     pl.plot([xy[0][0],xy[1][0]],[xy[0][1],xy[1][1]],'.',c='g')
        # else:
        #     pl.plot([xy[0][0],xy[1][0]],[xy[0][1],xy[1][1]],'.',c='c')
    #for k,xy in enumerate(xy_data):
    #    pl.plot([xy[1][0]],[xy[1][1]],'.',c='r')
        #if 'start_transit' in data['Queues'][k]:
        #    pl.plot([xy[0][0]],[xy[0][1]],'.',c='m')
        #if 'stop_transit' in data['Queues'][k]:
        #    pl.plot([xy[1][0]],[xy[1][1]],'.',c='c')

    if plot_transits:
        print 'Transits'
        for transit in data['Transits']:
            offsets = [0]
            xoffsets = []
            yoffsets = []
            offset = 0
            for n in transit['links'][1:-1]:
                l = nodes[n]['light']
                for transit_l_data in data['Lights'][l]['transits']:
                    if transit['id'] == transit_l_data['id']:
                        offset += transit_l_data['offset']
                        offsets.append(transit_l_data['offset'] * sim_time_factor)
                sched = transit_l_data
            for n in transit['links']:
                xoffsets.append(transit_nodes[n]['p'][0])
                yoffsets.append(transit_nodes[n]['p'][1])
            offsets[0] = offsets[1] - (offsets[2] - offsets[1])
            offsets.append(offsets[-1] + (offsets[-1] - offsets[-2]))
            num_transits = int(math.ceil(sim_duration / (sched['period'] * sim_time_factor)))
            print 'num_transits',num_transits
            print 'offsets',offsets
            print 'xoffsets',xoffsets
            print 'yoffsets',yoffsets
            zero_offsets = [x - offsets[0] for x in offsets]
            # zero_offsets_stops = []
            # xoffsets_stops = []
            # yoffsets_stops = []
            # for k in range(len(zero_offsets)-1):
            #
            #     zero_offsets_stops.append(zero_offsets[k])
            #     xoffsets_stops.append(xoffsets[k])
            #     yoffsets_stops.append(yoffsets[k])
            #
            #     zero_offsets_stops.append(zero_offsets[k] + (zero_offsets[k + 1] - zero_offsets[k]) * 0.4)
            #     xoffsets_stops.append(xoffsets[k] + (xoffsets[k + 1] - xoffsets[k]) * 0.7)
            #     yoffsets_stops.append(yoffsets[k] + (yoffsets[k + 1] - yoffsets[k]) * 0.7)
            #
            #     zero_offsets_stops.append(zero_offsets[k + 1] - (zero_offsets[k + 1] - zero_offsets[k]) * 0.2)
            #     xoffsets_stops.append(xoffsets[k] + (xoffsets[k + 1] - xoffsets[k]) * 0.7)
            #     yoffsets_stops.append(yoffsets[k] + (yoffsets[k + 1] - yoffsets[k]) * 0.7)
            #
            #     zero_offsets_stops.append(zero_offsets[k + 1])
            #     xoffsets_stops.append(xoffsets[k + 1])
            #     yoffsets_stops.append(yoffsets[k + 1])
            # transit['x_f'] = interp.interp1d(zero_offsets_stops,xoffsets_stops)
            # transit['y_f'] = interp.interp1d(zero_offsets_stops,yoffsets_stops)
            transit['x_f'] = interp.interp1d(zero_offsets,xoffsets)
            transit['y_f'] = interp.interp1d(zero_offsets,yoffsets)
            transit['transit_safe_time'] = transit_safe_time
            # print 'zero_offsets_stops',zero_offsets_stops
            # plt.plot(zero_offsets_stops,transit['x_f'](zero_offsets_stops))
            # plt.plot(zero_offsets_stops,transit['y_f'](zero_offsets_stops))
            # plt.show()
            transit['period'] = sched['period'] * sim_time_factor
            transit['offset'] = offsets[0]
            transit['duration'] = sched['duration'] * sim_time_factor
            transit['width'] = track_width
            transit['x_length'] = transit['x_f'](transit['duration']) - transit['x_f'](0) * 0.7
            transit['y_length'] = transit['y_f'](transit['duration']) - transit['y_f'](0) * 0.7
            t_0 = offsets[0]
            t_1 = offsets[-1]
            while t_0 < sim_duration:
                p = ax.add_patch( plt.Polygon([[0,0],[0,0],[0,0],[0,0]],
                                     closed=True,fill='y',color='None',ec='None',lw=1))
                transits.append({'t_0':t_0, 't_1': t_1, 'transit': transit, 'width': track_width,
                                 'duration': transit['duration'], 'poly': p, 'active': False, 'offset': transit['offset']})
                t_0 += transit['period']
                t_1 += transit['period']

    for k,car in enumerate(cars):
        j = 0
        #print car['link']
        while(car['link'][j] < 0):
            j += 1
        car_pos = car['position_f'](0)
        i = car['link'][j]
        rx0,ry0 = xy_data[i][0]
        rx,ry = xy_data[i][2]
        car_x = rx0 + rx * car_pos
        car_y = ry0 + ry * car_pos
        p = Circle((car_x,car_y), 1, fc="None",lw=1,ec='None')
        ax.add_patch(p)
        car_polys.append(p)
    #pl.plot(car_x,car_y,'.',c='None')

    ax.axis('scaled')
    ax.axis('off')


    #ax.set_ylim([x_min-10,x_max+10])
    #ax.set_xlim([y_min-10,y_max+10])
    #ax.set_ylim(ext[2:4])
    #ax.set_xlim(ext[0:2])
    ax.axis(ext)


    #cb1 = mp.colorbar.ColorbarBase(ax_bar, cmap=cmap,norm=cNorm,orientation='vertical')

    #pl.show()
    return cars, car_polys, xy_data, transits, qtm_q_in_polys, qtm_q_polys



# animation function.  This is called sequentially
def draw_frame(frame, time_lu, car_color, scalar_cmap, cars, car_polys, xy_data, transits,
               qtm_q_in_polys, qtm_q_polys, qtm_data):
    #frame_start_time = clock_time.time()
    updated_polys = []
    t = time_lu[frame]

    for k,car in enumerate(cars):
        offset = car['offset_f'](t)
        car_pos = car['position_f'](t) - offset
        delay = car['delay_f'](t)
        i = int(car['link_f'](t))
        if i >= 0 and not car['active']:
            car['active'] = True

        elif i < 0 and car['active']:
            car['active'] = False
            p=car_polys[k]
            p.set_facecolor('None')
            p.set_edgecolor('None')

        if car['active']:
            rx0,ry0 = xy_data[i][0]
            rx1,ry1 = xy_data[i][1]
            rx,ry = xy_data[i][2]
            car_x = rx0 + rx * car_pos
            car_y = ry0 + ry * car_pos
            p=car_polys[k]
            p.center = car_x,car_y
            # p.set_facecolor(scalar_cmap.to_rgba(delay))
            # p.set_edgecolor(scalar_cmap.to_rgba(delay))
            p.set_facecolor(car_color)
            p.set_edgecolor(car_color)
            updated_polys.append(p)

    # print
    # print 'frame:',frame,'time:',t
    for k,transit in enumerate(transits):

        if transit['t_0'] <= t <= transit['t_1'] + transit['duration']:
            transit['active'] = True
            if t <= transit['t_1']:
                t_start = t - transit['t_0']# - transit['offset']
            else:
                t_start = transit['t_1'] - transit['t_0']
            if t >= transit['t_0'] + transit['duration']:
                t_end = t - (transit['t_0'] + transit['duration']) # - transit['offset']
            else:
                t_end = 0 # transit['t_0'] - transit['offset']
            # print transit['transit']['id'],k
            # print 't0: %s t1: %s offset: %s ' % (transit['t_0'], transit['t_1'], transit['offset'])
            # print 't_start: %s t_stop: %s' %(t_start, t_end)
            transit_safe_time = transit['transit']['transit_safe_time']
            t_start += transit_safe_time
            t_end -= transit_safe_time
            if t_start > transit['t_1'] - transit['t_0']:
                t_start = transit['t_1'] - transit['t_0']
            if t_end< 0:
                t_end = 0
            x1 = transit['transit']['x_f'](t_start)
            y1 = transit['transit']['y_f'](t_start)
            x0 = transit['transit']['x_f'](t_end)
            y0 = transit['transit']['y_f'](t_end)
            # x0 = x1 - transit['transit']['x_length']
            # y0 = y1 - transit['transit']['y_length']
            # print transit['transit']['x_length'],transit['transit']['y_length']
            rx,ry = norm(x0,y0,x1,y1)
            tx = transit['width'] * rx
            ty = transit['width'] * ry
            transit['poly'].set_xy([[x0 - ty, y0 + tx], [x1 - ty, y1 + tx], [x1 + ty, y1 - tx], [x0 + ty, y0 - tx]])
            transit['poly'].set_facecolor('k')
            transit['poly'].set_edgecolor('k')
            updated_polys.append(transit['poly'])
        elif transit['active']:
            transit['active'] = False
            transit['poly'].set_facecolor('none')
            transit['poly'].set_edgecolor('none')
            updated_polys.append(transit['poly'])

    if qtm_data is not None:
        q_in_f = qtm_data['q_in_f']
        q_f = qtm_data['q_f']
        Q_DELAY = qtm_data['Q_DELAY']
        bins = qtm_data['bins']
        for i,q_in_f_i in enumerate(q_in_f):
            for k in range(bins[i]):
                t_q_in = t - Q_DELAY[i] * (k / bins[i])

                if t_q_in >= 0:
                    #print t_q_in,q_in_f_i(t_q_in)
                    if q_in_f_i(t_q_in) > 0:
                        col = scalar_cmap.to_rgba(0)
                    else:
                        col = 'none' #scalar_cmap.to_rgba(0)
                    qtm_q_in_polys[i][k].set_facecolor(col)
                    qtm_q_in_polys[i][k].set_edgecolor('none')
                    q_in = q_in_f_i(t_q_in)
                    if q_in > 1.0:
                        q_in = 1.0
                    if q_in < 0.0:
                        q_in = 0.0
                    qtm_q_in_polys[i][k].set_alpha(q_in)
                    updated_polys.append(qtm_q_in_polys[i][k])
        for i,q_f_i in enumerate(q_f):
            ((rx0,ry0),(rx1,ry1),(rx,ry),(tx,ty)) = xy_data[i]
            q = q_f_i(t)
            if q > 0:
                if q > 1.0:
                    q = 1.0
                col = scalar_cmap.to_rgba(0.25)
            else:
                col = 'none'
            rx0 = rx1 - (rx1 - rx0) * q
            ry0 = ry1 - (ry1 - ry0) * q
            qtm_q_polys[i].set_xy([[rx0 - ty, ry0 + tx], [rx1 - ty, ry1 + tx], [rx1 + ty, ry1 - tx], [rx0 + ty, ry0 - tx]])
            qtm_q_polys[i].set_facecolor(col)
            qtm_q_polys[i].set_edgecolor('none')
            qtm_q_polys[i].set_alpha(1.0)
            updated_polys.append(qtm_q_polys[i])

    return updated_polys

def animate(frame, time_lu, car_color, scalar_cmap,plots):
    updated_polys = []
    for plot in plots:
        updated_polys += draw_frame(frame,time_lu,car_color, scalar_cmap, **plot)
    return tuple(updated_polys)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="the model file to animate", nargs='+')
    parser.add_argument("-d", "--duration", type=float, help="the duration of the clip in seconds", default=10.0)
    parser.add_argument("--fps", type=int, help="frames per second", default=15)
    parser.add_argument("--car_color", help="color to render cars", default='k')
    parser.add_argument("--road_color", help="color to render road", default='0.8')
    parser.add_argument("--t1", type=float, help="end of simulation time to animate up to")
    parser.add_argument("--format", help="Video format to use. Options: vga, ,svga, xga, 720p, 1080p")
    parser.add_argument("--titles", help="titles to display under plots", nargs='*')
    parser.add_argument("--transit_safe_time", type = float, help="amount of dafe time before and after transit crossing intersection",default = 5.0)
    parser.add_argument("--save_fig", help="sve_fig to file")
    parser.add_argument("--dpi", help="DPI to save plots in", type=int, default = 300)
    parser.add_argument("--delay_plot", help="plot of delay", action="store_true", default=False)
    parser.add_argument("--model_overlay", help="overlay the model data with the microsimulation", action="store_true", default=False)

    args = parser.parse_args()

    green_red = LinearSegmentedColormap('green_red', {'red': ((0.0, 0.0, 0.0),
                                                            (0.5, 1.0, 1.0),
                                                          (1.0, 1.0, 1.0)),
                                                'green': ((0.0, 1.0, 1.0),
                                                          (0.5, 1.0, 1.0),
                                                          (1.0, 0.0, 0.0)),
                                                'blue':  ((0.0, 0.0, 0.0),
                                                          (1.0, 0.0, 0.0))})
    mp.cm.register_cmap('green_red',green_red)
    cmap_name = 'green_red'
    cmap = mp.cm.get_cmap(cmap_name)
    cNorm  = mp.colors.Normalize(vmin=0, vmax=1)
    scalar_cmap = mp.cm.ScalarMappable(norm=cNorm,cmap=cmap)

    dpi = None
    #print 'DPI:',plt.gcf().dpi
    if args.format is not None:
        dpi = plt.gcf().dpi * 1.5
        if args.format == 'vga':
            fig_size = (640.0/dpi,480.0/dpi)
        elif args.format == 'svga':
            fig_size = (800.0/dpi,600.0/dpi)
        elif args.format == 'xga':
            fig_size = (1024.0/dpi,768.0/dpi)
        elif args.format == '720p':
            fig_size = (1280.0/dpi,720.0/dpi)
        elif args.format == '1080p':
            fig_size = (1920.0/dpi,1080.0/dpi)
        else:
            print 'Unrecognised video format: %s' % args.format

    plot_data = []

    for file in args.file:
        f = open(str(file),'r')
        file_data = json.load(f)
        if file_data is not None:
            plot_data.append(file_data)
        f.close()

    if args.format is None:
        if 'Plot' in plot_data[0] and 'fig_size' in plot_data[0]['Plot']:
            fig_size = tuple(plot_data[0]['Plot']['fig_size'])
        else:
            fig_size = (10,5)

    data_time=plot_data[0]['Out']['t']
    sim = plot_data[0]['Out']['Microsim']
    sim_time=sim['time']
    duration=args.duration

    frames_per_second = float(args.fps)

    total_frames =  int(math.floor(duration * frames_per_second))

    frames_per_data_second = total_frames / (sim_time[-1] - sim_time[0])
    sim_time_f = interp.interp1d(sim_time,range(len(sim_time)),kind='zero')
    frame_DT = 1.0 / frames_per_second
    print 'frame_DT:',frame_DT
    if args.t1 is not None and args.t1 <= sim_time[-1]:
        t1_index = int(sim_time_f(args.t1))
        print 'Trimmed animation to %s sec of simulation' % sim_time[t1_index]
    else:
        t1_index = len(sim_time) - 1

    frame_f = interp.interp1d(np.linspace(0,total_frames,len(sim_time[:t1_index])),sim_time[:t1_index])
    print frame_f(total_frames)
    time_lu = frame_f(range(total_frames))

    fig, ax = plt.subplots(nrows=1, ncols=len(args.file), sharex=False, sharey=False)
    if len(args.file) == 1:
        ax = [ax]
    print ax
    #fig_size = (10,5)
    # line_width = 1
    # tail_width = 0
    # head_width = 5
    # line_color = 'k'
    # light_color = 'w'
    # text_color = 'k'
    # ext = [-200,200,-110,110]
    #if 'Plot' in plot_data[0]:
    #     if 'bg_image' in data['Plot']:
    #         if data['Plot']['bg_image'] != None:
    #             img = mpimg.imread(data['Plot']['bg_image'])
    #             bg_alpha = 1.0
    #             if 'bg_alpha' in data['Plot']:
    #                 if data['Plot']['bg_alpha'] != None:
    #                     bg_alpha = data['Plot']['bg_alpha']
    #             plt.imshow(img,extent=ext,alpha=bg_alpha)
    #     if 'extent' in data['Plot']:
    #         ext = data['Plot']['extent']
    #    if 'fig_size' in plot_data[0]['Plot']:
    #        fig_size = tuple(plot_data[0]['Plot']['fig_size'])
    fig.set_size_inches(fig_size)


    sub_plots = []

    for ax_i,data in enumerate(plot_data):
        print 'axis:',ax_i

        sim = data['Out']['Microsim']
        sim_time=sim['time']
        sim_free_flow_speed = sim['free_flow_speed']
        sim_time_factor = sim['time_factor']
        DT=sim['DT']
        cars = sim['cars']
        data_time=np.array(data['Out']['t']) * sim_time_factor
        q_length = []
        q_bins = 10
        q_delay_data = [[[] for n in range(q_bins)] for i in range(len(data['Queues']))]

        for i,q in enumerate(data['Queues']):
            q_length.append(q['Q_DELAY'] * sim_time_factor * sim_free_flow_speed)
        for j,car in enumerate(cars):
            position = car['position'][:-1]
            delay = []
            link = car['link'][:-1]
            offset = [0] * len(link)
            offset_i = 0
            prev_i = -1
            prev_pos = 0
            for k,i in enumerate(link):
                pos = position[k]
                speed = (pos - prev_pos) / DT
                #print 'speed',speed,'DT',DT
                #print 'free_flow_speed',(sim_free_flow_speed)
                norm_delay = (sim_free_flow_speed - speed) /  sim_free_flow_speed
                if norm_delay < 0:
                    norm_delay = 0.0
                if norm_delay > 1.0:
                    norm_delay = 1.0
                delay.append(norm_delay)
                prev_pos = pos
                if i >= 0:
                    if i != prev_i and prev_i != -1:
                        offset_i += 1
                        offset[k-1] = offset_i
                    position[k] = (position[k] / q_length[i]) + offset_i
                # if j == 22: print k,pos, i , offset_i, (pos / q_length[i]) + offset_i
                norm_pos = position[k] - offset_i
                offset[k] = offset_i
                link[k-1] = link[k]
                prev_i = i
                #print k,int(norm_pos * q_bins)
                if norm_pos >= 0 and link[k] >= 0:
                    #print k,len(q_delay),len(q_delay[link[k]])
                    if int(norm_pos * q_bins) >=  q_bins:
                        print int(norm_pos * q_bins)
                    q_delay_data[link[k]][int(norm_pos * q_bins)].append(norm_delay)
            #print len(sim_time),len(position),len(link)

            car['position_f'] = interp.interp1d(sim_time, position)
            car['offset_f'] = interp.interp1d(sim_time, offset,kind='zero')
            car['link_f'] = interp.interp1d(sim_time, link,kind='zero')
            car['active'] = False
            car['delay_f'] = interp.interp1d(sim_time, delay)

        if args.delay_plot:
            q_delay_f = [interp.interp1d(np.linspace(0,1,q_bins),
                                    [np.average(np.array(q_delay_data[i][n])) for n in range(q_bins)]) for i in range(len(data['Queues']))]
        else:
            q_delay_f = None

        if args.model_overlay:
            q,q_in,q_out,q_sig,Q_DELAY = calc_traffic(data,sim_time_factor)

            q_in_f = [interp.interp1d(data_time,q_in[i],kind='zero') for i in range(len(data['Queues']))]
            q_f = [interp.interp1d(data_time,q[i]) for i in range(len(data['Queues']))]
            bins = [int(Q_DELAY[i] / (frame_DT * 5)) for i in range(len(data['Queues']))]
            print 'bins:',bins
            qtm_data = {'q_in_f': q_in_f, 'q_f': q_f, 'Q_DELAY': Q_DELAY, 'bins': bins}
        else:
            qtm_data = None
        #print 'q_delay',q_delay[0]
        # print cars[0]['delay_f'](sim_time)[0:20]

        # test_car = 22
        # test_time = np.linspace(sim_time[0],sim_time[t1_index],300)
        # plt.clf()
        # plt.plot(sim_time,cars[test_car]['delay_f'](sim_time),'.-',label='delay')
        # pl.plot(test_time,cars[test_car]['link_f'](test_time)*0.1,'-',label='link_f')
        # pl.plot(test_time,cars[test_car]['offset_f'](test_time),'-',label='offset_f')
        # pl.plot(test_time,cars[test_car]['position_f'](test_time) - cars[test_car]['offset_f'](test_time),'-',label='position_f - offset_f')
        # pl.ylim(0,2)
        # pl.xlim(30,40)
        # pl.grid(True)
        # plt.legend()
        # plt.show()
        # plt.clf()

        start_time = clock_time.time()

        cars, car_polys, xy_data, transits, qtm_q_in_polys, qtm_q_polys = plot_network(data,ax[ax_i],cars,args.road_color,args.transit_safe_time,
                                                          q_delay_f,qtm_data, scalar_cmap)
        #plt.axis('scaled')

        car_color = args.car_color
        print "plot_network() completed at: %s seconds" % (clock_time.time() - start_time)
        sub_plots.append({'cars': cars, 'car_polys': car_polys, 'xy_data': xy_data, 'transits': transits,
                          'qtm_q_in_polys': qtm_q_in_polys, 'qtm_q_polys': qtm_q_polys,
                          'qtm_data': qtm_data})
        if args.titles is not None and len(args.titles) > ax_i:
            ax[ax_i].set_title(args.titles[ax_i])

        length = len(data['Out']['t'])

    #plt.axis('off')
    if args.save_fig is not None:
        plt.savefig(args.save_fig,bbox_inches='tight',dpi=args.dpi)
    plt.show()

    def init():
        updated_polys = []
        for plot in sub_plots:
            updated_polys += plot['car_polys']
            updated_polys += [transit['poly'] for transit in plot['transits']]
            updated_polys += [p for p in qtm_q_in_polys]
            updated_polys += [p for p in qtm_q_polys]
        return tuple(updated_polys)


    if not args.delay_plot:
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, fargs = (time_lu, car_color, scalar_cmap,sub_plots), init_func=init,
                                       frames=total_frames, interval=1000/frames_per_second, blit=True)



        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        #FFwriter = animation.FFMpegWriter()
        anim.save('%s.mp4' % sys.argv[1].split('.')[0], fps=frames_per_second, dpi = dpi, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])

        print "Animation completed at: %s seconds" % (clock_time.time() - start_time)