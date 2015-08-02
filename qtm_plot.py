from __future__ import division
import matplotlib.pyplot as pl
import matplotlib as mp
import matplotlib.image as mpimg
from matplotlib.patches import Arc,Arrow,Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import numpy as np
import json
import sys
import argparse
import os
import os.path
import numpy as np
import math as math
import zipfile
import zlib
import pandas as pd
from IPython.display import display

def calc_total_stops(data,results):
    Queues = data['Queues']
    time=results['t']
    stops=0
    for i,q in enumerate(Queues):
        for n,t in enumerate(time[1:-1]):
            stops+=abs(results['q_%d' %i ][n] - results['q_%d' %i ][n-1])
    return 0.5 * stops

def calc_total_travel_time(data,results):
    Queues = data['Queues']
    time=results['t']
    sum_in=0
    sum_out=0
    for i,q in enumerate(Queues):
        for n,t in enumerate(time):

            if Queues[i]['Q_IN'] > 0:
                sum_in+=(t*results['q_{%d,in}' %i ][n])
            if Queues[i]['Q_OUT'] > 0:
                sum_out+=(t*results['q_{%d,out}' %i ][n])
    return sum_out-sum_in

def calc_delay(data, results, args=[]):


    Queues = data['Queues']
    #if args:
    #    Queues=[data['Queues'][int(i)] for i in args]
    #Q = len(Queues)
    #results = data['Out']
    #if step != None:
    #    if 'Step' in results:
    #        results = data['Out']['Step'][step]
    time=results['t']
    cumu_in = []
    cumu_out= []
    cumu_delay=[]
    in_q=[]
    out_q=[]
    q_delay=0

    if not args:
        for i,q in enumerate(Queues):
            if Queues[i]['Q_IN'] > 0:
                in_q.append(results['q_{%d,in}' %i ])

            if Queues[i]['Q_OUT'] > 0:
                out_q.append(results['q_{%d,out}' %i ])
    else:
        in_q.append(results['q_{%d,in}' % int(args[0]) ])

        out_q.append(results['q_{%d,out}' % int(args[-1]) ])
        for i in args:
            q_delay+=Queues[i]['Q_DELAY']


    for i,t in enumerate(time):
        if i==0:
            cumu_in.append(0)
            cumu_out.append(0)

        else:
            cumu_in.append(cumu_in[-1])
            cumu_out.append(cumu_out[-1])


            for q in in_q:
                cumu_in[i]+=q[i]
            for q in out_q:
                cumu_out[i]+=q[i]

    n_out= 0
    for n_in,q in enumerate(cumu_in):
        if n_out<len(cumu_out):
            while cumu_out[n_out] < q:
                n_out+=1
                if n_out>=len(cumu_out): break
            if n_out<len(cumu_out):
                cumu_delay.append(time[n_out]-time[n_in] - q_delay)
            else:
                cumu_delay.append(cumu_delay[-1])
            if cumu_delay[-1]<0:
                cumu_delay[-1]=0

        else:
            cumu_delay.append(cumu_delay[-1])
    return time,cumu_in,cumu_out,cumu_delay

def calc_histogram_delay(data, step, width, queues=[],oversample=10):


    results = data['Out']


    if step != None:
        if 'Step' in results:
            results = data['Out']['Step'][step]
            label = results['label']
    time = results['t']
    total_time = time[-1] - time[0]
    N = int(total_time / width)
    bins = [0 for x in range(N)]
    bin_ranges = [width*x for x in range(N+1)]

    M = N * oversample
    Q = 1
    if not queues:
        queues = [range(len(data['Queues']))]
        Q = len(queues)

    for q in queues:

        cumu_time,cumu_in,cumu_out,cumu_delay = calc_delay(data,step, q)
        in_start=0
        in_end=1
        for j in range(len(cumu_in)):
            if cumu_in[j] > cumu_in[in_end]: in_end=j
        j=0
        while cumu_in[in_start+1]<1e-6 and in_start<in_end-1: in_start+=1


        t=time[in_start]
        dt = (time[in_end] - time[in_start])/M

        k=in_start
        while t<time[in_end]:
            alpha = (t- time[k]) / (time[k+1] - time[k])
            delay = (1-alpha) * cumu_delay[k] + alpha * cumu_delay[k+1]
            j=0
            while delay>bin_ranges[j+1] and j<N: j+=1
            w = 1/((cumu_in[-1]) /(bin_ranges[j+1]-bin_ranges[j]))
            bins[j]+=1/oversample
            t+=dt
            if t>time[k+1]: k+=1
    return bins,bin_ranges

def plot_delay_gt(data, step, width, threshold, queues=[],plots=None,line_style=['-'],colours='krgb'):
    pl.clf()
    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    fig.set_size_inches(15, 7)
    titles=[]

    i=0
    c_i=0

    if plots == None: plots = [1]
    offset=0
    for num_files in plots:
        plot_data = dict()
        plot_label=''
        for file in range(num_files):


            d=data[offset+file]
            title = d['Title']
            titles.append(d['Title'])
            results = d['Out']
            label =  d['Out']['label']
            if 'Step' in results:
                N = len(d['Out']['Step'][0]['t'])-1
            else:
                N = len(results['t'])-1
            if step != None:
                if 'Step' in results:
                    results = d['Out']['Step'][step]
                    label = results['label']
            if len(plot_label)==0:
                plot_label = '$'+label.split('$')[1]+'$'

            time = results['t']
            total_time = time[-1] - time[0]
            x_range = 0
            bins,bin_ranges = calc_histogram_delay(d, step, width, queues,10)
            total = sum(bins)
            #bins = [x/total /(bin_ranges[j+1]-bin_ranges[j]) for j,x in enumerate(bins)]
            above = 0
            below = 0
            for j,x in enumerate(bins):
                bins[j] = x * (bin_ranges[j+1]-bin_ranges[j])
                if bin_ranges[j]>threshold:
                    above+=bins[j]
                else:
                    below+=bins[j]
            plot_data[N]=above
        X = sorted(plot_data)
        Y = [plot_data[x] for x in X]
        ax.plot(X,Y,c=colours[c_i], label=plot_label)
        if i+1 < len(line_style): i += 1
        if c_i+1 < len(colours): c_i += 1
        offset+=num_files
    #ax.set_xlim(0, x_range)
    ax.grid()
    ax.set_xlabel('N (Samples)')
    ax.set_ylabel('Number of vehicles delayed > %0.8g' % threshold)
    ax.legend(loc='best')

    pl.title('Delay > %0.1g Histogram for Queues %s' % (threshold,', '.join([str(q) for q in queues])) )


def plot_delay_histogram(data, step, width, queues=[],line_style=['--'],colours='krgb',args=None):
    pl.clf()
    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    fig.set_size_inches(15, 7)
    titles=[]

    i=0
    c_i=0

    for d in data:
        title = d['Title']
        titles.append(d['Title'])
        results = d['Out']
        label = d['Out']['label']

        if step != None:
            if 'Step' in results:
                results = d['Out']['Step'][step]
                label = results['label']
        time = results['t']
        total_time = time[-1] - time[0]
        x_range = 0
        bins,bin_ranges = calc_histogram_delay(d, step, width, queues,10)
        total = sum(bins)
        for j,x in enumerate(bins):
            bins[j] = x *(bin_ranges[j+1]-bin_ranges[j])
            if bins[j]>1e-5 and x_range<bin_ranges[j+1]:
                if j<len(bin_ranges)-2:
                    x_range=bin_ranges[j+2]
                else:
                    x_range=bin_ranges[j+1]
        #bins = [x/total /(bin_ranges[j+1]-bin_ranges[j]) for j,x in enumerate(bins)]
        ax.plot(bin_ranges[:-1],bins,c=colours[c_i], linestyle=line_style[i], label=label)
        if i+1 < len(line_style): i += 1
        if c_i+1 < len(colours): c_i += 1
    #ax.set_xlim(0, x_range)
    ax.grid()
    ax.set_xlabel('Delay (sec)')
    ax.set_ylabel('Frequency')
    if args.x_limit != None:
        ax.set_xlim(args.x_limit[0], args.x_limit[1])
    if args.y_limit != None:
        ax.set_ylim(args.y_limit[0], args.y_limit[1])
    ax.legend(loc='best')

    pl.title('Delay Histogram for Queues %s' % ', '.join([str(q) for q in queues]) )

def label_distance(theta,i):
    dx = 0
    dy = 0
    if theta > 155 or theta < -155: # Vertical down
        if i>9: dx = -15
        else: dx = -4
    elif theta > -25 and theta < 25: # Vertical up
        dx = -22
    elif theta > -115 and theta < -65: # Vertical up
        dy = 16
    elif theta > 65 and theta < 115: # Vertical up
        dy = -8
    else: # Default
        dx =  -7
    return dx,dy

def plot_network_figure(data,figsize,type='arrow'):

    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)

    green_red = LinearSegmentedColormap('green_red', {'red': ((0.0, 0.0, 0.0),
                                                            (0.5, 1.0, 1.0),
                                                          (1.0, 1.0, 1.0)),
                                                'green': ((0.0, 1.0, 1.0),
                                                          (0.5, 1.0, 1.0),
                                                          (1.0, 0.0, 0.0)),
                                                'blue':  ((0.0, 0.0, 0.0),
                                                          (1.0, 0.0, 0.0))})

    black_green_red = LinearSegmentedColormap('black_green_red', {'red': ((0.0, 0.0, 0.0),
                                                            (0.5, 1.0, 1.0),
                                                          (1.0, 1.0, 1.0)),
                                                'green': ((0.0, 0.0,0.0),
                                                          (0.1, 1.0, 1.0),
                                                          (0.5, 1.0, 1.0),
                                                          (1.0, 0.0, 0.0)),
                                                'blue':  ((0.0, 0.0, 0.0),
                                                          (1.0, 0.0, 0.0))})

    mp.cm.register_cmap('green_red',green_red)
    mp.cm.register_cmap('black_green_red',black_green_red)

    cmap_name = args.colormap
    gamma = args.gamma

    cmap = pl.get_cmap(cmap_name)
    cmap.set_gamma(gamma)

    Z = [[0,0],[0,0]]
    levels = range(0,110,10)
    CS3 = pl.contourf(Z, levels, cmap=cmap)
    pl.clf()

    cNorm  = mp.colors.Normalize(vmin=0, vmax=1)
    scalarMap = mp.cm.ScalarMappable(norm=cNorm,cmap=cmap)

    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    fig.set_size_inches(10, 5)

    cNorm  = mp.colors.Normalize(vmin=0, vmax=1)
    scalarMap = mp.cm.ScalarMappable(norm=cNorm,cmap=cmap)

    if figsize != None:
        fig.set_size_inches(figsize)
    else:
        fig.set_size_inches((10,5))
    line_width = 1
    tail_width = 0
    head_width = 5
    r=15 # radius of intersection nodes
    label_space=15 # label spacing from line
    d=5 # distance to space two edges that share a pair of nodes
    width = 3 # width of bar
    lx = 3 # light label x offset
    ly = 3 # light label y offset
    line_color = 'k'
    edge_color = 'k'
    light_color = 'w'
    text_color = 'k'
    font_size = 16
    ext = [-200,200,-110,110]
    if 'Plot' in data:
        ext = data['Plot']['extent']
        if 'bg_image' in data['Plot']:
            if data['Plot']['bg_image'] != None:
                img = mpimg.imread(data['Plot']['bg_image'])
                bg_alpha = 1.0
                if 'bg_alpha' in data['Plot']:
                    if data['Plot']['bg_alpha'] != None:
                        bg_alpha = data['Plot']['bg_alpha']
                pl.imshow(img,extent=ext,alpha=bg_alpha)

        fig_size = tuple(data['Plot']['fig_size'])
        if 'line_width' in data['Plot']:
            line_width = data['Plot']['line_width']
        if 'head_width' in data['Plot']:
            head_width = data['Plot']['head_width']
        if 'tail_width' in data['Plot']:
            tail_width = data['Plot']['tail_width']
        if 'line_color' in data['Plot']:
            line_color = data['Plot']['line_color']
        if 'edge_color' in data['Plot']:
            edge_color = data['Plot']['edge_color']
        if 'light_color' in data['Plot']:
            light_color = data['Plot']['light_color']
        if 'text_color' in data['Plot']:
            text_color = data['Plot']['text_color']
        if 'font_size' in data['Plot']:
            font_size = data['Plot']['font_size']
        if 'label_space' in data['Plot']:
            label_space = data['Plot']['label_space']
        if 'r_light' in data['Plot']:
            r = data['Plot']['r_light']
        if 'd_edges' in data['Plot']:
            d = data['Plot']['d_edges']
        if 'width_bar' in data['Plot']:
            width = data['Plot']['width_bar']
        if 'lx' in data['Plot']:
            lx = data['Plot']['lx']
        if 'ly' in data['Plot']:
            ly = data['Plot']['ly']

    nodes = []
    edges = {}

    if "Annotations" in data:
        for a in data['Annotations']:
            p = a['point']
            ax.text(p[0],p[1],a['label'],fontsize=font_size,color=text_color)

    x_min=data['Nodes'][0]['p'][0]
    y_min=data['Nodes'][0]['p'][1]
    x_max=x_min
    y_max=y_min

    for i,n in enumerate(data['Nodes']):
        nodes.append({'n':n,'e':0,'l':None})
        x=n['p'][0]
        y=n['p'][1]
        x_min=min(x_min,x)
        y_min=min(x_min,y)
        x_max=max(x_max,x)
        y_max=max(x_max,y)

    for i,q in enumerate(data['Queues']):
        n0=q['edge'][0]
        n1=q['edge'][1]
        pair = (n0,n1)
        if n1<n0 : pair = (n1,n0)
        if pair in edges:
            edges[pair]+=1
        else:
            edges[pair]=1
        q['pair']=pair

    for i,l in enumerate(data['Lights']):
        n=data['Nodes'][l['node']]
        nodes[l['node']]['l']=i
        n['light']=i
        x=n['p'][0]
        y=n['p'][1]
        if type == 'arrow':
            p = Circle((x,y), r, fc=light_color)
            ax.add_patch(p)
            ax.text(x-lx,y-ly,r'$l_{%d}$' % int(i+1),fontsize=font_size)
        else:
            r=15

    for i,q in enumerate(data['Queues']):
        pair = edges[q['pair']] > 1
        if 'label' in q:
            label = q['label']
        else:
            label = ''
        n0= data['Nodes'][q['edge'][0]]
        n1= data['Nodes'][q['edge'][1]]
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
        if 'light' in n0:
            if pair:
                theta = -math.asin(d/r)
                trx = rx * math.cos(theta) - ry * math.sin(theta);
                ry = rx * math.sin(theta) + ry * math.cos(theta);
                rx=trx
            trx0-=rx * r; try0-=ry * r
        elif pair:
            trx0-=ry * d; try0+=rx * d
        rx = rx1-rx0
        ry = ry1-ry0
        lth = math.sqrt(rx*rx+ry*ry)
        rx/=lth
        ry/=lth
        if 'light' in n1:
            if pair:
                theta = math.asin(d/r)
                trx = rx * math.cos(theta) - ry * math.sin(theta);
                ry = rx * math.sin(theta) + ry * math.cos(theta);
                rx=trx
            if type == 'arrow':
                rx1-=rx * (r+line_width); ry1-=ry * (r+line_width)
            else:
                rx1-=rx * (r); ry1-=ry * (r)
        elif pair:
            rx1+=ry * d; ry1-=rx * d
        rx0=trx0
        ry0=try0
        rx = rx1-rx0
        ry = ry1-ry0
        lth = math.sqrt(rx*rx+ry*ry)
        theta = math.degrees(math.atan2(rx,ry))
        dx,dy = label_distance(theta,i)
        tx=ry/lth * label_space + dx; ty=rx/lth * label_space + dy
        tc = 0.5
        if 'text_pos' in q:
            tc = q['text_pos']
        rx = rx0+(rx1-rx0)*tc
        ry = ry0+(ry1-ry0)*tc

        if type == 'arrow':
            qtext_color = text_color
            qline_width = line_width
            qhead_width = head_width
            qtail_width = tail_width
            qedge_color = edge_color
            qline_color = line_color
            qfont_weight = None
            qfont_size = font_size
            qoutline = None
            if 'text_color' in q:
                qtext_color = q['text_color']
            if 'line_width' in q:
                qline_width = q['line_width']
            if 'head_width' in q:
                qhead_width = q['head_width']
            if 'tail_width' in q:
                qtail_width = q['tail_width']
            if 'edge_color' in q:
                qedge_color = q['edge_color']
            if 'line_color' in q:
                qline_color = q['line_color']
            if 'font_weight' in q:
                qfont_weight = q['font_weight']
            if 'font_size' in q:
                qfont_size = q['font_size']
            if 'outline' in q:
                qoutline = q['outline']
            if qfont_weight == 'bold':
                qlabel = r'$\mathbf{q_{%d}}$' % int(i+1)
            else:
                qlabel = r'$q_{%d}$' % int(i+1)
            if qoutline != None:
                ax.text(rx+(tx),ry-(ty), qlabel, fontsize=qfont_size+1, color=qoutline)
            ax.text(rx+(tx),ry-(ty), qlabel, fontsize=qfont_size, color=qtext_color)

            #plot([rx,rx+ty],[ry,ry-tx])
            arrow = ax.arrow(rx0,ry0,rx1-rx0,ry1-ry0, shape='full', lw=qline_width,color=qline_color,length_includes_head=True, head_width=qhead_width, width=qtail_width)
            arrow.set_ec(qedge_color)
            arrow.set_fc(qline_color)

        elif type == 'bar' or type == 'carrow':
            ax.text(rx+(tx),ry-(ty),'%d' % int(i+1),fontsize=font_size, fontweight=qfont_weight,color=scalarMap.to_rgba(0))
            N=len(q['cmap'])
            t=0
            #q=0.0
            rx = rx1-rx0
            ry = ry1-ry0
            tx=(rx/lth) * width; ty=(ry/lth) * width
            qrx0=rx1 #- rx * q
            qry0=ry1 #- ry * q
            if N == 1:
                dt = 1
            else:
                dt=(1.0)/(N)
            # for j in range(N):
            #     qrx0=rx0 + t * rx
            #     qry0=ry0 + t * ry
            #     qrx1=rx0 + (t+dt) * rx
            #     qry1=ry0 + (t+dt) * ry
            #     colorVal = scalarMap.to_rgba(q['cmap'][j])
            #     if type == 'bar':
            #         ax.add_patch(mp.patches.Polygon([[qrx0-ty,qry0+tx],[qrx1-ty,qry1+tx],[qrx1+ty,qry1-tx],[qrx0+ty,qry0-tx]],closed=True,fill='y',color=colorVal,ec='none',lw=0.9))
            #     else:
            #         arrow = ax.arrow(qrx0,qry0,rx1-qrx0,ry1-qry0, shape='full', lw=line_width,color=colorVal,length_includes_head=True, head_width=head_width, width=tail_width)
            #         arrow.set_ec(colorVal)
            #         arrow.set_fc(colorVal)
            #     t+=dt


            t = np.linspace(0,1,N)
            x =rx0 + t * rx
            y =ry0 + t * ry
            points = np.array([x,y]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            bar_cmap = np.array([scalarMap.to_rgba(q['cmap'][j]) for j in range(N)])
            bar_widths = 1 + np.array(q['cmap'])*2*width
            lc = LineCollection(segments,linewidths=bar_widths,colors=bar_cmap)
            ax.add_collection(lc)

        if type == 'bar' or type == 'carrow':
            for i,l in enumerate(data['Lights']):
                n=data['Nodes'][l['node']]
                nodes[l['node']]['l']=i
                n['light']=i
                x=n['p'][0]
                y=n['p'][1]
                p = Circle((x,y), r,ec=scalarMap.to_rgba(0),fc='w')
                ax.add_patch(p)
                ax.text(x-3,y-3,r'%d' % int(i+1),fontsize=10,color=scalarMap.to_rgba(0))



    pl.axis('scaled')
    #ax.set_ylim([x_min-10,x_max+10])
    #ax.set_xlim([y_min-10,y_max+10])
    #[-200,200,-110,110]
    #ax.set_ylim([-110,110])
    #ax.set_xlim([-200,200])
    ax.set_ylim(ext[2:4])
    ax.set_xlim(ext[0:2])
    pl.axis('off')
    if type == 'bar' or type == 'carrow':
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        pl.colorbar(CS3,cax=cax)

def plot_network(args):
    for f in args.files:
        data = open_data_file(f)
        if data is not None:
            plot_network_figure(data,args.figsize)

def plot_network_delay(args):
    """
    :param args:
    :return:
    """
    type = args.plot_network_delay
    if type == 'arrow': type = 'carrow'

    for f in args.files:
        data = open_data_file(f)
        if data is not None:

            results = data['Out']
            if 'Run' in results:
                if args.run:
                    results = data['Out']['Run'][args.run]
                else:
                    results = data['Out']['Run'][0]
                #if len(data['Out']['Run']) > 1:
                #        del data['Out']['Run'][1:-1]

            NQ=-1
            DT = float(results['DT'][0])
            max_delay = 10
            if args.delay_lt:
                max_delay = args.delay_lt
            if args.n <= 0:
                for i,q in enumerate(data['Queues']):

                    qi = results['q_%d' %i]
                    qi_in = np.array(results['q_{%d,in}' %i])
                    qi_out = results['q_{%d,out}' %i]
                    Q_DELAY = q['Q_DELAY']
                    N = len(qi)
                    c_in  = np.zeros(N)
                    c_out = np.zeros(N)
                    c_in[0] = qi_in[0]
                    c_out[0] = qi_out[0]
                    for n in range(1,N):
                        c_in[n]=c_in[n-1] + qi_in[n]
                        c_out[n]=c_out[n-1] + qi_out[n]
                    if i==NQ:
                        #pl.figure()
                        pl.plot(range(0,N),c_in)
                        pl.plot(range(0,N),c_out)
                    delay = ( np.sum((c_in - c_out) * DT) - (np.sum(qi_in) * Q_DELAY) ) / np.sum(qi_in)
                    q['cmap'] = [delay / max_delay]
                    #print 'q_%d' % i,'\tdelay: %f' % delay,'\tcars: %d' % np.sum(qi_in),'\tQ_DELAY: %d' %Q_DELAY
            else:
                N_dt = args.n

                dt = DT/N_dt
                print 'DT=',results['DT'][0]
                print 'dt=',dt
                t=dt
                nq = len(data['Queues'])
                qdat = []
                cdat = []
                #max_delay = 0
                for i,q in enumerate(data['Queues']):
                    qi = results['q_%d' %i]
                    qi_in = np.array(results['q_{%d,in}' %i]) * (dt/DT)
                    qi_out = np.array(results['q_{%d,out}' %i])* (dt/DT)
                    qi_in_total = np.sum(qi_in)*(DT/dt)
                    #print 'sum(q_in)=',np.sum(qi_in)
                    #print 'sum(q_out)=',np.sum(qi_out)
                    Q_DELAY = q['Q_DELAY']
                    Nt = int(len(qi_in)*N_dt)
                    N = int(math.ceil(Q_DELAY / dt))
                    #print 'N=',N
                    q_delay = float(Q_DELAY)/N
                    Q = q['Q_MAX']/N
                    #print i,N,Q
                    qs = np.zeros((N,Nt))
                    ys = np.zeros((N,Nt))
                    cy = np.zeros((N+1,Nt))
                    #ys[0,0] = qi_in[0]*q_delay
                    #ys[N-1,0] = qi_out[0]*dt_q
                    y0  = np.interp(np.linspace(0,len(qi),Nt,endpoint=False),np.linspace(0,len(qi),len(qi),endpoint=False),qi_in)
                    ys[N-1,:] = np.interp(np.linspace(0,len(qi),Nt,endpoint=False),np.linspace(0,len(qi),len(qi),endpoint=False),qi_out)
                    #print sum(qi_in)
                    for k in range(1,Nt):
                        #ys[0,k] = qi_in[int(k/N)]*dt
                        #ys[N-1,k] = qi_out[int(k/N)]

                        qs[0,k] = qs[0,k-1] + y0[k-1] - ys[0,k-1]
                        #qs[0,k] = qs[0,k-1] + qi_in[int(k/N)]*dt_q - ys[0,k-1]
                        for n in range(1,N):
                            qs[n,k] = qs[n,k-1] + ys[n-1,k-1] - ys[n,k-1]
                            qs[n,k] = max(qs[n,k],0)
                        ##qs[1:,k] = qs[1:,k-1] + ys[:-1,k-1] - ys[1:,k-1]
                        ##qs[1:,k] = np.fmax(qs[1:,k],0)
                        for n in range(0,N-1):
                            ys[n,k] = min(qs[n,k], Q-qs[n+1,k])
                            ys[n,k] = max(ys[n,k],0)
                        ##ys[:-1,k] = np.fmin(qs[:-1,k], Q-qs[1:,k])
                        ##ys[:-1,k] = np.fmax(ys[:-1,k],0)
                        #ys[N-1,k] = qi_out[int(k/N)]*dt_q

                    cy[0] = np.cumsum(y0)
                        #cy[1:,k] = cy[1:,k-1]+ys[0:,k]
                    cy[1:] = np.cumsum(ys,axis=1)
                    #print len(np.linspace(0,len(qi),len(qi)*N))
                    #y0  = np.interp(np.linspace(0,len(qi),len(qi)*N,endpoint=False),np.linspace(0,len(qi),len(qi),endpoint=False),qi_in*dt_q)
                    #print y0
                    #for k in range(1,ys.shape[1]):
                    #    cy[0,k] = cy[0,k-1]+y0[k]
                    #    cy[1:,k] = cy[1:,k-1]+ys[0:,k]
                    delay = (np.sum((cy[:-1,:] - cy[1:,:]) *dt ,axis=1 ) - qi_in_total*q_delay)# / qi_in_total
                    #print 'delay %d' %i, np.sum(delay)
                    #print 'q_%d' % i,'\tdelay: %f' % (np.sum(delay) ),'\tcars: %d' % (qi_in_total),'\tQ_DELAY: %d' % Q_DELAY
                    #pl.figure()
                    #pl.plot(range(0,len(delay)),delay)
                    #pl.title(r'$q_%d$' % i)
                    #pl.ylim(0,200)
                    #pl.figure()
                    q['cmap'] = delay / max_delay#qs[:,qs.shape[1]*0.25]/np.max(qs)#[j/float(N) for j in range(0,N)]
                    #max_delay = max(max_delay,np.max(delay))
                    #max_delay = 3.0
                    #print np.max(qs)
                    qdat.append(qs)
                    cdat.append(cy)

                    #if i == 0:
                    #    print len(qi)*N
                    #    print len(np.linspace(0,len(qi),len(qi)*N))
                    #    print cdat[0].shape[1]
                        #pl.plot(range(0,len(qi_in)),qi_in)
                        #pl.plot(range(0,len(qi_in)),qi_out)
                        #pl.plot(range(0,len(qi_in)),qi)


            #pl.figure()
            #pl.plot(range(0,qdat[0].shape[1]),qdat[0][0,:])
            #pl.plot(range(0,qdat[0].shape[1]),qdat[0][-1,:])
                #pl.figure()

                #pl.plot(np.linspace(0,len(qi),cdat[NQ].shape[1],endpoint=False),cdat[NQ][0,:])
                #pl.plot(np.linspace(0,len(qi),cdat[NQ].shape[1],endpoint=False),cdat[NQ][1,:])
                #pl.plot(np.linspace(0,len(qi),cdat[NQ].shape[1],endpoint=False),cdat[NQ][-2,:] - cdat[NQ][-1,:])
                print 'max_delay',max_delay
                #max_delay = 2
                #q['cmap'] = q['cmap'] / max_delay
            for i,q in enumerate(data['Queues']):

                qi = results['q_%d' %i]
                qi_in = np.array(results['q_{%d,in}' %i])
                qi_out = results['q_{%d,out}' %i]
                Q_DELAY = q['Q_DELAY']
                N = len(qi)
                c_in  = np.zeros(N)
                c_out = np.zeros(N)
                c_in[0] = qi_in[0]
                c_out[0] = qi_out[0]
                for n in range(1,N):
                    c_in[n]=c_in[n-1] + qi_in[n]
                    c_out[n]=c_out[n-1] + qi_out[n]
                if i==NQ:
                    #pl.figure()
                    pl.plot(range(0,N),c_in)
                    pl.plot(range(0,N),c_out)
                delay = np.sum((c_in - c_out) * DT) - (np.sum(qi_in) * Q_DELAY)
                #print 'q_%d' % i,'\tdelay: %f' % delay,'\tcars: %d' % np.sum(qi_in),'\tQ_DELAY: %d' %Q_DELAY
            #pl.xlim(0,600)
            #pl.ylim(75,100)
            plot_network_figure(data,args.figsize,type=type)


def plot_delay(data, step, queues=[],line_style=['--'],args=None):
    # print len(time),time
    # print len(cumu_in),cumu_in
    # print len(cumu_out),cumu_out
    pl.clf()
    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    if args.figsize:
        fig.set_size_inches(args.figsize[0], args.figsize[1])
    else:
        fig.set_size_inches(15, 7)
    ax2 = ax.twinx()

    i=0
    c_i=0
    l_i=0
    m_i=0
    lab_i=0
    labels=['']
    if args.labels: labels=args.labels




    if args.markerfill == 'n':
        mfc = ['None'] * len(args.color)
    else:
        mfc = [args.color[i] if args.markerfill[i]=='y' else 'None' for i in range(len(args.markerfill))]

    mevery = args.markevery
    if len(mevery) < len(args.marker):
        mevery = mevery + [1] * (len(args.marker) - len(mevery))
    i=0
    arrival_plot = False
    for d in data:
        results = d['Out']
        if 'Run' in results:
            results = d['Out']['Run'][0]
        if not queues:
            queues = [range(len(d['Queues']))]
        #label = results['Out']['label']
        if step != None:
            if 'Step' in results['Out']:
                label = results['Out']['Step'][step]['label']
        for q in queues:
            #title = d['Title']
            #titles.append(d['Title'])
            time,cumu_in,cumu_out,cumu_delay = calc_delay(d, results, args=q)
            if arrival_plot == False:
                ax.plot(time,cumu_in,c=args.color[c_i], linestyle=line_style[i], label='Cumulative arrivals', marker=args.marker[m_i],markeredgecolor=args.color[c_i], markerfacecolor=mfc[c_i],markevery=mevery[m_i])
                if i+1 < len(line_style): i += 1
                if c_i+1 < len(line_style): c_i += 1
                if m_i+1 < len(args.marker): m_i += 1
                arrival_plot = True
            ax.plot(time,cumu_out,c=args.color[c_i],linestyle=line_style[i], label='%s cumulative departures' % labels[lab_i], marker=args.marker[m_i],markeredgecolor=args.color[c_i],markerfacecolor=mfc[c_i],markevery=mevery[m_i])
            if i+1 < len(line_style): i += 1
            if c_i+1 < len(line_style): c_i += 1
            if m_i+1 < len(args.marker): m_i += 1
            ax2.plot(time,cumu_delay,c=args.color[c_i], linestyle=line_style[i], label='%s delay' % labels[lab_i], marker=args.marker[m_i],markeredgecolor=args.color[c_i],markerfacecolor=mfc[c_i],markevery=mevery[m_i])
            #ax.plot(time,[10 * cumu_delay[i]/q if q != 0 else 0 for i,q in enumerate(cumu_in)],label='norm delay')
            if i+1 < len(line_style): i += 1
            if c_i+1 < len(line_style): c_i += 1
            if m_i+1 < len(args.marker): m_i += 1

        if lab_i+1 < len(labels): lab_i += 1
    if args.x_limit:
        ax.set_xlim(args.x_limit[0], args.x_limit[1])
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Vehicles')
    ax2.set_ylabel('Delay')
    if args.y_limit != None:
        ax.set_ylim(args.y_limit[0], args.y_limit[1])
    if args.y_limit2 != None:
        ax2.set_ylim(args.y_limit2[0], args.y_limit2[1])
    #else:
    #    ax2.set_ylim(0, 25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper left')
    if args.title:
        #pl.title('Arrival Curves and Delay for Queues %s' % ', '.join([str(q) for q in queues]) )
        pl.title(args.title )

def calc_min_travel_time(data,args):
    print data.keys()
    results = data['Out']
    if 'Run' in results:
        results = d['Out']['Run'][0]
    if args.step != None:
        if 'Step' in results:
            if len(data['Out']['Step']) > args.step:
                results = data['Out']['Step'][args.step]
    min_travel_time=0
    total_q_in=0
    if args.queues != None:
        for qs in args.queues:
            delay = 0
            for q in qs:
                delay+=data['Queues'][q]['Q_DELAY']
            q_in = sum(results['q_{%d,in}' % qs[0] ])
            min_travel_time+=q_in*delay
            total_q_in+=q_in
    return min_travel_time,total_q_in

def plot_box_plot(args):
    pl.clf()
    npLots=len(args.files)
    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=False, sharey=False)

    if not isinstance(ax,np.ndarray): ax=[ax]

    if args.figsize:
        fig.set_size_inches(args.figsize[0], args.figsize[1])
    else:
        fig.set_size_inches(15, 7)
    titles=[]

    i=0
    c_i=0
    l_i=0
    lab_i=0
    labels=['Plot 1']
    if args.labels: labels=args.labels

    plot_data_files = read_files(args.files)

    for plot_i,plot in enumerate(plot_data_files):

        plot_data = []
        plot_labels=[]

        for d in plot:



            #titles.append(file['Title'])
            results = d['Out']

            #label =  file['Out']['label']
            if args.run != None:
                runs=args.run
            else:
                runs=[0]
            solve_time=[]
            #if 'Run' in results:
            #    runs = range(len(results['Run']))
            for run in runs:
                if 'Run' in results:
                    results = d['Out']['Run'][run]


                #print 'results',results.keys()

                #total_travel_time = calc_total_travel_time(d,results)
                #min_travel_time,total_q_in = calc_min_travel_time(d,args)
                delay = []
                for q in args.queues:
                    time,cumu_in,cumu_out,cumu_delay = calc_delay(d, results, args=q)
                    in_start=0
                    in_end=1
                    for j in range(len(cumu_in)):
                        if cumu_in[j] > cumu_in[in_end]: in_end=j
                    j=0
                    q_delay=0
                    for sub_q in q:
                        q_delay+=d['Queues'][sub_q]['Q_DELAY']

                    while cumu_in[in_start+1]<1e-6 and in_start<in_end-1: in_start+=1
                    delay+=[cumu_delay[i] for i in range(in_start,in_end)]
                #average = (total_travel_time - min_travel_time) / total_q_in
                #mean = sum(delay)/len(delay)
                #if len(plot_label)==0:
                #    plot_label = '$'+label.split('$')[1]+'$'

                time = results['t']
                total_time = time[-1] - time[0]

                if 'Step' in results:
                    if 'N' in results['Step'][0]:
                        N = results['Step'][0]['N']
                    else:
                        N = len(results['Step'][0]['t'])-1
                    for step in results['Step']:
                        if 'solver_runtime' not in step:
                            solve_time.appened(step['solve_time'])
                        else:
                            solve_time.append(step['solver_runtime'])
                else:
                     N = len(results['t'])-1
                if args.plot_cpu_time:
                    box_data=solve_time
                else:
                    box_data = delay
                Ni = N - 0.25 + plot_i
                #if Ni in plot_data:

#                    plot_data[Ni]=[plot_data[Ni][0]+box_data,0]
 #               else:
                plot_data.append(box_data) #[delay,average]
                plot_labels.append(labels[lab_i])

                if lab_i+1 < len(labels): lab_i += 1

        X = sorted(plot_data)
        Y = plot_data
        props = dict(linestyle=args.linestyle[l_i],color=args.color[c_i],markeredgecolor=args.color[c_i],
                      markerfacecolor=args.color[c_i])
        props2 = dict(color=args.color[c_i],markeredgecolor=args.color[c_i],
                      markerfacecolor=args.color[c_i])
        props3 = dict(markeredgecolor=args.color[c_i],
                      markerfacecolor=args.color[c_i])
        #print boxprops,c_i,len(args.color)
        ax[0].boxplot(Y, vert=True, showmeans=True, widths=0.5, whis=1e99) #, boxprops=props, whiskerprops=props3, capprops=props,
                      #medianprops=props, meanprops=props3)#,flierprops=props3,positions=X)
        #Y = [plot_data[x][1] for x in X]
        #ax.plot(X,Y,c=args.color[c_i], label=plot_label,marker='x')


        if args.x_limit:
            ax[0].set_xlim(args.x_limit[0], args.x_limit[1])
        if args.y_limit:
            ax[0].set_ylim(args.y_limit[0], args.y_limit[1])
        ax[0].grid()
        if args.plot_cpu_time:
            ax[0].set_ylabel('Solve Time')
        else:
            ax[0].set_ylabel('Delay')
        ax[0].set_xticklabels(plot_labels)
        if args.title:
            ax[0].set_title(args.title)
        if i+1 < len(args.linestyle): i += 1
        if c_i+1 < len(args.color): c_i += 1
        if l_i+1 < len(args.linestyle): l_i += 1


    #ax.set_ylabel('Total Travel Time')
    #ax.legend(loc='best')
    #if args.title:
    #    pl.title(args.title)
    #else:
    #    pl.title('Box Plot for Delay' )

def open_data_file(file):
    debug("opening data file:" + file)
    if os.path.isfile(file):
        if zipfile.is_zipfile(file):
            path,file_json = os.path.split(os.path.splitext(file)[0]+".json")
            debug('Loading zip file:' + file + '...')
            zf = zipfile.ZipFile(file)
            debug( 'Done.' )
            debug('   Decoding json file:'+file_json + '...')
            data = json.loads(zf.read(file_json))
            debug( 'Done.' )
        else:
            debug( 'Loading file:' + file + '...' )
            f = open(str(file),'r')
            debug('Done.')
            debug('   Decoding json file:' + file + '...')
            data  = json.load(f)
            debug('Done.')
            f.close()
    else:
        data = None
    return data

def read_files(plot_files):
    debug(str(plot_files))
    plots=[]
    for plot_file in plot_files:
        files = []

        path,name = os.path.split(plot_file)
        type = name.split('.')[-1]
        if type == 'json' or type == 'zip':
            files.append(plot_file)
        else:
            if len(path) > 0: path += '/'
            debug('Opening plot file: '+name)
            f = open(str(plot_file),'r')
            for line in f:
                fields = line.split(' ')
                if len(fields) > 0 and len(fields[0].strip()) > 0:
                    files.append(path+fields[0].strip())
                    if len(fields)>1:
                        labels.append(fields[1])
            f.close()
        data = []
        debug('  Opening files in plot file: '+str(files))
        for file in files:
            data.append(open_data_file(file))
            #if os.path.isfile(file):
                # if zipfile.is_zipfile(file):
                #     path,file_json = os.path.split(os.path.splitext(file)[0]+".json")
                #     #print 'Loading zip file:',file,'...',
                #     zf = zipfile.ZipFile(file)
                #     #print 'Done.'
                #     #print '   Decoding json file:',file_json,'...',
                #     data.append(json.loads(zf.read(file_json)))
                #     #print 'Done.'
                # else:
                #     #print 'Loading file:',file,'...',
                #     f = open(str(file),'r')
                #     #print 'Done.'
                #     #print '   Decoding json file:',file,'...',
                #     data.append(json.load(f))
                #     #print 'Done.'
                #     f.close()
        plots.append(data)
    return plots

def plot_av_travel_time(args):
    pl.clf()
    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    if args.figsize:
        fig.set_size_inches(args.figsize[0], args.figsize[1])
    else:
        fig.set_size_inches(15, 7)
    titles=[]
    i=0
    c_i=0

    l_i=0
    labels=[]
    if args.labels: labels=args.labels
    min_TT = 1e9
    plot_data = []
    plot_scatter_X = []
    plot_scatter_Y = []
    plot_data_mf = []
    plot_data_results = []

    # for file in args.files:
    #     files = []
    #
    #     path,name = os.path.split(file)
    #     if len(path) > 0: path += '/'
    #
    #     f = open(str(file),'r')
    #     for line in f:
    #         fields = line.split(' ')
    #         if len(fields) > 0:
    #             files.append(path+fields[0].strip())
    #             if len(fields)>1:
    #                 labels.append(fields[1])
    #     f.close()
    #     for file in files:
    #         if os.path.isfile(file):
    #             f = open(str(file),'r')
    #             data.append(json.load(f))
    #             f.close()

    plot_data_files = read_files(args.files)
    if args.plot_travel_time_ref != None:
        ref_file = open_data_file(args.plot_travel_time_ref)
    else:
        ref_file = None
    #print len(plot_data_files)
    for data in plot_data_files:
        #print len(data)

        plot_data.append( dict())
        plot_scatter_Y.append([])
        plot_scatter_X.append([])
        plot_data_mf.append( dict())
        plot_label=''
        if len(labels)>0: plot_label = labels[l_i]
        annotate=False
        if i < len(args.annotate) and args.annotate[i]=='y':
            annotate=True

        for d in data:
            #print d.keys()
            titles.append(d['Title'])
            results = d['Out']
            #print d.keys()
            #print results.keys()
            runs=1
            if 'Run' in results:
                runs = len(results['Run'])
                d_out = results['Run'][0]
                label =  d_out
            else:
                d_out = results

            #print 'Runs=',runs
            #print d_out.keys()
            if args.plot_travel_time_DT:
                N = int(1/d_out['DT'][0])
                #N = results['DT'][0]

            elif 'Step' in d_out:
                if 'N' in d_out['Step'][0]:
                    N = d_out['Step'][0]['N']
                else:
                    N = len(d_out['Step'][0]['t'])-1
                #print 'N=',N
            else:
                if 'N' in d_out:
                    N=d_out['N']
                else:
                    N = len(d_out['t'])-1
            #if args.step != None:
            #    if 'Step' in results:
            #        results = d['Out']['Step'][args.step]
            #        label = results['label']


            solve_time=0
            total_travel_time=0
            for run in range(runs):
                if 'Run' in d['Out']:
                    #print 'run=',run
                    results = d['Out']['Run'][run]
                #print results.keys()
                #total_travel_time = calc_total_travel_time(d,results)
                total_travel_time += results['total_travel_time']
                solve_time=0
                if args.plot_cpu_time:
                    if 'Step' in results:
                        for step in results['Step']:
                            if 'solver_runtime' not in step:
                                solve_time+=step['solve_time']
                            else:
                                solve_time+=step['solver_runtime']
                            #print step['solver_runtime']
                            if args.plot_cpu_time:
                                if 'solver_runtime' not in step:
                                    plot_scatter_X[-1].append(step['solve_time'])
                                else:
                                    plot_scatter_X[-1].append(step['solver_runtime'])
                                plot_scatter_Y[-1].append(results['total_travel_time'])
                        #print len(results['Step'])

                        solve_time /= len(results['Step'])
                        #print 'solver_runtime=',solve_time
                    else:
                        solve_time = d['Out']['solver_runtime']/runs

                #ax.plot(N,results['total_travel_time'],'x',c=args.color[c_i])
                if not args.plot_cpu_time:
                    plot_scatter_Y[-1].append(results['total_travel_time'])
                    plot_scatter_X[-1].append(N)
                #print results['total_travel_time']
                if len(plot_label)==0:
                    plot_label = '$'+label.split('$')[1]+'$'

                #time = results['t']
                #total_time = time[-1] - time[0]
                min_TT = min(results['total_travel_time'],min_TT)

            if args.plot_cpu_time:
                plot_data[-1][solve_time]=total_travel_time/runs
            else:
                plot_data[-1][N]=total_travel_time/runs
            #print total_travel_time/runs
            #print min_TT
            if annotate and 'plan' in d['Out']:
                if args.plot_cpu_time:
                    plot_data_mf[-1][solve_time]=int(d['Out']['plan']['major_frame'])
                else:
                    plot_data_mf[-1][N]=int(d['Out']['plan']['major_frame'])
                #print results['plan']['major_frame']
            else:
                if args.plot_cpu_time:
                    plot_data_mf[-1][solve_time]=0
                else:
                    plot_data_mf[-1][N]=0
        i+=1
        c_i+=1

    if ref_file != None:
        min_TT = ref_file['Out']['total_travel_time']

    i=0
    c_i=0
    m_i=0
    l_i=0
    labels=[]
    if args.labels: labels=args.labels
    for plot_i,plot in enumerate(plot_data):
        if len(labels)>0:
            plot_label = labels[l_i]
        else:
            plot_label = '$'+label.split('$')[1]+'$'
        X = sorted(plot)
        Y = [((plot[x]-min_TT)/min_TT)*100 for x in X]
        ax.plot(X,Y,c=args.color[c_i], linestyle=args.linestyle[l_i], label=plot_label,marker=args.marker[m_i],markerfacecolor='None')
        if args.plot_scatter:
            ax.plot(plot_scatter_X[plot_i],((np.array(plot_scatter_Y[plot_i])-min_TT)/min_TT)*100 ,'.',c=args.color[c_i])
        for j,x in enumerate(sorted(plot_data_mf[i])):
            if plot_data_mf[i][x]>0:
                label_test=True
                if args.x_limit and (X[j] < args.x_limit[0] or X[j] > args.x_limit[1]):
                    label_test=False
                if args.y_limit and (Y[j] < args.y_limit[0] or Y[j] > args.y_limit[1]):
                    label_test=False
                if label_test == True:
                    ax.text(X[j],Y[j],plot_data_mf[i][x],size=8,color=args.color[c_i])

        if l_i+1 < len(args.linestyle): l_i += 1
        if c_i+1 < len(args.color): c_i += 1
        if m_i+1 < len(args.marker): m_i += 1
        i += 1

    if args.annotate_labels is not None and args.annotate_x is not None and args.annotate_y is not None:
        assert len(args.annotate_labels) == len(args.annotate_x) == len(args.annotate_y)
        for i,_ in enumerate(args.annotate_labels):
            if args.annotate_size is not None:
                size = args.annotate_size[i]
            else:
                size = 8
            ax.text(args.annotate_x[i],args.annotate_y[i],args.annotate_labels[i],size=size)

    if args.x_limit:
        ax.set_xlim(args.x_limit[0], args.x_limit[1])
    if args.y_limit:
        ax.set_ylim(args.y_limit[0], args.y_limit[1])
    ax.grid()
    if args.plot_cpu_time:
        ax.set_xlabel('CPU Solve Time')
    else:
        ax.set_xlabel('N (number of time samples)')
    ax.set_ylabel('% increase in total travel time') #ax.set_ylabel('Total Travel Time')
    ax.legend(loc='best')
    if args.title:
        pl.title(args.title)
    #else:
    #    if args.plot_cpu_time:
    #        pl.title('CPU Time vs % increase in total travel time' )
    #    else:
    #        pl.title('Number of time samples vs % increase in total travel time' )



def plot_travel_time(args):
    pl.clf()
    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    if args.figsize:
        fig.set_size_inches(args.figsize[0], args.figsize[1])
    else:
        fig.set_size_inches(15, 7)
    titles=[]
    i=0
    c_i=0

    l_i=0
    labels=[]
    if args.labels: labels=args.labels
    min_TT = 1e9
    plot_data = []
    plot_data_mf = []
    plot_data_results = []
    for file in args.files:
        files = []

        path,name = os.path.split(file)
        if len(path) > 0: path += '/'

        f = open(str(file),'r')
        for line in f:
            fields = line.split(' ')
            if len(fields) > 0:
                files.append(path+fields[0].strip())
                if len(fields)>1:
                    labels.append(fields[1])
        f.close()
        data = []
        for file in files:
            if os.path.isfile(file):
                f = open(str(file),'r')
                data.append(json.load(f))
                f.close()

        plot_data.append( dict())
        plot_data_mf.append( dict())
        plot_label=''
        if len(labels)>0: plot_label = labels[l_i]
        annotate=False
        if i < len(args.annotate) and args.annotate[i]=='y':
            annotate=True

        for d in data:

            titles.append(d['Title'])
            results = d['Out']
            label =  d['Out']['label']
            if args.plot_travel_time_DT:
                N = int(1/results['DT'][0])
                #N = results['DT'][0]

            elif 'Step' in results:

                N = len(d['Out']['Step'][0]['t'])-1
            else:
                N = len(results['t'])-1
            if args.step != None:
                if 'Step' in results:
                    results = d['Out']['Step'][args.step]
                    label = results['label']

            total_travel_time = calc_total_travel_time(d,results)

            if len(plot_label)==0:
                plot_label = '$'+label.split('$')[1]+'$'

            time = results['t']
            total_time = time[-1] - time[0]

            plot_data[-1][N]=total_travel_time

            if annotate and 'plan' in results:
                plot_data_mf[-1][N]=results['plan']['major_frame']
            else:
                plot_data_mf[-1][N]=0
            min_TT = min(total_travel_time,min_TT)
        i+=1
    i=0
    c_i=0

    l_i=0
    labels=[]
    if args.labels: labels=args.labels
    for plot in plot_data:
        if len(labels)>0:
            plot_label = labels[l_i]
        else:
            plot_label = '$'+label.split('$')[1]+'$'
        X = sorted(plot)
        Y = [((plot[x]-min_TT)/min_TT)*100 for x in X]
        ax.plot(X,Y,c=args.color[c_i], linestyle=args.linestyle[l_i], label=plot_label,marker=args.marker)
        for j,x in enumerate(sorted(plot_data_mf[i])):
            if plot_data_mf[i][x]>0:
                label_test=True
                if args.x_limit and (X[j] < args.x_limit[0] or X[j] > args.x_limit[1]):
                    label_test=False
                if args.y_limit and (Y[j] < args.y_limit[0] or Y[j] > args.y_limit[1]):
                    label_test=False
                if label_test == True:
                    ax.text(X[j],Y[j],plot_data_mf[i][x],size=8,color=args.color[c_i])
        if l_i+1 < len(args.linestyle): l_i += 1
        if c_i+1 < len(args.color): c_i += 1
        i += 1

    if args.x_limit:
        ax.set_xlim(args.x_limit[0], args.x_limit[1])
    if args.y_limit:
        ax.set_ylim(args.y_limit[0], args.y_limit[1])
    ax.grid()
    ax.set_xlabel('N (number of time samples)')
    ax.set_ylabel('% increase in total travel time') #ax.set_ylabel('Total Travel Time')
    ax.legend(loc='best')
    if args.title:
        pl.title(args.title)
    else:
        pl.title('Number of time samples vs % increase in total travel time' )



def plot_cpu_time(args):
    pl.clf()
    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    if args.figsize:
        fig.set_size_inches(args.figsize[0], args.figsize[1])
    else:
        fig.set_size_inches(15, 7)
    titles=[]
    i=0
    c_i=0
    l_i=0
    labels=[]
    if args.labels: labels=args.labels
    for file in args.files:
        files = []
        path,name = os.path.split(file)
        if len(path) > 0: path += '/'
        f = open(str(file),'r')
        for line in f:
            fields = line.split(' ')
            if len(fields) > 0:

                files.append(path+fields[0].strip())
                if len(fields)>1:
                    labels.append(fields[1])
        f.close()
        data = []
        for file in files:
            if os.path.isfile(file):
                f = open(str(file),'r')
                data.append(json.load(f))
                f.close()

        plot_data = dict()
        plot_label=''
        if len(labels)>0: plot_label = labels[l_i]
        plot_X=[]
        plot_Y=[]
        for d in data:


            titles.append(d['Title'])
            results = d['Out']
            label =  d['Out']['label']
            if 'Step' in results:
                N = len(d['Out']['Step'][0]['t'])-1
            else:
                N = len(results['t'])-1



            if args.step != None:
                if 'Step' in results:
                    results = d['Out']['Step'][args.step]
                    label = results['label']
            #solve_time=0
            #if 'Step' in results:
            #    for step in results['Step']:
            #        solve_time+=step['solve_time']
            #solve_time/=len(results['Step'])
            total_travel_time = calc_total_travel_time(d,results)
            solve_time = results['solve_time']
            objval=results['objval']
            if len(plot_label)==0:
                plot_label = '$'+label.split('$')[1]+'$'




            time = results['t']
            total_time = time[-1] - time[0]

            plot_data[solve_time]=total_travel_time
            #if args.x_limit != None:
            #    if solve_time>args.x_limit[1]:
            #        break

            plot_X.append(solve_time)
            plot_Y.append(total_travel_time)
            #plot_Y.append(N)

        X = sorted(plot_data)
        Y = [plot_data[x] for x in X]
        #ax.plot(X,Y,c=args.color[c_i], label=plot_label,marker='x')
        ax.plot(plot_X,plot_Y,c=args.color[c_i], linestyle=args.linestyle[i], label=plot_label,marker=args.marker)
        if i+1 < len(args.linestyle): i += 1
        if c_i+1 < len(args.color): c_i += 1
        if l_i+1 < len(labels): l_i += 1

    if args.x_limit:
        ax.set_xlim(args.x_limit[0], args.x_limit[1])
    if args.y_limit:
        ax.set_ylim(args.y_limit[0], args.y_limit[1])
    ax.grid()
    ax.set_xlabel('CPU time')
    ax.set_ylabel('Total travel time')
    ax.legend(loc='best')
    if args.title:
        pl.title(args.title)
    else:
        pl.title('CPU time vs Total Travel Time for Network' )


def plot_phase_offset(args):
    pl.clf()
    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    if args.figsize:
        fig.set_size_inches(args.figsize[0], args.figsize[1])
    else:
        fig.set_size_inches(15, 7)
    titles=[]
    i=0
    c_i=0
    l_i=0
    labels=[]
    if args.labels: labels=args.labels
    for file in args.files:
        files = []
        path,name = os.path.split(file)
        if len(path) > 0: path += '/'
        f = open(str(file),'r')
        for line in f:
            fields = line.split(' ')
            if len(fields) > 0:
                files.append(path+fields[0].strip())
                if len(fields)>1:
                    labels.append(fields[1])

        f.close()
        data = []
        for file in files:
            if os.path.isfile(file):
                f = open(str(file),'r')
                data.append(json.load(f))
                f.close()

        plot_data = dict()
        plot_label=''
        if len(labels)>0: plot_label = labels[l_i]
        plot_X=[]
        plot_Y=[]
        for k,d in enumerate(data):


            titles.append(d['Title'])
            results = d['Out']
            label =  d['Out']['label']
            if 'Step' in results:
                N = len(d['Out']['Step'][0]['t'])-1
            else:
                N = len(results['t'])-1



            if args.step != None:
                if 'Step' in results:
                    results = d['Out']['Step'][args.step]
                    label = results['label']
            #solve_time=0
            #if 'Step' in results:
            #    for step in results['Step']:
            #        solve_time+=step['solve_time']
            #solve_time/=len(results['Step'])
            total_travel_time = calc_total_travel_time(d,results)
            #solve_time = results['solve_time']
            #objval=results['objval']

            if 'DT_offset' in results:

                if args.step == None and 'Step' in results:
                    phase = d['Out']['Step'][0]['DT_offset']

                else:
                    phase = results['DT_offset']

            else:

                phase = 0.1*k

            if len(plot_label)==0:
                plot_label = '$'+label.split('$')[1]+'$'




            time = results['t']
            total_time = time[-1] - time[0]

            #plot_data[solve_time]=total_travel_time
            #if args.x_limit != None:
            #    if solve_time>args.x_limit[1]:
            #        break

            plot_X.append(phase)
            plot_Y.append(total_travel_time)
            #plot_Y.append(N)

        #X = sorted(plot_data)
        #Y = [plot_data[x] for x in X]
        #ax.plot(X,Y,c=args.color[c_i], label=plot_label,marker='x')
        ax.plot(plot_X,plot_Y,c=args.color[c_i], linestyle=args.linestyle[i], label=plot_label,marker=args.marker)
        if i+1 < len(args.linestyle): i += 1
        if c_i+1 < len(args.color): c_i += 1
        if l_i+1 < len(labels): l_i += 1

    if args.x_limit:
        ax.set_xlim(args.x_limit[0], args.x_limit[1])
    if args.y_limit:
        ax.set_ylim(args.y_limit[0], args.y_limit[1])
    ax.grid()
    ax.set_xlabel('Phase offset (s)')
    ax.set_ylabel('Total travel time')
    ax.legend(loc='best')
    if args.title:
        pl.title(args.title)
    else:
        pl.title('Phase Offset vs Total Travel Time for Network' )

def plot_vars(args): #data_sets,params,colors,line_styles,steps):
    pl.clf()
    fig, axes = pl.subplots(nrows=len(args.plot_vars), ncols=1, sharex=True, sharey=False)
    fig.set_size_inches(15, 4*len(args.plot_vars))
    ax=axes
    if not isinstance(ax, np.ndarray): ax=[axes]

    labels=[]
    data_sets = []
    files=[]
    if args.labels: labels=args.labels

    for file in args.files:
        data_sets.append(open_data_file(file))

    for i,var in enumerate(args.plot_vars):
        k_step=0
        c_i=0
        l_i=0
        xl_i=0
        yl_i=0
        ls_i=0
        m_i=0
        for j,data in enumerate(data_sets):
            results = data['Out']

            if 'Run' in results:
                if args.run != None and len(args.run)>j:
                    results = results['Run'][args.run[j]]
                else:
                    results = results['Run'][0]
            if args.steps != None and len(args.steps)>j:
                if 'Step' in results:
                    results = results['Step'][args.steps[j]]
            ls=args.linestyle[ls_i]
            col=args.color[c_i]
            marker = args.marker[m_i]
            label = args.labels[l_i]
            xlabel=''
            ylabel=''
            if args.x_label is not None:
                xlabel = args.x_label[xl_i]
            if args.y_label is not None:
                ylabel = args.y_label[yl_i]
            t=results['t']

            if var[0:2]=='l_':
                l = int((var.split('_'))[1])

                if l < len(data['Lights']):
                    num_p = len(data['Lights'][l]['P_MAX'])
                    N=len(t)
                    p = [ [ results['d_{%d,%d}' % (l,k)][0] ]*N for k in range(num_p)]

                    for k in range(num_p):
                        P_MAX=data['Lights'][l]['P_MAX'][k]
                        P_MIN=data['Lights'][l]['P_MIN'][k]
                        for n,pk in enumerate(results['p_{%d,%d}' % (l,k)]):
                            if n==0 and p[k][n]==0: p[k][n]=P_MIN
                            if pk<1e-6:
                                p[k][n]=results['d_{%d,%d}' % (l,k)][n]

                            else:
                                if n>0: p[k][n]=p[k][n-1]

                    c = [ sum([p[k][n] for k in range(num_p) ]) for n in range(N)]

                    pl_p = [ [ 0 ]*N for k in range(num_p+1)]
                    if k>2:
                        pl_p[0] = [-c[n]/2 for n in range(N)]

                        ax[i].plot(t,[x+j*P_MAX*num_p+P_MAX for x in pl_p[0]],  marker=marker, linestyle = ls, color=col)
                        for k in range(1,num_p+1):
                            pl_p[k] = [pl_p[k-1][n]+p[k-1][n] for n in range(N)]
                            ax[i].plot(t,[x+j*P_MAX*num_p+P_MAX for x in pl_p[k]], marker=marker, linestyle = ls, color=col)
                    else:
                        ax[i].plot(t,[-x+j*P_MAX*2+P_MAX for x in p[0]],  marker=marker,  linestyle = ls, color=col)
                        ax[i].plot(t,[x+j*P_MAX*2+P_MAX for x in p[1]],  marker=marker, linestyle = ls, color=col)
                        ax[i].plot(t,[j*P_MAX*2+P_MAX for x in p[1]],  marker=marker, linestyle = ls, color=col)


            elif var[0:2]=='p_' and 1==2:

                l = int(((var[3:-1].split('_'))[0]).split(',')[0])
                k = int(((var[3:-1].split('_'))[0]).split(',')[1])
                if l < len(data['Lights']):
                    num_p = len(data['Lights'][l]['P_MAX'])
                    N=len(t)
                    p = [ results['d_{%d,%d}' % (l,k)][0] ]*N

                    P_MIN=data['Lights'][l]['P_MIN'][k]
                    for n,pk in enumerate(results['p_{%d,%d}' % (l,k)]):
                        if n==0 and p[n]==0: p[n]=P_MIN
                        if pk<1e-6:
                            p[n]=results['d_{%d,%d}' % (l,k)][n]

                        else:
                            if n>0: p[n]=p[n-1]

                ax[i].plot(t,p, label=results['label'], marker=marker, linestyle = ls, color=col)
                ax[i].set_ylim(-1,data['Lights'][l]['P_MAX'][k]+1 )
            elif var in results.keys():
                #if var[0] == 'l':
                if var[0:3]=='q_{':
                    DT = results['DT']
                    #ax[i].plot(t,[results[var][0]]+[results[var][x]/DT[x-1] for x in range(1,len(t))],marker=marker, label=label, linestyle = ls, color=col) #/DT[x_i]
                    ax[i].plot(t,[results[var][0]]+[results[var][x] for x in range(1,len(t))],marker=marker, label=label, linestyle = ls, color=col)
                else:
                    ax[i].plot(t,[results[var][x] for x in range(len(t))],marker=marker, label=label, linestyle = ls, color=col)
                #else:
                #    ax[i].plot(t,results[var], color=colors[j])

            if ylabel == ' ':
                ax[i].set_ylabel(r'$%s_{%s}$' % tuple(var.split('_')),fontsize=16)
            else:
                ax[i].set_ylabel(ylabel)
            if xlabel == ' ':
                ax[i].set_xlabel('time')
            else:
                ax[i].set_xlabel(xlabel)
            if ls_i+1 < len(args.linestyle): ls_i += 1
            if c_i+1 < len(args.color): c_i += 1
            if l_i+1 < len(labels): l_i += 1
            if xl_i+1 < len(args.x_label): xl_i += 1
            if yl_i+1 < len(args.y_label): yl_i += 1
            if m_i+1 < len(args.marker): m_i += 1
            if var in results.keys():
                if var[0] == 'q' or var[0] == 'f':
                    if var[2] != '{':
                        #ax[i].set_ylim(-1,data['Queues'][int(var[3:].split(',')[0])]['Q_MAX']+2)
                    #else:
                        ax[i].set_ylim(-1,data['Queues'][int(var[2:])]['Q_MAX']+2)
                elif var[0:2] == 'dq':
                    if var[3] == '{':
                        ax[i].set_ylim(-1,data['Queues'][int(var[4:].split(',')[0])]['Q_MAX']+2)
                    else:
                        ax[i].set_ylim(-1,data['Queues'][int(var[3:])]['Q_MAX']+2)
                elif var[0] == 'd' and var[1] != 'q':
                    lp = var[3:-1].split(',')
                    ax[i].set_ylim(-1,data['Lights'][int(lp[0])]['P_MAX'][int(lp[1])]+2)
                elif var[0] == 'l':
                    l = int(var[2:])
                    ax[i].set_ylim(-1,len(data['Lights'][l]['P_MAX']) )
                elif var[0] == 'p':
                    None
                #else:
                #    ax[i].set_ylim(0,1)
            ax[i].legend(loc='best')
            if args.x_limit != None:
                ax[i].set_xlim(args.x_limit[0],args.x_limit[1])
            ax[i].grid(True)
    if args.title:
        pl.title(args.title)


def dump_stats(args):

    data_files = read_files(args.files)
    data = []
    for i,file in enumerate(data_files):
        data.append(file)

    # for file in args.files:
    #     if file.endswith('.json'):
    #         files.append(file)
    #     else:
    #         path,name = os.path.split(file)
    #         if len(path) > 0: path += '/'
    #         f = open(str(file),'r')
    #         for line in f:
    #             fields = line.split(' ')
    #             if len(fields) > 0:
    #
    #                 files.append(path+fields[0].strip())
    #
    #         f.close()
    #     data = []
    #     loaded_files=[]
    #     for file in files:
    #         if os.path.isfile(file):
    #             loaded_files.append(os.path.split(file)[1])
    #             f = open(str(file),'r')
    #             data.append(json.load(f))
    #             f.close()

    # for file in args.files:
    #     if file.endswith('.json'):
    #         files.append(file)
    #     else:
    #         f = open(str(file),'r')
    #         for line in f:
    #             fields = line.split(' ')
    #             if len(fields) > 2:
    #                 files.append(fields[2])
    #             else:
    #                 files.append(fields[0].strip())
    #         f.close()
    # loaded_files=[]
    # for file in files:
    #     if os.path.isfile(file):
    #         loaded_files.append(file)
    #         f = open(str(file),'r')
    #         data.append(json.load(f))
    #         f.close()



    for d in data:
        if 'Run' in d['Out']:
            for run in d['Out']['Run']:
                min_travel_time,total_q_in = calc_min_travel_time({'Out':run},args)
        else:
            min_travel_time,total_q_in = calc_min_travel_time(d,args)
        if args.debug: print 'Min Travel Time for network: %f  Total volume of traffic through network: %f' % (min_travel_time,total_q_in)

    print 'File \t\t\t   Run time   Travel time   Delay   Av delay   Stops   N   Accu       Status        Obj  Objval'

    for i,d in enumerate(data):
        if 'Run' in d['Out']:
            results = d['Out']['Run'][0]
        else:
            results = d['Out']
        label = results['label']

        if args.step != None:
            if 'Step' in results:
                if len(results['Step']) > args.step:
                    results = d['Out']['Step'][args.step]
                    label = results['label']
        if 'status' in results:
            status = results['status']
        else:
            status = 'unknown'
        if 'Step' in results:
            N = len(results['Step'][0]['t'])-1
            if 'status' in results['Step'][0]:
                status=''
                for step in results['Step']:
                    status += step['status'][0]
        else:
            N = len(results['t'])-1
        total_travel_time = calc_total_travel_time(d,results)
        total_stops = calc_total_stops(d,results)
        min_travel_time,total_q_in = calc_min_travel_time(d,args)
        accuracy = results['accuracy']
        if 'obj' in results:
            obj = results['obj']
        else:
            obj = ''
        objval = results['objval']
        print '%25s %9.1f %13.1f %7.1f %10.1f %7.1f %4d %5.2f %12s %10s %7.1f' % (loaded_files[i],results['solve_time'],total_travel_time,
                                                                  total_travel_time-min_travel_time,
                                                                  (total_travel_time-min_travel_time) / total_q_in,
                                                                  total_stops,N,accuracy,status,obj,objval)

def dump_vars(args): #data_sets,params,colors,line_styles,steps):

    data_sets = []
    files=[]
    if args.labels: labels=args.labels

    for file in args.files:
        if file.endswith('.json'):
            files.append(file)
        else:
            f = open(str(file),'r')
            for line in f:
                fields = line.split(' ')
                if len(fields) > 2:
                    files.append(fields[2])
            f.close()
    loaded_files=[]
    for file in files:
        if os.path.isfile(file):
            loaded_files.append(file)
            f = open(str(file),'r')
            data_sets.append(json.load(f))
            f.close()


    pd.options.display.float_format = '{:20,.2f}'.format
    for j,data in enumerate(data_sets):

        results = data['Out']
        if args.step != None:
            if 'Step' in results:
                results = data['Out']['Step'][args.step]
        t=results['t']
        #for i,var in enumerate(args.dump_vars):
        #        if var in results.keys(): print var,'=',results[var]
        #for k in range(len(t)):
        #    for i,var in enumerate(args.dump_vars):
        #        if var in results.keys(): print '%5.2f' % results[var][k],
        #    print
        table = np.zeros((len(args.dump_vars),len(t)))
        for i,var in enumerate(args.dump_vars):
            if var in results.keys():
                var_data = results[var]
                if var == 'DT':  var_data = var_data[:-1]
                table[i:] = var_data
        #pd.set_option('display.width', 2000)
        pd.set_option('display.max_columns', 500)
        display(pd.DataFrame(table,index=args.dump_vars))

def dump_convergence_stats( args):
    files = [filename for filename in os.listdir('.') if filename.startswith(args.files[0])]
    print files
    data=[]
    for file in files:
        f = open(str(file),'r')
        data.append(json.load(f))
        f.close()
        print '''d['Out']['label'],d['Out']['solve_time'],travel_time,d['Out']['objval'] '''
    for d in data:
        travel_time = calc_total_travel_time(d,d['Out'])
        if 'Out' in d:
                print d['Out']['label'],d['Out']['solve_time'],travel_time,d['Out']['objval']


DEBUG = False

def debug(msg=""):
    """
    Display debug messege
    :param msg: The message to display
    :return:
    """
    if DEBUG:
        print 'DEBUG: '+msg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="the model file to plot", nargs='+')
    parser.add_argument("-q", "--queues", help="list of queues to plot the delay over", action='append', nargs='+', type=int)
    parser.add_argument("--t0", help="start point along queue of plot",type=float,default=0)
    parser.add_argument("--t1", help="end point along queue of plot",type=float,default=1)
    parser.add_argument("--linestyle", help="list of strings of matplotlib line styles for each plot",nargs='*', default=['_','_','_','_'])
    parser.add_argument("-c", "--color", help="list of matplotlib colors for each plot", default='rbgky')
    parser.add_argument("-o", "--out", help="save the plot as OUT")
    parser.add_argument("-n", help="number of points to plot", type=int, default=0)
    parser.add_argument("--histogram", help="plot a histogram of delay with HISTOGRAM wide bins", type=float)
    parser.add_argument("--step", help="Use output step STEP",type=int)
    parser.add_argument("--run", help="Use output run RUN",nargs='*', type=int)
    parser.add_argument("--steps", help="List of steps to use for each plot", nargs='*', type=int)
    parser.add_argument("--dump_stats", help="Dump stats from results files",action="store_true", default=False)
    parser.add_argument("--dump_convergence_stats", help="Dump stats from a set results files starting with FILE pattern",action="store_true", default=False)
    parser.add_argument("--delay_lt", help="plot delay less than DELAY_LT", type=float)
    parser.add_argument("--delay_gt", help="plot delay greater than DELAY_GT", type=float)
    parser.add_argument("--delay_gt_plots", help="list of number of results to group in each of plots delay_gt plots",  nargs='+', type=int)
    parser.add_argument("--plot_travel_time", help="Plots travel time for each file",action="store_true", default=False)
    parser.add_argument("--plot_travel_time_ref", help="file for travel time reference ")
    parser.add_argument("--plot_av_travel_time", help="Plots average travel time for each file",action="store_true", default=False)
    parser.add_argument("--plot_travel_time_DT", help="Plots travel time vs DT for each file",action="store_true", default=False)
    parser.add_argument("--plot_cpu_time", help="Plots CPU time vs travel time for each file",action="store_true", default=False)
    parser.add_argument("--plot_av_cpu_time", help="Plots average CPU time vs travel time for each file",action="store_true", default=False)
    parser.add_argument("--plot_scatter", help="Plots each minor frame or run as a point in a scatter plot",action="store_true", default=False)
    parser.add_argument("--plot_phase_offset", help="Plots phase offset vs travel time for each file",action="store_true", default=False)
    parser.add_argument("--plot_box_plot", help="Plots a box plot for each file",action="store_true", default=False)
    parser.add_argument("--plot_vars", help="Plots each var in list",nargs='*')
    parser.add_argument("--plot_network", help="Plots a network as a link and node graph",action="store_true", default=False)
    parser.add_argument("--plot_network_delay", help="Plots a network as a coloured link and node graph", default='')
    parser.add_argument("--colormap", help="the color map name to use", default='green_red')
    parser.add_argument("--gamma", type=float, help="the gamma of the color map", default=1.0)
    parser.add_argument("--x_limit", help="set the x limit of the plot",nargs='+', type=float)
    parser.add_argument("--y_limit", help="set the y limit of the plot",nargs='+', type=float)
    parser.add_argument("--y_limit2", help="set the y limit of the plot 2nd axis",nargs='+', type=float)
    parser.add_argument("--title", help="set the title of the plot")
    parser.add_argument("--labels", help="list of labels to use for each plot", nargs='+')
    parser.add_argument("--x_label", help="set the x axis label of the plot",nargs='+', default=' ')
    parser.add_argument("--y_label", help="set the y axis label of the plot",nargs='+', default=' ')
    parser.add_argument("--figsize", help="width and height of the plot", nargs='+',type=float)
    parser.add_argument("--marker", help="line maker for the plot",default=' ')
    parser.add_argument("--markerfill", help="List of y|n whether to fill maker",default='n')
    parser.add_argument("--markevery", help="maker every MARKEVERY points",nargs='+',type=int,default=[1])
    parser.add_argument("--debug", help="output debug messages", action="store_true", default=False)
    parser.add_argument("--dump_vars", help="Dump raw data for each var in the list",nargs='*')
    parser.add_argument("--annotate", help="List of y|n whether to annotate the points of each plot with the major frame time", default='n')
    parser.add_argument("--annotate_x", help="List of x points to plot each label in annotate_labels", nargs='*', type=float)
    parser.add_argument("--annotate_y", help="List of y points to plot each label in annotate_labels", nargs='*', type=float)
    parser.add_argument("--annotate_labels", help="List of labels to annotate on plot", nargs='*')
    parser.add_argument("--annotate_size", help="List of label font sizes for each label in annotate_label", nargs='*', type=int)

    args = parser.parse_args()

    linestyle_map = { '_': '-', '_ _': '--', '_.' : '-.'}

    if args.debug: DEBUG = True

    for i,style in enumerate(args.linestyle):
        if style in linestyle_map:
            args.linestyle[i]=linestyle_map[style]


    if args.plot_vars != None:
        plot_vars(args)
    elif args.plot_travel_time or args.plot_travel_time_DT:
        plot_travel_time(args)
    elif args.plot_av_travel_time:
        plot_av_travel_time(args)
    elif args.plot_phase_offset:
        plot_phase_offset(args)

    elif args.plot_box_plot:
        plot_box_plot(args)

    elif args.plot_cpu_time:
        plot_cpu_time(args)

    elif args.plot_network:
        plot_network(args)
    elif args.plot_network_delay != '':
        plot_network_delay(args)
    elif args.dump_stats:
            dump_stats(args)
    elif args.dump_vars != None:
        dump_vars(args)
    else:
        #files=[]
        #data=[]
        #for file in args.files:
        #    files.append(str(file))
        #    f = open(str(file),'r')
        #    data.append(json.load(f))
        #    f.close()

        plot_data_files = read_files(args.files)
        if args.histogram:
            if args.delay_gt:
                plot_delay_gt(data,args.step,args.histogram,args.delay_gt,args.queues,args.delay_gt_plots,args.linestyle)
            else:
                plot_delay_histogram(data,args.step,args.histogram,args.queues,args.linestyle,args=args)
        elif args.plot_travel_time:
            plot_tavel_time(data,args.step,args.delay_gt,args.queues,args.plot_travel_time,args.linestyle)
        else:
            plot_delay(plot_data_files[0],args.step,args.queues,args.linestyle,args)

    if args.out:
        pl.savefig(args.out, bbox_inches='tight')
    pl.show()






