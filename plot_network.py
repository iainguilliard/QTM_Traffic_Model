from __future__ import division
import matplotlib.pyplot as pl
from matplotlib.patches import Arc,Arrow,Circle
import matplotlib.image as mpimg
import math as math
import json
import sys
import argparse

def plot_network(data,output_file,arg_figsize):
    pl.clf()
    fig, ax = pl.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    fig_size = (10,5)
    line_width = 1
    tail_width = 0
    head_width = 5
    line_color = 'k'
    light_color = 'w'
    text_color = 'k'
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
            head_width = data['Plot']['head_width']
            tail_width = data['Plot']['tail_width']
            line_color = data['Plot']['line_color']
            light_color = data['Plot']['light_color']
            text_color = data['Plot']['text_color']

    if arg_figsize != None:
        fig.set_size_inches(arg_figsize)
    else:
        fig.set_size_inches(fig_size)
    r=15 # radius of intersection nodes
    d=5 # distance to space two edges that share a pair of nodes

    nodes = []
    edges = {}


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
        p = Circle((x,y), r, fc=light_color)
        ax.add_patch(p)
        ax.text(x-3,y-3,r'$l_{%d}$' % i,fontsize=16)

    for i,q in enumerate(data['Queues']):
        pair = edges[q['pair']] > 1
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
            rx1-=rx * (r+line_width); ry1-=ry * (r+line_width)
        elif pair:
            rx1+=ry * d; ry1-=rx * d
        rx0=trx0
        ry0=try0
        rx = rx1-rx0
        ry = ry1-ry0
        lth = math.sqrt(rx*rx+ry*ry)
        tx=rx/lth * r; ty=ry/lth * r
        rx = rx0+(rx1-rx0)/2
        ry = ry0+(ry1-ry0)/2
        ax.text(rx+(ty-7),ry-(tx),r'$q_{%d}$' % i,fontsize=16,color=text_color)
        #plot([rx,rx+ty],[ry,ry-tx])
        arrow = ax.arrow(rx0,ry0,rx1-rx0,ry1-ry0, shape='full', lw=line_width,color=line_color,length_includes_head=True, head_width=head_width, width=tail_width)
        arrow.set_ec('k')
        arrow.set_fc(line_color)

    pl.axis('scaled')
    #ax.set_ylim([x_min-10,x_max+10])
    #ax.set_xlim([y_min-10,y_max+10])
    #[-200,200,-110,110]
    #ax.set_ylim([-110,110])
    #ax.set_xlim([-200,200])
    ax.set_ylim(ext[2:4])
    ax.set_xlim(ext[0:2])
    pl.axis('off')
    if output_file:
        pl.savefig(output_file, bbox_inches='tight')
    pl.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="the model file to animate")
    parser.add_argument("-o", "--out", help="save the plot as OUT")
    parser.add_argument("--figsize", help="width and height of the plot", nargs='+',type=int)
    args = parser.parse_args()
    f = open(str(args.file),'r')
    data = json.load(f)
    f.close()
    plot_network(data,args.out,args.figsize)
