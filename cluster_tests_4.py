# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 08:00:35 2021

@author: patri
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

def create_data(nobs, vert_scale, horz_scale):

    t=np.linspace(0,2*np.pi,nobs)
    
    g1_x=np.array(horz_scale*np.cos(t)*np.random.uniform(.5,1,len(t)))
    g1_y=np.array(vert_scale*np.sin(t)*np.random.uniform(.5,1,len(t)))
    
    g2_x=np.array(horz_scale*np.cos(t)*np.random.uniform(0,.4,len(t)))
    g2_y=np.array(vert_scale*np.sin(t)*np.random.uniform(0,.4,len(t)))
    
    g3_x=np.array(np.cos(t)*np.random.uniform(0,.6,len(t))-2)
    g3_y=np.array(np.sin(t)*np.random.uniform(0,.8,len(t))-1)
    
    g4_x=np.array(np.cos(t*4)*.5+2+np.random.uniform(-.3,.3,len(t)))
    g4_y=np.array(t/(4*np.pi)*8-2)
    
    x_data=g1_x.tolist()+g2_x.tolist()+g3_x.tolist()+g4_x.tolist()
    y_data=g1_y.tolist()+g2_y.tolist()+g3_y.tolist()+g4_y.tolist()
    data=np.array([x_data,y_data])   
    
    fig=plt.figure(dpi=600)
    ax=fig.add_subplot(111)
    ax.set_aspect('equal')
    
    plt.plot(g1_x,g1_y,'-b',linestyle='',marker='o',markersize=1,label='group 1')
    plt.plot(g2_x,g2_y,'-g',linestyle='',marker='o',markersize=1,label='group 2')
    plt.plot(g3_x,g3_y,'-y',linestyle='',marker='o',markersize=1,label='group 3')
    plt.plot(g4_x,g4_y,'-r',linestyle='',marker='o',markersize=1,label='group 4')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    
    plt.legend()
    plt.title("plotting individual groups")
    
    plt.show()
    
    fig=plt.figure(dpi=600)
    ax=fig.add_subplot(111)
    ax.set_aspect('equal')
    
    plt.plot(data[0,:],data[1,:],'-b',linestyle='',marker='o',markersize=1,label='group 1')
    
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    
    plt.legend()
    plt.title("plotting combined data")
    
    plt.show()
 
    return data

def create_blob(nobs, a,b,sigma):
    x=np.linspace(-5,3.5,nobs)
    Ey1=a*x+b
    e=np.random.normal(loc=0,scale=sigma,size=nobs)
    y1=Ey1+e
    
    Ey2=(1*a)*x+(-10)
    y2=Ey2+e
    
    outx=x.tolist()+x.tolist()
    outy=y1.tolist()+y2.tolist()

    data=np.array([outx,outy])
    return data

def kmeans_fit(data,k):
    #try kmeans clustering
    from sklearn.cluster import KMeans
    
    model=KMeans(n_clusters=k)
    
    model.fit(data.T)
    
    fig=plt.figure(dpi=600)
    ax=fig.add_subplot(111)
    ax.set_aspect('equal')
    
    plt.scatter(data[0,:],data[1,:],marker='.',s=2,c=model.labels_)
    
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    
    plt.title("kmeans fit")
    
    plt.show()

#Kmeans clearly fails...
#now try spectral clustering
#First step: identify distances

def euclid_dist(source,target_list,thresh):
    targs=[]
    dists=[]
    targ_id=0
    for row in target_list:
        dist=np.sqrt((source[0]-row[0])**2+(source[1]-row[1])**2)
        if dist<=thresh:
            targs.append(targ_id)
            dists.append(dist)
        targ_id=targ_id+1
    return [targs,dists]

def identify_nearst_neighbors(data_list,threshold):
   
    sources=[]
    targets=[]
    distances=[]
    
    for i in range(data_list.shape[1]):
        print("source id=%d " %i)
        matched_target=euclid_dist(data_list[:,i],data_list.T,threshold)
        sources.extend([i]*len(matched_target[1]))
        targets.extend(matched_target[0])
        distances.extend(matched_target[1])
    
    
    dist_dict={'Source':sources,
               'Target':targets,
               'Distances:':distances,
               'Threshold':threshold}
    
    para_dist_df=pd.DataFrame(dist_dict)
    return para_dist_df
#create df

def create_graph_object(data):
    g = nx.Graph()
    for idx in data.index:
        g.add_edge(data['Source'][idx],data['Target'][idx])
    return g

def plot_graph(data):
    plt.figure(figsize=(18,18))
    graph_pos=nx.spring_layout(data)
    nx.draw_networkx_nodes(data,graph_pos, node_size=1, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(data,graph_pos)
#    nx.draw_networkx_labels(data,graph_pos,font_size=3, font_family='san-serif')
    plt.savefig("plot.pdf")            
            

if __name__ == '__main__':
    nobs=100
    data=create_data(nobs,3,1)
    data_dictionary={'ID':[i for i in range(data.shape[1])],
                     'X':list(data[0,:]),
                     'Y':list(data[1,:]),
                     'Group':[0]*nobs+[1]*nobs+[2]*nobs+[3]*nobs
                     }
    
    data_df=pd.DataFrame(data_dictionary)
    
    closeness_df=identify_nearst_neighbors(data,0.25)
    
    print(closeness_df.head(10))
    
    print(data_df.head(10))
    
    closeness_graph=create_graph_object(closeness_df)
    plot_graph(closeness_graph)
    

