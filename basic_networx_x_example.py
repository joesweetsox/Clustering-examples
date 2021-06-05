# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 08:21:18 2021

@author: patrick
"""

import networkx as nx
from networkx.algorithms import community
G=nx.barbell_graph(5,1)

communities_generator=community.asyn_lpa_communities(G)

top_level_communities=next(communities_generator)

next_level_communities=next(communities_generator)