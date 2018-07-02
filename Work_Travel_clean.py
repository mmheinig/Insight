##################################################################
##################################################################
#
#                   LIBRARIES
#
##################################################################
##################################################################
import collections
import csv
import datetime as dt
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
import requests
import seaborn as sns
import sys

from collections import Counter
from collections import defaultdict
from community import best_partition
from igraph import *
from itertools import count
from itertools import zip_longest
from networkx.drawing.nx_agraph import graphviz_layout
from neo4j.v1 import GraphDatabase
from operator import itemgetter, attrgetter






##################################################################
##################################################################
#
#                    DEFINITIONS
#
##################################################################
##################################################################


def create_dict(keys, values):
    return dict(zip_longest(keys, values[:len(keys)]))

def combinatorial(lst):
    count = 0
    index = 1
    pairs = []
    for element1 in lst:
        for element2 in lst[index:]:
            pairs.append((element1, element2))
        index += 1
    
    return pairs

def dedupe(data):
    result = Counter()
    for row in data:
        result.update(dict([row]))
    return result.items()

def edgelist_joshwt(data):
    edges = []
    weights = []
    counter = []
    start = int(data['user'].min())
    stop = int(data['user'].max())
    for u in range(start, stop):
        y = data[data['user'] == u]
        y = y.sort_values(by=[node_category])
        x = y[node_category].value_counts()
        z = list(combinatorial(y[node_category]))
        z = remove_selfloops(z)
        for i in range(0,len(z)):
            for item in z[i]:
                counter.append(x[item])
            if counter[0] == counter[1]:
                weights.append(round(counter[0],2))
            else:
                weights.append(round(max(counter)/abs(counter[0]-counter[1]),2))
            counter = []
        edges = edges + z

    edgesandwts = list(zip(edges, weights))
    edgesandwts = sorted(edgesandwts,key=itemgetter(0))
    edgesandwts_dedupe = list(dedupe(edgesandwts)) #gets rid of duplicate edges and adds their weights
    e_pairs, wts = zip(*edgesandwts_dedupe)
    e1, e2 = zip(*e_pairs)
    edgelist = list(zip(e1, e2, wts))

    return edgelist

def edgelist_usrcount(data):
    edges = []
    weights = []
    counter = []
    start = int(data['user'].min())
    stop = int(data['user'].max())
    for u in range(start, stop):
        y = data[data['user'] == u]
        y = y.sort_values(by=[node_category])
        y = y.drop_duplicates(node_category)
        z = list(combinatorial(y[node_category]))
        edges = edges + z

    edges = sorted(edges)   #creates a list of tuples (endnode1, endnode2)
    cntr = collections.Counter(edges)
    e_pairs, wts = cntr.keys(), cntr.values()
    e1, e2 = zip(*e_pairs) #unzips the list of tuples
    edgelist = list(zip(e1, e2, wts))   #(endnode1, endnode2, weight)

    return edgelist

# this is only really necessary if you use node_category = mcc_description
def edgelist_bynumber_forR(edgelst, node_labels):
    e1, e2, w = zip(*edgelst)
    src = replace(e1, node_labels)
    tgt = replace(e2, node_labels)
    wtd_edgelist_byid = list(zip(src, tgt, w))
    return wtd_edgelist_byid

def edgelist_withnames(edgelst, node_labels):
    e1, e2, w = zip(*edgelst)
    src = replace(e1, node_labels)
    tgt = replace(e2, node_labels)
    wtd_edgelist_withnames = list(zip(e1, e2, w, src, tgt))
    return wtd_edgelist_withnames

def mod(communities):
    modularity = {}
    for k,v in communities.items(): # Loop through the community dictionary
        if v not in modularity:
            modularity[v] = [k] # Add a new key for a modularity class the code hasn't seen before
        else:
            modularity[v].append(k)
    
    return modularity

def remove_duplicate(alist):
    return list(set(alist))

# Remove self loops! ie (a,a)
def remove_selfloops(listofpairs):
    new_list = []
    for a, b in listofpairs:
        if (a != b):
            new_list.append((a,b))
    return new_list

def replace(my_list, my_dict):
    return [x if x not in my_dict else my_dict[x] for x in my_list]

#def remove_duplicates(A):
#    [A.pop(count) for count,elem in enumerate(A) if A.count(elem)!=1]
#        return A

##################################################################
##################################################################
#
#                     Inputs
#
##################################################################
##################################################################

# input file
input_file = '/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/DataFromJosh/masked_transactions_with_year_of_birth.csv'
#choose which category to use for nodes of the network (merchant_id, merchant_details, merchant_name, mcc_code, mcc_description)
node_category = 'merchant_name'



##################################################################
##################################################################
#
#                     DATA PREP
#
##################################################################
##################################################################

#read in the data. Note there is already an index column in csv (hence index_col=False)
df = pd.read_csv('masked_transactions_with_year_of_birth.csv', encoding='latin', low_memory=False, index_col=False)

#rename the index column
df = df.rename(columns={'Unnamed: 0': 'index'})


########     CREATE NEW FIELDS        ########
#create new columns for unique identifier counts
df['user'] = df['anonymized_user_id']
df.user = pd.factorize(df.user)[0]
df["age"] = 2018 - df["year_of_birth"]

#clean up 'merchant_name'
df['merchant_name'] = df["merchant_name"].str.replace(r"[\"\',]", '')
df['merchant_name'] = df["merchant_name"].str.replace('(', '')
df['merchant_name'] = df["merchant_name"].str.replace(')', '')
df['merchant_name'] = df["merchant_name"].str.replace('#', '')
df.merchant_name = df.merchant_name.apply(lambda x: x.upper())
df_all.merchant_name.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)  # <-- removes non-ASCII characters
#df["merchant_name_countbyuser"] = df.groupby(["merchant_name", "user"])["user"].transform("count")
df["user_numofpurchases"] = df.groupby(["user"])["index"].transform("count")
                                                      
# clean up mcc_descriptions
df_all.mcc_description.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)  # <-- removes non-ASCII characters
df_all.mcc_details.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)  # <-- removes non-ASCII characters
       
# create a 'day of the week field' to potentially compare weekday activity v. weekend activity (M = 0, T = 1, W = 2, Th = 3, F = 4, Sat = 5, Sun = 6). So weekday < 5, weekend >=5.
df['authorization_timestamp'] = pd.to_datetime(df.authorization_timestamp)
df['date'] = [d.date() for d in df['authorization_timestamp']]
##df['authorization_time'] = [d.time() for d in df['authorization_timestamp']]
df['weekday'] = df['date'].apply( lambda x: x.weekday())
                                                      

# other counters
df["merch_name_count"] = df.groupby(["merchant_name"])["index"].transform("count")
df["merch_id_count"] = df.groupby(["merchant_id"])["index"].transform("count")
df["mcc_code_count"] = df.groupby(["mcc_code"])["index"].transform("count")
df["mcc_desc_count"] = df.groupby(["mcc_description"])["index"].transform("count")
df["mcc"] = df["mcc_code_count"] == df["mcc_desc_count"]
df["merch_match_count"] = df.groupby(["merchant_id", "merchant_name"])["index"].transform("count")
    
                                                      
########     DELETE FIELDS & RECORDS       ########
# remove rows/records based on criteria:
#    1. only want to keep ones with 'transaction_code == 1', so delete others
df = df.drop(df[(df.transaction_code == 0)].index)
df = df.drop(df[(df.transaction_code >= 2)].index)
#    2. one user has no DOB (was one of originial users and that information was never collected)
df = df.drop(df[(df.year_of_birth.isnull())].index)
#    3. remove users with only one purchase
df = df.drop(df[(df.user_numofpurchases == 1)].index)
#    4. Keep Travel purchases --
df = df.drop(df[(df.transaction_code == 0)].index)
#    5. official launch of koho card was March 2017, so remove records before then.
# NO IDEA HOW TO DO THIS --- tried a million times!
                                                      
# transaction_id is the exact same for all rows, so delete it
del df['transaction_id']
del df['anonymized_user_id']
del df['settle_timestamp']
del df['year_of_birth']
                      
                                                      
                                                      
#    ****  This NEEDS to be recalculated here (after deletions) because some of the user's transactional records may have been deleted due to transaction codes!!   ****
df["merchant_name_countbyuser"] = df.groupby(["merchant_name", "user"])["user"].transform("count")
df["merchant_id_countbyuser"] = df.groupby(["merchant_id", "user"])["user"].transform("count")

df_all = df.copy()
########     ORGANIZE THE DATA       ########
# organize the data fields
df_all = df_all[['index', 'user', 'age', 'date', 'weekday', 'amount', 'merchant_id', 'merchant_details', 'merchant_name',  'mcc_code', 'mcc_description', 'koho_category', 'date',  'user_numofpurchases', 'merch_id_count', 'merch_name_count', 'merchant_name_countbyuser', 'merch_match_count', 'mcc_code_count', 'mcc_desc_count', 'mcc', 'transaction_code', 'authorization_timestamp']]
                                                      

# Make different dataframes for weekday activity and weekend activity
df_wday = df[df['weekday'] <5].copy()
df_wknd = df[df['weekday'] >=5].copy()
                                                      
# Save out data frames to csv
df_all.to_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/TRAVEL/df_all.csv')
                               
                                                      
##################################################################
##################################################################
#
#                 Import Dataframe
#
##################################################################
##################################################################
#read in the data. Note there is already an index column in csv (hence index_col=False)
df_all = pd.read_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/TRAVEL/df_travel.csv', encoding='latin', low_memory=False)

df_trav = df_all[df_all['koho_category'] == 'Travel']
df_other = df_all[df_all['koho_category'] == 'Other']
df_travel = pd.concat([df_trav, df_other])

df_travel_category = df_travel[['mcc_code', 'mcc_description', 'koho_category', 'merchant_name']]
df_travel_category.to_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/TRAVEL/travel_category.csv', index=False)
#  --> manually labeled 347 mcc_codes for travel_category

df_travel_category = pd.read_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/TRAVEL/travel_category.csv')
df_travel_category_only = df_travel_category[['mcc_code', 'travel_category']]
df_travel = df_travel.merge(df_travel_category_only, on='mcc_code', how='right')
df_travel = df_travel.drop(df_travel[(df_travel.travel_category == 'Other')].index)
df_travel = df_travel.drop(df_travel[(df_travel.travel_category == 'Mislabeled')].index)
df_travel = df_travel.drop(df_travel[(df_travel.travel_category == 'Transportation')].index)

# save out final travel data set. size = 7,026
df_travel.to_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/TRAVEL/df_travel.csv', index=True)
# unique mcc_codes = 142
# unique mcc_description = 107




###################################
#    Travel ONLY Data
#
# by mcc_description & USER COUNT as edge weight
###################################
edgelist_travel_mcccode_usrcount = edgelist_usrcount(df_travel)
edgelist_travel_mcccode_usrcount.sort(key=operator.itemgetter(2), reverse=True)

#if using mcc_code, this is not necessary
#edgelist_travel_byid = edgelist_bynumber_forR(edgelist_travel)

nodes_travel_mcccode_usrcount = df_travel.copy()
nodes_travel_mcccode_usrcount = nodes_travel_mcccode_usrcount.drop_duplicates(node_category)
nodes_travel_mcccode_usrcount = nodes_travel_mcccode_usrcount.drop(columns = ['merchant_id', 'merchant_details', 'merchant_name', 'index', 'user', 'age', 'date', 'weekday', 'amount', 'user_numofpurchases',  'transaction_code', 'authorization_timestamp'])
# size = 142

# Create a dictionary for nodes & mcc_code (this is for R later on)
node_labels_mcccode_usrcount = create_dict(nodes_travel_mcccode_usrcount.mcc_code, nodes_travel_mcccode_usrcount.mcc_description.values)

edgelist_travel_mcccode_usrcount_withnames = edgelist_withnames(edgelist_travel_mcccode_usrcount, node_labels_usrcount)
edgelist_travel_mcccode_usrcount_withnames_asdf = pd.DataFrame(edgelist_travel_mcccode_usrcount_withnames, columns=['from', 'to', 'weight', 'from_name', 'to_name'])

#write to csv:
edgelist_travel_mcccode_usrcount_withnames_asdf.to_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/TRAVEL/mcc_code/edgelist_travel_mcccode_usrcount_withnames_asdf.csv', index=False)



###################################
#    Travel ONLY Data
#
# by mcc_code & JOSHWT as edge weight
###################################
edgelist_travel_mcccode_joshwt = edgelist_usrcount(df_travel)
edgelist_travel_mcccode_joshwt.sort(key=operator.itemgetter(2), reverse=True)

#if using mcc_code, this is not necessary
#edgelist_travel_byid = edgelist_bynumber_forR(edgelist_travel)

nodes_travel_mcccode_joshwt = df_travel.copy()
nodes_travel_mcccode_joshwt = nodes_travel_mcccode_joshwt.drop_duplicates(node_category)
nodes_travel_mcccode_joshwt = nodes_travel_mcccode_joshwt.drop(columns = ['merchant_id', 'merchant_details', 'merchant_name', 'index', 'user', 'age', 'date', 'weekday', 'amount', 'user_numofpurchases',  'transaction_code', 'authorization_timestamp'])
# size = 139

# Create a dictionary for nodes & mcc_code (this is for R later on)
node_labels_mcccode_joshwt = create_dict(nodes_travel_mcccode_joshwt.mcc_code, nodes_travel_mcccode_joshwt.mcc_description.values)

edgelist_travel_mcccode_joshwt_withnames = edgelist_withnames(edgelist_travel_mcccode_joshwt)
edgelist_travel_mcccode_joshwt_withnames_asdf = pd.DataFrame(edgelist_travel_mcccode_joshwt_withnames, columns=['from', 'to', 'weight', 'from_name', 'to_name'])
#write to csv:
edgelist_travel_mcccode_joshwt_withnames_asdf.to_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/TRAVEL/mcc_code/edgelist_travel_mcccode_joshwt_withnames_asdf.csv', index=False)




##################################################################
##################################################################
#
#               GRAPHING   -- using networkx
#
##################################################################
##################################################################

T = nx.Graph()
T.add_weighted_edges_from(edgelist_travel_mcccode_usrcount)


# Louvain Community Detection  <-- fix this to make isolated nodes their own community?
communities_travel_mcccode_usrcount = best_partition(T)
nx.set_node_attributes(T, communities_travel_mcccode_usrcount, 'modularity')
nodes_travel_mcccode_usrcount['community'] = nodes_travel_mcccode_usrcount[node_category].map(communities_travel_mcccode_usrcount)
modularity_travel_mcccode_usrcount = mod(communities_travel_mcccode_usrcount)  # <-- lets you look at each community, i.e.   cmd:  modularity_travel[#]

# degree
degree_dict_travel_mcccode_usrcount = dict(T.degree(T.nodes()))
nx.set_node_attributes(T, degree_dict_travel_mcccode_usrcount, 'degree')
nodes_travel_mcccode_usrcount['degree'] = nodes_travel_mcccode_usrcount[node_category].map(degree_dict_travel_mcccode_usrcount)
nodes_travel_mcccode_usrcount['degree'].fillna(0, inplace=True)  # < -- to deal with isolated nodes (degree = 0)

# betweenness centrality
btwn_centrality_travel_mcccode_usrcount = nx.betweenness_centrality(T)  # sets node attribute
nodes_travel_mcccode_usrcount['betweenness_centrality'] = nodes_travel_mcccode_usrcount[node_category].map(btwn_centrality_travel_mcccode_usrcount)  #adds attribute to nodes_travel dataframe

# closeness centrality
close_centrality_travel_mcccode_usrcount = nx.closeness_centrality(T)
nodes_travel_mcccode_usrcount['closeness_centrality'] = nodes_travel_mcccode_usrcount[node_category].map(close_centrality_travel_mcccode_usrcount)

# eigenvectory centrality
eigenvector_centrality_travel_mcccode_usrcount = nx.eigenvector_centrality(T)
nodes_travel_mcccode_usrcount['eigenvector_centrality'] = nodes_travel_mcccode_usrcount[node_category].map(eigenvector_centrality_travel_mcccode_usrcount)

# neighbors (list by mcc_code)
codes_travel_mcccode_usrcount = list(degree_dict_travel_mcccode_usrcount.keys())
nbrs_travel_mcccode_usrcount = {}
for c in codes_travel_mcccode_usrcount:
    nbrs_travel_mcccode_usrcount[c] = [n for n in T.neighbors(c)]
nodes_travel_mcccode_usrcount['neighbors'] = nodes_travel_mcccode_usrcount[node_category].map(nbrs_travel_mcccode_usrcount)



##################################################################
#        SAVE OUT INFO/NODE ATTRIBUTES TO CSV FILES
##################################################################
nodes_travel_mcccode_usrcount.to_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/TRAVEL/mcc_code/nodes_travel_mcccode_usrcount.csv')




##################################################################
##################################################################
#
#               GRAPHING   -- using networkx  -- joshwt
#
##################################################################
##################################################################

J = nx.Graph()
J.add_weighted_edges_from(edgelist_travel_mcccode_joshwt)


# Louvain Community Detection  <-- fix this to make isolated nodes their own community?
communities_travel_mcccode_joshwt = best_partition(J)
nx.set_node_attributes(J, communities_travel_mcccode_joshwt, 'modularity')
nodes_travel_mcccode_joshwt['community'] = nodes_travel_mcccode_joshwt[node_category].map(communities_travel_mcccode_joshwt)
modularity_travel_mcccode_joshwt = mod(communities_travel_mcccode_joshwt)  # <-- lets you look at each community, i.e.   cmd:  modularity_travel[#]

# degree
degree_dict_travel_mcccode_joshwt = dict(J.degree(J.nodes()))
nx.set_node_attributes(J, degree_dict_travel_mcccode_joshwt, 'degree')
nodes_travel_mcccode_joshwt['degree'] = nodes_travel_mcccode_joshwt[node_category].map(degree_dict_travel_mcccode_joshwt)
nodes_travel_mcccode_joshwt['degree'].fillna(0, inplace=True)  # < -- to deal with isolated nodes (degree = 0)

# betweenness centrality
btwn_centrality_travel_mcccode_joshwt = nx.betweenness_centrality(J)  # sets node attribute
nodes_travel_mcccode_joshwt['betweenness_centrality'] = nodes_travel_mcccode_joshwt[node_category].map(btwn_centrality_travel_mcccode_joshwt)  #adds attribute to nodes_travel dataframe

# closeness centrality
close_centrality_travel_mcccode_joshwt = nx.closeness_centrality(J)
nodes_travel_mcccode_joshwt['closeness_centrality'] = nodes_travel_mcccode_joshwt[node_category].map(close_centrality_travel_mcccode_joshwt)

# eigenvectory centrality
eigenvector_centrality_travel_mcccode_joshwt = nx.eigenvector_centrality(J)
nodes_travel_mcccode_joshwt['eigenvector_centrality'] = nodes_travel_mcccode_joshwt[node_category].map(eigenvector_centrality_travel_mcccode_joshwt)

# neighbors (list by mcc_code)
codes_travel_mcccode_joshwt = list(degree_dict_travel_mcccode_joshwt.keys())
nbrs_travel_mcccode_joshwt = {}
for c in codes_travel_mcccode_joshwt:
    nbrs_travel_mcccode_joshwt[c] = [n for n in J.neighbors(c)]
nodes_travel_mcccode_joshwt['neighbors'] = nodes_travel_mcccode_joshwt[node_category].map(nbrs_travel_mcccode_joshwt)





##################################################################
##################################################################
#
#               SAVE OUT INFO TO CSV FILES -- joshwt
#
##################################################################
##################################################################
nodes_travel_mcccode_joshwt.to_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/TRAVEL/mcc_code/nodes_travel_mcccode_joshwt.csv')










######################################################################################################
######################################################################################################
#                                                                                                    #
#                                R CODE                                                              #
#                                                                                                    #
######################################################################################################
######################################################################################################
######################################################################################################

######################################################
######################################################
#
#              Libraries
#
######################################################
######################################################
library(R.matlab)
library(readxl)
library(igraph)
library(foreign)
library(readr)
library(networkD3)
library(widgetframe)
library(visNetwork)

nodes_travel_usrcount <- read_csv("~/Desktop/Insight/KOHO_Financial/TRAVEL/mcc_code/nodes_travel_mcccode_usrcount.csv",
                         col_types = cols(Travel_related = col_skip(),
                                          X1 = col_skip(), koho_category = col_skip()))
nodes <- nodes_travel_usrcount[,1:5]
colnames(nodes) <- c("id", "label", "koho", "travel", "group")


edgelist_travel_usrcount_withnames <- read_csv("~/Desktop/Insight/KOHO_Financial/TRAVEL/mcc_code/edgelist_travel_mcccode_usrcount_withnames_asdf.csv")

edges <- edgelist_travel_usrcount[,1:3]


#To color by louvain community:
v1 <- visNetwork(nodes, edges, height = "700px", width = "100%") %>%
    visOptions(selectedBy = "group",
               highlightNearest = TRUE,
               nodesIdSelection = TRUE) %>%
               n
    visPhysics(stabilization = FALSE, maxVelocity=3)
frameWidget(v1)

#to color by Travel type (Hotel, Air, etc.)
v2 <- visNetwork(nodes_travel, wtd_edgelist_travel_usrcount_byid, height = "700px", width = "100%") %>%
    visOptions(selectedBy = "travel",
               highlightNearest = TRUE,
               nodesIdSelection = TRUE) %>%
    visPhysics(stabilization = FALSE, maxVelocity = 5)
frameWidget(v2)

# Simple Graph
p5 <- simpleNetwork(wtd_edgelist_travel_usrcount_byid, nodeColour = "red", zoom=T)
frameWidget(p5)





##################################################################
##################################################################
#
#          COMMUNITY library -- needed for community detection
#
##################################################################
##################################################################
__PASS_MAX = -1
__MIN = 0.0000001

def partition_at_level(dendrogram, level):
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition

def induced_graph(partition, graph, weight="weight"):
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())
    
    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})
    
    return ret

def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
        """
    count = 0
    ret = dictionary.copy()
    new_values = dict([])
    
    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:
            new_values[value] = count
            new_value = count
            count += 1
        ret[key] = new_value
    
    return ret

def __insert(node, com, weight, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
                           status.internals[com] = float(status.internals.get(com, 0.) +
                                                         weight + status.loops.get(node, 0.))


def __modularity(status):
    """
        Fast compute the modularity of the partition of the graph using
        status precomputed
        """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree / links - ((degree / (2. * links)) ** 2)
    return result

def __remove(node, com, weight, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
                           status.internals[com] = float(status.internals.get(com, 0.) -
                                                         weight - status.loops.get(node, 0.))
                           status.node2com[node] = -1

def __neighcom(node, graph, status, weight_key):
    """
        Compute the communities in the neighborhood of node in the graph given
        with the decomposition node2com
        """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

return weights

def __randomly(seq, randomize):
    """ Convert sequence or iterable to an iterable in random order if
        randomize """
    if randomize:
        shuffled = list(seq)
        random.shuffle(shuffled)
        return iter(shuffled)
    return seq

def __modularity(status):
    """
        Fast compute the modularity of the partition of the graph using
        status precomputed
        """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree / links - ((degree / (2. * links)) ** 2)
    return result

def __one_level(graph, status, weight_key, resolution, randomize):
    """Compute one level of communities
        """
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status)
    new_mod = cur_mod
    
    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1
        
        for node in __randomly(graph.nodes(), randomize):
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            neigh_communities = __neighcom(node, graph, status, weight_key)
            remove_cost = - resolution * neigh_communities.get(com_node,0) + \
                (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)
                     best_com = com_node
                     best_increase = 0
                         for com, dnc in __randomly(neigh_communities.items(),
                                                    randomize):
                             incr = remove_cost + resolution * dnc - \
                                 status.degrees.get(com, 0.) * degc_totw
                                     if incr > best_increase:
                                         best_increase = incr
                                             best_com = com
                                                 __insert(node, best_com,
                                                          neigh_communities.get(best_com, 0.), status)
                                                          if best_com != com_node:
                                                              modified = True
                                                          new_mod = __modularity(status)
    if new_mod - cur_mod < __MIN:
        break

def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=False):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")
    
    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for node in graph.nodes():
            part[node] = node
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    __one_level(current_graph, status, weight, resolution, randomize)
    new_mod = __modularity(status)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

while True:
    __one_level(current_graph, status, weight, resolution, randomize)
    new_mod = __modularity(status)
    if new_mod - mod < __MIN:
        break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
    return status_list[:]

def best_partition(graph, partition=None,
                   weight='weight', resolution=1., randomize=False):
    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                randomize)
    return partition_at_level(dendo, len(dendo) - 1)

###

class Status(object):
    """
        To handle several data in one struct.
        
        Could be replaced by named tuple, but don't want to depend on python 2.6
        """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    gdegrees = {}
    
    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])
    
    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))
    
    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight
    
    def init(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                if deg < 0:
                    error = "Bad node degree ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc
