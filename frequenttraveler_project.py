##################################################################
##################################################################
#
#                     Inputs
#
##################################################################
##################################################################

# input file
input_file = 'masked_transactions_with_year_of_birth.csv'
#choose which category to use for nodes of the network (merchant_id, merchant_details, merchant_name, mcc_code, mcc_description)
node_category = 'mcc_code'
# choose weight option (usercount OR joshwt)
weight = 'usercount'




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
    """ creates a dictionary from a list of keys & a list of values
        """
    return dict(zip_longest(keys, values[:len(keys)]))

def combinatorial(lst):
    """ creates every combinatorial pair from items in list.
        It is used to create edges from transactions
        """
    count = 0
    index = 1
    pairs = []
    for element1 in lst:
        for element2 in lst[index:]:
            pairs.append((element1, element2))
        index += 1
    
    return pairs

def dedupe(data):
    """removes duplicates
        """
    result = Counter()
    for row in data:
        result.update(dict([row]))
    return result.items()


def create_edgelist(data, weight):
    """ creates edge list from dataframe of transactions (data) with
        weight specified by input (can be either usercount (where the
        weight is the number of users who made purchases at both endnode
        locations, aka usrcount) or joshweight,(if a user made x purchases
        at merchant A and y purchases at Merchant B, then
        joshwt = max(x, y) / abs(x-y) ). where weight is the number of users
        who made purchases as both endnode locations, aka usercount.
        It is used to create edge weights from multiple transactions
        """
    edges = []
    weights = []
    counter = []
    start = int(data['user'].min())
    stop = int(data['user'].max())
    
    if weight == 'usercount':
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

    elif weight == 'joshwt':
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
    else:
        print("weight value not defined")


def edgelist_bynumber_forR(edgelst, node_labels):
    """This is only really necessay if node_category == mcc_description..
        It changes are source and targets from mcc_description to mcc_code.
        This was necessary because R needed srouce and target as numerical values
        for the visNetwork to run properly.
        """
    e1, e2, w = zip(*edgelst)
    src = replace(e1, node_labels)
    tgt = replace(e2, node_labels)
    wtd_edgelist_byid = list(zip(src, tgt, w))
    return wtd_edgelist_byid


def addnames_to_edgelist(edgelst, node_labels):
    """Adds names to edgelist. List goes from (source, target, weight) to
        (source, target, weight, source_name, target_name)
        This is helpful when source/targets are numerical values such as mcc_code;
        the addition of names make the edgelist more interpretable/informative
        """
    e1, e2, w = zip(*edgelst)
    src = replace(e1, node_labels)
    tgt = replace(e2, node_labels)
    wtd_edgelist_withnames = list(zip(e1, e2, w, src, tgt))
    return wtd_edgelist_withnames


def mod(communities):
    """ separates each comunity into individual sets for inspection of  at each community,
        i.e.   cmd:  modularity_travel[#]
        """
    modularity = {}
    for k,v in communities.items(): # Loop through the community dictionary
        if v not in modularity:
            modularity[v] = [k] # Add a new key for a modularity class the code hasn't seen before
        else:
            modularity[v].append(k)
    
    return modularity


def remove_duplicate(alist):
    """ removes duplicates
        """
    return list(set(alist))


def remove_selfloops(listofpairs):
    """Removes self loops, ie removes (a,a)
        """
    new_list = []
    for a, b in listofpairs:
        if (a != b):
            new_list.append((a,b))
    return new_list

def replace(my_list, my_dict):
    return [x if x not in my_dict else my_dict[x] for x in my_list]





##################################################################
##################################################################
#
#                     DATA PREP
#
##################################################################
##################################################################

#read in the data. Note there is already an index column in csv (hence index_col=False)
df = pd.read_csv(input_file, encoding='latin', low_memory=False, index_col=False)

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
                                                      
# clean up mcc_description & merchant_details. removes non-ASCII characters
df_all.mcc_description.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df_all.merchant_details.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
       
# create a 'day of the week field' to potentially compare weekday activity v. weekend activity (M = 0, T = 1, W = 2, Th = 3, F = 4, Sat = 5, Sun = 6). So weekday < 5, weekend >=5.
df['authorization_timestamp'] = pd.to_datetime(df.authorization_timestamp)
df['date'] = [d.date() for d in df['authorization_timestamp']]
##df['authorization_time'] = [d.time() for d in df['authorization_timestamp']]
df['weekday'] = df['date'].apply( lambda x: x.weekday())
    
                                                      
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
#    not sure how to do this.
                                                      
# transaction_id is the exact same for all rows, so delete it
del df['transaction_id']
del df['anonymized_user_id']
del df['settle_timestamp']
del df['year_of_birth']
                      
                                                      
df_all = df.copy()
########     ORGANIZE THE DATA       ########
# organize the data fields
df_all = df_all[['index', 'user', 'age', 'date', 'weekday', 'amount', 'merchant_id', 'merchant_details', 'merchant_name',  'mcc_code', 'mcc_description', 'koho_category', 'date',  'user_numofpurchases', 'merch_id_count', 'merch_name_count', 'merchant_name_countbyuser', 'merch_match_count', 'mcc_code_count', 'mcc_desc_count', 'mcc', 'transaction_code', 'authorization_timestamp']]
                                                      

# Make different dataframes for weekday activity and weekend activity
df_wday = df[df['weekday'] <5].copy()
df_wknd = df[df['weekday'] >=5].copy()
            
######################################################
#        SAVE OUT DF_ALL TO CSV FILES
######################################################
# Save out data frames to csv
df_all.to_csv('df_all.csv', index=False)
                                                      
                                                      
                                                      
                                                      
                                                      
##################################################################
##################################################################
#
#                 Import Dataframe
#
##################################################################
##################################################################
#read in the data. Note there is already an index column in csv (hence index_col=False)
df_all = pd.read_csv('df_all.csv', encoding='latin', low_memory=False)

df_trav = df_all[df_all['koho_category'] == 'Travel']
df_other = df_all[df_all['koho_category'] == 'Other']
df_travel = pd.concat([df_trav, df_other])

df_travel_category = df_travel[['mcc_code', 'mcc_description', 'koho_category', 'merchant_name']]
df_travel_category.to_csv('travel_category.csv', index=False)

#  --> manually labeled 347 mcc_codes for travel_category

df_travel_category = pd.read_csv('travel_category.csv')
df_travel_category_only = df_travel_category[['mcc_code', 'travel_category']]
df_travel = df_travel.merge(df_travel_category_only, on='mcc_code', how='right')
df_travel = df_travel.drop(df_travel[(df_travel.travel_category == 'Other')].index)
df_travel = df_travel.drop(df_travel[(df_travel.travel_category == 'Mislabeled')].index)
df_travel = df_travel.drop(df_travel[(df_travel.travel_category == 'Transportation')].index)

###########################################################
#        SAVE OUT TRAVEL TRANSACTIONS TO CSV FILES
###########################################################
# save out final travel data set. size = 7,026
df_travel.to_csv('df_travel.csv', index=False)
# unique mcc_codes = 142
# unique mcc_description = 107




#####################################################################
#
#     Creating the Network (edge list & node list)
#
#####################################################################

# create edge list & sort by weight
edgelist_travel = create_edgelist(df_travel, weight)
edgelist_travel.sort(key=operator.itemgetter(2), reverse=True)

#if using mcc_code, this is not necessary:
#edgelist_travel_byid = edgelist_bynumber_forR(edgelist_travel)

# create node DB (to be filled with node attributes later)
nodes_travel = df_travel.copy()
nodes_travel = nodes_travel.drop_duplicates(node_category)
nodes_travel = nodes_travel.drop(columns = ['merchant_id', 'merchant_details', 'merchant_name', 'index', 'user', 'age', 'date', 'weekday', 'amount', 'user_numofpurchases',  'transaction_code', 'authorization_timestamp'])
# size = 142

# Create a dictionary for nodes & mcc_code (this is for R later on)
node_labels = create_dict(nodes_travel.mcc_code, nodes_travel.mcc_description.values)

# add names to edgelist - do this if node category is numerical (ie mcc_code or merchant_id)
edgelist_travel_withnames = addnames_to_edgelist(edgelist_travel, node_labels)
edgelist_travel_withnames_asdf = pd.DataFrame(edgelist_travel_withnames, columns=['from', 'to', 'weight', 'from_name', 'to_name'])

                                                      
###########################################################
#        SAVE OUT EDGELIST TO CSV FILES
###########################################################
#write to csv:  -- make it a parameter -- check out style guide. pep8.
edgelist_travel_withnames_asdf.to_csv('edgelist_travel_mcccode_usrcount_withnames_asdf.csv', index=False)







##################################################################
##################################################################
#
#               GRAPHING   -- using networkx
#
##################################################################
##################################################################

T = nx.Graph()
T.add_weighted_edges_from(edgelist_travel)


# Louvain Community Detection  <-- fix this to make isolated nodes their own community?
communities_travel = best_partition(T)
nx.set_node_attributes(T, communities_travel, 'modularity')
nodes_travel['community'] = nodes_travel[node_category].map(communities_travel)
modularity_travel = mod(communities_travel)  # <-- lets you look at each community, i.e.   cmd:  modularity_travel[#]

# degree
degree_dict_travel = dict(T.degree(T.nodes()))
nx.set_node_attributes(T, degree_dict_travel, 'degree')
nodes_travel['degree'] = nodes_travel[node_category].map(degree_dict_travel)
nodes_travel['degree'].fillna(0, inplace=True)  # < -- to deal with isolated nodes (degree = 0)

# betweenness centrality
btwn_centrality_travel = nx.betweenness_centrality(T)  # sets node attribute
nodes_travel['betweenness_centrality'] = nodes_travel[node_category].map(btwn_centrality_travel)  #adds attribute to nodes_travel dataframe

# closeness centrality
close_centrality_travel = nx.closeness_centrality(T)
nodes_travel['closeness_centrality'] = nodes_travel[node_category].map(close_centrality_travel)

# eigenvectory centrality
eigenvector_centrality_travel = nx.eigenvector_centrality(T)
nodes_travel['eigenvector_centrality'] = nodes_travel[node_category].map(eigenvector_centrality_travel)

# neighbors (list by mcc_code)
codes_travel = list(degree_dict_travel.keys())
nbrs_travel = {}
for c in codes_travel:
    nbrs_travel[c] = [n for n in T.neighbors(c)]
nodes_travel['neighbors'] = nodes_travel[node_category].map(nbrs_travel)



##################################################################
#        SAVE OUT INFO/NODE ATTRIBUTES TO CSV FILES
##################################################################
nodes_travel.to_csv('nodes_travel_mcccode_usrcount.csv', index=False)











                                                      

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

nodes_travel_usrcount <- read_csv("nodes_travel_mcccode_usrcount.csv",
                         col_types = cols(Travel_related = col_skip(),
                                          X1 = col_skip(), koho_category = col_skip()))
nodes <- nodes_travel_usrcount[,1:5]
colnames(nodes) <- c("id", "label", "koho", "travel", "group")


edgelist_travel_usrcount_withnames <- read_csv("edgelist_travel_mcccode_usrcount_withnames_asdf.csv")

edges <- edgelist_travel_usrcount[,1:3]


#To color by louvain community:
v1 <- visNetwork(nodes, edges, height = "700px", width = "100%") %>%
    visOptions(selectedBy = "group",
               highlightNearest = TRUE,
               nodesIdSelection = TRUE) %>%
    visPhysics(stabilization = FALSE, maxVelocity = 3)
frameWidget(v1)

#to color by Travel type (Hotel, Air, etc.)
v2 <- visNetwork(nodes_travel, wtd_edgelist_travel_usrcount_byid, height = "700px", width = "100%") %>%
    visOptions(selectedBy = "travel",
               highlightNearest = TRUE,
               nodesIdSelection = TRUE) %>%
    visPhysics(stabilization = FALSE, maxVelocity = 3)
frameWidget(v2)

# Simple Graph
p5 <- simpleNetwork(wtd_edgelist_travel_usrcount_byid, nodeColour = "red", zoom=T)
frameWidget(p5)



