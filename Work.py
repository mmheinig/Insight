##################################################################
##################################################################
#
#                   LIBRARIES
#
##################################################################
##################################################################
import pandas as pd
import numpy as np
import networkx as nx
import os
import sys
import requests
import re
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
from operator import itemgetter, attrgetter
from neo4j.v1 import GraphDatabase
import datetime as dt
from collections import Counter
from igraph import *
import collections
import csv



##################################################################
##################################################################
#
#                    DEFINITIONS
#
##################################################################
##################################################################

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def dedupe(data):
    result = Counter()
    for row in data:
        result.update(dict([row]))
    return result.items()

def remove_duplicates(A):
    [A.pop(count) for count,elem in enumerate(A) if A.count(elem)!=1]
        return A

##################################################################
##################################################################
#
#                     Inputs
#
##################################################################
##################################################################

#choose which category to use for nodes of the network (merchant_id, merchant_details, merchant_name, mcc_code, mcc_description)
node_category = 'merchant_name'
# choose which category to use for node colors of the network (koho_category (10), mcc_code (409), mcc_description (373), merchant_name (??))
color_category = 'koho_category'







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
df.merchant_name = df.merchant_name.apply(lambda x: x.upper())
#df["merchant_name_countbyuser"] = df.groupby(["merchant_name", "user"])["user"].transform("count")
df["user_numofpurchases"] = df.groupby(["user"])["index"].transform("count")

       
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
#    3. official launch of koho card was March 2017, so remove records before then.
# NO IDEA HOW TO DO THIS --- tried a million times!
                                                      
# transaction_id is the exact same for all rows, so delete it
del df['transaction_id']
del df['anonymized_user_id']
del df['settle_timestamp']
del df['year_of_birth']
                      
                                                      
                                                      
#    ****  This NEEDS to be recalculated here (after deletions) because some of the user's transactional records may have been deleted due to transaction codes!!   ****
df["merchant_name_countbyuser"] = df.groupby(["merchant_name", "user"])["user"].transform("count")
df["merchant_id_countbyuser"] = df.groupby(["merchant_id", "user"])["user"].transform("count")


########     ORGANIZE THE DATA       ########
# organize the data fields
df = df[['index', 'user', 'age', 'date', 'weekday', 'amount', 'merchant_id', 'merchant_details', 'merchant_name',  'mcc_code', 'mcc_description', 'koho_category', 'date',  'user_numofpurchases', 'merch_id_count', 'merch_name_count', 'merchant_name_countbyuser', 'merch_match_count', 'mcc_code_count', 'mcc_desc_count', 'mcc', 'transaction_code', 'authorization_timestamp']]
                                                       

# Make different dataframes for weekday activity and weekend activity
df_wday = df[df['weekday'] <5].copy()
df_wknd = df[df['weekday'] >=5].copy()


                                                      
                                                      
                                                      
##################################################################
#                  Create NODE DATABASES!!
##################################################################
# can be saved out and used by all 3 categories/graphs
                                                      
merchant_nodes_df = df.copy()
nodedf = merchant_nodes_df.drop_duplicates('merchant_id')
nodedf = nodedf.drop(columns = ['index', 'user', 'age', 'date', 'weekday', 'amount', 'user_numofpurchases', 'mer_id_count', 'merch_name_count', 'merch_name_countbyuser', 'merch_match_count', 'merch_code_count', 'mcc_desc_count', 'mcc', 'transaction_code', 'authorization_timestamp', 'merchant_name_countbyuser', 'mcc_code_count', 'merch_id_count'])
nodedf.to_csv('nodedf_bymerchantid.csv', index=False)
#nodedf.to_csv('nodedf_all.csv', index=False)
                                                      
                                                      
nodedf_bymerchantname = df.copy()
nodedf_bymerchantname = nodedf_bymerchantname.drop_duplicates('merchant_name')
nodedf_bymerchantname = nodedf_bymerchantname.drop(columns = ['index', 'user', 'age', 'date', 'weekday', 'amount', 'user_numofpurchases', 'mer_id_count', 'merch_name_count', 'merch_name_countbyuser', 'merch_match_count', 'merch_code_count', 'mcc_desc_count', 'mcc', 'transaction_code', 'authorization_timestamp', 'merchant_name_countbyuser', 'mcc_code_count', 'merch_id_count','merchant_id', 'merchant_details'])
nodedf_bymerchantname.to_csv('nodedf_bymerchantname.csv', index=False)
                                                      
nodedf_bymcccode = df.copy()
nodedf_bymcccode = nodedf_bymcccode.drop_duplicates('mcc_code')
nodedf_bymcccode = nodedf_bymcccode.drop(columns = ['merchant_id', 'merchant_details', 'merchant_name', 'index', 'user', 'age', 'date', 'weekday', 'amount', 'user_numofpurchases', 'mer_id_count', 'merch_name_count', 'merch_name_countbyuser', 'merch_match_count', 'merch_code_count', 'mcc_desc_count', 'mcc', 'transaction_code', 'authorization_timestamp', 'merchant_name_countbyuser', 'mcc_code_count', 'merch_id_count'])
nodedf_bymcccode.to_csv('nodedf_bymcccode.csv', index = False)
                                                      
nodedf_bymccdesc = df.copy()
nodedf_bymccdesc = nodedf_bymccdesc.drop_duplicates('mcc_description')
nodedf_bymccdesc = nodedf_bymccdesc.drop(columns = ['mcc_code', 'merchant_id', 'merchant_details', 'merchant_name', 'index', 'user', 'age', 'date', 'weekday', 'amount', 'user_numofpurchases', 'mer_id_count', 'merch_name_count', 'merch_name_countbyuser', 'merch_match_count', 'merch_code_count', 'mcc_desc_count', 'mcc', 'transaction_code', 'authorization_timestamp', 'merchant_name_countbyuser', 'mcc_code_count', 'merch_id_count'])
nodedf_bymccdesc.to_csv('nodedf_bymccdesc.csv', index = False)
                                                      
nodedf['koho_category'] = pd.Categorical(nodedf['koho_category'])
nodedf['koho_category'].cat.codes
                                                      

# nodedf is written to csv as "node_df.csv" later on (see SAVE OUT DATA section)
                                                      
                                                      
##################################################################
#        Turn prepared data into lists of edges & weights!
##################################################################
                                                      
###################################
#    ALL  DATA
###################################
edges = []
weights = []
counter = []
start = df['user'].min()
stop = df['user'].max()
for u in range(start, stop):
    y = df[df['user'] == u]
    x = y[node_category].value_counts()
    z = list(pairwise(y[node_category]))
    for i in range(0,len(z)):
        for item in z[i]:
             counter.append(x[item])
        weights.append(min(counter))
        counter = []
    edges = edges + z
                                                      
edges   #creates a list of tuples (endnode1, endnode2)
weights #creates a list of weights

# ZIP edges & weights together to DEDUPLICATE
edgesandwts = list(zip(edges, weights)) #list of pairs where first term is the edge as a pair of endnodes (v1, v2) and second term is the weight of that edge
edgesandwts = sorted(edgesandwts,key=itemgetter(0))
# Get rid of duplicates and add their weights together
edgesandwts_dedupe = list(dedupe(edgesandwts)) #gets rid of duplicate edges and adds their weights
edgesandwts_dedupe   #list of edges + weights, with duplicates edge pairs removed but weight is sum of weights for all edge pairs originally found in edgesandwts
                                                      
#UNZIP to be able to switch (b,a) to (a,b)
e_pairs, wts = zip(*edgesandwts_dedupe)
# this creates a blank list "temp". assume a < b. it goes though the edges if (a,b) in  e_pairs_wknd, append (a,b) to temp; if (b,a) in e_pairs_wknd, append (a,b) to temp (ie switch the order around). then set e_pairs_wknd to temp.
temp = []
for a, b in e_pairs:
    if a > b:
        if (b,a) in e_pairs:
            temp.append((b,a))
        else:
            temp.append((a,b))
    else:
        temp.append((a,b))
e_pairs = temp

# re-ZIP edges & weights together to un-DEDUPLICATE AGAIN (this will take care of all the (b,a)s we switched to (a,b). Note, this is done here after the initial DEDUP because it takes a while and is faster on a smaller, deduplicated set.
e_w_rezipped = list(zip(e_pairs, wts))
e_w_dedupe2 = list(dedupe(e_w_rezipped))
e_pairs2, wts2 = zip(*e_w_dedupe2)
e1, e2 = zip(*e_pairs2) #unzips the list of tuples
edgelist = list(zip(e1, e2, wts2))   #(endnode1, endnode2, weight)

# Remove self loops! ie (a,a)
wtd_edgelist = []
for u, v, w in edgelist:
    if (u != v):
        wtd_edgelist.append((u, v, w))
                                                      
# This creates a sorted wtd_edgelist from highest weight down.
wtd_edgelist.sort(key=operator.itemgetter(2), reverse=True)
                                                      '''
edgesandwts = list(zip(edges, weights)) #list of pairs where first term is the edge as a pair of endnodes (v1, v2) and second term is the weight of that edge
# Get rid of duplicates and add their weights together
edgesandwts_dedupe = list(dedupe(edgesandwts)) #gets rid of duplicate edges and adds their weights
edgesandwts_dedupe   #list of edges + weights, with duplicates edge pairs removed but weight is sum of weights for all edge pairs originally found in edgesandwts
e_pairs, wts = zip(*edgesandwts_dedupe)
e1, e2 = zip(*e_pairs) #unzips the list of tuples
edgelist = list(zip(e1, e2, wts))   #(endnode1, endnode2, weight)
# note the above (edgelist) will have self loops, i.e., endnode1 == endnode2). the next for loop will remove those:
wtd_edgelist = []
for u, v, w in edgelist:
    if (u != v):
        wtd_edgelist.append((u, v, w))
# This creates a sorted wtd_edgelist from highest weight down.
wtd_edgelist_sorted = wtd_edgelist.sort(key=operator.itemgetter(2), reverse=True)
'''
                                                       
###################################
#   WEEKDAY DATA  (Monday - Friday)
###################################
edges_wday = []
weights_wday = []
counter_wday = []
start_wday = df_wday['user'].min()
stop_wday = df_wday['user'].max()
for u in range(start_wday, stop_wday):
    y = df_wday[df_wday['user'] == u]
    x = y[node_category].value_counts()
    z = list(pairwise(y[node_category]))
    for i in range(0,len(z)):
        for item in z[i]:
            counter_wday.append(x[item])
        weights_wday.append(min(counter_wday))
        counter_wday = []
    edges_wday = edges_wday + z
                                                      
edges_wday   #creates a list of tuples (endnode1, endnode2)
weights_wday #creates a list of weights
 
# ZIP edges & weights together to DEDUPLICATE
edgesandwts_wday = list(zip(edges_wday, weights_wday)) #list of pairs where first term is the edge as a pair of endnodes (v1, v2) and second term is the weight of that edge
edgesandwts_wday = sorted(edgesandwts_wday,key=itemgetter(0))
# Get rid of duplicates and add their weights together
edgesandwts_dedupe_wday = list(dedupe(edgesandwts_wday)) #gets rid of duplicate edges and adds their weights
edgesandwts_dedupe_wday   #list of edges + weights, with duplicates edge pairs removed but weight is sum of weights for all edge pairs originally found in edgesandwts
                                                      
#UNZIP to be able to switch (b,a) to (a,b)
e_pairs_wday, wts_wday = zip(*edgesandwts_dedupe_wday)
# this creates a blank list "temp". assume a < b. it goes though the edges if (a,b) in  e_pairs_wknd, append (a,b) to temp; if (b,a) in e_pairs_wknd, append (a,b) to temp (ie switch the order around). then set e_pairs_wknd to temp.
temp = []
for a, b in e_pairs_wday:
    if a > b:
        if (b,a) in e_pairs_wday:
            temp.append((b,a))
        else:
            temp.append((a,b))
    else:
        temp.append((a,b))
e_pairs_wday = temp

# re-ZIP edges & weights together to un-DEDUPLICATE AGAIN (this will take care of all the (b,a)s we switched to (a,b). Note, this is done here after the initial DEDUP because it takes a while and is faster on a smaller, deduplicated set.
e_w_rezipped_wday = list(zip(e_pairs_wday, wts_wday))
e_w_dedupe2_wday = list(dedupe(e_w_rezipped_wday))
e_pairs2_wday, wts2_wday = zip(*e_w_dedupe2_wday)
e1_wday, e2_wday = zip(*e_pairs2_wday) #unzips the list of tuples
edgelist_wday = list(zip(e1_wday, e2_wday, wts2_wday))   #(endnode1, endnode2, weight)

# Remove self loops! ie (a,a)
wtd_edgelist_wday = []
for u, v, w in edgelist_wday:
    if (u != v):
        wtd_edgelist_wday.append((u, v, w))
                                                      
# This creates a sorted wtd_edgelist from highest weight down.
wtd_edgelist_wday.sort(key=operator.itemgetter(2), reverse=True)
                                                      
''' edgesandwts_wday = list(zip(edges_wday, weights_wday)) #list of pairs where first term is the edge as a pair of endnodes (v1, v2) and second term is the weight of that edge
# Get rid of duplicates and add their weights together
edgesandwts_dedupe_wday = list(dedupe(edgesandwts_wday)) #gets rid of duplicate edges and adds their weights
edgesandwts_dedupe_wday   #list of edges + weights, with duplicates edge pairs removed but weight is sum of weights for all edge pairs originally found in edgesandwts
e_pairs_wday, wts_wday = zip(*edgesandwts_dedupe_wday)
e1_wday, e2_wday = zip(*e_pairs_wday) #unzips the list of tuples
edgelist_wday = list(zip(e1_wday, e2_wday, wts_wday))   #(endnode1, endnode2, weight)
# note the above (edgelist_wday) will have self loops, i.e., endnode1 == endnode2). the next for loop will remove those:
wtd_edgelist_wday = []
for u, v, w in edgelist_wday:
    if (u != v):
        wtd_edgelist_wday.append((u, v, w))
# This creates a sorted wtd_edgelist from highest weight down.
wtd_edgelist_wday_sorted = wtd_edgelist_wday.sort(key=operator.itemgetter(2), reverse=True) '''
                                                      
    
                                                      
###################################
#   WEEKEND DATA  (Saturday & Sunday)
###################################
edges_wknd = []
weights_wknd = []
counter_wknd = []
start_wknd = df_wknd['user'].min()
stop_wknd = df_wknd['user'].max()
for u in range(start_wknd, stop_wknd):
    y = df_wknd[df_wknd['user'] == u]
    x = y[node_category].value_counts()
    z = list(pairwise(y[node_category]))
    for i in range(0,len(z)):
        for item in z[i]:
            counter_wknd.append(x[item])
        weights_wknd.append(min(counter_wknd))
        counter_wknd = []
    edges_wknd = edges_wknd + z
                                                      
edges_wknd   #creates a list of tuples (endnode1, endnode2)
weights_wknd #creates a list of weights
                                                      
# ZIP edges & weights together to DEDUPLICATE
edgesandwts_wknd = list(zip(edges_wknd, weights_wknd)) #list of pairs where first term is the edge as a pair of endnodes (v1, v2) and second term is the weight of that edge
edgesandwts_wknd = sorted(edgesandwts_wknd,key=itemgetter(0))
# Get rid of duplicates and add their weights together
edgesandwts_dedupe_wknd = list(dedupe(edgesandwts_wknd)) #gets rid of duplicate edges and adds their weights
edgesandwts_dedupe_wknd   #list of edges + weights, with duplicates edge pairs removed but weight is sum of weights for all edge pairs originally found in edgesandwts
                                                      
#UNZIP to be able to switch (b,a) to (a,b)
e_pairs_wknd, wts_wknd = zip(*edgesandwts_dedupe_wknd)
# this creates a blank list "temp". assume a < b. it goes though the edges if (a,b) in  e_pairs_wknd, append (a,b) to temp; if (b,a) in e_pairs_wknd, append (a,b) to temp (ie switch the order around). then set e_pairs_wknd to temp.
temp = []
for a, b in e_pairs_wknd:
    if a > b:
        if (b,a) in e_pairs_wknd:
            temp.append((b,a))
        else:
            temp.append((a,b))
    else:
        temp.append((a,b))
e_pairs_wknd = temp

# re-ZIP edges & weights together to un-DEDUPLICATE AGAIN (this will take care of all the (b,a)s we switched to (a,b). Note, this is done here after the initial DEDUP because it takes a while and is faster on a smaller, deduplicated set.
e_w_rezipped_wknd = list(zip(e_pairs_wknd, wts_wknd))
e_w_dedupe2_wknd = list(dedupe(e_w_rezipped_wknd))
e_pairs2_wknd, wts2_wknd = zip(*e_w_dedupe2_wknd)
e1_wknd, e2_wknd = zip(*e_pairs2_wknd) #unzips the list of tuples
edgelist_wknd = list(zip(e1_wknd, e2_wknd, wts2_wknd))   #(endnode1, endnode2, weight)

# Remove self loops! ie (a,a)
wtd_edgelist_wknd = []
for u, v, w in edgelist_wknd:
    if (u != v):
        wtd_edgelist_wknd.append((u, v, w))
                                                      
# This creates a sorted wtd_edgelist from highest weight down.
wtd_edgelist_wknd.sort(key=operator.itemgetter(2), reverse=True)
                                                      
                                                      
                                                      
#NOTES:::
len(wtd_edgelist) = 6566
len(wtd_edgelist_wday) = 6192
len(wtd_edgelist_wknd) = 4670
                                                      
                                                      
                                                      
##################################################################
##################################################################
#
#                       SAVE OUTPUTS
#
##################################################################
##################################################################
                                                      
##################################################################
#          Save out node dataframe, df, df_wday, & df_wknd
##################################################################
# Node data frame (nodedf) <-- by merchant_id
#nodedf.to_csv('node_df.csv', index=False)
nodedf.to_csv('nodedf_all.csv', index=False)
# Node data frame by merchant_name
nodedf_bymerchantname.to_csv('nodedf_bymerchantname.csv', index=False)
# Node data frame by mcc_code
nodedf_bymcccode.to_csv('nodedf_bymcccode.csv', index=False)
# Node data frame by mcc_description
nodedf_bymccdesc.to_csv('nodedf_bymccdesc.csv', index=False)

# DATAFRAMES (transactional data)
# All data (df)
df.to_csv('df_all.csv', sep=',', index=False)
# Weekday data
df_wday.to_csv('df_weekday.csv', index=False)
# Weekend data
df_wknd.to_csv('df_weekday.csv', index=False)
                                                      
                                                      
                                                      
                                                      
##################################################################
#                 Save out weighted edge lists
##################################################################
###################################
#   ALL DATA
###################################
writer = csv.writer(open("weighted_edgelist_all_bymerchantname.csv", 'w+'))
for row in wtd_edgelist:
    writer.writerow(row)
###################################
#   WEEKDAY DATA  (Monday - Friday)
###################################
del row
with open('weighted_edgelist_weekday_bymerchantname.csv', 'w+') as outfile:
    writer = csv.writer(outfile)
    for row in wtd_edgelist_wday:
        writer.writerow(row)
###################################
#   WEEKEND DATA  (Saturday & Sunday)
###################################
del row
with open('weighted_edgelist_weekend_bymerchantname.csv', 'w+') as outfile:
    writer = csv.writer(outfile)
    for row in wtd_edgelist_wknd:
        writer.writerow(row)
          
###################################################################
#                 Open saved output (ie for new notebooks,
# add Libraries, Inputs and Definitions from above. then start here.
###################################################################
df = pd.read_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/Dataframes_savedout/df_all.csv', encoding='latin', low_memory=False)
edgelist = pd.read_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/Dataframes_savedout/weighted_edgelist_all_bymerchantname.csv')
#make the dataframe wtd_edgelist a list of 3-tuples (endnode, endnode, weight)
wtd_edgelist = list(zip(edgelist[edgelist.columns[0]], edgelist[edgelist.columns[1]], edgelist[edgelist.columns[2]]))
        #### THIS UPLOAD AS 215,260 edges --- something is wrong!!!!!!!!
nodedf_merchname = pd.read_csv('/Users/MonikaHeinig/Desktop/Insight/KOHO_Financial/Dataframes_savedout/node_df_bymerchantname.csv')
                                                      
                                                      
                                                      
                                                      
##################################################################
##################################################################
#
#               GRAPHING   -- using networkx
#
##################################################################
##################################################################
                                                      
       
##################################################################
# Making the Graphs  (G (all data), D (weekday), & N (weekend)
##################################################################
                                                      
# https://networkx.github.io/documentation/networkx-1.9/examples/drawing/degree_histogram.html
# http://www.ic.unicamp.br/~wainer/cursos/1s2012/mc906/grafos.pdf
                                                      
######################
#     ALL DATA
######################
G = nx.Graph()
#create graph from edge list above (for loop)
#G.add_edges_from(edges)
G.add_weighted_edges_from(wtd_edgelist)
G.remove_edges_from(G.selfloop_edges())  #edgelist -> wtd_edgelist should have taken care of this, but this is just a double check
G.number_of_nodes()   #373   #69,868
G.number_of_edges()   #6256 #158,725
#nx.connected_components(G)
nx.number_connected_components(G)
nx.draw(G)
plt.show()

#only one connected component - so giant is not necessary
largecomp = max(nx.connected_component_subgraphs(G), key=len)
     
                                                      
                                                      
###################################
#   WEEKDAY DATA  (Monday - Friday)
###################################
D = nx.Graph()
#create graph from edge list above (for loop)
#D.add_edges_from(edges)
D.add_weighted_edges_from(wtd_edgelist_wday)
D.remove_edges_from(D.selfloop_edges())   #edgelist -> wtd_edgelist should have taken care of this, but this is just a double check
D.number_of_nodes()     #360 nodes
D.number_of_edges()     #5901 edges
#nx.connected_components(D)
nx.number_connected_components(D)
nx.draw(D)
plt.show()

largecop_wday = max(nx.connected_component_subgraphs(D), key=len)
         
# define nodes:
nodes = G.nodes()
#nodes = list(df[df.columns[8]])
#nodes = list(set(nodes))
                                                      
                                                      
# define node attributes. Set dictoraries = {}, then define.
merchant_id = {}
merchant_details = {}
merchant_name = {}
mcc_code = {}
mcc_desc = {}
koho_category = {}
                                                      
# this is wayyyyyyyyyyyyy too slow, dont do this
''' #for i in range(0,len(nodes)):
    merchant_id[nodes[i]] = df.iloc[i][6]
    merchant_details[nodes[i]] = df.iloc[i][7]
    merchant_name[nodes[i]] = df.iloc[i][8]
    mcc_code[i]] = df.iloc[i][9]
    mcc_desc[nodes[i]] = df.iloc[i][10]
    Koho_category[nodes[i]] = df.iloc[i][10] '''

#BE CAREFULL!! THIS NEXT STEP DEPENDS ON WHAT YOU USE AS NODES. In this case we have "merchant_name" as the nodes.
koho_category = nodedf_bymerchantname.set_index(node_category)['koho_category'].to_dict()
                                                      
degree_dict = dict(G.degree(G.nodes()))
                                                      
                                                      
nx.set_node_attributes(G, merchant_id, 'merchant_id')
nx.set_node_attributes(G, merchant_details, 'merchant_details')
nx.set_node_attributes(G, merhant_name, 'merchant_name')
nx.set_node_attributes(G, mcc_code, 'mcc_code')
nx.set_node_attributes(G, mcc_desc, 'mcc_desc')
nx.set_node_attributes(G, koho_category, 'koho_category')
nx.set_node_attributes(G, degree_dict, 'degree')

###################################
#   WEEKEND DATA  (Saturday & Sunday)
###################################
N = nx.Graph()
N.add_nodes_from(nodedf) # sets nodes to nodes from "nodedf". So there may be some isolated nodes
#create graph from edge list above (for loop)
#N.add_edges_from(edges)
N.add_weighted_edges_from(wtd_edgelist_wknd)
N.remove_edges_from(N.selfloop_edges())    #edgelist -> wtd_edgelist should have taken care of this, but this is just a double check

N.remove_nodes_from(nx.isolates(N))  # <-- doesnt work!!!!!!!!
                                                      
N.number_of_nodes()       #321 nodes
N.number_of_edges()       #4431 edges
#nx.connected_components(N)
nx.number_connected_components(N)
nx.draw(N)
plt.show()

largecomp_wknd = max(nx.connected_component_subgraphs(N), key=len)
                                                      ## Notes about results

                                                      
                                                      
##################################################################
#           Analyzing the Graphs
##################################################################
                                                      
######################
#     ALL DATA
######################
pos = nx.spring_layout(G)
del u
del v

nodedf_merchname = nodedf_merchname.set_index('merchant_name')
nodedf_merchname = nodedf_merchname.reindex(G.nodes())
# And I need to transform my categorical column in a numerical value: group1->1, group2->2...
nodedf_merchname['koho_category']=pd.Categorical(nodedf_merchname['koho_category'])
nodedf_merchname['koho_category'].cat.codes
                                                      
 print(nx.info(G))
                                                      
### makes pretty picture
#thresh = 20   #  <--- can update/change this
#weights = [G[u][v]['weight'] for u,v in G.edges(data=True)]
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 1500 ]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 1500 ]
nx.draw_networkx_nodes(G, pos, node_size=3)
nx.draw_networkx_edges(G, pos, edges=edges, edgelist=elarge, width = 3)
nx.draw_networkx_edges(G, pos, edges=edges, edgelist=esmall, width = 0.5, alpha=0.1, edge_color='b', style='dashed')
plt.axis('off')
plot.show()
 
                                                      
###  Makes histograph of degrees (with graph in background, but commented out for now
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
dmax = max(degree_sequence)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.6 for d in deg])
ax.set_xticklabels(deg)

# draw graph in inset
#plt.axes([0.4, 0.4, 0.5, 0.5])
#Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
#pos = nx.spring_layout(G)
#plt.axis('off')
#nx.draw_networkx_nodes(G, pos, node_size=20)
#nx.draw_networkx_edges(G, pos, alpha=0.4)
plt.show()
  
                                                      
# USE THIS!!! lists nodes of highest degree (high - low) with node and degree
sorted(G.degree, key=lambda x: x[1], reverse=True)
                                                      
                                                      
# Not really useful:
degrees = G.degree()
plt.plot(degrees)
plt.xlabel('Degree')
plt.title('All data')
plt.show()

                                                      
#betweenness centrality
btwn_cent = nx.betweenness_centrality(G)
#closeness centrality
close_cent = nx.closeness_centrality(G)
# eignevector centrality
eig_cent = nx.eigenvector_centrality(G)
                                                      
#This didnt work. idk why:
def highest_centrality(cent_dict):
     """Returns a tuple (node,value) with the node
 with largest value from Networkx centrality dictionary."""
     # Create ordered tuple of centrality data
     cent_items=[(b,a) for (a,b) in cent_dict.iteritems()]
     # Sort in descending order
     cent_items.sort()
     cent_items.reverse()
     return tuple(reversed(cent_items[0]))
                                                      
                                                      
                                                      
###################################
#   WEEKDAY DATA  (Monday - Friday)
###################################
#pos = nx.spring_layout(G) <-- same as above

                                                      
### makes pretty picture
#thresh = 20   #  <--- can update/change this
#weights = [D[u][v]['weight'] for u,v in D.edges(data=True)]
elarge_wday = [(u, v) for (u, v, d) in D.edges(data=True) if d['weight'] > 50 ]
esmall_wday = [(u, v) for (u, v, d) in D.edges(data=True) if d['weight'] <= 50 ]
nx.draw_networkx_nodes(D, pos, node_size=3)
nx.draw_networkx_edges(D, pos, edges=edges, edgelist=elarge_wday, width = 3)
nx.draw_networkx_edges(D, pos, edges=edges, edgelist=esmall_wday, width = 0.5, alpha=0.1, edge_color='b', style='dashed')
plt.axis('off')
plt.savefig("graph_weekday.png") # save as png
plot.show()
 
                                                      
###  Makes histograph of degrees (with graph in background, but commented out for now
degree_sequence_wday = sorted([d for n, d in D.degree()], reverse=True)
dmax_wday = max(degree_sequence)
degreeCount_wday = collections.Counter(degree_sequence_wday)
deg_wday, cnt_wday = zip(*degreeCount_wday.items())
fig_wday, ax_wday = plt.subplots()
plt.bar(deg_wday, cnt_wday, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.6 for d in deg_wday])
ax.set_xticklabels(deg_wday)

# draw graph in inset
#plt.axes([0.4, 0.4, 0.5, 0.5])
#Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
#pos = nx.spring_layout(G)
#plt.axis('off')
#nx.draw_networkx_nodes(G, pos, node_size=20)
#nx.draw_networkx_edges(G, pos, alpha=0.4)
plt.savefig("degreehistogram_weekday.png") # save as png
plt.show()
  
                                                      
# USE THIS!!! lists nodes of highest degree (high - low) with node and degree
sorted(D.degree, key=lambda x: x[1], reverse=True)
                                                      
                                                      
                                                      
###################################
#   WEEKEND DATA  (Saturday & Sunday)
###################################
#pos = nx.spring_layout(G) <-- same as above
                                                      
### makes pretty picture
#thresh = 20   #  <--- can update/change this
#weights = [N[u][v]['weight'] for u,v in N.edges(data=True)]
elarge_wday = [(u, v) for (u, v, d) in N.edges(data=True) if d['weight'] > 50 ]
esmall_wday = [(u, v) for (u, v, d) in N.edges(data=True) if d['weight'] <= 50 ]
nx.draw_networkx_nodes(N, pos, node_size=3)
nx.draw_networkx_edges(N, pos, edges=edges, edgelist=elarge_wday, width = 3)
nx.draw_networkx_edges(N, pos, edges=edges, edgelist=esmall_wday, width = 0.5, alpha=0.1, edge_color='b', style='dashed')
plt.axis('off')
plt.savefig("graph_weekend.png") # save as png
plot.show()
 
                                                      
###  Makes histograph of degrees (with graph in background, but commented out for now
degree_sequence_wknd = sorted([d for n, d in N.degree()], reverse=True)
dmax_wknd = max(degree_sequence)
degreeCount_wknd = collections.Counter(degree_sequence_wknd)
deg_wknd, cnt_wknd = zip(*degreeCount_wknd.items())
fig_wknd, ax_wknd = plt.subplots()
plt.bar(deg_wknd, cnt_wknd, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.6 for d in deg_wknd])
ax.set_xticklabels(deg_wknd)

# draw graph in inset
#plt.axes([0.4, 0.4, 0.5, 0.5])
#Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
#pos = nx.spring_layout(G)
#plt.axis('off')
#nx.draw_networkx_nodes(G, pos, node_size=20)
#nx.draw_networkx_edges(G, pos, alpha=0.4)
plt.savefig("degreehistogram_weekend.png") # save as png
plt.show()
  
                                                      
# USE THIS!!! lists nodes of highest degree (high - low) with node and degree
sorted(N.degree, key=lambda x: x[1], reverse=True)
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
#################################
#
#        GRAPHING   -- igraph
#
#################################
g = igraph.Graph.Read_Ncol(wtd_edgelist, directed=False)
d = igraph.Graph.Read_Ncol(wtd_edgelist_wday, directed=False)
n = igraph.Graph.Read_Ncol(wtd_edgelist_wknd, directed=False)
                                                      
   
                                                      
                                                      
                                                      
                                                      
##################################################################
##################################################################
#             OLD STUFF BELOW    --- can be ignored.
##################################################################
##################################################################
                                                      
                                                      
#get the fields & their types
df.info()
df.describe
df[df['merchant_id'].isnull()]
df[df['user_numofpurchases'] == 1]


#
df['merchant_name2'] = df['merchant_name']
df['merchant_name2'] = df["merchant_name2"].str.replace(r"[\"\',]", '')

                                                        
# creates better user ID's
df['user_id'] = df['anonymized_user_id']
df.user_id = pd.factorize(df.user_id)[0]

# change order of columns
df = df[['index','anonymized_user_id', 'user_id', 'year_of_birth', 'age', 'authorization_timestamp', 'settle_timestamp', 'transaction_code', 'amount', 'merchant_id', 'merchant_details', 'merchant_name', 'merchant_name2', 'mcc_code', 'mcc_description', 'koho_category', 'user_numofpurchases', 'merch_id_count', 'merch_name_count', 'merch_match_count', 'mcc_code_count', 'mcc_desc_count', 'mcc']]
                                                        
                                                        
                                                        
                                                        
                                                        
                                                        
                                                        
                                                        ##Airlines, Hotels/Motels/Inns/Resorts, Car Rentals (in mcc_description) all have several mcc_codes. This can separate airline companies from each other (ie China Eastern Air v Jetstar Air v Norwegian.com, etc) as well as Hotels & Car Rental companies.
                                                        ##Note, there are quite a few merchant_id == NaN!!
                                                        
                                                        df[df['age'].isnull()]
                                                        df.loc[df['mcc_description'] == 'DELTA']
                                                        
                                                        
                                                        tuples = list(zip(df['user'], df['merchant_name'], df['merchant_id']))
                                                        t = sorted(tuples, key=itemgetter(0))
                                                        index = pd.MultiIndex.from_tuples(sorted(tuples, key=itemgetter(0)), names=['user', 'merchant_name2', 'merchant_id'])
                                                        s = pd.Series(index=index)
                                                        s
                                                        

                                                        
                                                        

df['anonymized_user_id'].value_counts()
''' 
    2a30297ee42a953fc936f45c4be02b2ba77c05010382755883ed8264fc12f5c8    3246
    b05c2faebb6d54bf19403750acb8bc8ed048d29458c9650f4a6bc0eec5762ec2    2754
    68be28fa9bcaf139fc33fedbd5541afc35ece674f86fb750c927ee66d17f51a7    2634
    359b6307c300f7a2c13a1e92fd411fe0d7ed5563cb836884ea230a39205c38cf    2604
    e6c383839ef5881627272891aa40bb7c3ccc1e4eece2a225231c7457e6228241    2104
    a61c88c801b40d35bd74336066f97cc3bbcf0f714a03a327344cbac9084e8a75    1756
    fda341ac258f03e6a83e0a994d31e9487c472c57b689d0f0d2c1f812b267f51e    1756
    d27a56c0344b1176c1a0de760d2be69ddef670813586db14246c725f66d89360    1684
    cbeffd8e23902aec2dfa107ad5462bb58e3239b1cab2f56bea1ee5491a815e2b    1635
    9c893eef23b5688cd231493af0e61b0207cbfa58fb363c4f1630949205c5f3e5    1540
    5276b1b0e20fdf16978fb9f78ac4c2b1a8997e9c4a5c5843e46c26b4878bf590    1406
    f440e892574889f259b650a90b464257e81ce987c2cff33cf3163d9093327002    1374
    40ed07682f2e186a99a1d7ecde483422cb981f413b7f2038d21252f3c07d8ee1    1344
    967ee19973dd0867b2a3e7e5575894265efb3e478b1075f7319c0337a95d82c2    1284
    08c2fea29f4158c67c37ace7fb45619bb43c54d1246721b165690cdeeab51d2c    1279
    d77e0544f24dd73f040e67991f42c784efdd566ee2ad9fe707bf2ff2e24ae44e    1267
    b8ff7b9c47cdb9b4fbde874083fce348c94e0d4e61897309b2b0b7f22c8899b3    1262
    67ab75aaa45c729bcc181068595713a723161ff647d231ed19f60f858a31b21d    1254
    27c05462bcb7b9333d7522fb3005699dec04f13b74dac372ca6a1d41880d9c4a    1188
    9fec5d6c0a620abd351d6756841ceb4c94b1bf8faea54c8f5966aa5b6027c238    1160
    7b20ff82f667b2f467f8fa1af1b8a1420467e3157df89f6812e7712d22c5441d    1111
    65e80ddeaab80082a003c697a97f482ae992d9b778a3dfd31db9669947b61900    1104
    cb2c031d75afc3c08192cf3d2da9b3dacebe2940750adbe02bcf7b9c14b3d00d    1095
    531e7deabc354e94b1ea02d3fc2bbc1b979a2c0917173d134bf2cc876b357788    1071
    4b953c22a5e74707285573d6ff6287430ac2f29faad134526943afbc0e1feba4    1057
    1c5f13f7f1e34c52abe5f1efcf07d8835cb689be641cc7b9aa51e609fde3f1dd    1050
    957b93b8eabef9e7d5f980fbec6dfd2c4557d07a1e16a7c04c9147adffd01160    1041
    9149be283d3be51fe9b2dccf6c73199fbeb0b42ac1be27283b7ff28d8c1d6c10    1035
    2112fc88d2dda717098feacdf8397b3770ffaffac1056c4755bbc731ba738fa4     930
    3d615440726cf7108269fed2d13a1d4dc822e5e25134a22faee05bf1cc9f5468     912
    ...
    6d3c3e3ddd8ed7e39367ad29597212fd2fabb3c4239e514ca2f564a530827910       1
    51bc700ac3db6a1eb1be49955605d369812eb8df1357be4867e3b902743d41d2       1
    99bb5597487d7fcd27dd4a39d559dd3de4c8d0b51dd8db4dd24872be6b682a74       1
    d43ce5d5c863e48a48a4dc23bf6f385ee1b986e2a77175091af62f9b15621904       1
    43ec9374db6b0923761e9dfd007b4778a3bbbbbb9f5585e599cd7e85f4505941       1
    afaae1d4384fd7a73c285ba6ead8aa6c39a96a31bb654dff4643b4ccc4adebdb       1
    14e8124492662ee08a05155e1f541175ff6dd5e7169b84223f00bbef38ee92d9       1
    7a10c8deae02ec6a295988a0040d52dad0cc00834a4f84720d8b2aee5d390aaf       1
    dfdb6d49dcc2defca0aabebadb056c10f651fdb9f3ef209b42848a5aaaaa525c       1
    77f34e136ef4f8858b551438c56216bd4d79ca658de1990597bf543bb02d4a49       1
    c7bd803df9e40fef682758d4b398881dd7045f9ad287e5ce10087b2905065f0d       1
    b54e1fea42b9556689cfc7660e29b539304ed8df1d9158a581aed0909b71ab0e       1
    dc46ff20d04bc76cd7861c62838f0187cf607e3105c28be8b5064cbd901e89d6       1
    565287114f335ec88dd90cfe39ed46247a030e4e6016b13c902e61e41fcee760       1
    0283f8435f00b0f579a34e55cb77c236d20fcba0951d15cc0ddd458630e64633       1
    7fa0ade234126234bd1bf4e456cbcce871d38fec057c427da2d6ec23e43f1492       1
    ce1aff56e202723cae49662303eecaf4a75b68c6fe94528e8538c89910a6f7d6       1
    480ceda31cd9c15760b6b98eba40a8471ea326971d7148ad4b09c30ffd2940df       1
    a8c48628522d0d654ec3f55b53ba8ee52a47add25e1046384b95764c35528578       1
    692e09b8654189dd3e3baa14ca1db1b5f0c235d36bd659fa46a2b1d99723b03f       1
    53b7f18f053d2b46aef4a1f393239728eda441dcf2a7580df8ccca91cb16d5e4       1
    d72a8d3b52d96d3cda947e371b47ff34dc1177a684a70ea6e68cf4bebe8a9528       1
    6124cfc9f195c493e59ec49c802f9d0407eb41dc61ff0e5bfcdac71432b921cb       1
    477897bba23ef82a97148c9eb6557d4e5941b961ff19793a46f69d06c0ea3ee1       1
    9c754aff1496ddf8ff375b88ab5b8430159069fae6e322d8bbb0942ad20e53aa       1
    b1eddde36073825db44b8cab0b529a0c39390ac213c1a3d3fa186f6a8bcc462a       1
    b223717b0d436438d3f88fb7e6b371242c4b0c8166c9846452b4e7a030ae8fb7       1
    8c6e84ac07da0274a49635f773fce9631b2782fe0841677d4e812620be2e575f       1
    d458df57e59cfcb993e685cbf18a4b7347e5ab65cda51148a6997d6feba909de       1
    dd2922eb4e54387f338b09d4bd62d9bd8cf1518867370577912efe73a17ae501       1
    Name: anonymized_user_id, Length: 10338, dtype: int64
'''


df['merchant_id'].value_counts()
'''
    000420291799       8917
    878918000156182    8202
    498750000002171    7495
    248726000110085    7186
    007274000108778    7067
    0030102489730      5399
    847566000156182    4623
    0030200305177      4295
    0030205550793      4098
    000980200178997    3730
    211366000053360    3055
    239732000192776    2752
    112150000108778    2714
    OBJGWI7KBUTWTG9    2557
    026239000593095    2432
    04844663           2357
    064310000557165    2275
    41728065704        2163
    000980200177999    2094
    498750000019399    1888
    0030200899153      1874
    000174030085990    1791
    0030102489748      1676
    0030200925040      1632
    000980020285998    1557
    0030207468275      1550
    151177000144509    1546
    849420000027792    1368
    248730000110085    1360
    030000041705001    1289
    ...
    030000087543031       1
    040080043926001       1
    040080041163102       1
    810000034260001       1
    40931055704           1
    132993                1
    84429892              1
    42017115704           1
    4445012062992         1
    0030401825710         1
    040080092756003       1
    0030409312885         1
    8015436275            1
    0030206304018         1
    0030221862552         1
    000423300210          1
    000010002492003       1
    0030407003031         1
    030000026987024       1
    0030421834015         1
    0014585491            1
    0030407667728         1
    810000070853001       1
    0030404833240         1
    000421847202          1
    0030408741852         1
    030000053356121       1
    040080092813006       1
    000390310             1
    0030206574776         1
    Name: merchant_id, Length: 101983, dtype: int64
'''


df['merchant_name'].value_counts()
'''
    Tim Hortons               32904
    Mcdonald's                23603
    Amazon                    16034
    Apl* Itunes.Com/Bill      10639
    Google                    10458
    Shoppers Drug Mart         9300
    Starbucks                  9128
    Wal-Mart                   9016
    Petro-Canada               7073
    Esso                       5491
    Dollarama                  5326
    Subway                     5314
    Safeway                    4360
    Skipthedishes.Com          4142
    Shell                      3990
    7 Eleven Store             3934
    Macs Conv. Store           3758
    Save On Foods              3638
    Impark                     3613
    LCBO                       3234
    Car2go                     3024
    Sobeys                     2951
    Tim Horton's               2861
    Netflix.Com                2807
    Lyft                       2570
    Real Cdn Superstore        2535
    City Of Van Paybyphone     2483
    Starbucks Card Reload      2432
    Coinbase Uk                2357
    A&W                        2133
    ...
    Sea Witch                     1
    The Local General Stor        1
    Uber *Trip Q2bke              1
    Barakat Downtown              1
    Computation                   1
    Departure Duty Free           1
    Jai Ho Restaraunt Ltd         1
    Uber Akpen Help.Uber.C        1
    35110697 Shopify Com C        1
    Mccoos Too                    1
    Pearce Hardware (1977)        1
    Uber Eats Lh534 Help.U        1
    Vallarta'S Mexican Res        1
    Buns Burger Shop Vsj          1
    Uber Trip Ulsfo Help.U        1
    Queen'S Isabel Bader C        1
    Subway 12544                  1
    Ezetop *28045191              1
    Laser Quest Regina            1
    Wix.Com*146470751             1
    Wu *8981133395                1
    Uber Trip 5pfrt Help.U        1
    Town And Country Food         1
    Mattamyathleticcentre         1
    Rob'S Bar & Grill Ltd         1
    Uber Eats Nsqxl Help.U        1
    Viva-Emmanoyhl Dhm A          1
    Pizza Nova 057                1
    Wpy*Help Esther Conque        1
    Rass Joehowe Hfx Tms          1
    Name: merchant_name, Length: 71920, dtype: int64
'''

                                                        
                                                        
df['merchant_name2'].value_counts()
'''
    Tim Hortons               35765
    Mcdonalds                 23772
    Amazon                    16034
    Apl* Itunes.Com/Bill      10639
    Google                    10458
    Shoppers Drug Mart        10313
    Starbucks                  9128
    Wal-Mart                   9016
    Petro-Canada               7073
    Esso                       5491
    Dollarama                  5326
    Subway                     5314
    Safeway                    4360
    Skipthedishes.Com          4142
    Shell                      3990
    7 Eleven Store             3934
    Macs Conv. Store           3758
    Save On Foods              3638
    Impark                     3613
    LCBO                       3234
    Car2go                     3024
    Sobeys                     2951
    Netflix.Com                2807
    Lyft                       2570
    Real Cdn Superstore        2535
    City Of Van Paybyphone     2483
    Starbucks Card Reload      2432
    Coinbase Uk                2357
    A&W                        2133
    Metro                      2086
    ...
    Departure Duty Free           1
    Computation                   1
    Paypal *Dbelectrica           1
    Uber Wpupt Help.Uber.C        1
    Uber B33l5 Help.Uber.C        1
    Gip-Loveyourmelo              1
    48268274 Shopify.Com/C        1
    Uber Eats 75bj5 Help.U        1
    Swamis                        1
    Uber *Trip Cl4ug              1
    Mulock Happy Mart             1
    BonnyS Taxi B 122             1
    Sq *Crackerztech.Com G        1
    Uber Eats 3v5kq Help.U        1
    Mop Donuts Cbd Pty Ltd        1
    Las Olas Hotel                1
    Uber Trip Ytbz2 Help.U        1
    Uber Trip 7x724 Help.U        1
    Uber Trip X3plm Help.U        1
    National Music Centre         1
    Mugg & Bean Bedford Ce        1
    Fabindia .                    1
    Teriyaki Exp J028 Qff         1
    A&W Osoyoos                   1
    Zalathai Thai Restaura        1
    Uber Trip Vs4vl Help.U        1
    Docusign Cad                  1
    Ftp*Fishing                   1
    Bqe Core Mgr-Mthly            1
    Rass Joehowe Hfx Tms          1
    Name: merchant_name2, Length: 71907, dtype: int64
    '''
                                                        
df['mcc_description'].value_counts()
'''
    Fast Food Restaurants                                                                                    119044
    Eating places and Restaurants                                                                             59456
    Grocery Stores, Supermarkets                                                                              48366
    Service Stations ( with or without ancillary services)                                                    32469
    Misc. Food Stores â Convenience Stores and Specialty Markets                                            20373
    Computer Network Services                                                                                 19273
    Taxicabs and Limousines                                                                                   17057
    Drug Stores and Pharmacies                                                                                16306
    Discount Stores                                                                                           16068
    Business Services, Not Elsewhere Classified                                                               14455
    Automobile Parking Lots and Garages                                                                       12800
    Package Stores â Beer, Wine, and Liquor                                                                 12161
    Record Shops                                                                                              10802
    Advertising Services                                                                                       5733
    Transportation Services, Not elsewhere classified)                                                         5630
    Telecommunications Equipment including telephone sales                                                     5471
    Local/Suburban Commuter Passenger Transportation â Railroads, Feries, Local Water Transportation.        5290
    Computer Software Stores                                                                                   5270
    Variety Stores                                                                                             5149
    Direct Marketing â Continuity/Subscription Merchant                                                      4708
    Miscellaneous and Specialty Retail Stores                                                                  4355
    Home Supply Warehouse Stores                                                                               3968
    Drinking Places (Alcoholic Beverages), Bars, Taverns, Cocktail lounges, Nightclubs and Discotheques        3890
    Government Services ( Not Elsewhere Classified)                                                            3863
    Professional Services ( Not Elsewhere Defined)                                                             3849
    Motion Picture Theaters                                                                                    3790
    Menâs and Womenâs Clothing Stores                                                                      3589
    Family Clothing Stores                                                                                     3574
    Pet Shops, Pet Foods, and Supplies Stores                                                                  3157
    Financial Institutions â Manual Cash Disbursements                                                       3113
    ...
    AIR-INDIA                                                                                                     2
    AUBERGE DES GOVERNEURS                                                                                        2
    LOEWS HOTELS                                                                                                  2
    ICELANDAIR                                                                                                    1
    SABENA                                                                                                        1
    Home2Suites                                                                                                   1
    LANCHILE                                                                                                      1
    TACA INTERNATIONAL                                                                                            1
    THAI AIRWAYS                                                                                                  1
    LOT (POLAND)                                                                                                  1
    RED ROOK INNS                                                                                                 1
    SOFITEL HOTELS                                                                                                1
    Betting (including Lottery Tickets, Casino Gaming Chips, Off-track Betting and Wagers at Race Tracks)         1
    THRIFTY RENT-A-CAR                                                                                            1
    AMERICANA HOTELS                                                                                              1
    Antique Reproductions                                                                                         1
    Electrical Contractors                                                                                        1
    SOUTH AFRICAN AIRWAYS                                                                                         1
    AUSTRAINLIAN AIRLINES                                                                                         1
    HAWAIIAN AIR                                                                                                  1
    Exterminating and Disinfecting Services                                                                       1
    FOUR SEASONS HOTELS                                                                                           1
    Intra â Government Transactions                                                                             1
    Furriers and Fur Shops                                                                                        1
    SONESTA HOTELS                                                                                                1
    SINGAPORE AIRLINES                                                                                            1
    CSA                                                                                                           1
    TAJ HOTELS INTERNATIONAL                                                                                      1
    JOURNEYâS END MOTLS                                                                                         1
    LACSA (COSTA RICA)                                                                                            1
    Name: mcc_description, Length: 377, dtype: int64
'''

df["count"] = df.groupby(["merchant_id", "merchant_name"])["index"].transform("count")




mcc_desrp = df['mcc_description'].value_counts()
merch_name = df['merchant_name'].value_counts()

sns.stripplot(x='mcc_description', y='mcc_count', data=df)
plt.show()




###########    TERRIBLE   R   Code
                                                        
# For R (ie transfer python stuff to csv so that it can be imported into R:
import csv
#out = csv.writer(open("wtd_edgelist.csv","w"), delimiter=',')
#out.writerow(wtd_edgelist)
with open('wtd_edge_list.csv','w+') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['endnode','endnode','weight'])
    for row in wtd_edgelist:
        csv_out.writerow(row)
                                                        
#Actual R code:
library(R.matlab)
library(readxl)
library(igraph)
library(foreign)
G <- graph.data.frame(wtd_edgelist_all,directed=FALSE);
A <- as_adjacency_matrix(G,type="both",names=TRUE,sparse=FALSE,attr="weight");
View(A)
diag(A) <- 0
G2 = graph_from_adjacency_matrix(A, weighted=TRUE, mode = "undirected")
plot.igraph(G2,vertex.size=2,main="All data")










