# Insight
Insight Data Science project - Community Detection

My Insight Data Science project was a consulting project for a Candian bank. I was asked to do exploratory network analysis on transactional data, focusing on merchants (rather than the users) and travel transactions. 

The initial data (masked_transactions_with_year_of_birth.csv) consisted of over 550,000 transactional records and had the following data fields:

      anonymized_user_id:	A unique user identifier that has been masked.
      year_of_birth: user's year of birth
      transaction_id:  	A unique transaction identifier that has been masked.
      authorization_timestamp:	The time at which the purchase was authorized by the user; i.e. the time at which the purchase was made at the merchant terminal.
      settle_timestamp:	The time at which the purchase was finalized; i.e. A gas station purchase settling for the final amount
      transaction_code:	Type of transaction. Description of codes:
             0	None
             1	Authorization
             2	Refund
             3	Force Post Settle-Could not match to an Auth
             4	Purchase with PIN
             5	Reversal-Credit Account
             6	ATM Withdrawal
             7	Debit or Credit Adjustment
             8	Pre-auth completion
      amount:	Dollar value of the transaction
      merchant_id:	A unique identifier fro each merchant
      merchant_details:	Text description of the merchant
      merchant_name:	Normalized version of the merchant details
      mcc_code:	Industry standard Merchant Category Code
      mcc_description:	Text describing the MCC code
      koho_category:	Category assgined by Koho's internal system


At a quick glance, here are the other files in the repository & what they are. (For more information about the steps of this project, please see below.)

df_travel: the final set of travel transactions that was used for the project

edgelist_travel_mcccode_usrcount_withnames_asdf.csv: final edge list from merchants being defined my mcc code & the weight defined as usercount. Since the merchants are simply numbers, I also added their mcc_description for informational purposes.

frequenttraveler_project.py: all of my code. (Includes some R code for making visualizations of the networks at the bottom of the file)

nodes_travel_mcccode_usrcount.csv: final node list & their attributes. Attributes include:
        
        mcc_description : (see description above)
        koho_category: (see description above)
        travel_category: classification of travel type. includes: Air Carrier, Hotel, Travel Agency, Tourist Attraction, Cruiseline, Busline
        community: specifies which community the merchant belongs to as determined by the Louvain Community Detection Algorithm
        degree:  the number of of other merchants a merchant is connected to
        betweenness_centrality: measures the centrality of the network based on shortest paths
        closeness_centrality: measures the mean distance from a node to other nodes
        eigenvector_centrality: measures the influence of that node in the network
        neighbors: list of the other nodes that node connected to
          
travel_category.csv: list of mcc codes & descriptions for which a travel_category was manually added based on description


The first step was data prep, which included restricting the transactions to "transaction_code == 1", as the focus was on authorized transactions. Next was some data clean up, which produced "df_all.csv". From this file, the remainder of the work completed, such as removing non-ASCII characters, the addition of some fields, etc. (Although it was not utilized in the analysis, df_all could be split up into weekday data (for transactions that occurred Monday - Friday) and weekend data (transactions occuring on Saturday and Sunday); see "df_weekday" and "df_weekend". And finally, from df_all, the data was restricted to travel transactions. 

To create a network, the merchants from the travel data became the nodes of the network. How exactly one is to choose to define the merchants is an input. For my analysis, I chose to make the nodes/merchants the mcc_code. The edges of the network are weighted and are defined as the number of customers who made transactions at both endnode merchants, called "usercount". This is another input, and I used usercount in my analysis. The other option for weight is "joshwt" in which each user contributes to an edge weight as follows: max(number of purchases made at merchant A (endnode1), number of purchases made at merchant B (endnode2)) / (absolute difference of the number of purchases made at merchant A (endnode1) and the number of purchases made at merchant B (endnode2)).

From the edgelist, I used NetworkX to create the graph that I could perform the analysis on. The analysis began with generating node attributes: Louvain Community detection, degree, betweenness centrality, closeness centrality, eigenvector centrality, & list of neighbors.

