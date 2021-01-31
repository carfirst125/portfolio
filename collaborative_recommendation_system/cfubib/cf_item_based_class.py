##############################################################
# Class: Collaborative Filtering - Item-based
#------------------------------------------------------------
# File name: cf_item_based_class.py
# Author:  Nhan Thanh Ngo


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import re
import datetime
import random
import pickle
import yaml
import math
import os
import sys
import shutil
import yaml
 
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree 
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
 

class cf_item_based():
    
  def __init__(self, df, root_path, name='item_based', item_colname='items', debug=False):
    
    self.N = 0
    self.CUS_COUNT = 0
    self.PERIOD    = 1000
    
    self.debug = debug
    self.df = df  
    self.root_path = root_path
    self.name = name
    self.temppath = ''
    self.outpath = ''
    self.item_colname = item_colname

    self.similar_items_col1 = ['customer_id','item_a','item_b','qty_a','qty_b']
    self.similar_items_col2 = ['item_a','item_b','qty_a','qty_b','customer_count']
    
    # create folder when declare object
    self.create_folder()    
    self.parameters = 'parameters.yaml'
    self.update_parameter()
    self.indicators = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'nineth', 'tenth', 'eleven','twelfth','thirthteen','forthteen','fifthteen']
  
  ############################################################################
  # Function: update_parameter(self)
  # Description:
  #   update parameter from parameters.yaml file

  def update_parameter(self):
    parameter_dict = {}
    with open(r'./'+self.parameters) as file:
      # The FullLoader parameter handles the conversion from YAML
      # scalar values to Python the dictionary format
      parameter_dict = yaml.load(file, Loader=yaml.FullLoader)
  
    for key in parameter_dict:
      if key == 'CLUSTER_DIV_PROCESS_THRESHOLD':
        self.CLUSTER_DIV_PROCESS_THRESHOLD = parameter_dict[key]
      elif key == 'NUM_NEARBY':
        self.NUM_NEARBY = parameter_dict[key]
      elif key == 'NUM_FAV_ITEMS':
        self.NUM_FAV_ITEMS = parameter_dict[key]
      elif key == 'NUM_BASE_RECOMMEND':
        self.NUM_BASE_RECOMMEND = parameter_dict[key]
      elif key == 'NUM_SOH_RECOMMEND':
        self.NUM_SOH_RECOMMEND = parameter_dict[key]
      elif key == 'NUM_FAV_SIZE':
        self.NUM_FAV_SIZE = parameter_dict[key] 
      elif key == 'NUM_TOP':
        self.NUM_TOP = parameter_dict[key]
    return True        
	
  ############################################################################
  # Function: create_folder(self)
  # Description:
  #   create neccessary folder 
  
  def create_folder(self):
    #------------------
    self.homepath = self.root_path+"/"+self.name
    if os.path.exists(self.homepath):
      print ("\'{}\' is already EXISTED!".format(self.homepath))
    else:
      os.mkdir(self.homepath)
      print ("\'{}\' is CREATED!".format(self.homepath))
      
    #------------------  
    self.temppath = self.root_path+"/"+self.name+"/temp"
    if os.path.exists(self.temppath):
      print ("\'{}\' is already EXISTED!".format(self.temppath))
    else:
      os.mkdir(self.temppath)
      print ("\'{}\' is CREATED!".format(self.temppath))
      
    #------------------  
    self.outpath = self.root_path+"/"+self.name+"/output"
    if os.path.exists(self.outpath):
      print ("\'{}\' is already EXISTED!".format(self.outpath))
    else:
      os.mkdir(self.outpath)
      print ("\'{}\' is CREATED!".format(self.outpath))  
    return True
  ###################################################
  # BACKUP

  def previous_recommendation_file_backup (self):
  
    source_path = self.outpath+"/OUTPUT_item_based_recommend.csv"
    destination_path = self.outpath+"/OUTPUT_item_based_recommend_backup.csv"
    
    if os.path.exists(source_path):
      print ("\'{}\' is currently EXISTED! Backup this file...".format(source_path))
      shutil.copyfile(source_path, destination_path)      
    return True
  ###################################################
  # DEBUG
 
  def print_debug (self, strcont):
    if self.debug:
      print(strcont)  
      with open(self.homepath+"/debug.log", "a") as logfile:
        logfile.write("\n"+strcont)
        
        
  ###########################################################################
  # Function: return_similar_item_pair_by_cus (row):
  #     sort_values of dataframe, and get the top values of them
  #     return the name of top venues 
  # Input: 
  #     row: Series (with index is items name, and value is rate of quantity/total)
  # Output:
  #     similar item pair array (item1, item2)
  #

  def return_similar_item_pair_by_cus(self, row):
  
    row_sorted = row.sort_values(ascending=False)

    df_row = pd.DataFrame(data=row_sorted).reset_index()
    df_row.columns=['items','rate']
    df_row['rate_sh1'] = df_row['rate'].shift(-1)
    
    df_row['similar'] = df_row.apply(lambda x: True if (x['rate_sh1'] >= 0.2) & ((x['rate'] - x['rate_sh1'])/(x['rate_sh1']+0.000001)<=0.3) else False, axis=1) # threshold for similar
    df_row['items_sh1'] = df_row['items'].shift(-1)
    
    similar_item_index = df_row[df_row.similar == True].index 
    if len(similar_item_index) == 0:
      similar_item_pair = [] 
    else:
      similar_item_pair = df_row.loc[df_row.index==similar_item_index[0], ['items','items_sh1']].values[0]
    return similar_item_pair


  ############################################################################
  # Function: cf_similar_item_pair_stats(self, df_pivot, group='')
  # Description:
  #   sweep df of group customer to find similar pair of all customer
  #   similar pair = (quantity of item/total_qty_of_cus >= 0.2) & (quantity of each item >= 2) & (diff <= 30%)
  # Inputs:
  #   - df_pivot: customer_id, itemA, itemB (pivot table)
  # Outputs:
  #   - dataframe [customer_id, itemA, itemB, qtyA, qtyB] --> groupby[itemA,itemB] --> [itemA, itemB, qtyA, qtyB, customer_count]
  #

  def cf_similar_item_pair_stats(self, df_pivot, group=''):
    
    # change absolute value of quantity to rate
    df_items = df_pivot[df_pivot.columns.values[1:]]    
    df_items['total'] = df_items.sum(axis=1)
    df_items = df_items[df_items.columns.values[:-1]].div(df_items['total'], axis=0) 
    
    ########################################
    # find similar items for each customer
    df_similar_items = pd.DataFrame(columns=self.similar_items_col1)
    df_similar_items['customer_id'] = df_pivot['customer_id']
    
    count = 0    
    for ind, cus_id in zip(df_pivot.index.values, df_pivot.customer_id.unique()):
      count+=1
      #print("[{}] item_based: Similar item-pair exploration for customer_id {}".format(self.CUS_COUNT,cus_id))
      if count%self.PERIOD==0:  
        print("[item_based] Similar item-pair exploration, completed for {} users".format(count))       
      similar_item_pair = self.return_similar_item_pair_by_cus(df_items.iloc[ind,:])
      
      if len(similar_item_pair) == 0:
        continue
      else:
        similar_item_pair = np.sort(similar_item_pair) # to arrange items name in same rule of order

        similar_qty_pair = df_pivot.loc[df_pivot.index==ind,similar_item_pair].values[0]
        if similar_qty_pair[0]<2:
          continue
        else:     
          df_similar_items.loc[df_similar_items.customer_id==cus_id,self.similar_items_col1[1:]] = np.concatenate([similar_item_pair,similar_qty_pair])
       
    ########################################
    # save file similar items by customer
    df_similar_items = df_similar_items[~df_similar_items[self.similar_items_col1[1]].isnull()]
    df_similar_items.to_csv(self.temppath+"/cus_similar_item_pair_group"+str(group)+".csv", index=False)
    
    ######################################################
    # dump similar item pair file and rated customer count
    if len(df_similar_items) == 0:
      df_similar_items_summary = pd.DataFrame(columns=self.similar_items_col2)
    else:
      
      df_similar_items_summary = df_similar_items.groupby(self.similar_items_col1[1:3]).agg({self.similar_items_col1[3]:'sum', self.similar_items_col1[4]:'sum', self.similar_items_col1[0]:'count'})
      df_similar_items_summary = df_similar_items_summary.reset_index()    
      df_similar_items_summary.columns = self.similar_items_col2
    
    df_similar_items_summary.to_csv(self.temppath+"/similar_item_pair_summary_group"+str(group)+".csv", index=False)

    #return df_similar_items, df_similar_items_summary
    return True  

  ############################################################################
  # Function: cf_similar_item_pair_stats_v2(self, df, group='')
  # Description:
  #   sweep df of group customer to find similar pair of all customer
  #   similar pair =  (quantity of item/total_qty_of_cus >= 0.2) & (quantity of each item >= 2) & (diff <= 30%)
  # Inputs:
  #   - df: customer_id, items, quantity
  # Outputs:
  #   - dataframe [customer_id, itemA, itemB, qtyA, qtyB] --> groupby[itemA,itemB] --> [item_a,item_b,qty_a,qty_b,customer_count]
  #
  
  def cf_similar_item_pair_stats_v2(self, group=''):    

    df_func = self.df[['customer_id','items','quantity']]
    df_func = df_func.sort_values('customer_id')    
    df_func = df_func.groupby(['customer_id','items'])['quantity'].sum().reset_index()
    
    # process to get df_funnc with items only having quantity/total_qty >= 0.2 
    col = df_func.columns.values
    df2 = df_func.groupby('customer_id')['quantity'].sum().reset_index()
    df2 = df2.rename({'quantity':'total_qty'},axis='columns')
    df_func = pd.merge(df_func,df2,on='customer_id',how='left')
    df_func['per_total'] = df_func['quantity']/df_func['total_qty']
    df_func = df_func[df_func.per_total>=0.2]
    df_func = df_func[col]

    # left join with df_func with itself
    df_combine = pd.merge(df_func, df_func, on='customer_id', how='left')
    df_combine = df_combine[(df_combine['items_x']!=df_combine['items_y'])]                                  # remove rows with same item values
    df_combine['min_qty'] = df_combine[['quantity_x','quantity_y']].min(axis=1)                              # new column of min(quantity_x,quantity_y)
    df_combine['per_relate'] = (df_combine['quantity_x'] - df_combine['quantity_y']) / df_combine['min_qty'] # new column percent btw 2 items
    df_combine['per_relate'] = df_combine['per_relate'].abs()                                                # get positive value
    df_combine = df_combine[(df_combine['per_relate'] <=0.3)&(df_combine['min_qty']>=2)]                     # get rows pass CONDITION
    df_combine = df_combine.sort_values(['customer_id','min_qty'], ascending=[True, False])                  # sort
    df_combine = df_combine.drop_duplicates(subset=['customer_id'], keep='first')                            # get BEST
    df_combine = df_combine[['customer_id','items_x','items_y','quantity_x','quantity_y']]                  

    df_qty_cus = df_combine.groupby(['items_x','items_y'])['customer_id'].nunique().reset_index()            # get customer_count for each item-pair
    df_qty_A_B = df_combine.groupby(['items_x','items_y'])[['quantity_x','quantity_y']].sum().reset_index()  # get total quantity of each item-pair
    df_similar_items_summary = pd.merge(df_qty_cus, df_qty_A_B, on=['items_x','items_y'], how='left')        # left join for table of [item_a,item_b,qty_a,qty_b,customer_count]
 
    # change column order, and change column name
    column = ['items_x', 'items_y', 'quantity_x', 'quantity_y', 'customer_id']
    df_similar_items_summary = df_similar_items_summary[column]  
    df_similar_items_summary.columns = ['item_a', 'item_b', 'qty_a', 'qty_b', 'customer_count']

    print("[item_based] Similar item-pair exploration, completed for {} users".format(df_func.customer_id.nunique()))

    df_similar_items_summary.to_csv(self.temppath+"/similar_item_pair_summary_group"+str(group)+".csv", index=False)
    
      
  ############################################################################
  # Function:   cf_similar_item_pair_summary(self, num_group):
  # Description:
  #   combine all result of each group into final file.
  # Inputs:
  #   - num_group: Num of group for sweep file
  # Outputs:
  #   - df_cus_similar_item_pair [customer_id, itemA, itemB, qtyA, qtyB] --> groupby[itemA,itemB] --> df_similar_items_summary [itemA, itemB, qtyA, qtyB, customer_count]
  #
  
  def cf_similar_item_pair_summary(self, num_group):
       
    # summarize for final result
    df_similar_items = pd.DataFrame(columns=self.similar_items_col1)
    df_similar_items_summary = pd.DataFrame(columns=self.similar_items_col2)
    for groupi in np.arange(num_group):
      df_similar_items_summary_i = pd.read_csv(self.temppath+"/similar_item_pair_summary_group"+str(groupi)+".csv", index_col=None)
      
      df_similar_items_summary = pd.concat([df_similar_items_summary,df_similar_items_summary_i])

    df_similar_items_summary = df_similar_items_summary.groupby(self.similar_items_col2[0:2]).agg({self.similar_items_col2[2]:'sum', self.similar_items_col2[3]:'sum', self.similar_items_col2[4]:'sum'})
    df_similar_items_summary = df_similar_items_summary.reset_index()
    df_similar_items_summary.to_csv(self.outpath+"/similar_item_pair_summary_group.csv", index=False)  

    return True
  
  ###########################################################################
  # Function: return_most_favorite_rm_noqty(row, num_top_favourite)
  #     sort_values of dataframe, and get the top values of them
  #     return the name of top venues 
  #

  def return_most_favorite_rm_noqty(self, row):

    row_sorted = row.sort_values(ascending=False)
    category_msb = row_sorted.index.values[0:self.NUM_TOP]
    qty_msb = row_sorted.values[0:self.NUM_TOP]
  
    qty_msb_bool = [True if i > 0 else False for i in qty_msb]
    category_msb_purchased = [ j if i else 0 for i,j in zip(qty_msb_bool, category_msb)]

    return [category_msb_purchased, qty_msb]
    
  ###########################################################################
  # Function: cus_similar_item_pair_lookup_and_return(self, item_array)
  # Description:
  #    sweep each items in array, looking similar item pair table get information
  # Input:
  #    item_array: array of item names
  # Output:
  #    get the list of item (self.NUM_BASE_RECOMMEND = 15)
  #    Mark score: 1-fav = N, ... ,N-fav = 1. 
  #    Score of item pair = sum of marks, if score is equal, customer_count will be considered.
  #
  
  def cus_similar_item_pair_lookup_and_return(self, item_array):

    # read similar item-pair list file for lookup
    df_item_pairs = pd.read_csv(self.outpath+"/similar_item_pair_summary_group.csv", index_col=None) 
    df_cus_item_pair = pd.DataFrame(columns = np.concatenate([self.similar_items_col2,['score']])) #score used to assess how items is suitable to user.
    
    #sweep each item in favorite array list, and lookup the item-pair
    count = len(item_array)+1
    for item_i in item_array:
      count-=1
      # user fav-item matches with item_a 
      df_temp = df_item_pairs[df_item_pairs[self.similar_items_col2[0]]==item_i]
      df_temp['score'] = count      
      df_cus_item_pair = pd.concat([df_cus_item_pair,df_temp])   
      
      # user fav-item matches with item_b 
      df_temp = df_item_pairs[df_item_pairs[self.similar_items_col2[1]]==item_i]
      df_temp['score'] = count     
      df_cus_item_pair = pd.concat([df_cus_item_pair,df_temp])
    
    # groupby ['item_a','item_b']
    df_cus_item_pair = df_cus_item_pair.groupby(self.similar_items_col2[:2]).agg({self.similar_items_col2[2]:'sum', self.similar_items_col2[3]:'sum', self.similar_items_col2[4]:'sum', 'score':'sum'})
    df_cus_item_pair = df_cus_item_pair.reset_index()

    # compute average quantity of pair purchased by a user (aqpu)
    df_cus_item_pair['aqpu'] = ((df_cus_item_pair['qty_a']+df_cus_item_pair['qty_b'])/2)/df_cus_item_pair['customer_count']
    df_cus_item_pair = df_cus_item_pair.sort_values(by=['score','customer_count','aqpu'], ascending=False).reset_index(drop=True)
    
    df_cus_item_pair.to_csv(self.temppath+"/cus_similar_item_pair_lookup.csv", index=False)   # for review only   
        
    return df_cus_item_pair

  ###########################################################################
  # Function: dump_cus_recommend_list(self, df_cus_item_pair, cus_top_purchased)    
  # Description:
  #    sweep each items in array, looking similar item pair table get information
  # Input:
  #    df_cus_item_pair: [item_a, item_b, qty_a, qty_b, score, customer_count, aqpu]
  #    cus_top_purchased: favourite item list [fav-1, fav-2,...,fav5]
  # Output:
  #    get the list of item (self.NUM_BASE_RECOMMEND = 15)
  #    Mark score: 1-fav = N, ... ,N-fav = 1. 
  #    Score of item pair = sum of marks, if score is equal, customer_count will be considered.
  #
  
  def dump_cus_recommend_list(self, df_cus_item_pair, cus_top_purchased):    
  
    # get customer recommend based on customer similar item-pair which was found
    # df_cus_item_pair is sorted by [score, customer_count, aqpu]
    cus_recommend_list = []
    for ind in df_cus_item_pair.index.values:
      # get each item-pair name, check is it in cus_top_purchased, if not, save it to recommend
      item_pair_arr = df_cus_item_pair.loc[ind,self.similar_items_col2[:2]].values
      if item_pair_arr[0] not in cus_top_purchased:
        cus_recommend_list.append(item_pair_arr[0])
      elif item_pair_arr[1] not in cus_top_purchased:
        cus_recommend_list.append(item_pair_arr[1])

    # unique items in recommend list
    cus_recommend_list_unique, indexx = np.unique(cus_recommend_list, return_index=True)
    cus_recommend_list = cus_recommend_list_unique[np.argsort(indexx)] # rearrange by original order

    # format length of recommend_list to parameter set.
    if len(cus_recommend_list) < self.NUM_BASE_RECOMMEND:
      cus_recommend_list = list(cus_recommend_list) + list(np.zeros(self.NUM_BASE_RECOMMEND-len(cus_recommend_list),dtype=int))
    else:
      cus_recommend_list = cus_recommend_list[:self.NUM_BASE_RECOMMEND]
      
    return cus_recommend_list[:self.NUM_BASE_RECOMMEND]
    
  ############################################################################
  # Function: cf_customer_recommend_main(self,group='')
  # Description:
  #     sweep each customer_id, lookup item-pair for recommendation list
  # Inputs:
  #   - group: mark group as name
  # Outputs:
  #   - df_cus_similar_item_pair [customer_id, itemA, itemB, qtyA, qtyB] --> groupby[itemA,itemB] --> df_similar_items_summary [itemA, itemB, qtyA, qtyB, customer_count]
  #
  
  def cf_customer_recommend_main(self, group=''):
    
    # user favorite table
    df_fav = pd.read_csv(self.root_path+"/output/OUTPUT_customer_favorite_insights"+str(group)+".csv", index_col=None) 
    df_fav = df_fav[df_fav.columns.values[:self.NUM_FAV_ITEMS+1]]
    df_fav.customer_id = df_fav.customer_id.astype(str)
    
    self.df.customer_id = self.df.customer_id.astype(str)        

    ###############################################
    # for recommend column name
  
    column_ib = ['customer_id']
    for i in np.arange(self.NUM_BASE_RECOMMEND):    
      column_ib.append('{}_cfib_recommend'.format(self.indicators[i]))
  
    df_item_based_recommend = pd.DataFrame(columns=column_ib)  
    df_item_based_recommend['customer_id'] = self.df['customer_id'].unique()
    
    count = 0
    for cus_id in self.df.customer_id.unique():
      count += 1
      #print("[{}] item_based: Get recommendation items for customer_id {}".format(count,cus_id))
      if count%self.PERIOD==0:  
        print("[item_based] Get recommendation items, completed for {} users".format(count)) 
        
      df_cus_i = self.df[self.df.customer_id==cus_id]
      df_cus_i_pivot = pd.pivot_table(df_cus_i, values='quantity', index=['customer_id'], columns=[self.item_colname], aggfunc=np.sum, fill_value=0)    
      
      # get customer favorite items
      cus_top_purchased = df_fav[df_fav.customer_id == cus_id].values[0][1:]

      # lookup item_pair table      
      df_cus_item_pair = self.cus_similar_item_pair_lookup_and_return(cus_top_purchased)    
      #print("[2] df_cus_item_pair = {}".format(df_cus_item_pair))      
      
      # get recommend item list      
      cus_recommend_list = self.dump_cus_recommend_list(df_cus_item_pair, cus_top_purchased)      
      #print("[3] cus_recommend_list = {}".format(cus_recommend_list))
      
      # append result to 
      df_item_based_recommend.loc[df_item_based_recommend.customer_id==cus_id,1:] = cus_recommend_list
      
    df_item_based_recommend.to_csv(self.outpath+"/OUTPUT_item_based_recommend"+str(group)+".csv", index=False)   
    
    return True

    
  ############################################################################
  # Function: ib_similar_item_lookup(self, df, pair_item, group)
  # Description:
  #     sweep each customer_id, lookup item-pair for recommendation list
  # Inputs:
  #   - df: ['customer_id', 'items', 'quantity'] 
  #   - pair_item: input (DataFrame) columns=['item_x','item_y','qty_cus_vote','qty_x','qty_y']
  #   - group: mark group as name
  # Outputs:
  #   - df_cus_similar_item_pair [customer_id, itemA, itemB, qtyA, qtyB] --> groupby[itemA,itemB] --> df_similar_items_summary [itemA, itemB, qtyA, qtyB, customer_count]
  #
  

  def ib_similar_item_lookup(self, df, pair_item, MAX_NUM_FAV_ITEMS,group=''):

    print('[ib_similar_item_lookup()] Initiating...')
    
    #df['indexing'] = df.sort_values(['customer_id','quantity'], ascending=[True, False]).groupby('customer_id').cumcount()+1
    df = df.sort_values(['customer_id','quantity'], ascending=[True, False])
    df['indexing'] = df.groupby('customer_id').cumcount()+1
          
    df = df[df['indexing']<=MAX_NUM_FAV_ITEMS]  
  
    # add score first fav, second fav, third fav
    df['score'] = df.groupby(['customer_id'])['quantity'].rank(method='max').astype('int')
   
    cateX = pair_item[['item_x','item_y','qty_cus_vote','qty_y']]
    cateX.rename(columns={'item_x':'items'}, inplace=True)
    cateX = cateX.sort_values(['items','qty_cus_vote'], ascending=[True, False])
    cateX['indexing'] = cateX.groupby('items').cumcount()
    cateX = cateX[cateX['indexing']<=20]

    cateY = pair_item[['item_y','item_x','qty_cus_vote','qty_x']]
    cateY.rename(columns={'item_y':'items'}, inplace=True)
    cateY = cateY.sort_values(['items','qty_cus_vote'], ascending=[True, False])
    cateY['indexing'] = cateY.groupby('items').cumcount()
    cateY = cateY[cateY['indexing']<=20]

    # left join on item to find pair
    # left join 2 times because have to look up item X and item Y in 2-way
    lookup_X = pd.merge(df, cateX, on='items', how='left')
    lookup_Y = pd.merge(df, cateY, on='items', how='left')

    lookup_X.rename(columns={'item_y':'item_lookup','qty_y':'qty_lookup'}, inplace=True)
    lookup_Y.rename(columns={'item_x':'item_lookup','qty_x':'qty_lookup'}, inplace=True)

    lookup_X = lookup_X[lookup_X['item_lookup'].notnull()]
    lookup_Y = lookup_Y[lookup_Y['item_lookup'].notnull()]
    lookup_X['item_lookup'] = lookup_X['item_lookup'].astype('str')#.astype('int')
    lookup_Y['item_lookup'] = lookup_Y['item_lookup'].astype('str')#.astype('int')

    lookup_X[['items','item_lookup']] = lookup_X[['items','item_lookup']].astype('str')
    lookup_Y[['items','item_lookup']] = lookup_Y[['items','item_lookup']].astype('str')

    lookup = pd.concat([lookup_X, lookup_Y])
    
    cal_score = lookup.groupby(['customer_id','item_lookup'])[['score','qty_cus_vote','qty_lookup']].sum().reset_index()
    
    # process score for customer having number of purchased items less than MAX_NUM_FAV_ITEMS
    cal_score['score_ceil'] = MAX_NUM_FAV_ITEMS
    cal_score = pd.merge(cal_score, cal_score.groupby('customer_id')['score'].max(), how='left',on='customer_id')
    cal_score.rename(columns={'score_x':'score','score_y':'score_max'}, inplace=True)    
    cal_score['score_plus'] = cal_score['score_ceil'] - cal_score['score_max']
    cal_score['score'] = cal_score['score'] + cal_score['score_plus']
    cal_score.drop(['score_ceil','score_max','score_plus'], axis=1, inplace=True)
    
    cal_score['item_per_cus'] = cal_score['qty_lookup'] / cal_score['qty_cus_vote']
    cal_score = cal_score.sort_values(by=['customer_id','score','qty_cus_vote','item_per_cus'], ascending=[True, False, False, False])
    
    #############################################################
    # add dummy for indexing NUMBER OF RECOMMEND ITEMS
    df_dummy = pd.DataFrame(columns = cal_score.columns.values)
    value = np.empty(2*self.NUM_BASE_RECOMMEND, dtype=np.str)
    value.fill('D')
    df_dummy['item_lookup'] = np.ones(2*self.NUM_BASE_RECOMMEND)
    df_dummy['customer_id'] = value
    #print(df_dummy)
    #print(cal_score)

    cal_score = pd.concat([cal_score,df_dummy])
    ##############################################################        
    
    cal_score['indexing'] = cal_score.groupby('customer_id').cumcount() + 1
    #print(cal_score)

    df_ib_recommend = cal_score[cal_score['indexing']<=3*self.NUM_BASE_RECOMMEND] # increasing multiply coef is 3 for case more favorite, many items might be removed
    
    df_ib_recommend = df_ib_recommend[['customer_id','item_lookup']]
    df_ib_recommend.columns = ['customer_id','items']
    df_ib_recommend.to_csv(self.outpath+"/item_based_recommend_BEFORE_REMOVE_FAV_ITEMS"+str(group)+".csv", index=False)   

    return df_ib_recommend    
    
  ############################################################################
  # Function: remove_favorite(self,df_rcm,df_favorite)
  # Description:
  #     remove favorite item out of recommend items list of item-based
  # Inputs:
  #   - df_rcm: ['customer_id', rcm] 
  #   - df_favorite: 
  # Outputs:
  #   - df recommend without fav
  #  
  
  def remove_favorite(self,df_rcm,df_favorite):
  
    df_favorite = df_favorite.rename(
      columns = {
        'items' : 'item_',
        'customer_id'  : 'cus_'
      }
    )
	
    df_rcm ['rating_rcm'] = np.arange(df_rcm.shape[0])
    df_favorite['rating_fav'] = np.arange(df_favorite.shape[0])

    result_find_rcm_cal = pd.merge(df_rcm,df_favorite,how = 'left', right_on=['cus_','item_'],left_on=['customer_id','items'])
    result_find_rcm_cal = result_find_rcm_cal[result_find_rcm_cal['rating_fav'].isnull()]

    result_find_rcm_cal = result_find_rcm_cal.drop(columns = ['cus_','item_','rating_fav'])
    result_find_rcm_cal = result_find_rcm_cal.sort_values(by = ['customer_id','rating_rcm'])

    result_find_rcm_cal['indexing'] = result_find_rcm_cal.groupby('customer_id').cumcount() + 1
    result_find_rcm_cal = result_find_rcm_cal[result_find_rcm_cal['indexing']<=2*self.NUM_BASE_RECOMMEND]
  
    df_ib_recommend = pd.pivot_table(result_find_rcm_cal, values='items', index=['customer_id'], columns=['indexing'], aggfunc=np.sum, fill_value=0).reset_index()    
    #df_ib_recommend.to_csv(self.temppath+"/item_based_recommend_ONLY.csv", index=False)   

    colum = ['customer_id']
    for i in np.arange(2*self.NUM_BASE_RECOMMEND):
      colum.append('{}_cfib_recommend'.format(self.indicators[i]))    
    df_ib_recommend.columns = colum
	
    #df_ib_recommend.to_csv(self.outpath+"/PRE_item_based_recommend.csv", index=False)   
	
    return df_ib_recommend
  
  ############################################################################
  # Function: cf_customer_recommend_main_v2(self,group='')
  # Description:
  #     - read customer favorite insights
  #     - lookup item-pair for recommendation list 
  #     - combine customer and recommend list into final result of ib
  # Inputs:
  #   - group: mark group as name
  # Outputs:
  #   - df_cus_similar_item_pair [customer_id, itemA, itemB, qtyA, qtyB] --> groupby[itemA,itemB] --> df_similar_items_summary [itemA, itemB, qtyA, qtyB, customer_count]
  #
  
  def cf_customer_recommend_main_v2(self, group=''):
    
    self.df.customer_id = self.df.customer_id.astype(str)        

    ###############################################
    # for recommend column name
 
    column_ib = ['customer_id']
    for i in np.arange(self.NUM_BASE_RECOMMEND):    
      column_ib.append('{}_cfib_recommend'.format(self.indicators[i]))
  
    df_item_based_recommend = pd.DataFrame(columns=column_ib)  
    df_item_based_recommend['customer_id'] = self.df['customer_id'].unique()
       
    df_similar_item = pd.read_csv(self.outpath+"/similar_item_pair_summary_group.csv", index_col=None)
    df_similar_item = df_similar_item[['item_a','item_b','customer_count','qty_a','qty_b']]    
    
    colum=['item_x','item_y','qty_cus_vote','qty_x','qty_y']    
    df_similar_item.columns = colum
    
    # process lookup table to get recommend item in pair
    df_ib_recommend  = self.ib_similar_item_lookup(self.df[['customer_id','items','quantity']], df_similar_item, 5, group)
    
    # df_fav by ['customer_id','items'] format which row of items is for first to last in favorite
    df_fav = pd.read_csv(self.root_path+'/temp/df_after_eda.csv',index_col=False)
    df_fav = df_fav.groupby(['customer_id','items'])['quantity'].sum().reset_index()
    df_fav = df_fav.sort_values(['customer_id','quantity'], ascending=[True,False]).reset_index()
    df_fav = df_fav[['customer_id','items']]
    df_fav['customer_id'] = df_fav['customer_id'].astype('str')
   	
    df_fav.to_csv(self.temppath+"/ib_df_fav_bf_rmfav.csv", index=False)   
    df_ib_recommend.to_csv(self.temppath+"/ib_df_rcm_BF_rmfav.csv", index=False)       
    df_ib_recommend = self.remove_favorite(df_ib_recommend,df_fav)
    df_ib_recommend.to_csv(self.temppath+"/ib_df_rcm_AF_rmfav.csv", index=False)    
  
    # user favorite table
    df_fav = pd.read_csv(self.root_path+"/output/OUTPUT_customer_favorite_insights"+str(group)+".csv", index_col=None)
    df_fav.customer_id = df_fav.customer_id.astype(str)        
    
    # merge customer favorite with recommendation result    
    df_item_based_recommend = pd.merge(df_fav, df_ib_recommend, how='left', on='customer_id')
    df_item_based_recommend.to_csv(self.outpath+"/OUTPUT_item_based_recommend"+str(group)+".csv", index=False)   

    return True    
    
    
    
    
    
    
    