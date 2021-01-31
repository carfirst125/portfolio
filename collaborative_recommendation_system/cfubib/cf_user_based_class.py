##############################################################
# Class: Collaborative Filtering - User-based
#------------------------------------------------------------
# File name: cf_user_based_class.py
# Author:  Nhan Thanh Ngo
    
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


class cf_user_based():
 
  def __init__(self, df, root_path, name='user_based', item_colname='items', debug=False):
    
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

    # create folder when declare object
    self.create_folder()
    self.column_items = self.df[item_colname].unique()

    self.parameters = 'parameters.yaml'
    self.update_parameter()
    
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
      #elif key == 'NUM_FAV_COLOR':
      #  self.NUM_FAV_COLOR = parameter_dict[key]
      elif key == 'NUM_FAV_SIZE':
        self.NUM_FAV_SIZE = parameter_dict[key] 
      elif key == 'NUM_TOP':
        self.NUM_TOP = parameter_dict[key]
        
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
    self.modelpath = self.root_path+"/"+self.name+"/model"
    if os.path.exists(self.modelpath):
      print ("\'{}\' is already EXISTED!".format(self.modelpath))
    else:
      os.mkdir(self.modelpath)
      print ("\'{}\' is CREATED!".format(self.modelpath))       
      
    #------------------  
    self.outpath = self.root_path+"/"+self.name+"/output"
    if os.path.exists(self.outpath):
      print ("\'{}\' is already EXISTED!".format(self.outpath))
    else:
      os.mkdir(self.outpath)
      print ("\'{}\' is CREATED!".format(self.outpath))  
    return True


  def save_item_list(self, group=''):
    
    items_arr = '#'.join(str(x) for x in self.column_items)
    with open(self.temppath+"/items_list"+str(group)+".log", mode='w') as file:
      file.write('%s' %(items_arr))


    
  ###################################################
  # BACKUP
  
  def previous_recommendation_file_backup (self):
  
    source_path = self.outpath+"/OUTPUT_user_based_recommend.csv"
    destination_path = self.outpath+"/OUTPUT_user_based_recommend_backup.csv"
    
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
        

  ###################################################################################################
  # Function: density_based_scaler(self)
  # Description:
  #   scaling data of multi-dimension in density based method
  # Inputs:
  #   - df_pivot
  # Outputs:
  #   - df_scale
  #
  
  def density_based_scaler(self, df_pivoti, cus_group=''):

    df_pivot = df_pivoti.copy()  # because df_pivoti will be change after run this function
    df_pivot.loc['std_dev',:] = df_pivot.std(axis=0)
    df_pivot.loc['average',:] = df_pivot[:df_pivot.shape[0]-1].mean(axis=0)

    df_std_mean = df_pivot[-2:]
    df_pivot = df_pivot[:-2]
   
    alpha = 2.25 
    for col in df_pivot.columns.values:
      mean_dim = df_std_mean.loc['average',col]
      std_dim = df_std_mean.loc['std_dev',col]
    
      dim_cuscount = df_pivot[(df_pivot[col]>=(mean_dim - alpha*std_dim))&(df_pivot[col]<=(mean_dim + alpha*std_dim))].count()[col]
      df_pivot[col] = ((df_pivot[col] - mean_dim + alpha*mean_dim)/(2*alpha*std_dim))*(dim_cuscount/len(df_pivot))
      self.print_debug("[density_based_scaler] Dense of customer in +-{}*sigma is {}, mean_dim = {}, std_dim = {}, dim_cuscount = {}".format(alpha,dim_cuscount/len(df_pivot),mean_dim, std_dim, dim_cuscount))
      
      df_std_mean.loc['rate',col] = dim_cuscount/len(df_pivot)
      
    df_std_mean.to_csv(self.temppath+'/density_based_scaler'+str(cus_group)+'.csv', index=False) 
    
    return df_pivot
    
  ############################################################################
  # Function: cf_user_based_kdtree_train_mono(self, df_pivot, cus_group):
  # Description:
  #   Train kdtree for df_group of customer and save model
  # Inputs:
  #   - df_pivot: customer_id, prodA, prodB, ...
  #   - savmodel_path: path and name of kdtree saved model 
  # Outputs:
  #   - out model in savepath
  #   - out scale transformation data file for cus-group
  #

  def cf_user_based_kdtree_train_mono(self, df_pivot, cus_group):

    with open(self.temppath+"/cus_group.log", mode='a') as file:
      file.write('%s#' %(str(cus_group)))
    
    
    savmodel_path = self.modelpath+"/kdtree_"+str(cus_group)+".sav" 
    target_arr = '' 
    column = df_pivot.columns.values
    df_pivot_dim = df_pivot[column[1:]]
  
    try:        
      df_pivot_scale = self.density_based_scaler(df_pivot_dim, cus_group)
      kdtree = KDTree(df_pivot_scale.values, leaf_size=40) #default: leaf_size=40   
   
      pickle.dump(kdtree, open(savmodel_path,'wb')) 
      self.print_debug("[cf_user-based] Saved kdtree model at {}".format(savmodel_path))
      df_pivot_scale['customer_id'] = df_pivot['customer_id']
      df_pivot_scale = df_pivot_scale[column]
      df_pivot_scale.to_csv(self.temppath+"/df_pivot_"+str(cus_group)+"_scale_transform.csv",index=False)
      return True
    
    except ValueError: 
      self.print_debug ("[cf_user_based_kdtree_train] get Error {} with ... \ndf_pivot = \n{} \n savmodel_path = {}".format(ValueError,df_pivot,savmodel_path))
      return False
    
  
  ###################################################################################################
  # Function: cf_user_based_kdtree_train(self):
  # Description:
  #   cf_user_based_kdtree_train() -->  cf_user_based_kdtree_train_mono()
  #   Train kdtree for df_group of customer and save model
  # Inputs:
  #   - df: customer_id, items, quantity
  # Outputs:
  #   - no return 
  #

  def cf_user_based_kdtree_train(self):

    for cus_group in self.df.cluster_group.unique():
        
      #save items_list files
      self.save_item_list(str(cus_group))          
          
      print("[TRAIN] kdtree train for cluster_group {}".format(cus_group))
      df_group = self.df[self.df.cluster_group==cus_group]
      df_group = df_group[['customer_id','items','quantity']]
      #pivot tablename
      df_group_pivot = pd.pivot_table(df_group, values='quantity', index=['customer_id'], columns=['items'], aggfunc=np.sum, fill_value=0)
  
      #############################################################
      # stable item columns
      cols_group = df_group_pivot.columns.values
      miss_cols = np.setdiff1d(self.column_items, cols_group)    
      #print("miss_cols: {}".format(miss_cols))
      if len(miss_cols)>0:
        self.print_debug("miss_cols = {},  catekmc_column = {}, columns = {}".format(miss_cols, self.column_items, cols_group))
        for col in miss_cols:
          df_group_pivot[col] = 0
    
      df_group_pivot = df_group_pivot[self.column_items]
      df_group_pivot.reset_index(inplace=True)
      #############################################################    

      response = self.cf_user_based_kdtree_train_mono(df_group_pivot, cus_group)
      if not response:
        self.print_debug ("[cf_user_based_kdtree_train] ERROR 01: Train kdtree model ...")
        sys.exit(1)
      else: 
        self.print_debug ("[cf_user_based_kdtree_train] Successful ...")
        
    return True 
    
  ###########################################################################
  # Function: return_most_favorite_rm_noqty(row, num_top_favourite)
  #     sort_values of dataframe, and get the top values of them
  #     return the name of top venues  
  #

  def return_most_favorite_rm_noqty(self, row, NUM_TOP):
 
    row_sorted = row.sort_values(ascending=False)
    category_msb = row_sorted.index.values[0:NUM_TOP]
    qty_msb = row_sorted.values[0:NUM_TOP]
  
    qty_msb_bool = [True if i > 0 else False for i in qty_msb]
    category_msb_purchased = [ j if i else 0 for i,j in zip(qty_msb_bool, category_msb)]
    if len(category_msb_purchased) < NUM_TOP:
      category_msb_purchased = list(category_msb_purchased) + list(np.zeros(NUM_TOP-len(category_msb_purchased),dtype=int))
      qty_msb = list(qty_msb) + list(np.zeros(NUM_TOP-len(qty_msb),dtype=int))
    return [category_msb_purchased, qty_msb]

  ###################################################################################################
  # Function: cf_user_based_kdtree_query(self, df_pivot, nearby_target_vector)
  # Description:
  #   Train kdtree for df_group of customer and save model
  # Inputs:
  #   - df_pivot: customer_id, prodA, prodB, ...
  #   - nearby_target_vector: np.array([[2,4,12,5]]) or dfx[dfx.customer_id==123132].values
  #   - savmodel_path: path and name of kdtree saved model
  # Outputs:
  #   - customer_id which nears the vector (df_kdtree_nearby.customer_id.values)  
  #

  def cf_user_based_kdtree_query(self, df_pivot, cus_vector, NUM_NEARBY, savmodel_path):
  
    kdtree_load = pickle.load(open(savmodel_path,'rb'))     

    distance, ind = kdtree_load.query(cus_vector, k=NUM_NEARBY)
    df_kdtree_nearby = df_pivot.iloc[ind[0],]

    return df_kdtree_nearby.customer_id.values, distance[0]
    
  ###################################################################################################
  # Function: cf_user_based_kdtree_nearest_neighbor_explore(self)
  # Description:
  #   cf_user_based_kdtree_train() --> cf_user_based_kdtree_nearest_neighbor_explore() --> cf_user_based_kdtree_query() --> cf_user_based_kdtree_train_mono()
  #   Find nearest neighbor of customer-id by kdtree
  # Inputs:
  #   - df [customer_id, items, quantity]
  #   - columns_item: list of columns that trained in kdtree
  # Outputs:
  #   - df_cus_nearby ['customer_id','nearby_cus']
  #

  def cf_user_based_kdtree_nearest_neighbor_explore(self, group=''):

    df_cus_nearby = pd.DataFrame(columns=['customer_id','nearby_cus'])
    df_cus_nearby['customer_id'] = self.df.customer_id.unique()
    df_cus_nearby['nearby_cus']=df_cus_nearby['nearby_cus'].astype(object)
    #df_cus_nearby['cus_vector']=df_cus_nearby['cus_vector'].astype(object)

    ###########################################
    # FIND NEARBY CUSTOMER IN TINY GROUPS
    ###########################################
    count=0
    for cluster_i in self.df.cluster_group.unique():

      df_pivot_scale = pd.read_csv(self.temppath+"/df_pivot_"+str(cluster_i)+"_scale_transform.csv",index_col=None) 
      #print('df_pivot_scale = ', df_pivot_scale)
      
      for cus_id in df_pivot_scale.customer_id.unique():
        count+=1
        if count%self.PERIOD==0: 
          print("[User-based] Find nearest neighbor, completed for {} users".format(count)) 

        df_pivot_scale_cus = df_pivot_scale[df_pivot_scale.customer_id==cus_id]
        df_pivot_scale_cus = df_pivot_scale_cus[df_pivot_scale_cus.columns.values[1:]]
        cus_vector = df_pivot_scale_cus.values
            
        # Run kdtree to find NUM_NEARBY customers
        savmodel_path = self.modelpath+"/kdtree_"+str(cluster_i)+".sav"
        group_nearby_cus_arr = self.cf_user_based_kdtree_query(df_pivot_scale,cus_vector, self.NUM_NEARBY+1, savmodel_path)[0]  
     
        group_nearby_cus_adj = []
        for i in group_nearby_cus_arr:
          if i!=cus_id:
            group_nearby_cus_adj.append(i)
    
        group_nearby_cus_arr = '#'.join(str(x) for x in group_nearby_cus_adj)     
  
        # append values of nearest neighbor customer_id to dataframe
        df_cus_nearby.loc[df_cus_nearby.customer_id==cus_id,'nearby_cus'] = str(group_nearby_cus_arr)        

    df_cus_nearby.to_csv(self.temppath+"/df_nearby_customer"+str(group)+".csv",index=False)

    return True
    
    
  ###################################################################################################
  # Function: cf_insight_nearby_cus_and_recommend(self, raw_file_name):
  # Description:
  #   ... --> cf_user_based_kdtree_nearest_neighbor_explore() -->  cf_insight_nearby_cus_and_recommend()
  #   find insights of kdtree nearby customer, find best items for recommend
  # Inputs: 
  # Outputs:
  #   - df_recommend ['customer_id',first_fav,...,fourth_fav,first_recommend,...,fifth_recommend]
  #

  def cf_insight_nearby_cus_and_recommend(self, group=''):

    # user favorite table
    df_fav = pd.read_csv(self.root_path+"/output/OUTPUT_customer_favorite_insights"+str(group)+".csv", index_col=None) 
    df_fav = df_fav[df_fav.columns.values[:self.NUM_FAV_ITEMS+1]]
    df_fav.customer_id = df_fav.customer_id.astype(str)
    
    self.df.customer_id = self.df.customer_id.astype(str)

    #############################################
    # Create df_recommend
    #############################################
    indicators = ['first','second','third','fourth','fifth','sixth','seventh','eighth','nineth','tenth','eleven','twelfth','thirthteen','forthteen','fifthteen']
    column = ['customer_id']

    for i in np.arange(self.NUM_BASE_RECOMMEND):
      column.append('{}_cfub_recommend'.format(indicators[i]))
    
    # create a new dataframe
    df_recommend = pd.DataFrame(columns=column)
    df_recommend['customer_id'] = self.df['customer_id'].unique() 
  
    #############################################
    # Create df_recommend
    #############################################  
    cus_nearby_file = self.temppath+"/df_nearby_customer"+str(group)+".csv"
    df_cus_nearby = pd.read_csv(cus_nearby_file, index_col=None)
    df_cus_nearby = df_cus_nearby[['customer_id','nearby_cus']]
    df_cus_nearby.customer_id = df_cus_nearby.customer_id.astype(str)
    
    #for each customer, investigate to nearby customer to get recommend    
    count=0
    for cus_id in df_cus_nearby.customer_id.unique():
      count+=1
      #print("[{}] User-based: Get Recommendation: Processing for customer_id {}".format(count, cus_id))  
      if count%self.PERIOD==0:  
        print("[User-based] Get recommendation items, completed for {} users".format(count))     
      # Get nearby customer list
      nearby_cus_id_str = df_cus_nearby.loc[df_cus_nearby.customer_id==cus_id,'nearby_cus'].values[0]
      nearby_cus_id_arr = (str(nearby_cus_id_str)).split("#")
    
      #####################################
      # Get Recommendation Items
      #####################################
      df_nearby_cus = self.df[self.df.customer_id.isin(nearby_cus_id_arr)]    
      df_nearby_cus_pivot = pd.pivot_table(df_nearby_cus, values='quantity', index=['customer_id'], columns=[self.item_colname], aggfunc=np.sum, fill_value=0)    

      total_quantity = df_nearby_cus_pivot.sum().sum()
      self.print_debug("[cus_id {}] kdtree nearest's total quantity: {} ".format(cus_id,total_quantity))
    
      df_item_qty_sum = pd.DataFrame(data=df_nearby_cus_pivot.sum())
      df_item_qty_sum.columns=['qty']
      df_item_qty_sum.sort_values(by='qty', ascending=False, inplace=True)
      top_recommend_list = df_item_qty_sum.index.values[:self.NUM_BASE_RECOMMEND]
      
      ###################################################
      # remove item in recommend list but still in user favorite list     
      ###################################################
      #print('df_fav[df_fav.customer_id == cus_id].values = ',df_fav[df_fav.customer_id == cus_id].values)
      top_fav_list = df_fav[df_fav.customer_id == cus_id].values[0]
      
      # Remove ones in recommend but still in favorite
      top_recommend_list_adj = []    
      for recom in top_recommend_list:
        check=False
        for fav in top_fav_list:
          if recom == fav:
            check=True
        if not check:
          top_recommend_list_adj.append(recom)

      # fill length of recommend list to self.NUM_BASE_RECOMMEND
      top_recommend_list = [top_recommend_list_adj[i] if i<len(top_recommend_list_adj) else 0 for i in np.arange(self.NUM_BASE_RECOMMEND)]    
      
      ###################################
      # Feed data to df_recommendation
      df_recommend.loc[df_recommend.customer_id==cus_id, 1:] = top_recommend_list
    
    df_recommend.to_csv(self.outpath+"/OUTPUT_user_based_recommend"+str(group)+".csv", index=False)
    
    return True

  ###################################################################################################
  # Function: cf_previous_recommend_ref(self, group=''):
  # Description:
  #    [update] use for ub recommend item update for customer have new purchases.
  # Inputs: 
  #    new customer fav and previous ub_recommendation file
  # Outputs:
  #   - df_recommend ['customer_id',first_fav,...,fourth_fav,first_recommend,...,fifth_recommend]
  #

  def cf_previous_recommend_ref(self, group=''):

    # user favorite table
    df_fav = pd.read_csv(self.root_path+"/output/OUTPUT_customer_favorite_insights"+str(group)+".csv", index_col=None) 
    df_fav = df_fav[df_fav.columns.values[:self.NUM_FAV_ITEMS+1]]
    df_fav.customer_id = df_fav.customer_id.astype(str)

    self.df.customer_id = self.df.customer_id.astype(str)

    #############################################
    # Create df_recommend
    #############################################
    indicators = ['first','second','third','fourth','fifth','sixth','seventh','eighth','nineth','tenth','eleven','twelfth','thirthteen','forthteen','fifthteen']
    #column = ['customer_id']
    column = []
    for i in np.arange(self.NUM_BASE_RECOMMEND):
      column.append('{}_cfub_recommend'.format(indicators[i]))
    
    # create a new dataframe
    df_recommend = pd.read_csv(self.outpath+"/OUTPUT_user_based_recommend.csv", index_col=None)
    df_recommend.customer_id = df_recommend.customer_id.astype(str)

    #############################################
    # Create df_recommend
    #############################################      
    #for each customer, investigate to nearby customer to get recommend    
    count=0
    for cus_id in self.df.customer_id.unique():
      count+=1
      if count%self.PERIOD==0:  
        print("[User-based] UPDATE recommendation items, completed for {} users".format(count))
        
      top_recommend_list = df_recommend[df_recommend.customer_id==cus_id][column].values[0]
      
      ###################################################
      # remove item in recommend list but still in user favorite list     
      ###################################################
      top_fav_list = df_fav[df_fav.customer_id == cus_id].values[0][1:]
      
      # Remove ones in recommend but still in favorite
      top_recommend_list_adj = []    
      for recom in top_recommend_list:
        check=False
        for fav in top_fav_list:
          if recom == fav:
            check=True
        if not check:
          top_recommend_list_adj.append(recom)

      # fill length of recommend list to self.NUM_BASE_RECOMMEND
      top_recommend_list = [top_recommend_list_adj[i] if i<len(top_recommend_list_adj) else '0' for i in np.arange(self.NUM_BASE_RECOMMEND)]    
      
    # update data to recommend df
    df_recommend.loc[df_recommend.customer_id==cus_id, 1:] = top_recommend_list
    df_oldcus_recommend = df_recommend[df_recommend.customer_id.isin(self.df.customer_id)]
    df_oldcus_recommend.to_csv(self.outpath+"/OUTPUT_user_based_recommend_old.csv", index=False)    
    df_recommend.to_csv(self.outpath+"/OUTPUT_user_based_recommend.csv", index=False)
    
    return True

  ###################################################################################################
  # Function: density_based_scaler(self)
  # Description:
  #   scaling data of multi-dimension in density based method
  # Inputs:
  #   - df_pivot
  # Outputs:
  #   - df_scale
  #
  
  def density_based_scaler_cus(self, df_pivoti, cus_group=''):

    df_std_mean = pd.read_csv(self.temppath+'/density_based_scaler'+str(cus_group)+'.csv', index_col=False)
    df_pivot = df_pivoti.copy()  # because df_pivoti will be change after run this function
   
    alpha = 2.25 
    for col in df_std_mean.columns.values:
      mean_dim = df_std_mean.loc[0,col]  #average
      std_dim = df_std_mean.loc[1,col]   #std_dev
      dim_cuscount = df_std_mean.loc[2,col] #rate
      
      df_pivot[col] = ((df_pivot[col] - mean_dim + alpha*mean_dim)/(2*alpha*std_dim))*(dim_cuscount/len(df_pivot))
      self.print_debug("[density_based_scaler] Dense of customer in +-{}*sigma is {}, mean_dim = {}, std_dim = {}, dim_cuscount = {}".format(alpha,dim_cuscount/len(df_pivot),mean_dim, std_dim, dim_cuscount))
      
    return df_pivot
    
    
    
  ###################################################################################################
  # Function: cf_newcus_lookup_kdtree_find_nearest_and_recommend_up_fullkdtree(self):
  # Description:
  #    [update] new customer: run kdtree, find nearest customer and get the recommendation result of this customer 
  # Inputs: 
  #    df of new customer (no pivot)
  # Outputs:
  #   - df_recommend ['customer_id',first_fav,...,fourth_fav,first_recommend,...,fifth_recommend]
  #

  def cf_newcus_lookup_kdtree_find_nearest_and_recommend_up_fullkdtree(self, group=''):
  
    with open(self.temppath+"/cus_group.log", mode='r') as file:
      cus_group_str = file.readline()
    cus_group_arr = cus_group_str.split("#")[:-1]  # remove the last one   
       
    df_cus_similar = pd.DataFrame(columns=['customer_id','similar_cus'])
    df_cus_similar['customer_id'] = self.df['customer_id'].unique()
	
    count = 0
	# sweep for each customer_id
    for cus_id in self.df.customer_id.unique():
      count+=1
      if count%self.PERIOD==0: 
        print("[User-based] UPDATE recommend for action users, completed for {} users".format(count)) 
  
      df_pivot_cus = self.df[self.df.customer_id==cus_id]
      df_pivot_cus = pd.pivot_table(df_pivot_cus, values='quantity', index=['customer_id'], columns=[self.item_colname], aggfunc=np.sum, fill_value=0)    
      cols_group = df_pivot_cus.columns.values      
      cus_nearby_id = []
      nearby_distance = []

      # sweep for different groups for KDTREE
	  
      for cus_group in cus_group_arr[:1]:   # 1 for short reference. If [:], kdtree reference for long times, so choose first kdtree model for reference.
	  
        df_pivot_cusg = df_pivot_cus.copy()
        with open(self.temppath+"/items_list"+str(cus_group)+".log", mode='r') as file:
          items_list_str = file.readline()
        items_list = items_list_str.split("#")  # remove the last one

        #############################################################
        # stable item columns
            
        miss_cols = np.setdiff1d(items_list, cols_group)    
        #print("miss_cols: {}".format(miss_cols))
        if len(miss_cols)>0:
          self.print_debug("miss_cols = {},  catekmc_column = {}, columns = {}".format(miss_cols, items_list, cols_group))
        for col in miss_cols:
          df_pivot_cusg[col] = 0
   
        df_pivot_cusg = df_pivot_cusg[items_list]
		
        #############################################################         

        df_cusi_scale = self.density_based_scaler_cus(df_pivot_cusg, cus_group)
        cus_vector = df_cusi_scale.values
        
        # Run kdtree to find NUM_NEARBY customers       
        df_pivot_scale = pd.read_csv(self.temppath+"/df_pivot_"+str(cus_group)+"_scale_transform.csv",index_col=None)    
        savmodel_path = self.modelpath+"/kdtree_"+str(cus_group)+".sav"
        group_nearby_cus_arr, distance = self.cf_user_based_kdtree_query(df_pivot_scale,cus_vector, 2, savmodel_path)

        # add to big array
        cus_nearby_id.extend(group_nearby_cus_arr)
        nearby_distance.extend(distance)
      
      # complete run kdtree for all group
      cus_nearby_cusid = np.array(cus_nearby_id)[(np.argsort(nearby_distance)).astype(int)]
      for id in cus_nearby_cusid:
        if not (str(id) == str(cus_id)):
          df_cus_similar.loc[df_cus_similar.customer_id==cus_id,'similar_cus'] = str(id)
          break

    df_cus_similar.to_csv(self.temppath+'/df_new_cus_similar.csv',index=None)   
   
    #################################################################
    # READ FILE RECOMMEND, COPY RECOMMEND RESULT OF SIMILAR CUS
    #################################################################
  
    df_ub_recommend = pd.read_csv(self.outpath+"/OUTPUT_user_based_recommend.csv", index_col=False)
	
    df_ub_recommend.customer_id = df_ub_recommend.customer_id.astype(str)
    df_cus_similar.columns = ['customer_idx','customer_id']	
    df_cus_similar.customer_id = df_cus_similar.customer_id.astype(str)
    df_new_recommend = pd.merge(df_cus_similar, df_ub_recommend, how='left', on='customer_id')   # error here

    df_new_recommend.drop(columns=['customer_id'], inplace=True)
    df_new_recommend.columns = np.concatenate((['customer_id'],np.array(df_new_recommend.columns[1:])))
	
    df_ub_recommend = pd.concat([df_ub_recommend, df_new_recommend])
    df_ub_recommend.to_csv(self.outpath+"/OUTPUT_user_based_recommend.csv",index=None)
	
    df_old_recommend = pd.read_csv(self.outpath+"/OUTPUT_user_based_recommend_old.csv", index_col=False)
    df_new_recommend = pd.concat([df_new_recommend, df_old_recommend])

    df_new_recommend.to_csv(self.outpath+"/OUTPUT_user_based_recommend"+str(group)+".csv",index=None)	

  ###################################################################################################
  # Function: cf_newcus_lookup_kdtree_find_nearest_and_recommend_up(self)# choosing 1 cluster group for refer
  # Description:
  #    [update] This only choose one kdtree of any clus_group and find nearest customer of new customer
  #             if you would like to refer all kdtrees to find better nearest customer, please use above function
  #                   cf_newcus_lookup_kdtree_find_nearest_and_recommend_up_fullkdtree(self, group='')
  # Inputs: 
  #    df of new customer (no pivot)
  # Outputs:
  #   - df_recommend ['customer_id',first_fav,...,fourth_fav,first_recommend,...,fifth_recommend]
  #

  def cf_newcus_lookup_kdtree_find_nearest_and_recommend_up(self, group=''):
  
    # cus_group file for information of all group in big file, each group have one kdtree.
    with open(self.temppath+"/cus_group.log", mode='r') as file:
      cus_group_str = file.readline()
    cus_group_arr = cus_group_str.split("#")[:-1]  # remove the last one   
       
    df_cus_similar = pd.DataFrame(columns=['customer_id','similar_cus'])
    df_cus_similar['customer_id'] = self.df['customer_id'].unique()
	
    cus_group = cus_group_arr[0]  # 0 for short reference. If [:], kdtree reference for long times, so choose first kdtree model for reference.	
    with open(self.temppath+"/items_list"+str(cus_group)+".log", mode='r') as file:
      items_list_str = file.readline()
    items_list = items_list_str.split("#")  # remove the last one

    df_pivot = pd.pivot_table(self.df, values='quantity', index=['customer_id'], columns=[self.item_colname], aggfunc=np.sum, fill_value=0) 
    cols_group = df_pivot.columns.values  
	
    #############################################################
    # stable item columns            
    miss_cols = np.setdiff1d(items_list, cols_group)    

    if len(miss_cols)>0:
      self.print_debug("miss_cols = {},  catekmc_column = {}, columns = {}".format(miss_cols, items_list, cols_group))
    for col in miss_cols:
      df_pivot[col] = 0
	  
    df_pivot = df_pivot[items_list]
    df_pivot.reset_index(inplace=True)
    #############################################################  
	  
    count = 0
	# sweep for each customer_id
    for cus_id in self.df.customer_id.unique():
      count+=1
      if count%self.PERIOD==0: 
        print("[User-based] Find nearest customer for refering, and get reference item list of nearest users, completed for {} users".format(count)) 
  
      df_pivot_cus = df_pivot[df_pivot.customer_id==cus_id]
      #df_pivot_cus = pd.pivot_table(df_pivot_cus, values='quantity', index=['customer_id'], columns=[self.item_colname], aggfunc=np.sum, fill_value=0)    

      cus_nearby_id = []
      nearby_distance = []   

      df_cusi_scale = self.density_based_scaler_cus(df_pivot_cus[df_pivot_cus.columns.values[1:]], cus_group)
      cus_vector = df_cusi_scale.values
  
      # Run kdtree to find NUM_NEARBY customers       
      df_pivot_scale = pd.read_csv(self.temppath+"/df_pivot_"+str(cus_group)+"_scale_transform.csv",index_col=None)    
      savmodel_path = self.modelpath+"/kdtree_"+str(cus_group)+".sav"
      group_nearby_cus_arr, distance = self.cf_user_based_kdtree_query(df_pivot_scale,cus_vector, 2, savmodel_path)

      # add to big array
      cus_nearby_id.extend(group_nearby_cus_arr)
      nearby_distance.extend(distance)
      
      # complete run kdtree for all group
      cus_nearby_cusid = np.array(cus_nearby_id)[(np.argsort(nearby_distance)).astype(int)]
      for id in cus_nearby_cusid:
        if not (str(id) == str(cus_id)):
          df_cus_similar.loc[df_cus_similar.customer_id==cus_id,'similar_cus'] = str(id)
          break

    df_cus_similar.to_csv(self.temppath+'/df_new_cus_similar.csv',index=None)   
   
    #################################################################
    # READ FILE RECOMMEND, COPY RECOMMEND RESULT OF SIMILAR CUS
    #################################################################
  
    df_ub_recommend = pd.read_csv(self.outpath+"/OUTPUT_user_based_recommend.csv", index_col=False)
	
    df_ub_recommend.customer_id = df_ub_recommend.customer_id.astype(str)
    df_cus_similar.columns = ['customer_idx','customer_id']	
    df_cus_similar.customer_id = df_cus_similar.customer_id.astype(str)
    df_new_recommend = pd.merge(df_cus_similar, df_ub_recommend, how='left', on='customer_id')   # error here

    df_new_recommend.drop(columns=['customer_id'], inplace=True)
    df_new_recommend.columns = np.concatenate((['customer_id'],np.array(df_new_recommend.columns[1:])))
	
    #####################################################################
    # get items for recommend, remove items in favorite
    #####################################################################
    # user favorite table
    df_fav = pd.read_csv(self.root_path+"/output/OUTPUT_customer_favorite_insights"+str(group)+".csv", index_col=None) 
    df_fav = df_fav[df_fav.columns.values[:self.NUM_FAV_ITEMS+1]]
    df_fav.customer_id = df_fav.customer_id.astype(str)	

    count=0
    column = df_new_recommend.columns.values[1:]
    #print('column=',column)
    for cus_id in df_new_recommend.customer_id.unique():
      #print('This cus_id = ',cus_id)
      count+=1
      if count%self.PERIOD==0:  
        print("[User-based] UPDATE recommendation items, completed for {} users".format(count))
        
      top_recommend_list = df_new_recommend[df_new_recommend.customer_id==cus_id][column].values[0]
	  
      ###################################################
      # remove item in recommend list but still in user favorite list     
      ###################################################
      top_fav_list = df_fav[df_fav.customer_id == cus_id].values[0][1:]
	  
      # Remove ones in recommend but still in favorite
      top_recommend_list_adj = []    
      for recom in top_recommend_list:
        check=False
        for fav in top_fav_list:
          if recom == fav:
            check=True
        if not check:
          top_recommend_list_adj.append(recom)
		  

      # fill length of recommend list to self.NUM_BASE_RECOMMEND
      top_recommend_list = [top_recommend_list_adj[i] if i<len(top_recommend_list_adj) else '0' for i in np.arange(self.NUM_BASE_RECOMMEND)]   
	  
      # update data to recommend df
      df_new_recommend.loc[df_new_recommend.customer_id==cus_id, 1:] = top_recommend_list	  

    df_new_recommend.to_csv(self.temppath+"/df_new_recommend.csv",index=None)

    #####################################################################    

    df_ub_recommend = pd.concat([df_ub_recommend, df_new_recommend])
    df_ub_recommend.to_csv(self.outpath+"/OUTPUT_user_based_recommend.csv",index=None)
	
    df_old_recommend = pd.read_csv(self.outpath+"/OUTPUT_user_based_recommend_old.csv", index_col=False)
    df_new_recommend = pd.concat([df_new_recommend, df_old_recommend])

    df_new_recommend.to_csv(self.outpath+"/OUTPUT_user_based_recommend"+str(group)+".csv",index=None)	    
    
    