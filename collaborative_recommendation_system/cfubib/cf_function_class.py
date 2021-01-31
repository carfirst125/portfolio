##############################################################
# Class: Collaborative Filtering - GENERAL FUNCTIONS
#------------------------------------------------------------
# File name: cf_function_class.py
# Author:    Nhan Thanh Ngo


import tensorflow as tf
print(tf.__version__)

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


class general_functions():
 
  def __init__(self, df, root_path, ub_name='user_based', ib_name='item_based', name=None, item_colname='items', mode='userbased', debug=False):
        
    self.CUS_COUNT = 0
    self.N = 0
    self.PERIOD    = 1000
    
    self.debug = debug
    self.df = df
    self.ub_name = ub_name
    self.ib_name= ib_name
    
    self.root_path = root_path
    self.name = name
    self.temppath = ''
    self.outpath = ''
    self.item_colname = item_colname
    self.mode=mode
    self.mode_initial()

    # create folder when declare object 
    self.declare_folderpath()
    self.parameters = 'parameters.yaml'
    self.update_parameter()
    
    self.indicators = ['first','second','third','fourth','fifth','sixth','seventh','eighth','nineth','tenth','eleven','twelfth','thirthteen','forthteen','fifthteen']
     
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
      elif key == 'NUM_FAV_SIZE':
        self.NUM_FAV_SIZE = parameter_dict[key] 
      elif key == 'NUM_TOP':
        self.NUM_TOP = parameter_dict[key]

  ############################################################################
  # Function: create_folder(self)
  # Description:
  #   create neccessary folder 
  
  def declare_folderpath(self):
    #print(self.root_path)  
    #sys.exit()
    self.homepath = self.root_path
    self.temppath = self.root_path+"/temp"
    self.modelpath = self.root_path+"/model" 
    self.recommend_base = self.root_path+"/recommend_base" 
    self.clusfile_path = self.root_path+"/clusfile"
    self.outpath = self.root_path+"/output"
    self.uboutpath = self.root_path+"/"+self.ub_name+"/output"
    self.iboutpath = self.root_path+"/"+self.ib_name+"/output"
    self.ubtemppath = self.root_path+"/"+self.ub_name+"/temp"
    self.ibtemppath = self.root_path+"/"+self.ib_name+"/temp"
    
  def mode_initial(self):
    self.userbased = False
    self.itembased = False    
       
    if self.mode == 'both':
       self.userbased = True
       self.itembased = True       
    elif self.mode == 'userbased':
       self.userbased = True
       self.itembased = False
    elif self.mode == 'itembased':
       self.userbased = False
       self.itembased = True    
       
  ###################################################
  # DEBUG
 
  def print_debug (self, strcont):
    if self.debug:
      print(strcont)  
      with open(self.homepath+"/debug.log", "a") as logfile:
        logfile.write("\n"+strcont)

  ###################################################################################################
  # Function: no_accent_vietnamese(self, s)
  # Description:
  #   replace vietnamese letters by unsigned letters
  # Inputs: 
  #   - s: string (vietnamese string)
  # Outputs:
  #   - s: string removed sign
  #

  def no_accent_vietnamese(self, s):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s
    
  ###################################################################################################
  # Function: product_eda(self)
  # Description:
  #   - process item value that does not map the main item name (eg.tra sen vang (S, co sua, co kem) --> tra sen vang)
  #   - remove item_names which are out of main item list
  # Inputs: 
  #   - df ['customer_id','date_order','items','product_id','quantity']
  # Outputs:
  #   - df  ['customer_id','date_order','items','product_id','quantity'] has been EDA
  #
  
  def product_eda(self):

    ############################################################
    # process item_name which does not match with standard
    ############################################################
    main_items_list = []

    for item_name in self.df['items'].unique():
      #print('Process: item_name: {}'.format(item_name))      

      matched = bool(re.match("(^EXTRA$)",item_name)) | \
                bool(re.match("(^COMBO$)",item_name)) | \
                bool(re.match("(^PHIN COFFEE$)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(200G)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(185ML)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(1KG)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(STICK)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(CHAI)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(GOI)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(HOP)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(LON)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(FREE PRODUCT)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(DISCOUNT)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("(^NUOC TINH KHIET$)",item_name)) 
                
      if matched:
        print('[REMOVE]item_name: {}'.format(item_name))
        continue
      else:
        print('[KEEP]item_name: {}'.format(item_name))
        main_items_list.append(item_name) 
        
    # remove items which is not in the main_items_list
    self.df = self.df[self.df[self.item_colname].isin(main_items_list)] 

  
    ###################################################################
    # PROCESS MISSING SIZE OF ITEM (NOT COOKIES)
    # Fill missing size with the Size equivalent to the product_id
    not_drink_list = ['BANH', 'BM', 'CAKE', 'BREAD', 'COOKIES ']
    drink_list = []
    append = True
    for item_name in main_items_list:
      for not_drink_sub in not_drink_list:
        if not_drink_sub in item_name:
            append = False
            break
        append = True
        
      if append:
        drink_list.append(item_name)
        
    for item_name in drink_list:
      proid_list = self.df[self.df['items']==item_name]['product_id'].unique()
      for proid in proid_list:   
        equivalent_size = np.sort(self.df[(self.df['items']==item_name)&(self.df['product_id']==proid)]['size'].unique())[::-1][0]
        self.df.loc[(self.df['size']=='0')&(self.df['items']==item_name)&(self.df['product_id']==proid),'size'] = equivalent_size
		
    #################################################################

    # save input dataframe after pre-EDA (after item_name adjust process)
    self.df.groupby(['customer_id','date_order','items','size'])['quantity'].sum().reset_index()
    
    # store features to file
    main_items_list_str = ','.join(main_items_list)    
    with open(self.modelpath+'/train_input_features.txt', 'w') as logfile:
      logfile.write(main_items_list_str)

    return self.df        
    
  ###################################################################################################
  # Function: density_based_scaler(self)
  # Description:
  #   scaling data of multi-dimension in density based method
  # Inputs:
  #   - df_pivot
  # Outputs:
  #   - df_scale
  #
  
  def density_based_scaler(self, df_pivoti):

    df_pivot = df_pivoti.copy()  # because df_pivoti will be change after run this function
    df_pivot.loc['std',:] = df_pivot.std(axis=0)
    df_pivot.loc['mean',:] = df_pivot[:df_pivot.shape[0]-1].mean(axis=0)

    df_std_mean = df_pivot[-2:]
    df_pivot = df_pivot[:-2]
   
    alpha = 2.25 
    for col in df_pivot.columns.values:
      mean_dim = df_std_mean.loc['mean',col]
      std_dim = df_std_mean.loc['std',col]
    
      dim_cuscount = df_pivot[(df_pivot[col]>=(mean_dim - alpha*std_dim))&(df_pivot[col]<=(mean_dim + alpha*std_dim))].count()[col]
      df_pivot[col] = ((df_pivot[col] - mean_dim + alpha*mean_dim)/(2*alpha*mean_dim))*(dim_cuscount/len(df_pivot))
      self.print_debug("[density_based_scaler] Dense of customer in +-{}*sigma is {}, mean_dim = {}, std_dim = {}, dim_cuscount = {}".format(alpha,dim_cuscount/len(df_pivot),mean_dim, std_dim, dim_cuscount))
      
    return df_pivot
  
  ###################################################################################################
  # Function: df_file_cluster(self)
  # Description:
  #   divide file into smaller file by amount of customer insight
  # Inputs:
  #   - self.df
  # Outputs:
  #   - cluster file in ./clusfile folder
  #

  def df_file_cluster(self):
    
    #######################################################
    # SPLIT TO SMALL GROUPs [reason by computer capacity]
    #######################################################
    
    total_customer = self.df.customer_id.nunique()
    customer_id_array = self.df.customer_id.unique()
    if (total_customer) > self.CLUSTER_DIV_PROCESS_THRESHOLD:
        NUMCLUS = int(total_customer/self.CLUSTER_DIV_PROCESS_THRESHOLD)+1
        CLUSTER_NUMCUS = int(total_customer/NUMCLUS)+1
        start = 0
        end = 0
        for i in np.arange(NUMCLUS):
          start = i*CLUSTER_NUMCUS
                
          if i==NUMCLUS:
            end = None
          else:
            end = (i+1)*CLUSTER_NUMCUS
        
          self.df.loc[self.df.customer_id.isin(customer_id_array[start:end]),'customer_group'] = '%s'%(str(i))
          
        # print cluster summary
        df_cus_cnt = self.df.groupby(['customer_group']).customer_id.nunique().reset_index()
        df_cus_cnt.columns = ['customer_group','num_cus']
        self.print_debug("[INFO] df_cus_cnt = {}".format(df_cus_cnt))
    
    self.df = self.df.rename({'customer_group':'cluster_group'},axis='columns')    

    for eachclus in self.df['cluster_group'].unique():      
      (self.df[self.df['cluster_group'] == eachclus]).to_csv(self.clusfile_path+"/splitting_input_cluster_"+str(eachclus)+".csv", index=False)       
    
    return True
    
  ###################################################################################################
  # Function: customer_fav_PID(self,NUM_GROUP):
  # Description:
  #   get customer favourites by product ID
  # Inputs:
  #   - self.df
  # Outputs:
  #   - 
  #
  
  def customer_fav_PID(self):
        
    total_customer = self.df.customer_id.nunique()
    customer_id_array = self.df.customer_id.unique()
    NUMCLUS = int(total_customer/self.CLUSTER_DIV_PROCESS_THRESHOLD)+1
    CLUSTER_NUMCUS = int(total_customer/NUMCLUS)+1
    
    if (total_customer) > self.CLUSTER_DIV_PROCESS_THRESHOLD:
        df_temp = self.df.copy()        
        start = 0
        end = 0
        for i in np.arange(NUMCLUS):
          start = i*CLUSTER_NUMCUS
                    
          if i==NUMCLUS:
            end = None
          else:
            end = (i+1)*CLUSTER_NUMCUS
          
          self.df = df_temp[df_temp.customer_id.isin(customer_id_array[start:end])]
          self.tf_customer_favorite_insights(group="_PID"+str(i))

        # summary
        df_fav_pid = pd.DataFrame()
        for i in np.arange(NUMCLUS):
            df_temp = pd.read_csv(self.outpath+"/OUTPUT_customer_favorite_insights_PID"+str(i)+".csv", index_col=False)
            df_fav_pid = pd.concat([df_fav_pid,df_temp])         

        df_fav_pid.to_csv(self.outpath+"/OUTPUT_customer_favorite_insights_PID.csv", index=False)
         
    else:
        self.tf_customer_favorite_insights(group='_PID')    
 
  ################################################################## 
  # this function use for update recommend result
  def customer_fav_PID_4up(self):
        
    total_customer = self.df.customer_id.nunique()
    customer_id_array = self.df.customer_id.unique()
    NUMCLUS = int(total_customer/self.CLUSTER_DIV_PROCESS_THRESHOLD)+1
    CLUSTER_NUMCUS = int(total_customer/NUMCLUS)+1
    
    if (total_customer) > self.CLUSTER_DIV_PROCESS_THRESHOLD:
        df_temp = self.df.copy()        
        start = 0
        end = 0
        for i in np.arange(NUMCLUS):
          start = i*CLUSTER_NUMCUS
                    
          if i==NUMCLUS:
            end = None
          else:
            end = (i+1)*CLUSTER_NUMCUS
          
          self.df = df_temp[df_temp.customer_id.isin(customer_id_array[start:end])]
          self.tf_customer_favorite_insights(group="_PID_4up"+str(i))

        # summary
        df_fav_pid = pd.DataFrame()
        for i in np.arange(NUMCLUS):
            df_temp = pd.read_csv(self.outpath+"/OUTPUT_customer_favorite_insights_PID"+str(i)+".csv", index_col=False)
            df_fav_pid = pd.concat([df_fav_pid,df_temp])         

        df_fav_pid.to_csv(self.outpath+"/OUTPUT_customer_favorite_insights_PID.csv", index=False)
         
    else:
        self.tf_customer_favorite_insights(group='_PID_4up')    
		
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

  ###########################################################################
  # Function: sort_values of dataframe, and get the top values of them
  # return the name of top venues

  def return_most_favorite_rm_noqty1(self, row, NUM_TOP):
    row_sorted = row.sort_values(ascending=False)
    category_msb = row_sorted.index.values[0:NUM_TOP]

    if len(category_msb) < NUM_TOP:
      category_msb = list(category_msb) + list(np.zeros(NUM_TOP-len(category_msb),dtype=int))

    return  [category_msb, 0]
    

  ###################################################################################################
  # TENSORFLOW
  # Function: tf_customer_favorite_insights(self, group='')
  # Description:
  #   find insights of customer such as favourite items, size
  # Inputs: 
  #   - group: for open data file name  [customer_id, items, quantity, 'size']
  # Outputs:
  #   - df_insights ['customer_id',1-4item_fav, 1-3 size]
  #
  
  def tf_customer_favorite_insights(self, group=''):
  
    print("[User Insights] Already got User Insights for cluster file _{} users".format(group))
    
    df_items = self.df[['customer_id',self.item_colname,'quantity']]    
    df_items.customer_id = df_items.customer_id.astype(str)

    ##########################################################
    # Favourite items
    ##########################################################
    
    # pivot data table and find top fav item for all customer
    df_items_pivot = pd.pivot_table(df_items, values='quantity', index=['customer_id'], columns=[self.item_colname], aggfunc=np.sum, fill_value=0) 
    df_items_pivot_val = tf.convert_to_tensor(df_items_pivot.values)
    df_top_val_ind = tf.math.top_k(df_items_pivot_val, self.NUM_FAV_ITEMS) #0: val, 1: index
    
    # get value of quantity and item col index of top k 
    #df_top_val = df_top_val_ind[0]
    df_top_ind_p1 = df_top_val_ind[1] + tf.constant([1]) #plus to ignored Zero index
    
    # clear all index having value = 0    
    df_items_ind = tf.where(tf.equal(tf.dtypes.cast(df_top_val_ind[0],tf.int32), tf.zeros_like(tf.dtypes.cast(df_top_val_ind[0],tf.int32))), 
                            tf.zeros_like(tf.dtypes.cast(df_top_ind_p1,tf.int32)),
                            tf.dtypes.cast(df_top_ind_p1,tf.int32))
    
    # map top favorite items name     
    items_fav_array=[]
    for i in np.arange(tf.shape(df_items_ind)[0].numpy()):      
      items_fav_array.append([df_items_pivot.columns.values[int(ind)-1] if int(ind) > 0 else str(0) for ind in df_items_ind[i]])
   
    # change to numpy --> pandas --> save file
    column = []
    for i in np.arange(self.NUM_FAV_ITEMS):
      column.append('{}_favorite'.format(self.indicators[i]))
    df_fav = pd.DataFrame(data=items_fav_array, columns=column)    
    df_fav['customer_id'] = df_items_pivot.index.values
    column = ['customer_id']+column
    df_fav = df_fav[column]

    ##########################################################
    # Favourite SIZE
    ##########################################################
    df_size = self.df[['customer_id','size','quantity']]    
    df_size.customer_id = df_size.customer_id.astype(str)
   
    # pivot data table and find top fav item for all customer
    df_size_pivot = pd.pivot_table(df_size, values='quantity', index=['customer_id'], columns=['size'], aggfunc=np.sum, fill_value=0) 
    if '0' in df_size_pivot.columns.values:
      df_size_pivot.drop(['0'], axis=1, inplace=True)

    df_size_pivot_val = tf.convert_to_tensor(df_size_pivot.values)
    df_top_val_ind = tf.math.top_k(df_size_pivot_val, self.NUM_FAV_SIZE) #0: val, 1: index
    
    # get value of quantity and item col index of top k 
    #df_top_val = df_top_val_ind[0]
    df_top_ind_p1 = df_top_val_ind[1] + tf.constant([1]) #plus to ignored Zero index
    
    # clear all index having value = 0    
    df_size_ind = tf.where(tf.equal(tf.dtypes.cast(df_top_val_ind[0],tf.int32), tf.zeros_like(tf.dtypes.cast(df_top_val_ind[0],tf.int32))), 
                            tf.zeros_like(tf.dtypes.cast(df_top_ind_p1,tf.int32)),
                            tf.dtypes.cast(df_top_ind_p1,tf.int32))
    
    # map top favorite items name     
    size_fav_array=[]
    for i in np.arange(tf.shape(df_size_ind)[0].numpy()):      
      size_fav_array.append([df_size_pivot.columns.values[int(ind)-1] if int(ind) > 0 else str(0) for ind in df_size_ind[i]])

    df_fav['first_fav_size'] = np.array(size_fav_array).flatten()      

    df_fav.to_csv(self.outpath+"/OUTPUT_customer_favorite_insights"+str(group)+".csv", index=False) # same quantity: diff btw (tf sort, df sort)   

  
  ###################################################################################################
  # Function: customer_favorite_insights(self, group='')
  # Description:
  #   find insights of customer such as favourite items, size
  # Inputs: 
  #   - raw_file_name [customer_id, items, quantity, 'size']
  # Outputs:
  #   - df_insights ['customer_id',1-4item_fav, 1-3 size]
  #

  def customer_favorite_insights(self, group=''):

    df = self.df
    df.customer_id = df.customer_id.astype(str)

    #############################################
    # Create df_recommend
    #############################################
    columns = ['customer_id']
    for i in np.arange(self.NUM_FAV_ITEMS):
      columns.append('{}_favorite'.format(self.indicators[i]))

    for i in np.arange(self.NUM_FAV_SIZE):
      columns.append('{}_fav_size'.format(self.indicators[i]))    
   
    # create a new dataframe
    df_fav = pd.DataFrame(columns=columns)
    df_fav['customer_id'] = df['customer_id'].unique() 
    
    #############################################
    # Create df_fav
    #############################################  

    count=0
    for cus_id in df.customer_id.unique():
      count+=1
      if count%self.PERIOD==0:
        print("[User Insights] Already got User Insights for {} users".format(count))

      #####################################    
      # Get Favorite Items
      #####################################
      df_cus_id = df[df.customer_id==cus_id]
      df_item_id_pivot = pd.pivot_table(df_cus_id, values='quantity', index=['customer_id'], columns=[self.item_colname], aggfunc=np.sum, fill_value=0) 

      #call function to get favourite list
      top_fav_list = self.return_most_favorite_rm_noqty(df_item_id_pivot.iloc[0,:], self.NUM_FAV_ITEMS)[0]
      del[df_item_id_pivot]
      
      #####################################    
      # Get Customer SIZE purchased
      #####################################
      top_fav_size = []
      if df_cus_id['size'].nunique() == 0: 
        top_fav_size = np.zeros(self.NUM_FAV_SIZE).astype(int)
      else:
        df_size_i_pivot = pd.pivot_table(df_cus_id, values='quantity', index=['customer_id'], columns=['size'], aggfunc=np.sum, fill_value=0) 
        top_fav_size = self.return_most_favorite_rm_noqty(df_size_i_pivot.iloc[0,:], self.NUM_FAV_SIZE)[0]   
        
      ###################################
      # Feed data to df_favation
    
      # write values to dataframe
      df_fav.loc[df_fav.customer_id==cus_id, 1:self.NUM_FAV_ITEMS+1] = top_fav_list
      df_fav.loc[df_fav.customer_id==cus_id, self.NUM_FAV_ITEMS+1:] = top_fav_size
      
    df_fav.to_csv(self.outpath+"/OUTPUT_customer_favorite_insights"+str(group)+".csv", index=False)
    return True

  ###################################################################################################
  # Function: generate_final_recommend_v2(self, input_name, group='') - VERSION2
  
  def generate_final_recommend_v2(self, input_name, group=''):
  
    cus_insight_path = self.outpath+'/OUTPUT_customer_favorite_insights'+str(group)+'.csv'
    df_cus_insights = pd.read_csv(cus_insight_path, index_col=None)  
    
    df_cus_recommend = pd.DataFrame()
    if self.userbased:
      cfub_path = self.uboutpath+'/OUTPUT_user_based_recommend'+str(group)+'.csv'
      df_cus_recommend = pd.read_csv(cfub_path, index_col=None)  
      df_cus_recommend = pd.merge(df_cus_insights,df_cus_recommend,how='left',on='customer_id')
      df_cus_recommend.to_csv(self.outpath+"/"+input_name+'_cfub_RECOMMEND'+str(group)+'.csv', index=False)      
  
    if self.itembased:
      cfib_path = self.iboutpath+'/OUTPUT_item_based_recommend'+str(group)+'.csv'
      df_cus_recommend = pd.read_csv(cfib_path, index_col=None)    
      df_cus_recommend = pd.merge(df_cus_insights,df_cus_recommend,how='left',on='customer_id')
      df_cus_recommend.to_csv(self.outpath+"/"+input_name+'_cfib_RECOMMEND'+str(group)+'.csv', index=False)      

    if self.userbased and self.itembased:
      #self.cf_ibub_combine_recommend()
      cfub_path = self.uboutpath+'/OUTPUT_user_based_recommend'+str(group)+'.csv'
      df_cus_recommend = pd.read_csv(cfub_path, index_col=None)      
      cfib_path = self.iboutpath+'/OUTPUT_item_based_recommend'+str(group)+'.csv'
      df_ib = pd.read_csv(cfib_path, index_col=None)
      df_cus_recommend = pd.merge(df_cus_recommend,df_ib,how='left',on='customer_id')
      
      columns = ['customer_id']
      for i in np.arange(self.NUM_BASE_RECOMMEND):
        columns.append('{}_cfub_recommend'.format(self.indicators[i]))
        columns.append('{}_cfib_recommend'.format(self.indicators[i]))
      
      df_cus_recommend = df_cus_recommend[columns]         
 
      #############################
      # process transpose column to row, remove duplicate then pivot to column again

      column_inrow = ['customer_id', 'itemval']
      df_cus_recommend_inrow = df_cus_recommend.T.unstack().reset_index() # dont use customer_id column
      df_cus_recommend_inrow = df_cus_recommend_inrow.drop(columns='level_1')
      df_cus_recommend_inrow.columns = column_inrow
      df_cus_recommend_inrow = df_cus_recommend_inrow.drop_duplicates(subset=column_inrow)
      df_cus_recommend_inrow = df_cus_recommend_inrow.fillna('0')
      df_cus_recommend_inrow['itemval'] = df_cus_recommend_inrow['itemval'].astype(str)
      df_cus_recommend_inrow = df_cus_recommend_inrow[(df_cus_recommend_inrow.itemval!='0')]  
      
      df_cus_recommend_inrow['indexing'] = df_cus_recommend_inrow.groupby('customer_id').cumcount()+1    
   
      #add dummy customer      
      df_dummy = pd.DataFrame(columns=['customer_id','indexing'])
      for i in np.arange(2*self.NUM_BASE_RECOMMEND+1):
        df_dummy.loc[i] = ['DM',i+1]
  
      # concat this dummy rows to df      
      df_cus_recommend_inrow = pd.concat([df_cus_recommend_inrow,df_dummy])#.reset_index(drop=True)
      df_cus_recommend_inrow = pd.pivot_table(df_cus_recommend_inrow, values='itemval', index=['customer_id'], columns=['indexing'], aggfunc=np.sum, fill_value=0).reset_index()
      
      column = df_cus_recommend_inrow.columns.values
      columnx = ['customer_id']
      for i in np.arange(2*self.NUM_BASE_RECOMMEND):
        columnx.append('{}_cf_recommend'.format(self.indicators[i])) 

      df_cus_recommend_inrow = df_cus_recommend_inrow[df_cus_recommend_inrow.customer_id!='DM']
      df_cus_recommend_inrow = df_cus_recommend_inrow[column[1:]]
      df_cus_recommend_inrow.columns = columnx      
      
      df_cus_recommend_inrow['customer_id'] = df_cus_recommend_inrow['customer_id'].astype(str)
      df_cus_insights['customer_id'] = df_cus_insights['customer_id'].astype(str)
      
      df_cus_recommend_inrow = pd.merge(df_cus_insights,df_cus_recommend_inrow,how='left',on='customer_id')      
      df_cus_recommend_inrow.to_csv(self.outpath+"/"+input_name+'_CFUBIB_RECOMMEND'+str(group)+'.csv', index=False)    
    


 
