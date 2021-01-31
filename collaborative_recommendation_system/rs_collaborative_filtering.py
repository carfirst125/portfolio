##############################################################################
# RECOMMENDATION SYSTEM: Collaborative Filtering (User-based & Item-based) 
#-----------------------------------------------------------------------------
# File name:   rs_collaborative_filtering.py 
# Author:      Nhan Thanh Ngo


#############################################################################
# PARAMETER
# Divide big group customer into smaller group before run cluster
# Suggest: define threshold for max size of matrix, depends on size of data (num_cus*num_items)
#          Only devide to N group, where matrix process not exceeding capacity of hardware.

CLUSTER_DIV_PROCESS_THRESHOLD = 50000 #example here, threshold is 50000 customers in a group for running cluster
NUM_NEARBY = 50
NUM_FAV_ITEMS = 4 #number of favorite items
NUM_BASE_RECOMMEND = 15
NUM_FAV_SIZE = 3

###################################################
# IMPORT LIBRARY
##
#import tensorflow as tf

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
import shutil
import os, os.path
#from OpenSSL import SSL

from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree 
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
#import tensorflow as tf

from cf_item_based_class import cf_item_based
from cf_user_based_class_2_0 import cf_user_based
from cf_function_class_2_6 import general_functions

####################################################
# General

####################################################
filename = "rs_collaborative_filtering.py"

####################################################
# Step 0: GetOptions
import sys, getopt
import glob
import time

#default
global debug
debug = False
input = "need_to_specified_table"
query = False
limit = ''
commit = False
commitonly = False
runall = False
model_path = ''
mode = 'both'
user_based = False
item_based = False
user_fav = False
FULL_RUN = True

try:
  opts, args = getopt.getopt(sys.argv[1:], 'hi:m:q:c:u:d:a', ['help','input=','mode=','query','commit','commitonly','debug','updateall'])

except getopt.GetoptError as err:
  print ("ERROR: Getoption gets error... please check!\n {}",err)
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-q', '--query'):
    query = True
  if opt in ('-i', '--input'):
    input = str(arg)
  if opt in ('-m', '--mode'):
    mode = str(arg)
  if opt in ('-c', '--commit'):
    commit = True
  if opt in ('-u', '--commitonly'):
    commitonly = True
  if opt in ('-d', '--debug'):
    debug = True
  if opt in ('-g', '--updateall'):
    FULL_RUN = True
  if opt in ('-h', '--help'):
    parser.print_help()
    sys.exit(2)
  
else:
   print("[Error] Please check --model=... to be sure that you enter the right value. If not, please leave this option.")
    
if mode == 'userbased':
  user_based = True
  item_based = False
  user_fav = True
  
elif mode == 'userbasednotfav':
  user_based = True
  item_based = False
  user_fav = False
 
elif mode == 'itembased':
  user_based = False
  item_based = True
  user_fav = True
 
elif mode == 'itembasednotfav':
  user_based = False
  item_based = True
  user_fav = False
 
elif mode == 'both':
  user_based = True
  item_based = True
  user_fav = True
  
elif mode=='test':
  user_based = True
  item_based = True
  user_fav = True 
  
print ("This is {}\n".format(filename))

print("##################################")
print("ARGUMENTS:")
print("Input Name  : {}".format(input))
print("Mode        : {}".format(mode))
print("Debug mode  : {}".format(debug))
print("Query mode  : {}".format(query))
print("Limit       : {}".format(limit))
print("Commit      : {}".format(commit))
print("CommitOnly  : {}".format(commitonly))
print("##################################")

parameter_file='parameters.yaml'
parameter_dict = {}
with open(r'./'+parameter_file) as file:
  # The FullLoader parameter handles the conversion from YAML
  # scalar values to Python the dictionary format
  parameter_dict = yaml.load(file, Loader=yaml.FullLoader)
  #print('[MAIN] parameter_dict = {}'.format(parameter_dict))
  
for key in parameter_dict:
  if key == 'CLUSTER_DIV_PROCESS_THRESHOLD':
    CLUSTER_DIV_PROCESS_THRESHOLD = parameter_dict[key]
  elif key == 'NUM_NEARBY':
    NUM_NEARBY = parameter_dict[key]
  elif key == 'NUM_FAV_ITEMS':
    NUM_FAV_ITEMS = parameter_dict[key]
  elif key == 'NUM_BASE_RECOMMEND':
    NUM_BASE_RECOMMEND = parameter_dict[key]
  elif key == 'NUM_SOH_RECOMMEND':
    NUM_SOH_RECOMMEND = parameter_dict[key]
  elif key == 'NUM_FAV_SIZE':
    NUM_FAV_SIZE = parameter_dict[key] 

###########################################
#create DEBUG directory
#

cwd = os.getcwd()
print (cwd)
homepath = "./"+input.split('.')[-1]+"_rscf"

#### create log folder path
if os.path.exists(homepath):
  print ("\'{}\' is already EXISTED!".format(homepath))
else:
  os.mkdir(homepath)
  print ("\'{}\' is CREATED!".format(homepath))

#### debug_log file path
debug_log = "debug.log"
if os.path.isfile(homepath+"/"+debug_log):
  os.remove(homepath+"/"+debug_log)

#### report_log file path
report_log = "report.log"
if os.path.isfile(homepath+"/"+report_log):
  os.remove(homepath+"/"+report_log)
  
#### where the model is saved
modelpath = homepath+"/model"
if os.path.exists(modelpath):
  print ("\'{}\' is already EXISTED!".format(modelpath))
else:
  os.mkdir(modelpath)
  print ("\'{}\' is CREATED!".format(modelpath))

#### where the model is saved
temppath = homepath+"/temp"
if os.path.exists(temppath):
  print ("\'{}\' is already EXISTED!".format(temppath))
else:
  os.mkdir(temppath)
  print ("\'{}\' is CREATED!".format(temppath))
  
# clusterfile
clusfile_path = homepath+"/clusfile"
if os.path.exists(clusfile_path):  
  print ("\'{}\' is already EXISTED!".format(clusfile_path))
  shutil.rmtree(clusfile_path)
  os.mkdir(clusfile_path)
else:
  os.mkdir(clusfile_path)
  print ("\'{}\' is CREATED!".format(clusfile_path))  

# store all predict output data
recommend_base = homepath+"/recommend_base"
if os.path.exists(recommend_base):
  print ("\'{}\' is already EXISTED!".format(recommend_base))
else:
  os.mkdir(recommend_base)
  print ("\'{}\' is CREATED!".format(recommend_base))
  
# store all predict output data
outpath = homepath+"/output"
if os.path.exists(outpath):
  print ("\'{}\' is already EXISTED!".format(outpath))
else:
  os.mkdir(outpath)
  print ("\'{}\' is CREATED!".format(outpath))

######################################################################################################
# FUNCTION LIBRARY
######################################################################################################

def print_debug (strcont):
  debugn = debug
  if debugn:
    print(strcont)  
    with open(homepath+"/"+debug_log, "a") as logfile:
      logfile.write("\n"+strcont)

def recommend_history(updateall, input_name):

  year = datetime.datetime.today().year
  month = datetime.datetime.today().month
  day = datetime.datetime.today().day
  CURRENT_DATE = datetime.datetime(int(year),int(month),int(day)).strftime("%Y-%m-%d") # + datetime.timedelta(days=-backsplit)
  subs = ''
  PREVIOUS_3DAY = (datetime.datetime(int(year),int(month),int(day)) + datetime.timedelta(days=-3)).strftime("%Y-%m-%d")
  
  if updateall:
    subs = '_FULL_RUN'
  
  # remove the previous 03Day backup data
  removepath = homepath+"/recommend_base/"+PREVIOUS_3DAY+subs
  if os.path.exists(removepath):
    os.system('chmod 777 {}'.format(homepath+"/recommend_base/*"))
    os.system('chmod 777 {}'.format(homepath+"/recommend_base/*/*"))
    shutil.rmtree(removepath)

  # create recommend history folder
  recommend_his = homepath+"/recommend_base/"+CURRENT_DATE+subs
  if os.path.exists(recommend_his):
    print ("\'{}\' is already EXISTED!".format(recommend_his))
  else:
    os.mkdir(recommend_his)
    print ("\'{}\' is CREATED!".format(recommend_his))
  
  source_path = homepath+"/item_based/output/similar_item_pair_summary_group.csv"
  if os.path.exists(source_path): 
    destination_path = recommend_his+"/similar_item_pair_summary_group.csv"
    shutil.copyfile(source_path, destination_path)
 
  source_path = homepath+"/item_based/output/OUTPUT_item_based_recommend.csv"
  if os.path.exists(source_path): 
    destination_path = recommend_his+"/OUTPUT_item_based_recommend.csv"
    shutil.copyfile(source_path, destination_path)      
      
  source_path = homepath+"/user_based/output/OUTPUT_user_based_recommend.csv"
  if os.path.exists(source_path):   
    destination_path = recommend_his+"/OUTPUT_user_based_recommend.csv"
    shutil.copyfile(source_path, destination_path)    

  source_path = homepath+"/output/OUTPUT_customer_favorite_insights.csv"
  if os.path.exists(source_path):   
    destination_path = recommend_his+"/OUTPUT_customer_favorite_insights.csv"
    shutil.copyfile(source_path, destination_path)    

  source_path = homepath+"/output/OUTPUT_customer_favorite_insights_PID.csv"
  if os.path.exists(source_path):   
    destination_path = recommend_his+"/OUTPUT_customer_favorite_insights_PID.csv"
    shutil.copyfile(source_path, destination_path)    
	
  source_path = homepath+"/output/cfib_RECOMMEND.csv"
  if os.path.exists(source_path): 
    destination_path = recommend_his+"/cfib_RECOMMEND.csv"
    shutil.copyfile(source_path, destination_path)     
  
  source_path = homepath+"/output/cfub_RECOMMEND.csv"
  if os.path.exists(source_path): 
    destination_path = recommend_his+"/cfub_RECOMMEND.csv"
    shutil.copyfile(source_path, destination_path)

  source_path = homepath+"/output/"+input_name+"_cfub_RECOMMEND.csv"
  if os.path.exists(source_path): 
    destination_path = recommend_his+"/"+input_name+"_cfub_RECOMMEND.csv"
    shutil.copyfile(source_path, destination_path)    
 
  source_path = homepath+"/output/result_FAV_CFIBUB_RECOMMEND_PID.csv"
  if os.path.exists(source_path): 
    destination_path = recommend_his+"/result_FAV_CFIBUB_RECOMMEND_PID.csv"
    shutil.copyfile(source_path, destination_path)   

  source_path = homepath+"/lastday_in_data_previous_run.log"
  if os.path.exists(source_path): 
    destination_path = recommend_his+"/lastday_in_data_previous_run.log"
    shutil.copyfile(source_path, destination_path)
  
  print("Backup history previous result completed!")  

#######################################################
# USER-BASED 
#######################################################

###################################################################################################
# Function: cf_user_based_recommend(df)
# Description:
#   input data after process --> cf_item_based_similar_items_explore() 
#   sweep input datafile to figure out item pair
# Inputs:
#   - df: partition dataframe
# Outputs:
#   - information of similar item pair in file [item_a, item_b, qty_a, qty_b, customer_count]
#

def cf_user_based_recommend(df, group=''):
    
  cfub_obj = cf_user_based(df, homepath)

  print ('####################################################')
  print ('# [User-based] KDTREE: TRAIN NEAREST NEIGHBOR MODEL')
  print ('####################################################')
  cfub_obj.cf_user_based_kdtree_train() 
  
  print ('#############################################')
  print ('# [User-based] KDTREE: FIND NEAREST NEIGHBOR ')
  print ('#############################################')  
  cfub_obj.cf_user_based_kdtree_nearest_neighbor_explore(group)   
  
  print ('##############################################################################')
  print ('# [User-based] RECOMMEND: EXPLORE IN NEAREST NEIGHBOR TO FIND RECOMMEND ITEMS')
  print ('##############################################################################')     
  cfub_obj.cf_insight_nearby_cus_and_recommend(group) 
  
  return True

###################################################################################################
# Function: lookup_product_id(df_rcm, df_product_id, NUM_RCM_ITEM)
# Description:
#   [items+size] --> [product_id]
#   from result of recommendation of collaborative filtering, and favorite size --> look up the product id in df_product_id
# Inputs:
#   - df: partition dataframe
# Outputs:
#   - return recommendation result in product id
#

def lookup_product_id(df_rcm, df_product_id, NUM_RCM_ITEM):

  print('[lookup_product_id] Product ID looking up...')
  
  df_size = df_rcm[['customer_id','first_fav_size']]
  df_rcm.drop('first_fav_size', axis=1, inplace=True)
  
  df_rcm.set_index('customer_id', inplace=True)
  df_rcm = df_rcm.T.unstack().reset_index()
  df_rcm = pd.merge(df_rcm, df_size, on='customer_id',how='left')

  df_rcm = df_rcm.drop(columns='level_1')
  df_rcm.columns = ['customer_id','items','size']
  df_rcm['items'] = df_rcm['items'].astype(str)
  df_rcm = df_rcm[df_rcm['items']!='0']
  #df_rcm = df_rcm[df_rcm['items']!=0]

  df_rcm.loc[df_rcm['items'].str.contains('BANH')| \
             df_rcm['items'].str.contains('BM')| \
             df_rcm['items'].str.contains('CAKE')| \
             df_rcm['items'].str.contains('BREAD')| \
             df_rcm['items'].str.contains('COOKIES '), 'size'] = 'B'
			 
  df_rcm = pd.merge(df_rcm, df_product_id, on=['items','size'], how='left')
  
  # dummy data
  df_dummy = pd.DataFrame(columns = df_rcm.columns.values)
  value = np.empty(NUM_RCM_ITEM, dtype=np.str)
  value.fill('D')
  df_dummy['items'] = np.ones(NUM_RCM_ITEM)
  df_dummy['customer_id'] = value
  df_rcm = pd.concat([df_rcm,df_dummy])
  
  df_rcm['indexing'] = df_rcm.groupby('customer_id').cumcount() + 1

  #df_rcm.to_csv('./df_rcm.csv', index=False)
  df_rcm = pd.pivot_table(df_rcm, values='product_id', index=['customer_id'], columns=['indexing'], aggfunc=np.sum, fill_value=0).reset_index()
  df_rcm = df_rcm[df_rcm['customer_id']!='D']
  df_rcm = df_rcm.iloc[:,:NUM_RCM_ITEM+1]
  df_rcm.to_csv(temppath+"/df_rcm.csv", index=False) 
  print('lookup_product_id completed!')
  
  return df_rcm


###################################################################################################
# Function: common_member(a, b)
# Description:
#     find elements in array a and also in array b
# Inputs:
#   - df: partition dataframe
# Outputs:
#   - information of similar item pair in file [item_a, item_b, qty_a, qty_b, customer_count]
#

def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        return list((a_set & b_set))
    else: 
        print("No common elements")
        return []
  
###################################################################################################
# Function: cleaning()
# Description:
#     clean all temporary files that generated by the script

def cleaning():

  # clean clusfile path
  if os.path.exists(clusfile_path):
    shutil.rmtree(clusfile_path)
  if os.path.exists(homepath+'/item_based/temp/'):
    shutil.rmtree(homepath+'/item_based/temp/')
  
  # clean in item based folder
  pattern = "^([\w]+)(_FAV_ITEMS)([\d]+).csv$"
  mypath = homepath+'/item_based/output/'
  for root, dirs, files in os.walk(mypath):
    for file in filter(lambda x: re.match(pattern, x), files):
      print(file)
      os.remove(os.path.join(root, file))
		
  pattern = "^([\w]+)(_recommend)([\d]+).csv$"
  mypath = homepath+'/item_based/output/'
  for root, dirs, files in os.walk(mypath):
    for file in filter(lambda x: re.match(pattern, x), files):
      print(file)
      os.remove(os.path.join(root, file))

  # clean in user based
  pattern = "^([\w]+)(_recommend)([\d]+).csv$"
  mypath = homepath+'/user_based/output/'
  for root, dirs, files in os.walk(mypath):
    for file in filter(lambda x: re.match(pattern, x), files):
      print(file)
      os.remove(os.path.join(root, file))
	  	  
  # remove in output folder
  pattern = "^([\w]+)(_insights_PID)([\d]+).csv$"
  mypath = homepath+'/output/'
  for root, dirs, files in os.walk(mypath):
    for file in filter(lambda x: re.match(pattern, x), files):
      print(file)
      os.remove(os.path.join(root, file))

  pattern = "^([\w]+)(_favorite_insights)([\d]+).csv$"
  mypath = homepath+'/output/'
  for root, dirs, files in os.walk(mypath):
    for file in filter(lambda x: re.match(pattern, x), files):
      print(file)
      os.remove(os.path.join(root, file))	
	  
###########################################################################################################################################
# MAIN
###########################################################################################################################################

df = pd.DataFrame()
if not commitonly:
  print ('#############################')
  print ('# QUERY DATA FROM BIGQUERY')
  print ('#############################')

  from google.cloud import bigquery
  from google.oauth2 import service_account

  client = bigquery.Client()

  sql = "SELECT * FROM "+input+limit
  bq_cus_purchase = input.split('.')[-1]+".csv"

  ###################################3
  # QUERY
  ###################################
  if query:
    # Run a Standard SQL query with the project set explicitly
    print("Querying data from sandbox...\nit will take a few minutes...")
    df = client.query(sql).to_dataframe() 
    
    if not glob.glob(bq_cus_purchase):
      print ("[INFO] No BigQuery datafile available")
    else:
      print("[INFO] Remove exist bq datafile")
      os.remove(bq_cus_purchase)
    
    print("[INFO] Store query data from Big Query to file")
    df.to_csv(bq_cus_purchase,index=False)
  else:
    print("[INFO] Read input data from offline file, need updateall please run again with -q to query new data from Big Query")
    print("Read offline input data file...\nit will take a few minutes...")
    df = pd.read_csv(bq_cus_purchase, index_col=None)
  
  ##############################################################################################################################
  ##############################################################################################################################
  
  print ('##########################################')
  print ('# INPUT DATAFRAME PROCESS')
  print ('##########################################') 
            
  # preprocessing
  df['date_order'] = pd.to_datetime(df.date_order)  
  df.customer_id = df.customer_id.astype('str')  
  
  # drop value of customer_id and items which have NULL value
  df.dropna(subset=['customer_id','items'], inplace=True)
  df['size'].fillna('0',inplace=True)
  df['size'] = df['size'].astype(str)
  print("[Before EDA] Total customer is {}".format(df.customer_id.nunique()))  
  
  # LIMIT NUM OF CUSTOMER FOR TEST
  #df = df[df.customer_id.isin(df.customer_id.unique()[:9000])]
  
  # get necessary table
  df = df[['customer_id','date_order','product_id','items','size','quantity']]  
  
  print ('##########################################')
  print ('# EDA                                     ')
  print ('##########################################') 
  
  # eda to get main products (remove ones that are not main product)
  func_obji = general_functions(df, homepath, mode=mode, debug=debug)    
  df = func_obji.product_eda() 
  df.to_csv(temppath+"/df_after_eda.csv", index=False)  
  print('[After EDA] Total customer is {}'.format(df.customer_id.nunique()))
  del func_obji
  
  print('Complete...')
    
  print ('##########################################')
  print ('# CUSTOMER FAV BY PRODUCT ID')
  print ('##########################################')    
    
  df_pid = df[['customer_id','product_id','size','quantity']] 
  df_pid.columns = ['customer_id','items','size','quantity']  
  func_obj = general_functions(df_pid, homepath, mode=mode, debug=debug)    
  func_obj.customer_fav_PID()
  del func_obj

  print('Complete...')    
 
  ##########################################  

  # back up all import file to new folder before run, not append
  if os.path.exists(homepath+"/lastday_in_data_previous_run.log"):
    print ("This is not initial run, lastday_in_data_previous_run.log exists.")
    with open(homepath+"/lastday_in_data_previous_run.log", mode='r') as file:
      previous_run_date = file.readline()
      PRE_RUNDATE = datetime.datetime.strptime(previous_run_date,'%Y-%m-%d')
    
    # having this file, it means not first time run --> backup
    recommend_history(FULL_RUN, input.split('.')[-1])   
  
  else:
    print ("This is INITIAL RUN of this data.")
    FULL_RUN = True
  
  ###########################################################################################################################################
  
  if FULL_RUN:
    cluster_column = 'items' # might be categories
    ###############################
    # PREPARATION                ##
    ###############################
    if os.path.exists(homepath+"/user_based/temp/cus_group.log"):
      os.remove(homepath+"/user_based/temp/cus_group.log")

    LAST_DAY = df.date_order.max().strftime("%Y-%m-%d")
    print('LAST_DAY:', LAST_DAY)    

    print ('##########################################')
    print ('# FILE CLUSTER BIG --> SMALL GROUPs')
    print ('##########################################')  
    # Declare general_function object
    df = pd.read_csv(temppath+"/df_after_eda.csv", index_col=False) 
    df['customer_group'] = '0'	
    df = df[['customer_id','customer_group','items','size','quantity']]
    df = df.groupby(['customer_id','customer_group','items','size'])['quantity'].sum().reset_index()
	
    print_debug ("[INFO] length of input data before process: {}".format(len(df)))
    print_debug ("[INFO] df.columns = {}".format(df.columns))
    print_debug ('[INFO] TOTAL CUSTOMER: {}'.format(df.customer_id.nunique()))
	
    # file clustering
    func_obj = general_functions(df, homepath, mode=mode, debug=debug)   
    func_obj.df_file_cluster() # cluster completed
    del func_obj
	
    print('Complete...')	 

    dir_list = os.listdir(clusfile_path)
    # check cluster file num of customer  
    for group, clusfile in zip(np.arange(len(dir_list)),dir_list):
      df = pd.read_csv(clusfile_path+"/"+clusfile, index_col=None) 
      if df.customer_id.nunique() < NUM_NEARBY:
        print("ERROR: in data, please re-cluster or reduce NUM_NEARBY, the number of customer_id in cluster must greater than NUM_NEARBY")
        sys.exit()
      else:
        print("[CHECKING DATA] [{}] cluster file {} having {} customer_id".format(group,clusfile, df.customer_id.nunique()))  
  
    ################################################################################
    # CUSTOMER FAVORITE 
    ################################################################################
    
    if user_fav:
      print ('####################################################################')
      print ('# CUSTOMER INSIGHTS: ITEMS, SIZE FAVORITE')
      print ('####################################################################')  
  
      # run user_based for each cluster_group
      for group, clusfile in zip(np.arange(len(dir_list)),dir_list):
        print ('[Cluster] {}'.format(clusfile))
        df = pd.read_csv(clusfile_path+"/"+clusfile, index_col=None)   
  
        func_obj = general_functions(df, homepath, mode=mode, debug=debug)   
        func_obj.tf_customer_favorite_insights(group)
		
      #--------------------------------
      # summary all
      df_fav = pd.DataFrame()
      for group in np.arange(len(dir_list)):
        df_temp = pd.read_csv(outpath+"/OUTPUT_customer_favorite_insights"+str(group)+".csv", index_col=None) 
        df_fav = pd.concat([df_fav,df_temp])
      df_fav.to_csv(outpath+"/OUTPUT_customer_favorite_insights.csv", index=False)

      print ('##################################################################')
      print ('# Customer insights: DISCOVERY COMPLETED')
      print ('##################################################################') 


    ################################################################################
    # USER-BASED
    ################################################################################
  
    if user_based:
      print ('####################################################################')
      print ('# USER-BASED')
      print ('####################################################################')  
  
      # run user_based for each cluster_group
      for group, clusfile in zip(np.arange(len(dir_list)),dir_list):
        df = pd.read_csv(clusfile_path+"/"+clusfile, index_col=None)    
        df = df[['customer_id','cluster_group',cluster_column,'quantity']]   
        df[cluster_column] = df[cluster_column].astype(str)         
        cf_user_based_recommend(df, group) 
        
      #-----------------------------------
      # RECOMMENDATION: Summarize all
      df_recommend = pd.DataFrame()
      for group in np.arange(len(dir_list)):
        df_temp = pd.read_csv(homepath+"/user_based/output/OUTPUT_user_based_recommend"+str(group)+".csv", index_col=None) 
        df_recommend = pd.concat([df_recommend,df_temp])
      df_recommend.to_csv(homepath+"/user_based/output/OUTPUT_user_based_recommend.csv", index=False)

      print ('##################################################################')
      print ('# User-based RECOMMENDATION COMPLETED')
      print ('##################################################################')  
    
    ################################################################################
    # ITEM-BASED
    ################################################################################
    if item_based:
      #if updateall:      
      #######################################
      # SIMILAR ITEM-PAIR: For each cluster
      #######################################
      for group, clusfile in zip(np.arange(len(dir_list)),dir_list):
        print ('[Cluster] {}'.format(clusfile))
        df = pd.read_csv(clusfile_path+"/"+clusfile, index_col=None)       

        df = df[['customer_id','cluster_group',cluster_column,'quantity']]
        cfib_obj = cf_item_based(df, homepath)

        cfib_obj.cf_similar_item_pair_stats_v2(group)
        
      #------------------------------------- 
      # Summarize all
      cfib_obj = cf_item_based(0, homepath)
      cfib_obj.cf_similar_item_pair_summary(len(dir_list))
      #cfib_obj.tf_item_and_recommend_list()

      #######################################
      # RECOMMENDATION: For each cluster
      #######################################
      for group, clusfile in zip(np.arange(len(dir_list)),dir_list):
        print ('[Cluster] {}'.format(clusfile))
        df = pd.read_csv(clusfile_path+"/"+clusfile, index_col=None)       
 
        df = df[['customer_id','cluster_group',cluster_column,'quantity']]
        cfib_obj = cf_item_based(df, homepath)
        cfib_obj.cf_customer_recommend_main_v2(group)
   
      #--------------------------------
      # Summarize all    
      df_recommend = pd.DataFrame()
      for group in np.arange(len(dir_list)):
        df_temp = pd.read_csv(homepath+"/item_based/output/OUTPUT_item_based_recommend"+str(group)+".csv", index_col=None) 
        df_recommend = pd.concat([df_recommend,df_temp])
      df_recommend.to_csv(homepath+"/item_based/output/OUTPUT_item_based_recommend.csv", index=False)
  
      print ('##################################################################')
      print ('# Item-based RECOMMENDATION COMPLETED')
      print ('##################################################################') 
  

    # write file lastday_in_data_previous_run.log last day of data
    with open(homepath+"/lastday_in_data_previous_run.log", mode='w') as file:
      file.write('%s' %(LAST_DAY))    
  
  ##############################################################
  # Product ID recommend
  ##############################################################
  print('[ProductID] REPLACE ITEM NAME by PRODUCT ID')
  # Product id lookup table
  df_product_id = pd.read_csv(temppath+"/df_after_eda.csv", index_col=False) 
  df_product_id = df_product_id[['items','size','product_id']]
  df_product_id['product_id'] = df_product_id['product_id'].astype(int)
  df_product_id['size'] = df_product_id['size'].astype(str)
  df_product_id.loc[df_product_id['size']=='0','size'] = 'B' #Banh + others (NO SIZE)
  
  df_product_id['rm'] = 1
  df_product_id = df_product_id.groupby(['items','size','product_id'])['rm'].sum().reset_index()
  df_product_id.drop('rm',axis=1,inplace=True)

  df_product_id.to_csv(temppath+'/df_product_id.csv', index=False) 
 
  indicators = ['first','second','third','fourth','fifth','sixth','seventh','eighth','nineth','tenth','eleven','twelfth','thirthteen','forthteen','fifthteen']
  
  #------------------------------------------------------------- 
  # Product_id lookup for item based result
  ib_output = homepath+"/item_based/output/OUTPUT_item_based_recommend.csv"
  df = pd.read_csv(ib_output, index_col=False) 
  df.loc[df.first_fav_size=='0', 'first_fav_size'] = 'Size M'
  
  column = ['customer_id', 'first_fav_size']
  for i in np.arange(2*NUM_BASE_RECOMMEND):
    column.append('{}_cfib_recommend'.format(indicators[i]))
  df = df[column]    

  # Call items + fav size -- product_id lookup
  df = lookup_product_id(df, df_product_id,NUM_BASE_RECOMMEND)  
  
  column = ['customer_id']
  for i in np.arange(NUM_BASE_RECOMMEND):
    column.append('cfib_'+str(i+1))  
  df.columns = column
  #df = df.iloc[:,0:NUM_BASE_RECOMMEND+1]
  
  df.to_csv(outpath+'/cfib_RECOMMEND.csv', index=False)    
  #------------------------------------------------------------- 
  # Product_id lookup for user based result
  cus_insight_path = outpath+'/OUTPUT_customer_favorite_insights.csv'
  df_fav = pd.read_csv(cus_insight_path, index_col=None)  
  df_fav = df_fav[['customer_id','first_fav_size']]
  
  ub_output = homepath+"/user_based/output/OUTPUT_user_based_recommend.csv"
  df = pd.read_csv(ub_output, index_col=False) 
  df = pd.merge(df,df_fav,how='left',on='customer_id')
  df.loc[df.first_fav_size=='0', 'first_fav_size'] = 'Size M' 
  # in EDA might cover all missing size, customer only buy cakes that might not have size.

  column = ['customer_id', 'first_fav_size']
  for i in np.arange(NUM_BASE_RECOMMEND):
    column.append('{}_cfub_recommend'.format(indicators[i]))
  df = df[column]    

  # Call items + fav size -- product_id lookup
  df = lookup_product_id(df, df_product_id,NUM_BASE_RECOMMEND)  
  
  column = ['customer_id']
  for i in np.arange(NUM_BASE_RECOMMEND):
    column.append('cfub_'+str(i+1))
  df.columns = column
  df = df.iloc[:,0:NUM_BASE_RECOMMEND+1]
  
  df.to_csv(outpath+'/cfub_RECOMMEND.csv', index=False)    
  #-------------------------------------------------------------  
  # combine all result ['customer_id','fav_size', 'fav1',...,'fav5','cfib1',...,'cfib5','cfub1',...,'cfub5']
  df = pd.read_csv(outpath+"/OUTPUT_customer_favorite_insights_PID.csv", index_col=False)   
  
  column = ['customer_id']
  for i in np.arange(NUM_FAV_ITEMS):
    column.append('fav_'+str(i+1))  
  column.append('fav_size')
  df.columns = column
  
  df_temp = pd.read_csv(outpath+'/cfib_RECOMMEND.csv', index_col=False)   
  df = pd.merge(df, df_temp, on='customer_id',how='left')
  df_temp = pd.read_csv(outpath+'/cfub_RECOMMEND.csv', index_col=False)   
  df = pd.merge(df, df_temp, on='customer_id',how='left')  
  df = df.fillna('0')
  
  df.to_csv(outpath+'/result_FAV_CFIBUB_RECOMMEND_PID.csv', index=False)
  
  #############################################################################
  # Cleaning all temporary files and folders
  
  cleaning()
  
if commit or commitonly:
  
  print ('##################################')
  print ('# COMMIT RESULT TO DW')
  print ('##################################')
  commit_filelist = ''
  commit_sandbox_name = ''
  
  if mode == 'both':
    if FULL_RUN:      
      commit_filelist = [outpath+'/result_FAV_CFIBUB_RECOMMEND_PID.csv', homepath+'/item_based/output/similar_item_pair_summary_group.csv']
      commit_sandbox_name = [input+'_CFUBIB_RECOMMEND',input+'_SIMILAR_ITEMS']
    else:
      commit_filelist = [outpath+'/result_FAV_CFIBUB_RECOMMEND_PID.csv']
      commit_sandbox_name = [input+'_CFUBIB_RECOMMEND']
  
  for i in range(len(commit_filelist)):
    os.system('python commit_bq.py --source {} --destination {}'.format(commit_filelist[i],commit_sandbox_name[i]))


print ('##################################################################')
print ('# RECOMMENDATION SUCESSFULLY COMPLETED!!!')
print ('##################################################################')     





  
  
