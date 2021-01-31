##############################################################################
# RECOMMENDATION SYSTEM: TRAIN PROCESS
#-----------------------------------------------------------------------------
# File name:   cfubib_recommend_train.py 
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
NUM_CUS_KDTREE = 50000

###################################################
# IMPORT LIBRARY
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
import datetime

import sys, getopt
import glob
import time
from optparse import OptionParser
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.neighbors import KDTree 
from cf_function_class import general_functions


####################################################
# General variable declaration
####################################################

#default
global debug
debug = False
input = "DW.table_name"
query = False
limit = ''
modelpath = None
mode = 'TRAIN'

####################################################
# GetOptions
####################################################
                       
try:
  opts, args = getopt.getopt(sys.argv[1:], 'hi:m:q:d', ['help','input=','mode=','query','debug'])

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
  if opt in ('-d', '--debug'):
    debug = True
  if opt in ('-h', '--help'):
    parser.print_help()
    sys.exit(2)
    
else:
   print("[Error] Please check {Options} to be sure that you enter the right value. If not, please leave this option.")
    
homepath = "./"+input.split('.')[-1]+"_train"
temppath = outpath = clusfile_path = ''
lbl_homepath = "./ml_"+input.split('.')[-1]+"_rscf"

######################################################
# Get configuration in .yaml file
######################################################
parameter_file='parameters.yaml'
parameter_dict = {}
with open(r'./'+parameter_file) as file:
  parameter_dict = yaml.load(file, Loader=yaml.FullLoader)

for key in parameter_dict:
  if key == 'CLUSTER_DIV_PROCESS_THRESHOLD':
    CLUSTER_DIV_PROCESS_THRESHOLD = parameter_dict[key]
  elif key == 'NUM_NEARBY':
    NUM_NEARBY = parameter_dict[key]
  elif key == 'NUM_FAV_ITEMS':
    NUM_FAV_ITEMS = parameter_dict[key]
  elif key == 'NUM_BASE_RECOMMEND':
    NUM_BASE_RECOMMEND = parameter_dict[key]
  elif key == 'NUM_FAV_SIZE':
    NUM_FAV_SIZE = parameter_dict[key] 
  elif key == 'NUM_CUS_KDTREE':
    NUM_CUS_KDTREE = parameter_dict[key] 

#######################################################
# FUNCTION
#######################################################

#######################################################
# Function: setup_folder(homepath)
# Description: 
#   set up neccessary folder for
#

def setup_folder(homepath):

  global temppath, outpath, clusfile_path, modelpath
  
  # home
  if os.path.exists(homepath):
    print ("\'{}\' is already EXISTED!".format(homepath))
  else:
    os.mkdir(homepath)
    print ("\'{}\' is CREATED!".format(homepath)) 
  
  # temp
  temppath = homepath+'/temp'
  if os.path.exists(temppath):
    print ("\'{}\' is already EXISTED!".format(temppath))
  else:
    os.mkdir(temppath)
    print ("\'{}\' is CREATED!".format(temppath))

  # model
  modelpath = homepath+'/model'
  if os.path.exists(modelpath):
    print ("\'{}\' is already EXISTED!".format(modelpath))
  else:
    os.mkdir(modelpath)
    print ("\'{}\' is CREATED!".format(modelpath))
	
  # output
  outpath = homepath+'/output'
  if os.path.exists(outpath):
    print ("\'{}\' is already EXISTED!".format(outpath))
  else:
    os.mkdir(outpath)
    print ("\'{}\' is CREATED!".format(outpath))

  # clusterfile
  clusfile_path = homepath+"/clusfile"
  if os.path.exists(clusfile_path):  
    print ("\'{}\' is already EXISTED!".format(clusfile_path))
    shutil.rmtree(clusfile_path)
    os.mkdir(clusfile_path)
  else:
    os.mkdir(clusfile_path)
    print ("\'{}\' is CREATED!".format(clusfile_path))  
  
######################################################
# Function: query_data(input,query)
# Description:
#     query data from DW to local PC
# 
# Input: 
#    - input: table name in DW (eg. DW.input_table)
#    - query: True/False
# Output:
#    query data and store in local location
#    

def query_data(input, query):

  print ('#############################')
  print ('# QUERY DATA FROM BIGQUERY')
  print ('#############################')

  from google.cloud import bigquery
  from google.oauth2 import service_account

  # query in DW (credentials is available in production machine)
  client = bigquery.Client()
  
  sql = "SELECT * FROM "+input
  bq_cus_purchase = input.split('.')[-1]+".csv"

  if query:
    # Run a Standard SQL query with the project set explicitly
    print("Querying data from sandbox...\nit will take a few minutes...")
    df = client.query(sql).to_dataframe() 
    #df = client.query(sql, project=project_id).to_dataframe()
    
    if not glob.glob(bq_cus_purchase):
      print ("[INFO] No BigQuery datafile available")
    else:
      print("[INFO] Remove exist bq datafile")
      os.remove(bq_cus_purchase)
    
    print("[INFO] Store query data from Big Query to file")
    df.to_csv(bq_cus_purchase,index=False)

######################################################
# Function: data_preprocessing(df)
# Description:
#     pre-processing input data 
# Input: 
#    - df: input dataframe
# Output:
#    df after cleaning, wrangling
# 

def data_preprocessing(df):

  # preprocessing
  df['date_order'] = pd.to_datetime(df.date_order)

  df.customer_id = df.customer_id.astype(str)
  df.dropna(subset=['customer_id','items'], inplace=True)
  df['size'].fillna('0',inplace=True)  
  df['size'] = df['size'].astype(str)
  print("[Before EDA] Total customer is {}".format(df.customer_id.nunique()))
  
  # remove extra
  df = df[df.check_main=='main'] 
  # get necessary table
  df = df[['customer_id','date_order','items','size','quantity']]

  # eda to get main products (remove ones that are not main product)
  func_obji = general_functions(df, homepath, mode=mode, debug=debug)    
  df = func_obji.product_eda() 
  
  #df.to_csv(temppath+"/df_after_eda.csv", index=False)  
  print('[After EDA] Total customer is {}'.format(df.customer_id.nunique()))

  return df
  
  
######################################################
# Function: train_model(df_in)
# Description:
#     train model 
# Input: 
#    - df_in: input data in dataframe which will be used to kdtree
# Output:
#    - the kdtree model
#

def train_model(df_in):

  print('[train_model()] model training...')
  model = KDTree(df_in.values)
  print('[train_model()] training completed')
  
  return model

######################################################
# Function: model_input_features(df_in)
# Description:
#    get input vector of kdtree
#    this input vector will be saved to file, and recall for using when predicting
# Input: 
#    - df_in: input data in dataframe which will be used to kdtree
# Output:
#    - outfile: modelpath+'/model_input_features.txt'
#

def model_input_features(df_in):

  # get input feature (item vector of customer)
  all_col = df_in.columns.values
  all_col = all_col[all_col != 'customer_id']
  # save input feature array
  with open(modelpath+'/model_input_features.txt', 'w') as file:
    file.write(','.join(all_col))

  
######################################################
# Function: model_train_type(df, type)
# Description:
#    train model for Userbased, Itembased with customer item input vector
#    (two models will be train here - UB, IB)
# Input: 
#    - df: all user in vector qty of items (df_pivot by items)
# Output:
#    - saved kdtree model, and input features
#

def model_train_type(df):

  indicators = ['first','second','third','fourth','fifth','sixth','seventh','eighth','nineth','tenth','eleven','twelfth','thirthteen','forthteen','fifthteen']
  
  # recommend itembased
  columns_cfib = ['customer_id']
  for i in np.arange(2*NUM_BASE_RECOMMEND):
    columns_cfib.append('{}_cfib_recommend'.format(indicators[i]))
	
  df_recommend = pd.read_csv(lbl_homepath+'/item_based/output/OUTPUT_item_based_recommend.csv',index_col=None)
  df_recommend = df_recommend[columns_cfib]
  df = pd.merge(df,df_recommend,on='customer_id',how='left')

  
  # recommend userbased
  columns_cfub = ['customer_id']
  for i in np.arange(NUM_BASE_RECOMMEND):
    columns_cfub.append('{}_cfub_recommend'.format(indicators[i]))
	
  df_recommend = pd.read_csv(lbl_homepath+'/user_based/output/OUTPUT_user_based_recommend.csv',index_col=None)
  df_recommend = df_recommend[columns_cfub]
  df = pd.merge(df,df_recommend,on='customer_id',how='left')
 
  # merge eda data with recommend result data on customer_id to be sure index of them is correct when train and look up recommend result
  df.reset_index(drop=True, inplace=True)
  
  # Save cfub label to file
  df[columns_cfub].to_csv(modelpath+'/cfub_label_lookup.csv',index=False)

  # Save cfib label to file
  df[columns_cfib].to_csv(modelpath+'/cfib_label_lookup.csv',index=False)
  
  
  #############################################################
  # Feature Engineering: is customer favourite
  
  print("All columns are: \n{}".format(df.columns.values))
  
  col_cus_vector = [col for col in df.columns.values if (len(col.split('_'))==1 and col!='customer_id')] # get list of items only
  print("customer item vector: \n{}".format(col_cus_vector))
  df = df[col_cus_vector]

  # train kdtree is finding nearest customers
  #df = pd.read_csv(input_file, index_col=None)
  model = train_model(df)

  # save model
  pickle.dump(model, open(modelpath+'/c360_kdtree_model.pkl', 'wb'))
  
  print("Train kdtree and save model completed!")
   

######################################################
# Function: kdtree_input_data_samples(df_pivot, NUM_CUS_KDTREE)
# Description:
#    get a number of customer from df_pivot.
#    those customer will be used for kdtree model train.
#    Why? because using full number of customer creates large matrix --> need more resources
#    
# Input: 
#    - df_pivot: customer_id, items vector
#    - NUM_CUS_KDTREE: number of customer need to be trained by kdtree
# Output:
#    - chosen customer list
#

def kdtree_input_data_samples(df_pivot, NUM_CUS_KDTREE):

  DIV_COEF = int(df_pivot.shape[0]/NUM_CUS_KDTREE)

  # get list of items ONLY (remove customer_id)
  items_col = df_pivot.columns.values[np.where(df_pivot.columns.values!='customer_id')]
  
  # mark True for cell that has MAX value in row
  df_pivot[items_col] = df_pivot[items_col].apply(lambda x: x==np.max(x), axis=1)
  
  kdtree_cuslist = []
  for items in items_col:
   
    df_temp = df_pivot[df_pivot[items] == True] 
	
    customer_list = df_temp.customer_id.unique()
	
    print("**items: ",items, 'number of customer: ', len(customer_list))	
	
    np.random.seed(0)
    chosen_list = random.sample(customer_list.tolist(), int(len(customer_list)/DIV_COEF))
    kdtree_cuslist = np.concatenate((kdtree_cuslist, chosen_list), axis=None)	
	
    kdtree_cuslist = np.unique(kdtree_cuslist)
	
  print("Total number of customer will be used for training Kdtree: {}".format(len(kdtree_cuslist))) 
  return kdtree_cuslist

###################################################################################################
# Function: cleaning()
# Description:
#     clean all temporary files that generated by the script

def cleaning():
  import os, re, os.path
  if os.path.exists(clusfile_path):
    shutil.rmtree(clusfile_path)

	  
#######################################################
# MAIN
#######################################################
# [1] Setup folder/Query data
# [2] Find customer Favorite
# [3] Labelling (CF run)
# [4] Training Model (kdtree)

print ('##################################################################')
print ('# [1] FOLDER SETUP/QUERY DATA')
print ('##################################################################')

setup_folder(homepath)        # create folder for running
query_data(input,query)  # query data from data source

print ('##################################################################')
print ('# [2] LABELING DATA')
print ('##################################################################')
# call recommedation labelling code (long time)
os.system('python rs_collaborative_filtering.py -i input --updateall --commit')

print ('##################################################################')
print ('# [3] TRAIN')
print ('##################################################################')
# customer item pivot
df = pd.read_csv(lbl_homepath+'/temp/df_after_eda.csv', index_col=None)

df = df.drop(columns=['date_order','size'],axis=1)
df = df.drop(columns= ["product_id"])
all_col = pd.unique(df['items'])
df = df.groupby(["customer_id","items"]).sum().reset_index()
df = df.pivot(index="customer_id",columns='items',values="quantity").reset_index()
df = df.fillna(0)  
df.reset_index(drop=True,inplace=True)
df.to_csv(modelpath+'/df_input_train.csv',index=False)

# get smaller group of customer for train kdtree
# If Large amount of customer --> do this
# If Not --> No Need
if df.customer_id.nunique() > NUM_CUS_KDTREE:
  chosen_cuslist_for_kdtree = kdtree_input_data_samples(df, NUM_CUS_KDTREE)
  df = df[df.customer_id.isin(chosen_cuslist_for_kdtree)]

# PIVOT TABLE --> INPUT VECTOR
# save the customer item vector to file
model_input_features(df) # df now is df_pivot

# TRAIN TWO KDTREE MODELS FOR UB AND IB
# train kdtree models
model_train_type(df)  # kdtree model is ONE used for both cfub, cfib. The different of them is the labelling lookup table.
                      # cfib: look up the itembased labelling table
					  # cfub: look up the userbased labelling table
  
#############################################################################
# Cleaning all temporary files and folders
  
cleaning()
  
print ('##################################################################')
print ('# RECOMMENDATION TRAIN COMPLETED!!!')
print ('##################################################################')



