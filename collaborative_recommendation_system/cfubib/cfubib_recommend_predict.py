##############################################################################
# RECOMMENDATION SYSTEM: QUERY PROCESS
#-----------------------------------------------------------------------------
# File name:   cfubib_recommend_predict.py 
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

import sys, getopt
import glob
import time
from optparse import OptionParser
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from cf_function_class import general_functions
from cf_item_based_class import cf_item_based

####################################################
# General variable declaration
####################################################

#default
global debug
debug = False
input = "DW.input_table"
query = False
commit = False
commitonly = False
limit = ''
modelpath = None
updatemodel = False
customer_id = ''
cusid_list_en = False

####################################################
# GetOptions
####################################################
                       
try:
  opts, args = getopt.getopt(sys.argv[1:], 'hi:s:q:c:u:l:d:', ['help','input=','customer_id=','query','commit','commitonly','updatemodel','debug'])

except getopt.GetoptError as err:
  print ("ERROR: Getoption gets error... please check!\n {}",err)
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-q', '--query'):
    query = True
  if opt in ('-i', '--input'):
    input = str(arg)
  if opt in ('-s', '--customer_id'):
    customer_id = str(arg)
    if customer_id != None:
      cusid_list_en = True	
  if opt in ('-c', '--commit'):
    commit = True
  if opt in ('-u', '--commitonly'):
    commitonly = True
  if opt in ('-u', '--updatemodel'):
    updatemodel = True
  if opt in ('-d', '--debug'):
    debug = True
  if opt in ('-h', '--help'):
    parser.print_help()
    sys.exit(2)
    
else:
   print("[Error] Please check option.")   


modelpath_train = "./ml_"+input.split('.')[-1]+"/model"
homepath = "./"+input.split('.')[-1]+"_predict"
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

  global cusid_list_en, customer_id
  
  print ('#############################')
  print ('# QUERY DATA FROM BIGQUERY')
  print ('#############################')

  from google.cloud import bigquery
  from google.oauth2 import service_account

  # query in DW (credentials is available in production machine)
  client = bigquery.Client()
  
  sql = ''  
  if cusid_list_en:
    sql = 'SELECT * FROM '+input + ' where customer_id in ('+ customer_id +')'
  else:
  
    sql = 'SELECT * FROM '+input #+' where customer_id in (368056, 491676, 55394, 90224, 323196)'
	
  bq_cus_purchase = input.split('.')[-1]+".csv"
 
  if query:
    # Run a Standard SQL query with the project set explicitly
    print("Querying data from sandbox...\nit will take a few minutes...")
    print("Command: {}".format(sql))	
	
    df = client.query(sql).to_dataframe() 
    #df = client.query(sql, project=project_id).to_dataframe()
  
    if not glob.glob(bq_cus_purchase):
      print ("[INFO] No BigQuery datafile available")
    else:
      print("[INFO] Remove exist bq datafile")
      os.remove(bq_cus_purchase)
  
    print("[INFO] Store query data from Big Query to file")
    df.to_csv(bq_cus_purchase,index=False)
  else:
    print("[INFO] Read input data from offline file, need update please run again with -q to query new data from Big Query")
    print("Read offline input data file...\nit will take a few minutes...")
    df = pd.read_csv(bq_cus_purchase, index_col=None)
  
  return df

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

  # drop value of customer_id and items which have NULL value
  df.dropna(subset=['customer_id','items'], inplace=True)
  df['size'].fillna('0',inplace=True)
  df['size'] = df['size'].astype(str)
  print("[Before EDA] Total customer is {}".format(df.customer_id.nunique()))
  
  # get necessary table
  df = df[['customer_id','date_order','product_id','items','size','quantity']]

  # eda to get main products (remove ones that are not main product)
  func_obji = general_functions(df, homepath, debug=debug)    
  df = func_obji.product_eda() 
  
  #df.to_csv(temppath+"/df_after_eda.csv", index=False)  
  print('[After EDA] Total customer is {}'.format(df.customer_id.nunique()))

  return df

######################################################
# Function: find_rcm(model,new_query,look_up,all_col,k_nearest=10,total_rcm=10)
# Description:
#     find nearest neighbor for recommend
# Input: 
#     - model : KDtree model
#     - new_query: list of new cus
#     - look up : file rcm
#     - k_nearest: k customer similarity with a new customer
#     - total_rcm: total item rcm for new cus. 
# Output:
#    - list 
#

def find_rcm(model, data, new_query, look_up, all_col, k_nearest=3, total_rcm=10):

  if new_query.shape[0] == 0 :
    return 

  # get new_query ['customer_id', 'items', 'quantity']
  new_query = new_query.drop(columns=['date_order','size'],axis=1)
  favorite_new_query = new_query.copy()
  col_rcm = [col for col in look_up.columns if col.split('_')[1]!="favorite" and col!="first_fav_size"]
  score = {}
  for idx,col in enumerate(col_rcm):
    score[col] = len(col_rcm)-idx+1
  
  # create dummy customer with full purchased items 
  df_dummy = pd.DataFrame({
        'items'       :all_col,
        'customer_id' :['D']*(len(all_col)),
        'quantity'    : np.ones(len(all_col)),
  })
  
  # concat dummy df into new query df
  new_query = pd.concat([new_query,df_dummy])
  
  # pivot new query to get customer item purchased vector
  new_query = pd.pivot_table(new_query, values='quantity', index=['customer_id'], columns=['items'], aggfunc=np.sum, fill_value=0).reset_index()
  
  # NHAN NEW ADDING
  new_query = new_query[['customer_id']+all_col]  # remove items is not in Features of kdtree

  new_query.drop(new_query.loc[new_query['customer_id']=='D'].index, axis=0, inplace=True)
  customer_id_new_query = new_query.pop('customer_id')
  
  # kdtree model query the result
  dist,index = model.query(new_query.values, k_nearest)
  df_output = pd.DataFrame(
    index,
    index =  customer_id_new_query
  )

  # reset index of data
  data = data.reset_index().reset_index()

  data.rename(columns = {
    'customer_id':'customer_id_rcm',
    'index':'customer_index'
  },inplace=True)

  df_look_up = pd.DataFrame(
    index, 
    columns = np.arange(index.shape[1]),
    index=customer_id_new_query
  )
  df_look_up = df_look_up.unstack().reset_index()

  df_look_up.rename({
      0: 'customer_index',
      'level_0':'priority_rcm'
  },inplace=True,axis=1)
  
  df_look_up = df_look_up[['customer_id','priority_rcm','customer_index']]
  df_look_up.sort_values(by = ['customer_id','priority_rcm'],ascending=[True,True],inplace=True)
  reference_dataset = pd.merge(df_look_up,data, on='customer_index', how='left')
  reference_dataset = reference_dataset[['customer_id','priority_rcm','customer_index','customer_id_rcm']]
  current_cus_index = []
  find_rcm_validate = []
  for idx,distance in enumerate(dist):
    if distance[0] ==0:
      current_cus_index.append(customer_id_new_query[idx])
    else:
      find_rcm_validate.append(index[idx])

  df_current_rcm  = reference_dataset[reference_dataset.customer_id.isin(current_cus_index)]   # get cus avaiable rcm
  df_find_rcm     = reference_dataset[~reference_dataset.customer_id.isin(current_cus_index)]  # cus must find rcm
  
  #k-nearest cutomer of the new cus ... index[0] ->  
  if len(find_rcm_validate)>0:
    id_ = df_find_rcm[df_find_rcm.customer_index.isin(find_rcm_validate[0])]['customer_id_rcm'].values

  df_current_rcm = df_current_rcm.drop_duplicates(subset= ['customer_id'],keep='first')

  look_up = look_up.rename(columns = {
      "customer_id" : "customer_id_rcm"
  })

  result_rcm_current = pd.merge(df_current_rcm,look_up,on ="customer_id_rcm",how = "left" )
  result_rcm_current = result_rcm_current.sort_values(by=['customer_id','priority_rcm'],ascending=[True,True])
  result_rcm_current = result_rcm_current[col_rcm]
  
  # get favorite
  favorite_new_cus = favorite_new_query[favorite_new_query.customer_id.isin(current_cus_index)]
  favorite_new_cus = favorite_new_cus[['customer_id','items','quantity']]
  favorite_new_cus = favorite_new_cus.groupby(by=['customer_id','items'])['quantity'].sum().reset_index()
  result_current_rcm = pd.DataFrame([])
  if len(current_cus_index) != 0 :

    final_result_all_unstack = result_rcm_current.set_index('customer_id').unstack().reset_index()


    final_result_all_unstack = final_result_all_unstack.rename(columns = {
        0: "item_rcm",
        "level_0" : "score", 
      })
    final_result_all_unstack['score'] = final_result_all_unstack['score'].apply(lambda x: score[x])
    
    final_result_all_unstack['item_rcm'] = final_result_all_unstack['item_rcm'].astype('str')
    final_result_all_unstack  = final_result_all_unstack[final_result_all_unstack['item_rcm']!='0']

    final_result_all_unstack = pd.merge(final_result_all_unstack, favorite_new_cus,how = 'left',right_on=['customer_id','items'],left_on=['customer_id','item_rcm'])
    
    final_result_all_unstack = final_result_all_unstack[final_result_all_unstack['quantity'].isnull()]
    final_result_all_unstack = final_result_all_unstack.drop(columns = {'items','quantity'})

    final_result_all_unstack['indexing'] = final_result_all_unstack.sort_values("score",ascending=False).groupby('customer_id').cumcount() + 1
    final_result_all_unstack = final_result_all_unstack[final_result_all_unstack['indexing']<=total_rcm]
    final_result_all_unstack = final_result_all_unstack.rename(columns={"item_rcm":"items"})
    final_result_all_unstack.sort_values(by = ['customer_id', 'indexing'], ascending=[True, True])

    df_dummy = pd.DataFrame({
          'indexing'   :np.arange(len(col_rcm)-1)+1,
          'customer_id' :['D']*(len(col_rcm)-1),
          'score'       : np.ones(len(col_rcm)-1),
          'items'        :['D']*(len(col_rcm)-1),
    })
    
    result_rcm  = pd.concat([final_result_all_unstack,df_dummy])
    result_rcm = pd.pivot_table(result_rcm, values='items', index=['customer_id'], columns=['indexing'], aggfunc=np.sum, fill_value=0).reset_index()
    result_rcm.drop(result_rcm.loc[result_rcm['customer_id']=='D'].index, axis=0, inplace=True)

    miss = []
    # rcm_idx_idx = []
    for value in df_current_rcm.customer_id.values:
      if value not in result_rcm.customer_id.values:
        miss.append([value]+[0]*(result_rcm.shape[1]-1))
        # rcm_idx_idx.append(value)

    df_miss = pd.DataFrame(
        miss,
        columns = result_rcm.columns
    )
    result_current_rcm = pd.concat([result_rcm,df_miss])
  
    result_current_rcm.columns = col_rcm

  if df_find_rcm.values.shape[0] == 0:
    return result_current_rcm

  df_output = df_output.reset_index()
  df_cus_id = df_output[~df_output.customer_id.isin(current_cus_index)]
  df_cus_id = df_cus_id.set_index('customer_id')
  df_cus_id = df_cus_id.unstack().reset_index()
  df_cus_id = df_cus_id.rename(columns={
        0:'customer_id_rcm',
  })
  
  df_cus_id = df_cus_id.drop(columns=['level_0'])

  col_rcm[0] = "customer_id_rcm"
  look_up_unstack = look_up.copy()
  look_up_unstack = look_up_unstack[col_rcm]
  look_up_unstack = look_up_unstack.set_index("customer_id_rcm").unstack().reset_index()

  look_up_unstack['level_0'] = look_up_unstack['level_0'].apply(lambda x: score[x])
  look_up_unstack = look_up_unstack.rename(columns = {'level_0':'score',0:'item_rcm'})

  result_find_rcm = pd.merge(df_find_rcm,look_up_unstack,on ="customer_id_rcm",how = "left" )
  result_find_rcm = result_find_rcm.drop(columns = {'priority_rcm'})

  result_find_rcm_cal = result_find_rcm.groupby(by = ['customer_id','item_rcm'])['score'].sum().reset_index()
  result_find_rcm_cal  =result_find_rcm_cal.sort_values("score",ascending=False)

  result_find_rcm_cal['item_rcm'] = result_find_rcm_cal['item_rcm'].astype('str')
  result_find_rcm_cal  = result_find_rcm_cal[result_find_rcm_cal['item_rcm']!='0']

  favorite_new_cus = favorite_new_query[~favorite_new_query.customer_id.isin(current_cus_index)]
  favorite_new_cus = favorite_new_cus[['customer_id','items','quantity']]
  favorite_new_cus = favorite_new_cus.groupby(by=['customer_id','items'])['quantity'].sum().reset_index()

  result_find_rcm_cal = pd.merge(result_find_rcm_cal, favorite_new_cus,how = 'left',right_on=['customer_id','items'],left_on=['customer_id','item_rcm'])
  result_find_rcm_cal = result_find_rcm_cal[result_find_rcm_cal['quantity'].isnull()]

  
  result_find_rcm_cal = result_find_rcm_cal.drop(columns = {'items','quantity'})

  result_find_rcm_cal['indexing'] = result_find_rcm_cal.sort_values("score",ascending=False).groupby('customer_id').cumcount() + 1
  result_find_rcm_cal = result_find_rcm_cal[result_find_rcm_cal['indexing']<=total_rcm]
  result_find_rcm_cal = result_find_rcm_cal.rename(columns={"item_rcm":"items"})
  result_find_rcm_cal.sort_values(by = ['customer_id', 'indexing'], ascending=[True, True])

  df_dummy = pd.DataFrame({
        'indexing'   :np.arange(len(col_rcm)-1)+1,
        'customer_id' :['D']*(len(col_rcm)-1),
        'score'       : np.ones(len(col_rcm)-1),
        'items'        :['D']*(len(col_rcm)-1),
  })
  final_result  = pd.concat([result_find_rcm_cal,df_dummy])
  final_result = pd.pivot_table(final_result, values='items', index=['customer_id'], columns=['indexing'], aggfunc=np.sum, fill_value=0).reset_index()
  final_result.drop(final_result.loc[final_result['customer_id']=='D'].index, axis=0, inplace=True)

  col_rcm[0] = "customer_id"
  final_result.columns = col_rcm

  final_result_all = pd.concat([final_result,result_current_rcm])
  
  return final_result_all.reset_index(drop=True)
  
######################################################
# Function: customer_favourite_items(df)
# Description:
#     find favorite items of customer
# Input: 
#    - df: dataframe
# Output:
#    - splited data in folder
#  
  
def customer_favourite_items(df):  
  
  global clusfile_path
  
  df['customer_group'] = '0'
  
  # Declare general_function object
  func_obj = general_functions(df, homepath, debug=True)    
  func_obj.df_file_cluster() # cluster completed
  del func_obj
    
  print ('####################################################################')
  print ('# CUSTOMER INSIGHTS: ITEMS, SIZE FAVORITE')
  print ('####################################################################')  
  
  # run user_based for each cluster_group
  dir_list = os.listdir(clusfile_path)
  for group, clusfile in zip(np.arange(len(dir_list)),dir_list):
    df = pd.read_csv(clusfile_path+"/"+clusfile, index_col=None)   
  
    func_obj = general_functions(df, homepath, debug=debug)    
    func_obj.tf_customer_favorite_insights(group)

  ############################
  # summary all
  df_fav = pd.DataFrame()
  for group in np.arange(len(dir_list)):
    df_temp = pd.read_csv(outpath+"/OUTPUT_customer_favorite_insights"+str(group)+".csv", index_col=None) 
    df_fav = pd.concat([df_fav,df_temp])
  df_fav.to_csv(outpath+"/OUTPUT_customer_favorite_insights.csv", index=False)
  
######################################################
# Function: customer_favourite_items_proid(df)
# Description:
#     find favorite items of customer
# Input: 
#    - df: dataframe
# Output:
#    - splited data in folder
#   
  
def customer_favourite_items_proid(df):

  print ('##########################################')
  print ('# CUSTOMER FAV BY PRODUCT ID')
  print ('##########################################')  
  
  df_pid = df[['customer_id','product_id','size','quantity']] 
  df_pid.columns = ['customer_id','items','size','quantity']
  func_obj = general_functions(df_pid, homepath, mode='both', debug=debug)    
  func_obj.tf_customer_favorite_insights(group='')
  del func_obj
  os.system('mv {}/output/OUTPUT_customer_favorite_insights.csv {}/output/OUTPUT_customer_favorite_insights_PID.csv'.format(homepath,homepath))
  
  print('Complete...')  

###################################################################################################
# Function: lookup_product_id(df)
# Description:
#   from result of recommendation of collaborative filtering, and favorite size, look up the product id
# Inputs:
#   - df: partition dataframe
# Outputs:
#   - return recommendation result in product id
#

def lookup_product_id(df_rcm, df_product_id):

  #print('[lookup_product_id] Product ID looking up...')
  #df_rcm.to_csv('./df_rcm.csv',index=False)
  
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
  value = np.empty(2*NUM_BASE_RECOMMEND, dtype=np.str)
  value.fill('D')
  df_dummy['items'] = np.ones(2*NUM_BASE_RECOMMEND)
  df_dummy['customer_id'] = value
  df_rcm = pd.concat([df_rcm,df_dummy])
  
  df_rcm['indexing'] = df_rcm.groupby('customer_id').cumcount() + 1

  df_rcm = pd.pivot_table(df_rcm, values='product_id', index=['customer_id'], columns=['indexing'], aggfunc=np.sum, fill_value=0).reset_index()
  df_rcm = df_rcm[df_rcm['customer_id']!='D']
  df_rcm = df_rcm.iloc[:,:NUM_BASE_RECOMMEND+1]

  return df_rcm

###################################################################################################
# Function: recommend_items_to_pid(method)
# Description:
#   from items result, lookup and change result to PID 
# Inputs:
#   - method: cfib, cfub
# Outputs:
#   - ['customer_id',cfxb_1,cfxb_2,cfxb_3,cfxb_4,cfxb_5]  x:i/b
#

def recommend_items_to_pid(method):

  print('Items to PID Processing for {}'.format(method.upper()))
  # read customer favorite
  df = pd.read_csv(outpath+'/OUTPUT_customer_favorite_insights.csv',index_col=False)

  # Read recommend label file, fill all '0' (customer purchased baker only) in fav_size with 'Size M'
  df_recommend = pd.read_csv(outpath+'/result_'+method.upper()+'_RECOMMEND.csv',index_col=False) 

  # merge dataframe 
  df = pd.merge(df[['customer_id','first_fav_size']],df_recommend,on='customer_id',how='left')
  df.loc[df.first_fav_size=='0', 'first_fav_size'] = 'Size M'
  
  # read product_id lookup table (items+size --> product_id)
  df_product_id = pd.read_csv(lbl_homepath+'/temp/df_product_id.csv', index_col=False)  

  # Call items + fav size -- product_id lookup
  CLUSTER_MEM = 1000
  NUM_FILE = int(df.customer_id.nunique()/CLUSTER_MEM)+1
  print('NUM_FILE:',NUM_FILE)
  for i in np.arange(NUM_FILE):
    end = (i+1)*CLUSTER_MEM    

    if i == int(df.customer_id.nunique()/CLUSTER_MEM):
      end = None
      print('Completed looking up Product ID for {} customers'.format(df.customer_id.nunique()))  
    
    df_temp = df[df.customer_id.isin(df.customer_id.unique()[i*CLUSTER_MEM:end])]
    df_temp.to_csv(temppath+'/rm-df_temp'+str(i)+'.csv', index=False)
    df_product_id.to_csv(temppath+'/rm-df_product_id'+str(i)+'.csv',index=False)
  
    df_proid = lookup_product_id(df_temp, df_product_id)
    df_proid.to_csv(temppath+'/recommend_proid'+str(i)+'.csv',index=False)

  df = pd.DataFrame()  
  for i in np.arange(NUM_FILE):
    df_temp = pd.read_csv(temppath+'/recommend_proid'+str(i)+'.csv', index_col=None) 
    df = pd.concat([df, df_temp])
    
  column = ['customer_id']
  for i in np.arange(NUM_BASE_RECOMMEND):
    column.append('{}_{}'.format(method, str(i+1)))
  print('column: ',column)
  df.columns = column

  df.to_csv(outpath+'/result_'+method.upper()+'_RECOMMEND_PID.csv',index=False) 
  print('recommend_items_to_pid() run for {} completed!'.format(method.upper()))

###################################################################################################
# Function: cleaning()
# Description:
#     clean all temporary files that generated by the script

def cleaning():
  import os, re, os.path
  if os.path.exists(clusfile_path):
    shutil.rmtree(clusfile_path)
	  
  pattern = "^([\w-]+)([(_id)*(temp)*]+)([\d]+).csv$"
  mypath = temppath
  for root, dirs, files in os.walk(mypath):
    for file in filter(lambda x: re.match(pattern, x), files):
      os.remove(os.path.join(root, file))
	  
#######################################################
# MAIN
#######################################################
# [1] Setup folder/Query data
# [2] EDA
# [3] Customer favorite by product_id
# [4] Customer favorite by items
# [5] Predict (get recommendation result)
# [6] Commit

# FOLDER SETUP
setup_folder(homepath)        # create folder for running
  
if not commitonly:
  print ('##################################################################')
  print ('# [1] QUERY DATA')
  print ('##################################################################')

  df = query_data(input,query)  # query data from data source

  if cusid_list_en:
    cusid_list = customer_id.split(',')
    df = df[df.customer_id.isin(cusid_list)]

  print ('##################################################################')
  print ('# [2] EDA')
  print ('##################################################################')
  print('Running...')
  df = data_preprocessing(df)
  df.to_csv(temppath+"/df_after_eda.csv", index=False)  
  print('Completed.')
  
#------------------------------------------------------------------------

# get old customer
# copy all old data
# action customer = update fav result_
# new customer = kdtree
# 
  print ('##################################################################')
  print ('# [3] PROCESS FOR CUSTOMER OLD OR NEW')
  print ('##################################################################')
  # Get lastest day of Labeling RUN
  # back up all import file to new folder before run, not append
  PRE_RUNDATE = ''
  if os.path.exists(lbl_homepath+"/lastday_in_data_previous_run.log"):
    print ("This is not initial run, lastday_in_data_previous_run.log exists.")
    with open(lbl_homepath+"/lastday_in_data_previous_run.log", mode='r') as file:
      previous_run_date = file.readline()
      PRE_RUNDATE = datetime.datetime.strptime(previous_run_date,'%Y-%m-%d')    
	  
  print('PRE_RUNDATE:',PRE_RUNDATE)
  #	get the previous output and find the new or old customer
  LAST_DAY = df.date_order.max().strftime("%Y-%m-%d")  
  df_maxdate = df.groupby('customer_id').date_order.max().reset_index()
  cusid_newaction_list = df_maxdate[df_maxdate.date_order>PRE_RUNDATE]['customer_id'].values
  cusid_noaction_list = df_maxdate[df_maxdate.date_order<=PRE_RUNDATE]['customer_id'].values
  
  print("Number of customer still be active     : {}".format(len(cusid_newaction_list)))
  print("Number of customer have no new purchase: {}".format(len(cusid_noaction_list)))

  print ('##################################################################')
  print ('# [3A] PROCESS FOR NO ACTION CUSTOMER')
  print ('##################################################################')

  df_noaction = pd.read_csv(lbl_homepath+"/output/result_FAV_CFIBUB_RECOMMEND_PID.csv",index_col=False)
  df_noaction = df_noaction[df_noaction.customer_id.isin(cusid_noaction_list)]
  df_noaction.to_csv(outpath+"/result_CFIBUB_RECOMMEND_PID_noaction.csv",index=False)  
  del df_noaction
  
  # fav+ub+ib
#----------------------------------------------------------------------------     
  print ('##################################################################')
  print ('# [3B] PROCESS FOR CUSTOMER HAVING NEW ACTIONS')
  print ('##################################################################')  

  # df here is df of newaction 
  df = df[df.customer_id.isin(cusid_newaction_list)] 
 

  print (' [3B.1] FIND CUSTOMER FAVORITE (By product_id)')
  print('Running...')
  customer_favourite_items_proid(df) # find customer favorite product_id
  print('Completed.')

  print (' [3B.2] FIND CUSTOMER FAVORITE (By items)')
  print('Running...')
  customer_favourite_items(df) # find customer favorite items
  print('Completed.')

  print (' [3B.3] CFUB PREDICT')
  print('Predicting...')

  #-------------------------------------------------------------
  # [1] AUTO COPY MODEL: First time of Predict
  # initial run, the model is not in folder --> auto copy model 
  if len(os.listdir(modelpath)) < 3:
    updatemodel = True

  # update model or not
  if updatemodel:
    for file in os.listdir(modelpath_train):
      filename = file.split('/')[-1]
      shutil.copy2(modelpath_train+'/'+file, modelpath+'/'+file)
  #--------------------------------------------------------------
  # [2] LOAD MODEL, FIND NEAREST NEIGHBOR, LOOKUP FOR RECOMMEND ITEMS
  #     
  # this look up will include two parts
  # part 1: cfub
  # part 2: cfib

  # Load input vector values
  all_col=''
  with open(modelpath+'/model_input_features.txt', 'r') as file:
    all_col = file.readline().split(',')
  #df = df[all_col]  # 

  # Load kdtree model
  model = pickle.load(open(modelpath+'/c360_kdtree_model.pkl','rb')) 

  # Load previous input data train
  data = pd.read_csv(modelpath+'/df_input_train.csv',index_col=None)

  # CFUB Predict
  df_look_up = pd.read_csv(modelpath+'/cfub_label_lookup.csv',index_col=None)
  final_result = find_rcm(model, data, df, df_look_up, all_col,k_nearest=10,total_rcm=5)
  final_result.to_csv(outpath+'/result_CFUB_RECOMMEND.csv', index=False)  
  
  print('UB Items Completed...')

  #------------------------------------------------------------------------------
  # CFIB Predict    
  print (' [3B.4] CFIB PREDICT')
  print('Predicting...')  
  
  shutil.copy2(lbl_homepath+"/item_based/output/similar_item_pair_summary_group.csv", homepath+'/item_based/output/similar_item_pair_summary_group.csv')
  group = ''
  cfib_obj = cf_item_based(df, homepath)        
  cfib_obj.cf_customer_recommend_main_v2(group)
  del cfib_obj	

  print('IB Items Completed...')
  
  print (' [3B.5] POST PROCESSING (From Item to ProductID)')
  print('Running...')
  indicators = ['first','second','third','fourth','fifth','sixth','seventh','eighth','nineth','tenth','eleven','twelfth','thirthteen','forthteen','fifthteen']

  # CFUB: exchange from items to PID recommend
  recommend_items_to_pid('cfub')
  
  # create file result_CFIB_RECOMMEND.csv ['customer_id','first_cfib_recommend',...]
  df_cfib = pd.read_csv(homepath+"/item_based/output/OUTPUT_item_based_recommend.csv", index_col=False)
  cfib_cols = [col for col in df_cfib.columns.values if (col.split('_')[-1]!='favorite') and (col!='first_fav_size') ]
  df_cfib = df_cfib[cfib_cols]
  df_cfib.to_csv(outpath+'/result_CFIB_RECOMMEND.csv', index=False)

  # CFIB: exchange from items to PID recommend  
  recommend_items_to_pid('cfib')

  # form the output in one file
  df = pd.read_csv(outpath+"/OUTPUT_customer_favorite_insights_PID.csv", index_col=False)  

  colrm = ['customer_id']
  for i in np.arange(NUM_FAV_ITEMS):
    colrm.append('fav_{}'.format(str(i+1)))
  colrm.append('fav_size')
  df.columns = colrm
  
  print (' [3B.6] MERGE FAV+UB+IB')
  
  df_temp = pd.read_csv(outpath+'/result_CFUB_RECOMMEND_PID.csv',index_col=False) 
  df = pd.merge(df, df_temp, on='customer_id',how='left')
  df_temp = pd.read_csv(outpath+'/result_CFIB_RECOMMEND_PID.csv',index_col=False) 
  df = pd.merge(df, df_temp, on='customer_id',how='left')
  df.to_csv(outpath+'/result_CFIBUB_RECOMMEND_PID_newaction.csv', index=False)


  print ('##################################################################')
  print ('# [4] FINAL MERGE (NO ACTION + NEW ACTION)')
  print ('##################################################################')

  df_temp = pd.read_csv(outpath+'/result_CFIBUB_RECOMMEND_PID_noaction.csv',index_col=False) 
  df = pd.concat([df, df_temp])
  df.to_csv(outpath+'/result_CFIBUB_RECOMMEND_PID.csv', index=False)
  
  # cleaning temporary
  cleaning()
  
# commit data to DW
if commit or commitonly:
  
  print ('##################################')
  print ('# [7] COMMIT RESULT TO DW')
  print ('##################################')
  print('Commiting...')
  commit_filelist = [outpath+'/result_CFIBUB_RECOMMEND_PID.csv']
  commit_sandbox_name = [input+'_CFUBIB_PRED']
  
  db = None
  if debug:
    db = '--debug'
  for i in range(len(commit_filelist)):
    os.system('python commit_bq.py --source {} --destination {}'.format(commit_filelist[i],commit_sandbox_name[i]))
  print('Completed.')
  
print ('##################################################################')
print ('# RECOMMENDATION PREDICTION COMPLETED!!!')
print ('##################################################################')





  
