########################################################################################################################
# CHURN PREDICTION PREDICT
# File name : churn_predict.py
# Author    : Nhan Thanh Ngo
#######################################################################

###################################################
# IMPORT LIBRARY
#
#

import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import sys
import shutil

import yaml
from general_func import eda_data
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

####################################################
# General

NUM_OBSERVED_DAY = 365

modelpath_train = "./ml_customer_input_recommendation_truncate_churn_train/model"
####################################################
# GetOptions
import getopt
import glob
import time

global debug
debug = False
input = "need_to_specified_table"
query = False
limit = ''
commit = False
commitonly = False
updatemodel = False

                      
try:
  opts, args = getopt.getopt(sys.argv[1:], 'hq:i:l:c:u:d', ['help','query','input=','updatemodel','commit','commitonly','debug'])
except getopt.GetoptError as err:
  print ("ERROR: Getoption gets error... please check!\n {}",err)
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-q', '--query'):
    query = True
  if opt in ('-i', '--input'):
    input = str(arg)
  if opt in ('-c', '--commit'):
    commit = True
  if opt in ('-u', '--commitonly'):
    commitonly = True
  if opt in ('-l', '--updatemodel'):
    updatemodel = True
  if opt in ('-d', '--debug'):
    debug = True
  if opt in ('-h', '--help'):
    parser.print_help()
    sys.exit(2)

if commitonly:
  query = False
  limit = ''
  commit = True


print ("CHURN PREDICTION")
print ("This is {}\n".format(filename))
  
###########################################
#create DEBUG directory
#
homepath = "./"+input.split('.')[-1]+"_churn_predict"
temppath = outpath = clusfile_path = modelpath = ''

###################################################
# GENERAL FUNCTION
#

def print_debug (strcont):
  debugn = debug
  if debugn:
    print(strcont)  
    with open(log_path+"/"+debug_log, "a") as logfile:
      logfile.write("\n"+strcont)

def print_report (strcont):
  debugn = debug
  if debugn:
    print(strcont)  
  with open(temppath+"/report.log", "a") as logfile:
    logfile.write("\n"+strcont)



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
  
  # homepath
  if os.path.exists(homepath):
    print ("\'{}\' is already EXISTED!".format(homepath))
  else:
    os.mkdir(homepath)
    print ("\'{}\' is CREATED!".format(homepath)) 

  # temppath
  temppath = homepath+'/temp'
  if os.path.exists(temppath):
    print ("\'{}\' is already EXISTED!".format(temppath))
  else:
    os.mkdir(temppath)
    print ("\'{}\' is CREATED!".format(temppath))

  # modelpath
  modelpath = homepath+'/model'
  if os.path.exists(modelpath):
    print ("\'{}\' is already EXISTED!".format(modelpath))
  else:
    os.mkdir(modelpath)
    print ("\'{}\' is CREATED!".format(modelpath))

  # outpath
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
#    - input: table name in DW (eg. DW.customer_recommend_datas
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
  else:
    print("[INFO] Read input data from offline file, need update please run again with -q to query new data from Big Query")
    print("Read offline input data file...\nit will take a few minutes...")
    df = pd.read_csv(bq_cus_purchase, index_col=None)

  return df

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


def gen_dummy_customer_in_date_order_range(base, N):
  '''
    generate dummy customer 'D' with date_order range from 'base' back to N days
    input: 
      base: datetime.datetime.today() - now
            datetime.datetime(2019,9,13)
      N: N rows back to past from base day
    output: 
      ['customer_id', 'date_order']
           D           2020-10-16
           D           2020-10-15
  '''
  
  #base = datetime.datetime.today()
  #base = datetime.datetime(2019,9,13)
  date_list = [base - datetime.timedelta(days=x) for x in range(N)]
  date_list = [date_list[i].strftime('%Y-%m-%d') for i in range(len(date_list))]

  df_dummy = pd.DataFrame(columns=['customer_id','date_order'])
  df_dummy['date_order'] = date_list
  df_dummy['customer_id'] = ['D1']*(len(date_list))
  df_dummy['quantity'] = 1
  df_dummy['date_order'] = pd.to_datetime(df_dummy.date_order)
  
  return df_dummy
  
  
def feature_engineering(df, INPUT_LEN):

  '''
    feature engineering for time series model (LSTM)
    input:
      df: ['customer_id', 'date_order', 'product_id', 'items', 'size', 'quantity']
    output:
      df: ['customer_id','churn','alert']
  '''
  
  column = ['customer_id','date_order','quantity']
  df = df.groupby(['customer_id','date_order'])['quantity'].sum().reset_index().sort_values(['customer_id','date_order'])

  # CREATE TABLE OF TIME RANGE FOR MIN DAY TO MAX DAY
  # AND TIMESCALES FOR PERIOD set by PARAMETERs
  # contineous day list 
  TIMESERIES_TIMESCALES = 7
  
  #create Dummy customer for date_order fill from last purchase day to now
  lastday = datetime.datetime.now()
  NUMDAY_GEN = (datetime.datetime.now() - df.date_order.max()).days
  df_dummy = gen_dummy_customer_in_date_order_range(datetime.datetime.now(), NUMDAY_GEN)
  df = pd.concat([df,df_dummy])
  
  total_numday = (df.date_order.max() - df.date_order.min()).days + 1

  if (total_numday%TIMESERIES_TIMESCALES)!= 0:
    total_numday = TIMESERIES_TIMESCALES*(int(total_numday/TIMESERIES_TIMESCALES)+1)

  total_timescales = int(total_numday/TIMESERIES_TIMESCALES)

  # date_list is list of day from lastday to N day
  #datetime.datetime(2019,9,13)
  date_list = [lastday - datetime.timedelta(days=x) for x in range(total_numday)]
  date_list = [date_list[i].strftime('%Y-%m-%d') for i in range(len(date_list))]

  # df_day 
  df_day = pd.DataFrame()
  df_day['date_order'] = date_list
  #df_day = df_day.sort_values(['date_order'], ascending=True)

  df_day['indexing'] = np.arange(total_numday)
  #df_day['indexing'] = df_day.groupby('customer_id').cumcount()+1
  df_day['timescales'] = df_day['indexing'].apply(lambda x: int(x/TIMESERIES_TIMESCALES))
  df_day['date_order'] = pd.to_datetime(df_day.date_order)

  # MERGE TABLE
  df = pd.merge(df,df_day[['date_order','timescales']],on='date_order',how='left')

  # df quantity by timescales instead of date_order
  df_ts = df.groupby(['customer_id','timescales'])['quantity'].sum().reset_index()
  df_ts.quantity = df_ts.quantity.astype(int)  

  # dummy customer
  df_dummy = pd.DataFrame({
    'customer_id' : ['D']*(INPUT_LEN),
    'timescales'  : np.arange(INPUT_LEN)
  })

  # CONCAT DUMMY
  df = pd.concat([df_ts,df_dummy])

  # PIVOT FOR TIME SERIES INPUT OF LSTM
  df_pivot = pd.pivot_table(df, values='quantity', index=['customer_id'], columns=['timescales'], aggfunc=np.sum, fill_value=0).reset_index()
  df_pivot = df_pivot[~df_pivot.customer_id.isin(['D','D1'])] # remove dummy customer
  
  column = [str(col) for col in df_pivot.columns.values]
  df_pivot.columns = column

  # RE-ORDER COLUMNS
  column = df_pivot.columns.values
  column = column[::-1]
  adj_col = np.concatenate((['customer_id'],column[:-1]),axis=0)
  df_pivot = df_pivot[adj_col]
  
  return df_pivot

################################################################################################

def build_model(INPUT_LEN,OUTPUT_NUM,BATCH_SIZE):

    model = keras.Sequential()
    # Add an Embedding layer expecting input vocab of size TIMESCALES_DIM*BATCH_SIZE, and
    # output embedding dimension of size OUTPUT_NUM.

    model.add(layers.Embedding(input_dim=INPUT_LEN*BATCH_SIZE, output_dim=OUTPUT_NUM))

    # Add a LSTM layer with TIMESCALES_DIM internal units.
    model.add(layers.LSTM(INPUT_LEN))

    # Add a Dense layer with OUTPUT_NUM units.
    model.add(layers.Dense(OUTPUT_NUM))

    # model.summary()
    return model 


def model_predict(label_name, feature_matrix, model_info_dict):

  global INPUT_LEN,OUTPUT_NUM,BATCH_SIZE
  
  loadmodel = build_model(INPUT_LEN,OUTPUT_NUM,BATCH_SIZE)
  # load transfromer and model
  loadmodel.load_weights(modelpath+'/'+label_name+'_model.ckpt')
  transformer = pickle.load(open(modelpath+'/transformer.pkl', 'rb'))

  feature_matrix  = feature_matrix.astype(np.int32)       # converse to int32
  feature_matrix  = transformer.transform(feature_matrix) # normalize
  feature_matrix  = tf.convert_to_tensor(feature_matrix, dtype=tf.float32) # converse to tensor    

  prediction_rs_array = loadmodel(feature_matrix).numpy()
  #print('prediction result: {}'.format(prediction_rs_array))

  # post prediction processing
  THRESHOLD = model_info_dict[label_name+'_THRESHOLD']  
  #print(THRESHOLD)
  prediction_rs_array = [1 if i > THRESHOLD else 0 for i in prediction_rs_array]  
  #print('prediction result: {}'.format(prediction_rs_array))  
  #sys.exit()
  return prediction_rs_array

############################################################################################################################################
### MAIN     ###############################################################################################################################
############################################################################################################################################

if not commitonly:

  print ('##################################################################')
  print ('# [1] FOLDER SETUP/QUERY DATA')
  print ('##################################################################')

  setup_folder(homepath) 
  df = query_data(input, query)
  print('Total number of customers is: {}'.format(df.customer_id.nunique()))
  
  print ('##################################################################')
  print ('# [2] UPDATE MODEL')
  print ('##################################################################')
  # update model or not
  if updatemodel:
    for file in os.listdir(modelpath_train):
      filename = file.split('/')[-1]
      shutil.copy2(modelpath_train+'/'+filename, modelpath+'/'+filename)

  # read pkl file
  model_info_dict = {}
  model_info_dict =  pickle.load(open(modelpath+'/model_info_dict.pkl', 'rb'))
  print('model_info_dict:\n',model_info_dict)
  
  INPUT_LEN = model_info_dict['INPUT_LEN']
  
  '''
  for key in model_info_dict:
    if key == 'INPUT_LEN':
      INPUT_LEN = model_info_dict[key]
    elif key == 'churn_ACCURACY':
      CHURN_ACCURACY_CURRENT = model_info_dict[key]    
    elif key == 'critical_ACCURACY':
      CRITICAL_ACCURACY_CURRENT = model_info_dict[key]
    elif key == 'alert_ACCURACY':
      ALERT_ACCURACY_CURRENT = model_info_dict[key]
    elif key == 'churn_THRESHOLD':
      CHURN_THRESHOLD = model_info_dict[key]    
    elif key == 'critical_THRESHOLD':
      CRITICAL_THRESHOLD = model_info_dict[key]
    elif key == 'alert_THRESHOLD':
      ALERT_THRESHOLD = model_info_dict[key]
  '''

  print ('##################################################################')
  print ('# [3] EDA DATA')
  print ('##################################################################')

  df = eda_data(df)
  df.date_order = pd.to_datetime(df.date_order) 

  # only train with data in one nearest year
  #start_date = df.date_order.max() - datetime.timedelta(days=NUM_OBSERVED_DAY)
  #df = df[df.date_order>=start_date]
  #EXPERIMENTAL RESULT: limit data in 12 months in not good, it bias customer insight.
  
  print ("[Predict] Number of customer is used to predict {}, date from {} to {}".format(df.customer_id.nunique(), df.date_order.min(), df.date_order.max()))

  print ('##################################################################')
  print ('# [4] FILE CLUSTERING AND ITERATION RUN')
  print ('##################################################################')

  OUTPUT_NUM = 1
  BATCH_SIZE = 100 #number of customers

  # divide file into cluster file
  NUMCUS_PER_CLUSTER = 10000
  end = None
  NUM_FILE_DIV = int(df.customer_id.nunique()/NUMCUS_PER_CLUSTER)+1
  for i in np.arange(NUM_FILE_DIV):
    if i == NUM_FILE_DIV:
      end = None
    else:
      end = (i+1)*NUMCUS_PER_CLUSTER
      
    df_i = df[df.customer_id.isin(df.customer_id.unique()[i*NUMCUS_PER_CLUSTER:end])]
    df_i.to_csv(clusfile_path+'/input_file_cluster'+str(i)+'.csv',index=None)
    
  del df
  
  # read cluster file and process
  for i in np.arange(NUM_FILE_DIV):
  
    df_i = pd.read_csv(clusfile_path+'/input_file_cluster'+str(i)+'.csv',index_col=False)
    df_i.date_order = pd.to_datetime(df_i.date_order)     
    
    print ('##################################################################')
    print ('# [4.1] FEATURE ENGINEERING')
    print ('##################################################################')

    df_feature_eng = feature_engineering(df_i, INPUT_LEN)
    df_feature_eng.to_csv(temppath+'/df_features'+str(i)+'.csv',index=None)

    df_predict = df_feature_eng[['customer_id']]

    df_feature_eng = df_feature_eng.drop('customer_id', axis=1)
  
    # process for case of feature array is longer than feature input array of model
    if len(df_feature_eng.columns) > INPUT_LEN:
      col_truncate = df_feature_eng.columns.values[-INPUT_LEN:]
      df_feature_eng = df_feature_eng[col_truncate]
    
    feature_matrix = df_feature_eng.values
    del df_feature_eng

    print ('##################################################################')
    print ('# [4.2] PREDICT')
    print ('##################################################################')

    # predict for 'churn'
    prediction_rs_array = model_predict('churn', feature_matrix, model_info_dict)
    df_predict['churn'] = prediction_rs_array

    # predict for 'critical'
    prediction_rs_array = model_predict('critical', feature_matrix, model_info_dict)
    df_predict['critical'] = prediction_rs_array

    # predict for 'alert'
    prediction_rs_array = model_predict('alert', feature_matrix, model_info_dict)
    df_predict['alert'] = prediction_rs_array

    df_predict.to_csv(temppath+'/result_CHURN_PREDICTION'+str(i)+'.csv',index=False) 
  
  
  df_predict = pd.DataFrame()
  for i in np.arange(NUM_FILE_DIV):
    df_predict_i = pd.read_csv(temppath+'/result_CHURN_PREDICTION'+str(i)+'.csv',index_col=None)
    df_predict = pd.concat([df_predict, df_predict_i])
    
  df_predict.to_csv(outpath+'/result_CHURN_PREDICTION.csv',index=False)
   
  
####################################################################################################  
# commit data to DW
if commit or commitonly:
  
  print ('##################################')
  print ('# COMMIT RESULT TO DW')
  print ('##################################')
  print('Commiting...')
  commit_filelist = [outpath+'/result_CHURN_PREDICTION.csv']
  commit_sandbox_name = [input.split('.')[-1]+'_CHURN_PRED']
  
  db = None
  if debug:
    db = '--debug'
  for i in range(len(commit_filelist)):
    os.system('python commit_bq.py -s {} -d {} --numrow 500 --pause 50 --verify {}'.format(commit_filelist[i],commit_sandbox_name[i],db)) 

  print('Completed.')

print ('##################################################################')
print ('# PREDICTION COMPLETED!')
print ('##################################################################')