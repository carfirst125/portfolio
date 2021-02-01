########################################################################################################################
# CHURN PREDICTION TRAIN AUTO
# File name : churn_train_auto.py
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
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle
import random

####################################################
# General
NUM_OBSERVED_DAY = 365 # numday in year

####################################################
# GetOptions
import sys, getopt
import glob
import time

debug = False
input = "need_to_specified_table"
query = False
limit = ''
first_run = False

from optparse import OptionParser

usage = "usage: %prog [options] arg1 arg2 ...\n\n"\
        "Example:"\
        "\n\tpython %prog  -i HLC.ml_customer_input_recommendation [--query] [--first_run]"\
        "\n\tAllow select model build (LSTM)"

parser = OptionParser(usage=usage)

parser.add_option("-i", "--input",
                  default="`Aldo.nodefaultyet`",
                  metavar="SANDBOX", help="Sandbox dataname"
                                     "[default: %default]")        
       
parser.add_option("-q", "--query",
                  default="False",
                  help="Query mode: query data from Big Query"
                       "[default: %default]")     
parser.add_option("-f", "--first_run",
                  default="False",
                  help="First script running, not check previous model accuracy"
                       "[default: %default]") 
					   
parser.add_option("-d", "--debug",
                  default="False",
                  help="Enable debug mode"
                       "[default: %default]")

                       
try:
  opts, args = getopt.getopt(sys.argv[1:], 'hq:i:f:d', ['help','query','input=','first_run','debug'])
except getopt.GetoptError as err:
  print ("ERROR: Getoption gets error... please check!\n {}",err)
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-q', '--query'):
    query = True
  if opt in ('-i', '--input'):
    input = str(arg)
  if opt in ('-f', '--first_run'):
    first_run = True
  if opt in ('-d', '--debug'):
    debug = True
  if opt in ('-h', '--help'):
    parser.print_help()
    sys.exit(2)

print ("CHURN PREDICTION TRAIN")
print ("This is {}\n".format(filename))
  
###########################################
#create DEBUG directory
#
homepath = "./"+input.split('.')[-1]+"_churn_train_auto"

homepath_train = "./"+input.split('.')[-1]+"_truncate_churn_train"

temppath = outpath = clusfile_path = modelpath = ''

###################################################
# GENERAL FUNCTION
#

def print_debug (strcont):
  global debug
  if debug:
    print(strcont)  
    with open(log_path+"/"+debug_log, "a") as logfile:
      logfile.write("\n"+strcont)

def print_report (strcont):
  #debug = True
  #if debug:
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
#    - input: table name in DW (eg. HLC.customer_recommend_datas
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

def labelling(df):
  '''
    labelling churn and alert signal for all customers.
    input:
      df: ['customer_id', 'date_order', 'product_id', 'items', 'size', 'quantity']
    output:
      df: ['customer_id','churn','alert']
  '''

  # get relevant columns for processing
  column = ['customer_id','date_order','quantity']
  df = df.groupby(['customer_id','date_order'])['quantity'].sum().reset_index().sort_values(['customer_id','date_order'])

  # compute distance between two nearest purchases
  df['date_sh1'] = df.groupby(['customer_id']).date_order.shift(1)
  df['pur_dist'] = df['date_order'] - df['date_sh1']

  # fill NaT cell with 0
  df.fillna(pd.Timedelta(seconds=0),inplace=True)
  df['pur_dist'] = df['pur_dist'].apply(lambda x:  int(x.days))
  
  # compute purchasing frequency of customer [customer_id, freq]
  df_freq = df.groupby('customer_id').date_order.count().reset_index()
  df_freq.columns = ['customer_id','freq'] #############################################freq  

  # max distance btw 2 purchases by customer [customer_id, pur_dist]
  df_purdist = df.groupby('customer_id').pur_dist.max().reset_index()
  
  # combine [customer_id, freq, pur_dist] pur_dist: max distance
  df_label = pd.merge(df_freq,df_purdist,on='customer_id',how='left')

  # recency of each customer  [customer_id, R]
  df_recency = df.groupby(['customer_id']).date_order.max().reset_index()
  df_recency['today'] = pd.to_datetime(datetime.datetime.now().strftime("%Y-%m-%d")) # f"{datetime.datetime.now():%Y-%m-%d}"
  df_recency['R'] = (df_recency['today'] - df_recency['date_order']).astype('timedelta64[D]').astype(int)

  df_label = pd.merge(df_label, df_recency[['customer_id','R']], on='customer_id', how='left')
  df_label.to_csv(temppath+'/df_label_cus_freq_r.csv', index=False)  
  
  # read lookup critical data of model for labeling.
  df_lookup = pd.read_csv(homepath_train+'/model/label_data.csv', index_col=False)
  
  df_label = pd.merge(df_label, df_lookup, on='freq', how='left')
  
  df_label.to_csv(temppath+'/df_label_cus_freq_r_lookup.csv', index=False)    
  
  df_label['numday_critical'] = df_label[["pur_dist", "numday_prechurn"]].max(axis=1)
  df_label['numday_alert'] = df_label['numday_critical'].apply(lambda x: np.round(x*0.8))
  df_label['numday_churn'] = df_label['numday_critical'].apply(lambda x: np.round(x*1.2))
  
  df_label['alert'] = (df_label['R'] >= df_label['numday_alert'])
  df_label['critical'] = (df_label['R'] >= df_label['numday_critical'])
  df_label['churn'] = (df_label['R'] >= df_label['numday_churn'])

  df_label['alert']    = df_label['alert'].apply(lambda x: 1 if x else 0)  
  df_label['critical'] = df_label['critical'].apply(lambda x: 1 if x else 0)
  df_label['churn']    = df_label['churn'].apply(lambda x: 1 if x else 0)

  df_label['churn'] = df_label['churn'].astype(int)
  df_label['critical'] = df_label['critical'].astype(int)
  df_label['alert'] = df_label['alert'].astype(int)

  df_label.to_csv(temppath+'/df_label.csv', index=False)    
  
  return df_label[['customer_id','churn','critical','alert']]
 

############################################################################################################################################
### MAIN ####################################################################################################################################
############################################################################################################################################


# print working day to report.log
current_day_str = datetime.datetime.now().strftime('%y-%m-%d')
print_report('\n##########################################\n[{}]\n'.format(current_day_str))

print ('##################################################################')
print ('# [1] FOLDER SETUP/QUERY DATA')
print ('##################################################################')
setup_folder(homepath) 
df = query_data(input, query)
#df = df[df.customer_id.isin(df.customer_id.values[0:100000])]

# change date_order to datetime datatype
df.date_order = pd.to_datetime(df.date_order) 

#--------------------------------------------------------
# just for Test
df = df[df.date_order <= datetime.datetime(2020,11,7)]
#--------------------------------------------------------

# trancate data 12 months from the last day of data  
#EXPERIMENTAL RESULT: limit data in 12months in not good, it bias customer insight.
#start_date = df.date_order.max() - datetime.timedelta(days=NUM_OBSERVED_DAY)
#df = df[df.date_order>=start_date]

df.to_csv(input.split('.')[-1]+'_truncate.csv', index=None)

print ('##################################################################')
print ('# [2] EDA DATA')
print ('##################################################################')
df = eda_data(df)

if not first_run:
  print ('##################################################################')
  print ('# [3] RANDOMIZE 10% OF TOTAL DATA FOR TEST')
  print ('##################################################################')
  # get 10% of total data run prediction to check accuracy
  print('Total number of customers is: {}'.format(df.customer_id.nunique()))

  NUMCUS_TEST = int(0.1*df.customer_id.nunique())
  print('Quantity of customers for test: {}'.format(NUMCUS_TEST))

  customer_list = df.customer_id.unique()
  np.random.seed(0)
  list_cus = random.sample(customer_list.tolist(), NUMCUS_TEST)

  df_test = df[df.customer_id.isin(list_cus)]
  df_test.to_csv('./test.csv',index=None)
  #print('Number cus in df_test:',df_test.customer_id.nunique())

  print ('##################################################################')
  print ('# [4] LABELING - TRUTH VALUE')
  print ('##################################################################')

  df_label_test = labelling(df_test) 
  #df_label_test = df_label[df_label.customer_id.isin(list_cus)]
  df_label_test = df_label_test[['customer_id','churn','critical','alert']]
  df_label_test.to_csv(temppath+'/df_label_test.csv',index=None)
  # return rs is ['customer_id','churn','critical','alert']
  
  print ("[Label] Number of customer is used to predict {}, date from {} to {}".format(df_label_test.customer_id.nunique(), df_test.date_order.min(), df_test.date_order.max()))
  
  print ('##################################################################')
  print ('# [5] PREDICT')
  print ('##################################################################')

  os.system('python churn_predict.py -i DW.test --updatemodel')
  # after run this, the outpath+'/result_CHURN_PREDICTION.csv' result is returned at ...predict_auto... folder.

  print ('##################################################################')
  print ('# [6] GET TEST ACCURACY')
  print ('##################################################################')
  df_predict = pd.read_csv('./test_churn_predict/output/result_CHURN_PREDICTION.csv',index_col=False)

  df_merge = pd.merge(df_predict,df_label_test, on='customer_id',how='left')
  df_merge['alert_match']    = df_merge['alert_x']   == df_merge['alert_x']
  df_merge['critical_match'] = df_merge['critical_x']== df_merge['critical_y']
  df_merge['churn_match']    = df_merge['churn_x']   == df_merge['churn_y']

  alert_acc_test    = df_merge[df_merge['alert_match']==True]['customer_id'].nunique()/df_merge.shape[0]
  critical_acc_test = df_merge[df_merge['critical_match']==True]['customer_id'].nunique()/df_merge.shape[0]
  churn_acc_test    = df_merge[df_merge['churn_match']==True]['customer_id'].nunique()/df_merge.shape[0]
  print_report(' - alert_acc_test:{}\n - critical_acc_test:{}\n - churn_acc_test:{}\n'.format(alert_acc_test,critical_acc_test,churn_acc_test))
  
  print('alert_acc_test   : {}'.format(alert_acc_test))
  print('critical_acc_test: {}'.format(critical_acc_test))
  print('churn_acc_test   : {}'.format(churn_acc_test))

  print ('##################################################################')
  print ('# [7] RETRAIN TRIGGER RELEASE FOR 3 MODELS')
  print ('#     Compare test accuracy with model accuracy for trigger')
  print ('##################################################################')

  # read pkl file
  model_info_dict = {}
  model_info_dict =  pickle.load(open(homepath_train+'/model/model_info_dict.pkl', 'rb'))
  print('model_info_dict:\n',model_info_dict)

  churn_acc_model    = model_info_dict['churn_ACCURACY']
  critical_acc_model = model_info_dict['critical_ACCURACY']
  alert_acc_model    = model_info_dict['alert_ACCURACY'] 
  print_report(' - alert_acc_model:{}\n - critical_acc_model:{}\n - churn_acc_model:{}\n'.format(alert_acc_model,critical_acc_model,churn_acc_model))

  # if 5% reduction of accuracy or less than 81% in absolute value, the RETRAIN Trigger active
  CHURN_RETRAIN_TRIGGER = False
  CRITICAL_RETRAIN_TRIGGER = False
  ALERT_RETRAIN_TRIGGER = False

  if ((alert_acc_test-alert_acc_model)/alert_acc_model<-0.05)|\
   (alert_acc_test<0.81):
    ALERT_RETRAIN_TRIGGER = True
  print_report('ALERT_RETRAIN_TRIGGER: {} (True: Test accuracy < 0.81)'.format(ALERT_RETRAIN_TRIGGER))

  if ((critical_acc_test-critical_acc_model)/critical_acc_model<-0.05)|\
   (critical_acc_test<0.81):
    CRITICAL_RETRAIN_TRIGGER = True
  print_report('CRITICAL_RETRAIN_TRIGGER: {} (True: Test accuracy < 0.81)'.format(CRITICAL_RETRAIN_TRIGGER))
  
  if ((churn_acc_test-churn_acc_model)/churn_acc_model<-0.05)|\
   (churn_acc_test<0.81):
    CHURN_RETRAIN_TRIGGER = True
  print_report('CHURN_RETRAIN_TRIGGER: {} (True: Test accuracy < 0.81)\n'.format(CHURN_RETRAIN_TRIGGER))


  print ('##################################################################')
  print ('# [8] RETRAIN')
  print ('##################################################################')

  if CHURN_RETRAIN_TRIGGER | CRITICAL_RETRAIN_TRIGGER | ALERT_RETRAIN_TRIGGER:
    os.system('python churn_train.py -i DW.{}_truncate --weight_inherit'.format(input.split('.')[-1]))
  
    # READ ACCURACY and CHECK
    model_info_dict = {}
    model_info_dict =  pickle.load(open(homepath_train+'/model/model_info_dict.pkl', 'rb'))
    print('model_info_dict:\n',model_info_dict)
  
    churn_acc_model_new    = model_info_dict['churn_ACCURACY']
    critical_acc_model_new = model_info_dict['critical_ACCURACY']
    alert_acc_model_new    = model_info_dict['alert_ACCURACY'] 

    print('New train model completed with acc of churn({}), critical({}), alert({})'.format(churn_acc_model_new,critical_acc_model_new,alert_acc_model_new))
    print('comparing to Previous acc result of churn({}), critical({}), alert({})'.format(churn_acc_model,critical_acc_model,alert_acc_model))
  
    #if (churn_acc_model_new>=churn_acc_model) & (critical_acc_model_new>=critical_acc_model) & (alert_acc_model_new>=alert_acc_model):
    if (churn_acc_model_new>=0.85) & (critical_acc_model_new>=0.85) & (alert_acc_model_new>=0.85):
      print('New model is UPDATED!!!')
    else:
      print('Reload previous model')
      ##################################################################
      # NOT BETTER MODEL: load backup model back to model folder
      current_day_str = datetime.datetime.now().strftime('%y-%m-%d')
      backup_subdir = homepath_train+"/backup/"+current_day_str
      for file in os.listdir(backup_subdir):
        filename = file.split('/')[-1]
        filename = file.split('/')[-1]
        shutil.copy2(backup_subdir+'/'+filename, modelpath+'/'+filename)
      ##################################################################

  else:
    print('No Retrain ...')
else:
  os.system('python churn_train.py -i DW.{}_truncate'.format(input.split('.')[-1]))

print ('##################################################################')
print ('# AUTO TRAINING COMPLETED!')
print ('##################################################################')