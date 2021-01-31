##############################################################################
# AWS PERSONALIZE: User Personalization + Rerank
#-----------------------------------------------------------------------------
# File name:   aws_personalize_evaluation_nday.py 
# Author:      Nhan Thanh Ngo
#############################################################################

import json
import pandas as pd
from io import StringIO 
import boto3
import time
from time import sleep
import json
from datetime import datetime
import numpy as np
import datetime
import sys
import os
import re
import io
import argparse
import shutil
import glob
import matplotlib.pyplot as plt

  
parser = argparse.ArgumentParser()
parser.add_argument("--input", default='DW.input_table', type=str)
parser.add_argument("--customer_id", default=None, type=str)
parser.add_argument("--output", default='', type=str)
parser.add_argument("--query", default=False, type=bool)
parser.add_argument("--limit", default=0, type=int)
parser.add_argument("--numdayback", default=1, type=int) 
parser.add_argument("--commit", default=False, type=bool)
parser.add_argument("--commitonly", default=False, type=bool)
args = parser.parse_args()

print('AWS Personalize Query PROGRAM')

###########################################
# Function: query_bq_sql(sql, input, query)
# Input:
#   - sql  : sql statement
#   - input: input table name in DW
#   - query: enable signal
# Output:
#   - df: the input data 

def query_bq_sql(sql, input, query):

  print ('#############################')
  print ('# QUERY DATA FROM BIGQUERY')
  print ('#############################')

  from google.cloud import bigquery
  from google.oauth2 import service_account

  client = bigquery.Client()
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
    
  return df

##################################################
# Function: verify_top_N(df_actual, df_model, N=3)
# Input:
#   - df_actual: df_pivot customer purchase in the day
#   - df_model : data returns from AWS
#   - N        : number of items in AWS result are used
# Output:
#   - accuracy rate

def verify_top_N(df_actual, df_model, N=3):
  
  df_model_N = df_model.iloc[:,:N+1]
  df_model_N = df_model_N.rename({'customer_id':'USER_ID'},axis='columns') 
  print(df_model_N)
  df_merge = pd.merge(df_model_N, df_actual, on = 'USER_ID', how='left')
  print(df_merge)

  df_merge['verified'] = [len(np.intersect1d(df_merge.iloc[i,1:N+1].values, df_merge.iloc[i,N+1:].values)) != 0 for i in np.arange(df_merge.shape[0])]
  
  df_merge.to_csv("./temp/df_merge.csv", index=False)

  if df_merge[df_merge.verified == True].shape[0]!=0:
    rate = df_merge[df_merge.verified == True].shape[0]/df_merge.shape[0]
  else:
    rate = -1
  print("Number of items recommend: {}, accuracy_rate: {}".format(N,rate))
  
  return rate
  
##############################################################################  
def main():

  print('Initiating...') 
  bucket_name =  'ABC-rcm-project'
  cusid_list_en = False
  cusid_list = []  

  from_date = datetime.datetime.now()- datetime.timedelta(days=args.numdayback)
  from_date = from_date.strftime("%Y-%m-%d")
  print("Get data from date: {}".format(from_date))
  
  sql = "select distinct customer_id as USER_ID, product_id as ITEM_ID, date_order from " + args.input + \
        " where date_order >= \'" + from_date + "\'"\
        " group by customer_id, product_id, date_order"
        
  print('sql statement: {}'.format(sql))

  ###########################################################
  # QUERY CUSTOMER ID LIST
  # query and get list user in the particular period of days
  df_query_org = query_bq_sql(sql, args.input, args.query)
  df_query_org.date_order = pd.to_datetime(df_query_org.date_order)
  
  cusid_list = df_query_org.USER_ID.unique() 
  cusid_list_str = ",".join([str(item) for item in cusid_list])

  ###########################################################
  # API QUERY AWS PERSONALIZE RERANK
  #    User personalization + Rerank 
  print('Total amout of customer_id RERANK: {}'.format(len(cusid_list)))
  print("Request AWS API for rerank result...")
  os.system("python aws_personalize_user_personalization_rerank_api.py --customer_id {}".format(cusid_list_str))  

  #################################################################################
  # SWEEP EACH DAY IN THE DAYBACK and CHECK HOW ACTUAL PURCHASE MATCHES AWS RESULT
  for dayback in np.arange(1,args.numdayback+1):

    check_date = datetime.datetime.now()- datetime.timedelta(days=int(dayback))
    check_date = check_date.strftime("%Y-%m-%d")
    print("Evaluation Date: {}".format(check_date))
    df_query = df_query_org[df_query_org.date_order == check_date]
     
    # get list of customer_id
    cusid_list = df_query.USER_ID.unique() 
    print(cusid_list)
    print("Number of customer: {}".format(len(cusid_list)))

    # get Reranking result of a particular day
    df_aws_rerank = pd.read_csv('./result/aws_rerank_result.csv',index_col=False)
    df_aws_rerank = df_aws_rerank[df_aws_rerank.customer_id.isin(cusid_list)]
  
    df_query['indexing'] = df_query.groupby('USER_ID').cumcount()+1	
    df_query.to_csv("./temp/df_query.csv", index=False)  

    df_query = pd.pivot_table(df_query, values='ITEM_ID', index=['USER_ID'], columns=['indexing'], aggfunc=np.sum, fill_value=0).reset_index()
    df_query.to_csv("./temp/df_query_pivot.csv", index=False)  

    # get accuracy rate 
    accuracy_rate = []
    for i in np.arange(1,20):
      accuracy_rate.append(verify_top_N(df_query, df_aws_rerank, i))
    print(accuracy_rate) 
	
    # plot the chart for accuracy for N items choice.
    x = np.arange(1,20,1)
    plt.figure(figsize=(20, 10))
    plt.plot(x , accuracy_rate, color='r')
    plt.xlabel("Number of recommend items")
    plt.ylabel("Accuracy")  
    plt.title("Accuracy based on Number of Recommend items. Observed Population: {} customers".format(df_query.USER_ID.nunique()))
    plt.savefig("./temp/accuracy_chart_"+str(check_date)+".png")

if __name__ == "__main__":    
    main()    
  
  
  

  
  
  
  
  
  