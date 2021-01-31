##############################################################################
# AWS PERSONALIZE: User Personalization 
#-----------------------------------------------------------------------------
# File name:   aws_personalize_user_personalization_api.py 
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

parser = argparse.ArgumentParser()
parser.add_argument("--input", default='DW.input_table', type=str)
parser.add_argument("--customer_id", default=None, type=str)
parser.add_argument("--output", default='', type=str)
parser.add_argument("--query", default=False, type=bool)
parser.add_argument("--limit", default=0, type=int)
parser.add_argument("--commit", default=False, type=bool)
parser.add_argument("--commitonly", default=False, type=bool)
args = parser.parse_args()
   
print ('AWS Personalize Query PROGRAM')

AWS_ACCESS_KEY_ID = 'S3key'
AWS_SECRET_ACCESS_KEY = 's3secretkey'
REGION_NAME = 'ap-southeast-1'
userpersonalization_campaign_arn = "arn:aws:personalize:ap-southeast-1:111111111:campaign/personalize-ABC-userpersonalization"
rerank_campaign_arn = "arn:aws:personalize:ap-southeast-1:1111111:campaign/personalize-ABC-rerank"
                     
personalize_runtime = boto3.client('personalize-runtime',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=REGION_NAME
                      )
  
###########################################################################
# Function: get_new_recommendations_df_users(recommendations_df,user_id)
# Description:
#    input user_id and recommendations df for adding the result into.
#

def get_new_recommendations_df_users(recommendations_df,user_id):

   
    # get recommend result from campaign
    get_recommendations_response = personalize_runtime.get_recommendations(
        campaignArn = userpersonalization_campaign_arn,
        userId = str(user_id),
    )
    
    # get itemList. In itemList, get itemId
    item_list = get_recommendations_response['itemList']

    recommendation_list = []
    for item in item_list:
        recommendation_list.append(str(item['itemId']))

    new_rec_DF = pd.DataFrame(recommendation_list, columns = [user_id])

 
    # Add this dataframe to the old one
    recommendations_df = pd.concat([recommendations_df, new_rec_DF], axis=1)
    
    return recommendations_df
  

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
  
##############################################################################  
def main():

  print('Initiating...') 
  bucket_name =  'ABC-rcm-project'
  cusid_list_en = False
  cusid_list = []  
  
  ###############################################
  # GetOptions process
  ###############################################  
  if not args.output: 
    args.output = args.input+'_AWS_USER_PERSONALIZATION'
  print ('args.output=',args.output)
  
  if args.customer_id != None:
    cusid_list = args.customer_id.split(',')
    print("Run User Personalization for customer_id ID: {}".format(cusid_list))
    cusid_list_en = True
   
  if (not args.input) and (not cusid_list_en):
    print("Please feed --input DW.tablename, or --product_id 44454,33223 for running")
  
  ###############################################
  # Folder and Path
  ###############################################  
  result_path = './result/aws_user_personalization_result_transpose.csv'    
  
  if not os.path.exists('./result'):
    os.mkdir('./result')  
    
  if not os.path.exists('./temp'):
    os.mkdir('./temp') 
  else:
    shutil.rmtree('./temp')
    os.mkdir('./temp')


  ###############################################
  # Main Stream
  ###############################################  
  if not args.commitonly:
    
    ###########################################################
    # query if input is table on Big Query
    if not cusid_list_en:      
        
      if args.limit:
        limit = " LIMIT "+str(args.limit)
      else:
        limit = ""
      # read data from bigquery  
      sql = "select distinct customer_id as USER_ID from " + args.input + \
            " group by customer_id"+limit 
    
      df_query = query_bq_sql(sql, args.input, args.query)
      df_query.USER_ID = df_query.USER_ID.astype(str)
    
      cusid_list = df_query.USER_ID.values
    ###########################################################
    
    print('Number of customer_id need do API Request: {}'.format(len(cusid_list)))
    
    # get recommend result return from model
    CUSTOMER_BLOCK = 100;
    recommendations_df = pd.DataFrame()
    for ind, user_id in zip(np.arange(1,len(cusid_list)+1),cusid_list):

      if ((ind%CUSTOMER_BLOCK)==0):
        recommendations_df = get_new_recommendations_df_users(recommendations_df,user_id)
        result_path_pre = './temp/aws_user_personalization_result'+str(int(ind/CUSTOMER_BLOCK))+'.csv'
        recommendations_df = recommendations_df.transpose().reset_index()
        recommendations_df.to_csv(result_path_pre,index=False)
        recommendations_df = pd.DataFrame()
      else:
        recommendations_df = get_new_recommendations_df_users(recommendations_df,user_id)      
    
    # save file the rest
    recommendations_df = recommendations_df.transpose().reset_index()
    result_path_pre = './temp/aws_user_personalization_result'+str(int(ind/CUSTOMER_BLOCK)+1)+'.csv'
    recommendations_df.to_csv(result_path_pre,index=False)

    # concat all file in directory to file
    recommendations_df = pd.DataFrame()
    listfile = os.listdir('./temp')
    for i, file in zip(np.arange(len(listfile)),listfile):
      df_temp = pd.read_csv("./temp/"+file, index_col=None)
      recommendations_df = pd.concat([recommendations_df,df_temp])    
    
    # change columns name
    print(recommendations_df.columns.values[1:])
    recommendations_df.columns = ['customer_id'] + list(recommendations_df.columns.values[1:])    
    recommendations_df.to_csv(result_path,index=False) 
 
  ###########################################################
  # up data to Bigquery
  if (args.commit or args.commitonly):
  
    print('Commiting...')
    commit_bq_table_name = args.output
    print(commit_bq_table_name)   
    
    print('python commit_bq.py -s {} -d {}'.format(result_path,commit_bq_table_name))
    os.system('python commit_bq.py --source {} --destination {}'.format(result_path,commit_bq_table_name))

  ############################################################     
  print('Completed!')        

if __name__ == "__main__":    
    main()




