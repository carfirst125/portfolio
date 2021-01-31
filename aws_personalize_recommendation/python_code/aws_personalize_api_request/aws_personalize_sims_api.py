##############################################################################
# AWS PERSONALIZE: SIMS
#-----------------------------------------------------------------------------
# File name:   aws_personalize_sims_api.py 
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
parser.add_argument("--product_id", default=None, type=str)
parser.add_argument("--output", default='', type=str)
parser.add_argument("--query", default=False, type=bool)
parser.add_argument("--limit", default=0, type=int)
parser.add_argument("--commit", default=False, type=bool)
parser.add_argument("--commitonly", default=False, type=bool)
args = parser.parse_args()

if not args.output: 
   args.output = args.input+'_AWS_SIMS'

proid_list_en = False
proid_list = []
if args.product_id != None:
   proid_list = args.product_id.split(',')
   print("Run SIMS for product ID: {}".format(proid_list))
   proid_list_en = True
   
if (not args.input) and (not proid_list_en):
   print("Please feed --input DW.tablename, or --product_id 44454,33223 for running")

print ('AWS Personalize Query PROGRAM')
print ('args.output=',args.output)

AWS_ACCESS_KEY_ID = 'S3key'
AWS_SECRET_ACCESS_KEY = 'S3secretkey'
REGION_NAME = 'ap-southeast-1'
sims_campaign_arn = "arn:aws:personalize:ap-southeast-1:11111111111:campaign/personalize-ABC-SIMS"

personalize_runtime = boto3.client('personalize-runtime',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=REGION_NAME
                      )
    
def get_new_recommendations_df_items(recommendations_df, item_id):

    # Get the recommendations
    get_recommendations_response = personalize_runtime.get_recommendations(
        campaignArn = sims_campaign_arn,
        itemId = str(item_id)
    )
    
    # Build a new dataframe of recommendations
    item_list = get_recommendations_response['itemList']

    recommendation_list = []    
    for item in item_list:
        recommendation_list.append(item['itemId'])
    new_rec_DF = pd.DataFrame(recommendation_list, columns = [str(item_id)])
    
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
  
  
def main():
  
  global proid_list, proid_list_en
  result_path = './result/aws_sims_result_transpose.csv'
  bucket_name =  'ABC-rcm-project'  
  
  if not os.path.exists('./result'):
    os.mkdir('./result')  
    
  if not os.path.exists('./temp'):
    os.mkdir('./temp') 
  else:
    shutil.rmtree('./temp')
    os.mkdir('./temp')

  print('Initiating...')  
  
  if not args.commitonly:
    ###########################################################
    if not proid_list_en:      

      sql = "SELECT distinct product_id as ITEM_ID FROM "+args.input+" GROUP BY product_id"
      df_query = query_bq_sql(sql, args.input, args.query)
      df_query.ITEM_ID = df_query.ITEM_ID.astype(str)
    
      proid_list = df_query.ITEM_ID.values
        
    print('Number of product_id need do API Request: {}'.format(len(proid_list)))
    ###########################################################
        
    # get recommend result return from model
    recommendations_df = pd.DataFrame()
    for product_id in proid_list:
      recommendations_df = get_new_recommendations_df_items(recommendations_df,product_id)

    # save file the rest
    recommendations_df = recommendations_df.transpose().reset_index()
    result_path_pre = './temp/aws_sims_result.csv'
    recommendations_df.to_csv(result_path_pre,index=False)
 
    # change columns name
    print(recommendations_df.columns.values[1:])
    recommendations_df.columns = ['product_id'] + list(recommendations_df.columns.values[1:])    
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




