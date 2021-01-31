##############################################################################
# AWS PERSONALIZE TRAIN FOR ALL USER PERSONALIZATION, SIMS, and PERSONAL RANKING
#-----------------------------------------------------------------------------
# File name:   aws_personalize_train.py 
# Author:      Nhan Thanh Ngo

#############################################################################

import json
import pandas as pd
import numpy as np
from io import StringIO
import boto3
import time
from time import sleep
from datetime import datetime

import datetime
import sys
import os
import re
import io
import argparse
import yaml
import glob

from my_lib import query_bq
from my_lib import data_eda

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="DW.input_table", type=str)
parser.add_argument("--output", default='user_personalization_recommend_result.csv', type=str)
parser.add_argument("--query", default=False, type=bool)
parser.add_argument("--solution", default='userpersonalization', type=str) # userpersonalization/sims/personalizedranking/all
args = parser.parse_args()


########################################################################################################
AWS_ACCESS_KEY_ID = 'S3key'
AWS_SECRET_ACCESS_KEY = 'S3secretkey'
REGION_NAME = 'ap-southeast-1'


# ARN (role, dataset group, interacts_metadata, items_metadata)
role_arn = 'arn:aws:iam::111111111:role/PersonalizeRole'
dataset_group_arn = "arn:aws:personalize:ap-southeast-1:1111111:dataset-group/personalize-project"

# [Import dataset]

# ARN interacts metadata
interaction_schema_arn = "arn:aws:personalize:ap-southeast-1:1111111:schema/personalize-project-interact-schema"
interactions_dataset_arn = "arn:aws:personalize:ap-southeast-1:1111111:dataset/personalize-project/INTERACTIONS"

# ARN items metadata
itemmetadataschema_arn = "arn:aws:personalize:ap-southeast-1:1111111:schema/personalize-project-items-schema"
items_dataset_arn = "arn:aws:personalize:ap-southeast-1:1111111:dataset/personalize-project/ITEMS"

# [Solution]

# ARN User Personalization Solution
user_personalization_recipe_arn = "arn:aws:personalize:::recipe/aws-user-personalization"
user_personalization_solution_arn = "arn:aws:personalize:ap-southeast-1:1111111:solution/personalize-project-userpersonalization"

# ARN SIMS Solution
sims_recipe_arn = "arn:aws:personalize:::recipe/aws-sims"
sims_solution_arn  = "arn:aws:personalize:ap-southeast-1:1111111:solution/personalize-project-sims"

# ARN Personal Ranking Solution
rerank_recipe_arn = "arn:aws:personalize:::recipe/aws-personalized-ranking"
rerank_solution_arn = "arn:aws:personalize:ap-southeast-1:1111111:solution/personalize-project-rerank"

# [Campaign]

# ARN Campaign 
userpersonalization_campaign_arn = "arn:aws:personalize:ap-southeast-1:907079353792:campaign/personalize-project-userpersonalization"
sims_campaign_arn = "arn:aws:personalize:ap-southeast-1:907079353792:campaign/personalize-project-SIMS"
rerank_campaign_arn = "arn:aws:personalize:ap-southeast-1:907079353792:campaign/personalize-project-rerank"    

############################################################################################
session = boto3.Session(
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        region_name=REGION_NAME)
s3_resource = session.resource('s3')
           
personalize = boto3.client('personalize', 
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=REGION_NAME
                      )
          
personalize_runtime = boto3.client('personalize-runtime',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=REGION_NAME
                      )
                      
personalize_events = boto3.client(service_name='personalize-events',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=REGION_NAME
                      )        
bucket_name = 'ABC-rcm-project'

#################################################################################
userpersonalization_active = False;
sims_active = False;    
personalizedranking_active = False;

#################################################################################
s3upload_path = './s3upload'
temp_path = './temp'
extern_path = './extern'

# create folder structure 
def folder_structure():
    global s3upload_path, temp_path, extern_path
    
    if not os.path.exists(s3upload_path):
        os.mkdir(s3upload_path)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    if not os.path.exists(extern_path):
        os.mkdir(extern_path)

# connect to AWS
def connect_aws():
    # get a handle on s3
    session = boto3.Session(
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        region_name='ap-southeast-1')
                        
    return session.resource('s3')

# read data from S3
def read_s3(bucketname, filename):

    # get a handle on s3
    s3 = connect_aws()
    # get a handle on the bucket that holds your file
    s3_bucket = s3.Bucket(bucketname) # bucket = s3.# example: energy_market_procesing
    for obj in s3_bucket.objects.all():
        print('[Read file] Bucket: {}, filename: {}'.format(obj.bucket_name, obj.key))
        key = obj.key
        
        matched = re.match("^{}$".format(filename),obj.key)    
        if bool(matched):
            body = obj.get()['Body'].read()
            df = pd.read_csv(io.BytesIO(body))
            #print('The dataframe is: {}'.format(df))
    
    return df

# upload file to S3 bucket
def upload_file(bucketname,filepath):
    
    s3_resource = connect_aws()
    file_dir, filename = os.path.split(filepath)
    
    bucket = s3_resource.Bucket(bucketname)

    bucket.upload_file(
            Filename=filepath,
            Key=filename,
            ExtraArgs={'ACL':'public-read'}
    )
##############################################################
# WAITING FUNCTION

# IMPORT JOB
def waiting_import_job_complete(dataset_import_job_arn):
    '''
    # WAITING IMPORT JOB COMPLETE
    '''
    print('Waiting import job complete...')
    global personalize

    max_time = time.time() + 6*60*60 # 6 hours
    while time.time() < max_time:
        describe_dataset_import_job_response = personalize.describe_dataset_import_job(
            datasetImportJobArn = dataset_import_job_arn
        )
        status = describe_dataset_import_job_response["datasetImportJob"]['status']
        print("DatasetImportJob: {}".format(status))
        
        if status == "ACTIVE" or status == "CREATE FAILED":
            break
        
        time.sleep(60)

'''
# SOLUTION VERSION
def waiting_solution_train(solution_version_arn):

    print('Waiting solution train...')
    global personalize
    
    max_time = time.time() + 10*60*60 # 10 hours
    while time.time() < max_time:
        version_response = personalize.describe_solution_version(
            solutionVersionArn = solution_version_arn
        )
        status = version_response["solutionVersion"]["status"]
            
        if status == "ACTIVE":
            print("Build succeeded for {}".format(solution_version_arn))
            break
        elif status == "CREATE FAILED":
            print("Build failed for {}".format(solution_version_arn))
        break    

        time.sleep(60)
'''
        
# SOLUTION VERSION
def waiting_solution_train(in_progress_solution_versions):

    print('Waiting solution train...')
    global personalize
    
    max_time = time.time() + 10*60*60 # 10 hours
    while time.time() < max_time:
        for solution_version_arn in in_progress_solution_versions:
            version_response = personalize.describe_solution_version(
                solutionVersionArn = solution_version_arn
            )
            status = version_response["solutionVersion"]["status"]
            
            if status == "ACTIVE":
                print("Build succeeded for {}".format(solution_version_arn))
                in_progress_solution_versions.remove(solution_version_arn)
            elif status == "CREATE FAILED":
                print("Build failed for {}".format(solution_version_arn))
                in_progress_solution_versions.remove(solution_version_arn)
        
        if len(in_progress_solution_versions) <= 0:
            break
        else:
            print("At least one solution build is still in progress")

        time.sleep(60)


# CAMPAIGN
def waiting_campaign_update(in_progress_campaigns):

    print('Waiting campaign update...')
    global personalize
    
    max_time = time.time() + 3*60*60 # 3 hours
    while time.time() < max_time:
        for campaign_arn in in_progress_campaigns:
            version_response = personalize.describe_campaign(
                campaignArn = campaign_arn
            )
            status = version_response["campaign"]["status"]
            
            if status == "ACTIVE":
                print("Build succeeded for {}".format(campaign_arn))
                in_progress_campaigns.remove(campaign_arn)
            elif status == "CREATE FAILED":
                print("Build failed for {}".format(campaign_arn))
                in_progress_campaigns.remove(campaign_arn)
        
        if len(in_progress_campaigns) <= 0:
            break
        else:
            print("At least one campaign build is still in progress")
            
        time.sleep(60)        
    
#################################################################################    
def main(args):

    global s3upload_path, temp_path, extern_path
    global role_arn, dataset_group_arn 
    global interactions_dataset_arn, items_dataset_arn 
    global user_personalization_recipe_arn, sims_recipe_arn, rerank_recipe_arn
    global user_personalization_solution_arn, sims_solution_arn, rerank_solution_arn 
    global userpersonalization_campaign_arn, sims_campaign_arn, rerank_campaign_arn
    global personalize, personalize_runtime, personalize_events
     
    userpersonalization_active = True;
    sims_active = False;    
    personalizedranking_active = False; 
        
    if (args.solution == 'userpersonalization'):
        userpersonalization_active = True;
    elif (args.solution == 'sims'):
        sims_active = True;
    elif (args.solution == 'personalizedranking'):
        personalizedranking_active = True;    
    elif (args.solution == 'all'):
        userpersonalization_active = True;
        sims_active = True;    
        personalizedranking_active = True; 
    
    #folder_structure()
    bucket_name =  'ABC-rcm-project'    
    
    # transform data to AWS Personalize data input format
    # upload transform data to S3
    # import data and train
    ##################################################################
    # Step1. GET DATA FROM BIGQUERY + TRANSFORM
	
    query_bq(args.input, args.query)
	
    print('python train_data_preparation.py --input {}'.format(args.input))
    os.system('python train_data_preparation.py --input {}'.format(args.input))   
    
    ##################################################################
    # Step2. UPLOAD DATA TO S3   

    # upload result of train_data_preparation to S3
    interactions_filename = 'interacts_metadata.csv'
    items_filename = 'items_metadata.csv'
    items_info_filename = 'items_info.csv'
    
    interacts_metadata_path = s3upload_path+'/'+interactions_filename
    items_metadata_path = s3upload_path+'/'+items_filename
    items_info_path = s3upload_path+'/'+items_info_filename

    upload_file(bucket_name, interacts_metadata_path)    
    upload_file(bucket_name, items_metadata_path)    
    upload_file(bucket_name, items_info_path)  
    
    interacts_metadata_s3path = 's3://{}/{}'.format(bucket_name,interactions_filename)
    items_metdata_s3path = 's3://{}/{}'.format(bucket_name,items_filename)
    items_info_s3path = 's3://{}/{}'.format(bucket_name,items_info_filename)
      
    ####################################################################################    
    # Step3. IMPORT INTERACTS, ITEMS, USERS METADATA   
    
    ####################################################################################    
    #    [3.1] IMPORT INTERACT METADATA
    df_interact = read_s3(bucket_name, interactions_filename)
    df_interact = df_interact.sort_values('TIMESTAMP', ascending = True)
    
    today = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')
   
    # [1] import interacts metadata 
    create_dataset_import_job_response = personalize.create_dataset_import_job(
        jobName = "personalize-project-ABC-rcm-interact-import-"+today,
        datasetArn = interactions_dataset_arn,
        dataSource = {
            "dataLocation": interacts_metadata_s3path   # path to interact data [REAL DATA HERE]
        },
        roleArn = role_arn
    )
    dataset_import_job_arn = create_dataset_import_job_response['datasetImportJobArn']
    print(json.dumps(create_dataset_import_job_response, indent=2))
    
    # waiting import job complete
    waiting_import_job_complete(dataset_import_job_arn)
    
    ####################################################################################    
    #    [3.2] IMPORT ITEMS METADATA    
    create_dataset_import_job_response = personalize.create_dataset_import_job(
        jobName = "personalize-project-ABC-rcm-item-import-"+today,
        datasetArn = items_dataset_arn,
        dataSource = {
            "dataLocation": items_metdata_s3path   
        },
        roleArn = role_arn
    )
    dataset_import_job_arn = create_dataset_import_job_response['datasetImportJobArn']
    print(json.dumps(create_dataset_import_job_response, indent=2))
    
    # waiting import job complete
    waiting_import_job_complete(dataset_import_job_arn)

    ####################################################################################    
    #    [3.3] IMPORT USERS METADATA    
    # NO-USER METADATA DEPLOY YET
    
    ####################################################################################    
    # Step4. SOLUTION TRAIN    

    ####################################################################################    
    #    [4.1] USER PERSONALIZATION   
    '''
    user_personalization_create_solution_response = personalize.create_solution(
        name = "personalize-project-userpersonalization",
        datasetGroupArn = dataset_group_arn,
        recipeArn = user_personalization_recipe_arn
    )
    user_personalization_solution_arn = user_personalization_create_solution_response['solutionArn']
    print(json.dumps(user_personalization_solution_arn, indent=2))
    '''    
    
    if userpersonalization_active:
        # Create solution version
        userpersonalization_create_solution_version_response = personalize.create_solution_version(
            solutionArn = user_personalization_solution_arn
        )
        userpersonalization_solution_version_arn = userpersonalization_create_solution_version_response['solutionVersionArn']
        print(json.dumps(userpersonalization_create_solution_version_response, indent=2))
    
    ####################################################################################    
    #    [4.2] SIMS    
    '''
    # create SIMS
    sims_create_solution_response = personalize.create_solution(
        name = "personalize-project-sims-"+today,
        datasetGroupArn = dataset_group_arn,
        recipeArn = sims_recipe_arn
    )
    sims_solution_arn = sims_create_solution_response['solutionArn']
    print(json.dumps(sims_create_solution_response, indent=2))
    '''
    if sims_active:
        # create solution version
        sims_create_solution_version_response = personalize.create_solution_version(
            solutionArn = sims_solution_arn
        )
        sims_solution_version_arn = sims_create_solution_version_response['solutionVersionArn']
        print(json.dumps(sims_create_solution_version_response, indent=2))

    ####################################################################################    
    #    [4.3] PERSONAL RANKING    
    '''
    # create PERSONALIZED RANKING
    rerank_create_solution_response = personalize.create_solution(
        name = "personalize-project-rerank-"+today,
        datasetGroupArn = dataset_group_arn,
        recipeArn = rerank_recipe_arn
    )
    rerank_solution_arn = rerank_create_solution_response['solutionArn']
    print(json.dumps(rerank_create_solution_response, indent=2))
    '''

    if personalizedranking_active:
        # create solution
        rerank_create_solution_version_response = personalize.create_solution_version(
            solutionArn = rerank_solution_arn
        )
        rerank_solution_version_arn = rerank_create_solution_version_response['solutionVersionArn']
        print(json.dumps(rerank_create_solution_version_response, indent=2))    
    
    ###########################################################
    # WAITING MODEL TRAIN
    ###########################################################
    
    # Train 3 models
    in_progress_solution_versions = []
    
    if userpersonalization_active:
        in_progress_solution_versions.append(userpersonalization_solution_version_arn)
        
    if sims_active:
        in_progress_solution_versions.append(sims_solution_version_arn)        
        
    if personalizedranking_active:
        in_progress_solution_versions.append(rerank_solution_version_arn)

    # waiting for solution train
    waiting_solution_train(in_progress_solution_versions)    
    
    ####################################################################################        
    # Step5. EVALUATION

    # [5.1] User Personalization metrics        
    if userpersonalization_active:    
        user_personalization_solution_metrics_response = personalize.get_solution_metrics(
            solutionVersionArn = userpersonalization_solution_version_arn
        )
        print(json.dumps(user_personalization_solution_metrics_response, indent=2))
    
    # [5.2] SIMS metrics
    if sims_active:
        sims_solution_metrics_response = personalize.get_solution_metrics(
            solutionVersionArn = sims_solution_version_arn
        )
        print(json.dumps(sims_solution_metrics_response, indent=2))    
            
    # [5.3] Personalized ranking metrics
    if personalizedranking_active:
        rerank_solution_metrics_response = personalize.get_solution_metrics(
            solutionVersionArn = rerank_solution_version_arn
        )
        print(json.dumps(rerank_solution_metrics_response, indent=2))    
   
    ####################################################################################
    # Step6. CREATE CAMPAIGN    
    
    # [6.1] User Personalization Campaign    
    if userpersonalization_active:    
        userpersonalization_update_campaign_response = personalize.update_campaign(
            campaignArn=userpersonalization_campaign_arn,
            solutionVersionArn=userpersonalization_solution_version_arn,
            minProvisionedTPS=1,
        )    
        userpersonalization_campaign_arn = userpersonalization_update_campaign_response['campaignArn']
        print(json.dumps(userpersonalization_update_campaign_response, indent=2))
    
    # [6.2] SIMS Campaign    
    if sims_active:
        sims_update_campaign_response = personalize.update_campaign(
            campaignArn=sims_campaign_arn,
            solutionVersionArn=userpersonalization_solution_version_arn,
            minProvisionedTPS=1,
        )    
        sims_campaign_arn = sims_update_campaign_response['campaignArn']
        print(json.dumps(sims_update_campaign_response, indent=2))    
    
    # [6.3] Personalized Ranking Campaign
    if personalizedranking_active:
        rerank_update_campaign_response = personalize.update_campaign(
            campaignArn=rerank_campaign_arn,
            solutionVersionArn=rerank_solution_version_arn,
            minProvisionedTPS=1,
        )        
        rerank_campaign_arn = rerank_update_campaign_response['campaignArn']
        print(json.dumps(rerank_update_campaign_response, indent=2))    
            
    
    # create campaign for 3 models
    in_progress_campaigns = []
    
    if userpersonalization_active:
        in_progress_campaigns.append(userpersonalization_campaign_arn)
        
    if sims_active:
        in_progress_campaigns.append(sims_campaign_arn)        
        
    if personalizedranking_active:
        in_progress_campaigns.append(rerank_campaign_arn)  
        
    # waiting 
    waiting_campaign_update(in_progress_campaigns)
    
if __name__ == "__main__":  
    main(args)
   



