## Project Management Overall

Project management could be implemented through 4 major steps:

   * Project Inititating

   * Project Planning

   * Project Executing

   * Project Closing


#### Project Initiating

In the project initiating, project manager (PM) gets the project request/proposal from senior organization leader or company director board. 
PM needs to work closely with the stakeholders to clarify Objectives, Goals, Scope, Delivarables and Timeline of the project.
Those are discussed significantly and noted into the Project Charter which aligned among stakeholders.

![alt text](https://github.com/carfirst125/portfolio/blob/main/Guildline%20-%20Project%20Management/diagram/PM-Initiating.png?raw=true {width=10, height=10})

#### Personalized Ranking (Reranking)

Input is list of ITEM_IDs and USER_ID. The model computes and re-ranks ITEM_ID in list. So we could know which ITEM_ID is most potential to the customer.

#### SIMS

Input is ITEM_ID. The model returns list of ITEM_IDs which are highest correlation with the input ITEM_ID (no User behavior reference)

## General Block Diagram

![alt text](https://github.com/carfirst125/portfolio/blob/main/aws_personalize_recommendation/images/aws_personalize_recommendation_BlockDiagram.png?raw=true)

## Implementation

1. Data for Training

The input data includes three types of data

#### Interacts metadata

This data could include: 

*   *Customer purchasing history (product, quantity and moneytary)*

*   *Interaction in website, mobile app (views, clicks, carting, checkout)*

#### Items metadata

Comprising data relating to items *(item ingredients, attributes, features)*.

#### Users metadata

This can be demographics information of customers *(gender, age, job, income, etc.)*

**Note:** Those data **MUST** be sent to **S3 Storage** before create group data set in AWS Peronalize and train.

2. Solution (Model), Campaign and Filter

The training is implemented separately with each solution. The Train operation takes resource and charged until training end. 

The saved model as result of training operation is not charged. 

The campaign created a container (endpoint) where hardware is allocated, and is ready for query. The existence of campaign will be charged based on the used resource (EC2). Besides, based on the amount of query, the extra charge is computed.

AWS Personalize also supports **Filter** feature that permit you contrainst in the output result flexiblely.

Example Code

You can refer the example code of AWS at

  *  https://github.com/aws-samples/amazon-personalize-samples/tree/master/next_steps/workshops/POC_in_a_box

Besides, you can refer my diagram and upload code above for working flows.

- - - - - 
Feeling Free to contact me if you have any question around.

    Nhan Thanh Ngo
    Email: ngothanhnhan125@gmail.com
    Skype: ngothanhnhan125
    Phone: (+84) 938005052
