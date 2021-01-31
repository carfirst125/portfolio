## AWS Personalize Recommendation (User Personalization, Personalized Ranking, SIMS)

AWS Personalize provides various recommendation methods for different questions. This project three recommendation method are discussed including **User Personalization**, **Personalized Ranking** and **SIMS**. Those are mentioned further as below.

AWS Personalize uses HRNN model to implement this application.

#### User Personalization

Input is USER_ID. The model returns list of ITEM_IDs which have highest probability in purchasing at the next transaction.

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



#### Item metadata

#### User metadata


