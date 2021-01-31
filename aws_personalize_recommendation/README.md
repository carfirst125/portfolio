## AWS Personalize Recommendation (User Personalization, Personalized Ranking, SIMS)

Collaborative Filtering method refer other customers to find the candidate product items.
There are two major methods could be considered to implement. They are user-based and items-based.

#### User Personalization

Input is USER_ID. The model returns list of ITEM_IDs which have highest probability in purchasing at the next transaction.

#### Personalized Ranking (Reranking)

Input is list of ITEM_IDs of a USER_ID. The model computes and re-ranks ITEM_ID in list. So we could know which ITEM_ID is most potential to the customer.

#### SIMS

Input is ITEM_ID. The model returns list of ITEM_IDs which are highest correlation with the input ITEM_ID (no User behavior reference)

#### General Block Diagram

![alt text](https://github.com/carfirst125/portfolio/blob/main/aws_personalize_recommendation/images/aws_personalize_recommendation_BlockDiagram.png?raw=true)

#### Case Study
