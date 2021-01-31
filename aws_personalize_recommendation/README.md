## AWS Personalize Recommendation (User Personalization, Personalized Ranking, SIMS)

Collaborative Filtering method refer other customers to find the candidate product items.
There are two major methods could be considered to implement. They are user-based and items-based.

#### User Personalization

Based on purchasing history of customer likes purchased item-name and quantity, a number of nearby customer are found (eg. N nearby customer). Next, investigating to those customers to get the most favorite item list, then the top ones will be chosen for recommendation.

#### Personalized Ranking

From the purchasing history data of customer, similar items-pair are picked. Similar items-pair is defined as the pair of items which got the same favorite by a customer.
Based on the customer purchased items, and looking up them in similar items-pair table, we have recommendation item list.

#### SIMS

#### General Block Diagram

![alt text](https://github.com/carfirst125/portfolio/blob/main/aws_personalize_recommendation/images/aws_personalize_recommendation_BlockDiagram.png?raw=true)

#### Case Study
