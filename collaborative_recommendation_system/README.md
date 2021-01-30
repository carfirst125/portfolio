## Recommendation System: Collaborative Filtering

Collaborative Filtering method refer other customers to find the candidate product items.
There are two major methods could be considered to implement. They are user-based and items-based.

#### User-based method

Based on purchasing history of customer likes purchased item-name and quantity, a number of nearby customer are found (eg. N nearby customer). Next, investigating to those customers to get the most favorite item list, then the top ones will be chosen for recommendation.

#### Items-based method

From the purchasing history data of customer, similar items-pair are picked. Similar items-pair is defined as the pair of items which got the same favorite by a customer.
Based on the customer purchased items, and looking up them in similar items-pair table, we have recommendation item list.

#### General Block Diagram

![alt text](https://github.com/carfirst125/portfolio/blob/main/collaborative_recommendation_system/hlc_cfubib_c360_BlockDiagram.png?raw=true)

#### Case Study

A Food and Beverage band need to get insight of which products are the most favourite ones for each customer. Besides, the company also would like to know that beyond the favorite products, whether there are any other products that might are potential to customers, but they have not tried before. Besides the marketing campaign focuses on customer's favourite items, company also would like to have the campaign to recommend new recipes to customers. So customer has the chance to expand their favourite list in their brand. That is great! 

The Collaborative Filter can be used to solve this issue.

#### Implementation Proposal

Belong to which data source that you have, you can define the most suitable solution to solve the issue. 
In this proposal, the simplest method is discussed as your reference which delivers beneficial in this case.

The input data is **the purchasing history of customers**.

The implementation therefore could be two parts. 

The first part can be so-called as **TRAIN process**. This process trains model for user-based method, and does statistical analysis to from similar item pair used by item-based method.

The second part is the **QUERY process**, so you can input the data of a number of customers and get the favourite items, and recommend result in both methods of user-based and item-based.

Refer the below Block Diagram for details.

![alt text](https://github.com/carfirst125/portfolio/blob/main/collaborative_recommendation_system/hlc_cfubib_c360_BlockDiagramDetails.png?raw=true)

The block diagram looks straightforward to understant, doesn't it? However, you will face with critical issue when implement this. Those are discussed further in the next session of important notes.

#### IMPORTANT NOTES

1. The large customer input files

When the amount of customer is small (<100.000), everything is normal. However, when the amount of customers is large, you might got the problem when processing by tremendous matrix process. The kdtree model might be over-memory.

Solution: 
      - increase the resource
      - find another algorithm to find the nearest neighbors          
      - split large input files to smaller files (choose the suitable spliting method)
 
