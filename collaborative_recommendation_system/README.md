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

#### Aplication

Case Study: A Food and Beverage band need to get insight of which products are the most favourite ones for each customer. Besides, they also would like to know beyond those product whether there are other products that might are potential to them, but they have not tried before. 

#### Implementation Proposal

Case
You can refer the below details Block Diagram for implement  the Collaborative Filtering 

![alt text](https://github.com/carfirst125/portfolio/blob/main/collaborative_recommendation_system/hlc_cfubib_c360_BlockDiagramDetails.png?raw=true)
