## Recommendation System: Collaborative Filtering

Collaborative Filtering method refer other customers to find the candidate product items.
There are two major methods could be considered to implement. They are user-based and items-based.

#### User-based method

Based on purchasing history of customer likes purchased item-name and quantity, a number of nearby customer are found (eg. N nearby customer). Next, investigating to those customers to get the most favorite item list, then the top ones will be chosen for recommendation.

#### Items-based method

From the purchasing history data of customer, similar items-pair are picked. Similar items-pair is defined as the pair of items which got the same favorite by a customer.
Based on the customer purchased items, and looking up them in similar items-pair table, we have recommendation item list.

#### General Block Diagram

![alt text](https://github.com/carfirst125/portfolio/blob/main/collaborative_recommendation_system/images/cfubib_BlockDiagram.png?raw=true)

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

![alt text](https://github.com/carfirst125/portfolio/blob/main/collaborative_recommendation_system/images/cfubib_BlockDiagramDetails.png?raw=true)

The block diagram looks straightforward to understant, doesn't it? However, you will face with critical issue when implement this. Those are discussed further in the next session of important notes.

#### IMPORTANT NOTES

1. The large number of customers

When the amount of customer is small (<100.000), everything is normal. However, when the amount of customers is large, you might get the problem with tremendous matrix process. The kdtree model might be over-memory.

*Solution:* 

      - increase the computing resource      
      - find another algorithm to find the nearest neighbors instead of kdtree       
      - split large input files to smaller files (choose the suitable spliting method)
 
 2. Fair Normalization
 
In kdtree, the minkowski distance with p=2 is used. It is crucial to normalize data which ensure the fair treatment among multi-dimensions.

For the max-min normalization, it is suggested to analyze the distribution of each dimension data in order to choose max-min which covers 85% of data. The value out of range should be adjusted to the max or min depending on their actual value.

This process reduces the bias effect of minority (outliers) to majority.

3. Query Kdtree model

Need to pay attention on the input matrix when query kdtree. The quantity of items or the order of items in input matrix is not right when query causes the error or wrong result. It is needs to save the input items list at TRAIN process, and ensure the input matrix in QUERY process is same meaning.

4. Similar item-pair lookup

In the item-based method, based on the consuming history of customer, the equavalent item in the item-pair would be extracted and ranked for the potential recommend item list. So which criteria used to rank the items in different pairs is the important thing to discuss. Recommend using number of customer vote for each similar item-pair, folowing with quantity of used items per customer for ranking.

5. Product multi-level tree (Categories and SKU)

The product tree usually has some levels from categories to sub-categories and SKU/Product-ID at the end of tree. The SKU indicates particular product like type of product, size, color, season, etc. 

Based on your insight in the industry, defining the suitable strategies in choosing which data will be used to analyze. For example, in this case study, you can choose sub-categories of product name for analysis, then lookup for SKU/Product ID if needing to recommend by product ID. 

- - - - - 
Feeling Free to contact me if you have any question around.

    Nhan Thanh Ngo (Mr.)
    Email: ngothanhnhan125@gmail.com
    Skype: ngothanhnhan125
    Phone: (+84) 938005052


