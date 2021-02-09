## Customer Segmentation by Product

### Overview

In this case study, the customers, after clustering using RFM (Recency, Frequency, Monetary) method, need to be grouped to smaller group of similar favourite. So the customer segmentation by product is implemented to solve this request.

The RFM method supports clustering customer by three dimension Recency, Frequecy, Monetary. Based on the location of customer in RFM map, we can analyse and assess customers to choose the suitable marketing campaign for each RFM group. 

However, if we would like to approach customer in another dimension, for eg. who and what are their favourite products. The customer segment by product is investigated. We can combine RFM method and Customer segment by product for more insights.

### Implementation

#### General Block Diagram

Based on the requirement, the K-means algorithm is chosen as solution for this issue. Following that, the customer having similar favourite in product purchasing would be grouped in the same segment. Investigating to each segment by statistics analysis for which products purchased most, we get the insights of the segment.

Please take a look on Block Diagram below for detail flow.

<img src="https://github.com/carfirst125/portfolio/blob/main/customer_segmentation/images/customer_cluster_by_product_BlockDiagram.png?raw=true"/>

#### Input data 

In this example, the input data need just simple like below. That is the customer summary of purchasing information.

<img src="https://github.com/carfirst125/portfolio/blob/main/customer_segmentation/images/customer_input_data.png?raw=true"/>

#### Segment Info Summary

The figure is an example of kmeans cluster result.

<img src="https://github.com/carfirst125/portfolio/blob/main/customer_segmentation/images/segment_summary.png?raw=true"/>

#### Segment Insight Investigation

Each pie chart performs the statistics of segment product purchased by customer in the group. Based on this information, you can assign a name/label for the segment. 

<img src="https://github.com/carfirst125/portfolio/blob/main/customer_segmentation/images/segment_labeling.png?raw=true"/>

    For instance, 
        Cluster 18 can be labeled as Tra Sen (Lotus Tea) group.
        Cluster 11 can be labeled as Tra (Tea) group.
        Cluster 07 can be labeled as Phin den da (Black Coffee) group.
    
#### Which customer in which segment result

Finally, we need to know which customer in which segment. The result can be as below.

<img src="https://github.com/carfirst125/portfolio/blob/main/customer_segmentation/images/customer_in_segment.png?raw=true"/>
