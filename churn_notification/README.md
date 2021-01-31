## Churn Notification

There are a variety of churn model proposals. However, it is not important what it is. The importance is your question, your business issue that you need to find the solution for that. 

In this project, I would like to propose a simple churn model that support you in alert, provide the notification that which customer need contacting by long time without action, and this also means that the customer is nearly going to churn.

## General Block Diagram

![alt text](https://github.com/carfirst125/portfolio/blob/main/churn_notification/diagram/churn_prediction_BlockDiagram.png?raw=true)

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

