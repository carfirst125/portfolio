## Churn Notification

There are a variety of churn model proposals. However, it is not important what it is. The importance is your question, your business issue that you need to find the solution for that. 

In this project, I would like to propose a simple churn model that support you in alert, provide the notification that which customer need contacting by long time without action, and this also means that the customer is nearly going to churn.

## General Block Diagram

![alt text](https://github.com/carfirst125/portfolio/blob/main/churn_notification/diagram/churn_prediction_BlockDiagram.png?raw=true)

## Implementation

One of the most important questions of churn prediction or notification is how to get the value of the **Distance to Churn**, where a customer could be consulted as churn or not.

This approach suggests using statistical method on the lengths between two continuous purchases of customers for deciding the **Distance to Churn**.

There are different frequencies in purchases among customers. Let try to exploratory data analysis for first purchasing distance, second, third, and so on, and get your own finding. You might recognize that the customer with higher frequency in purchasing seems to have shorter return time (means higher return rate in time). You can find the period of time that almost customer return. This can be compared with actual activity of particular customer to define the suitable Distance to Churn period for them. Based on the value, you can mark the time when the customer need alerting because of long time without purchasing. This is the labeling for churn notification.

This is the time series issue, so I recommend you using LSTM for model train. The simplest input feature is the mark in time series purchasing status of customer.

#### Example Code

You can refer the example code of AWS at

  *  https://github.com/aws-samples/amazon-personalize-samples/tree/master/next_steps/workshops/POC_in_a_box

Besides, you can refer my diagram and upload code above for working flows.

- - - - - 
Feeling Free to contact me if you have any question around.

    Nhan Thanh Ngo
    Email: ngothanhnhan125@gmail.com
    Skype: ngothanhnhan125
    Phone: (+84) 938005052

