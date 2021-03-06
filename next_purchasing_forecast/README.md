## Next Purchasing Forecast

#### Overview

Predicting when customer returns to purchase is very important. This forecast supports in deciding which group of customers priors to interact, based on anticipating their next purchasing date.

#### General Block Diagram

This is the simple block digram of train and query from the model.

<img src="https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/next_purchase_forecast_BlockDiagram.png?raw=true"/>

#### EDA 

The **Exploratory Data Analysis (EDA)** is the crucial step that lets you understand the data, therefore having relevant solution including choosing machine learning algorithm and features. 

This begins with finding outliers and gets rid of unreal customer accounts. For example, common accounts record data of multicustomer instead of a particular customer should be eliminated. Accounts that has the actual high frequency abnormally also need to be ignored. In short, the accounts, which are not real as the transaction of a customer, should be abandoned to prevent the bias.

**Example:** The figure below shows the EDA result for customer purchasing frequency. The customer is grouped by **one purchased customer**, **two or more purchased customer**, and **abnormal/weird customer**. Based on your understanding in the business, you can define the rules for classifying abnormal customers. For instances, in this case of shoes and hand bag product, the customer who had more than 200 times of purchasing in data period or having purchase every 4 days, or having 3 purchases per day, will be considered as abnormal one. 

<img src="https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/EDA.png?raw=true" width="600"/>

#### ML Model

The supervised model will be used to do the next purchase forecast. So the input features and label are neccesary for supervised model train.

Depending on the attributes of product and business industry, the **SPLIT TIME** is declared. The **SPLIT TIME** is the date before the the last date of data. **SPLIT TIME** is applied for purpose of creating the model label. (Eg. in food and beverage, the **SPLIT TIME** could be one or two months back. However, in the fashion industry, the SPLIT TIME might be longer from three to six months back from the last date in data.)

* **Model for customer with only one purchase**

There are no more criteria for new customers with only one purchase, that the model can learn to infer the next purchasing. The model used for one purchasing customer is trained by the data of 2 purchasing customers. 

<img src="https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/onepur-cus-train.png?raw=true" width="600"/>

<img src="https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/onepur-cus-predict.png?raw=true" width="600"/>

* Model for customer with 2 purchased or more

The model, used to predict customers which have two purchases or more, is trained by customers having at least three purchases.
When preparing data for Train, the SPLIT TIME is chosen and used as the supposed current time. The transaction data after the SPLIT TIME can be processed for the label.

Depending on the kind of algorithm you use (traditional or time series), the feature engineering is implemented in different way.

The below chart is an example of using traditional model. The input features of the model could be **[average purchase distance, standard deviation of purchase distances, recency, monetary, gender, type of customer (vip/normal), purchase channel]**

The label is the time distance between two contineous purchases, the last purchase before SPLIT TIME and first purchase after SPLIT TIME.

<img src="https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/gt2pur-cus-train.png?raw=true" width="600"/>

<img src="https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/gt2pur-cus-predict.png?raw=true" width="600"/>

If you use LSTM model, the features might be **[monetary amount, type of customer (vip/normal), purchase channel, gender] by time series**

The label is also marked in time series or the same as above for the first transaction of customer after SPLIT TIME.

#### Output

Data out could be shown as below, including the day when the forecast runs (current day), the next purchasing date forecast result following by the error range, and confident score respectively. 

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/output.png?raw=true)

- - - - - 
Feeling Free to contact me if you have any question around.

    Nhan Thanh Ngo
    Email: ngothanhnhan125@gmail.com
    Skype: ngothanhnhan125
    Phone: (+84) 938005052
