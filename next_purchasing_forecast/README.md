## Next Purchasing Forecast

#### Overview

Predicting when customer returns is very important. This supports you in deciding which group of customers prior to interact, based on forecasting of their return day.

#### General Block Diagram

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/Overview.png?raw=true)

#### EDA 

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/EDA.png?raw=true)

#### ML Model

The supervised model will be used to do the next purchase forecast. So it has to have the features and label for supervised model train.

Depend on the attribute of product and industry, the SPLIT TIME is declared before the max-date of data for create the label.

Eg. in food an beverage, the SPLIT TIME could be from one to two monthes back. However, in the fashion industry, the SPLIT TIME might be longer from three to six monthes back from the max-date of data.


* Model for customer with only on purchase


![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/onepur-cus-train.png?raw=true)

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/onepur-cus-predict.png?raw=true)

* Model for customer with 2 purchased or more

The model, used to predict customers which have two purchases or more, is trained by customer having at least three purchases.
When preparing data for Train, the SPLIT TIME is choosen as the the current time. The transaction data after the SPLIT TIME can be processed for the label.

Depend on the model that you use (traditional model or time series), the feature engineering is implemented.

The chart below is an example of using traditional model. The features engineering of the model could be **[average purchase distance, standard deviation of purchase distances, recency, monetary, gender, type of customer (vip/normal), purchase channel]**

The label is the time distance from the last purchase before SPLIT TIME and first purchase after SPLIT TIME.

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/gt2pur-cus-train.png?raw=true)

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/gt2pur-cus-predict.png?raw=true)


If you use LSTM model, the features might be **[monetary amount, type of customer (vip/normal), purchase channel, gender] by time series**

The Label is the first transaction of customer after SPLIT TIME. Of cource, the label is in the sequence in time series marked where customer makes purchasing.

#### Output

Data out could be shown as below, including the day when the forecast runs (current day), the next purchasing date forecast result following by the error range, and confident score respectively. 

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/output.png?raw=true)
