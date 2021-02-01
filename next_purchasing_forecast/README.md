## Next Purchasing Forecast

#### Overview

Predicting when customer returns is very important. This supports you in deciding which group of customers prior to interact, based on forecasting of their return day.

Data input

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/Overview.png?raw=true)



![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/EDA.png?raw=true)

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/onepur-cus-train.png?raw=true)

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/onepur-cus-predict.png?raw=true)

The model, used to predict customers which have two purchases or more, is trained by customer having at least three purchases.
When preparing data for Train, the SPLIT TIME is choosen as the the current time. The transaction data after the SPLIT TIME can be processed for the label.

Depend on the model that you use (traditional model or time series), the feature engineering is implemented.

The chart below is an example of using traditional model, the features of model could be [average distance between two purchases, purchase frequency, recency, gender,   ]

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/gt2pur-cus-train.png?raw=true)

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/gt2pur-cus-predict.png?raw=true)

#### Output

Data out could be as below.

![alt text](https://github.com/carfirst125/portfolio/blob/main/next_purchasing_forecast/images/output.png?raw=true)
