## Customer Data Platform (CDP)

CDP in Figure 1 can be considered as a customer data-driven platform which is used to deploy digital transformation. Through the CDP platform, new deployed MLAI applications can be experienced by customer as soon as deployed. The company also can get the experiment result by real interaction.

CDP helps collecting customer data from Omni-channels from our data source such as website, mobile app, CRM system, or from association partners such as Ads Publishers (Google Ads, FB Ads, etc.), or purchase data of 3rd Party such as Bluekai, Lotame, etc.

The data are collected to the data lake, and extracted into various data marts for particular usage purpose. Those data marts are used by MLAI applications and Dashboard generation as well.

_Figure 1: Customer Data Platform architecture_

<img src="https://github.com/carfirst125/portfolio/blob/main/Business%20Information%20System%20-%20Customer%20Orientation/figures/BIS_CDP.png" width="90%" height="90%" alt="CDP">

The Dashboard plays an important role for staff users in getting insight, and making decision. The Dashboard update automatically daily for latest information showed in dashboard.

### Customer Unique ID

<img src="https://github.com/carfirst125/portfolio/blob/main/Business%20Information%20System%20-%20Customer%20Orientation/figures/BIS_CUS_UUID.png" width="20%" height="20%" alt="UUID" align="left" style="margin-right: 100px;"/>

One of the most advantages of CDP is ability to unique recognize customer when they interact with any managed user interface (app, website, fanpage, etc.). 
When customers use their account with phone number, email, google account, social network account, the device with recorded Device ID, or PC with saved cookies, they are recognized.

This helps collect customer data real-time, and is the foundation for implement Personalize model. The Personalize is proven method that get high effective in accelerating the conversion. This method is present in next session.

.

.

.

### Personalize

_Figure 3: Personalize, how it works_

<img src="https://github.com/carfirst125/portfolio/blob/main/Business%20Information%20System%20-%20Customer%20Orientation/figures/BIS-Personalize.png" width="60%" height="60%" alt="Personalize"/>

The idea of Personalize is: 

based on our customer understanding and recent customer interact, do real-time provide the suitable content or notification to customer when they are surfing in our terminal interface such as our app, website, fanpage, etc.

For example, customer are visiting our website, and viewing page of Loan. It is likely that customer are having demand of loan. If our business information system can track, immediately process and pop-up special offer information of loan to customer at the moment. It is high ability that customer will click to read, and also increase the registration conversion.

Opposite with prediction based on history data, this method does forecasting based on recent observation of customer interact in content.

### RFM (Recency, Frequency, Monetary)

The RFM is the well-known method using to assess and segment customers based on their transaction history. By the method, three factors (Recency, Frequency and Monetary) are used to measure the potential and classify customer at the current time.

  *	Recency: the time length from last transaction to present.

  *	Frequency: how many transactions that customer had in the observed time range (eg. last 12 months)

  *	Monetary: it is the amount of money the customer spend. In the case of loan or credit card, monetary could be the used amount of credit.

Depending on value of Recency, Frequency, Monetary, customer will be arrange into a particular segments. 
Customers have low recency means they still are in retention, and are using the service. Whereas high recency means that the customers have not used the service for a long time.

The frequency expresses how often the customer uses services. If frequency is high, it means customer really loyalty.

Monetary performs how customer are potential through amount of money they spend for the service.

_Figure 4: RFM explanation_

<img src="https://github.com/carfirst125/portfolio/blob/main/Business%20Information%20System%20-%20Customer%20Orientation/figures/BIS_MLAI_RFM1.png" width="80%" height="80%" alt="RFM"/>

As shown in the Figure 4, the best customer are marked from (1) to (3) with low recency, high frequency. They are loyal customers, frequent using service and still active. By this way, it is straightforward to put the label name for customers in various location of RFM map.

It is not always the shape of RFM tower looking like in Figure 5. However, it might have the triangle shape. Beside the amount of customers are active in last 12 months, there are a large amount of customer which are no action more than 12 months. The responsibity of us that is efforting to pull up customers from bottom layers to top layer, changing shape of RFM tower as shown in the figure.

_Figure 5: RFM tower and Conversion Target_

<img src="https://github.com/carfirst125/portfolio/blob/main/Business%20Information%20System%20-%20Customer%20Orientation/figures/BIS-MLAI_RFM2.png" width="80%" height="80%" alt="RFM tower"/>

To achieve the conversion target, we analyze the RFM segmentation, choose the relevant customer group for action.
The RFM analysis supports increasing conversion based on own customer data. For expanding service to external customer (new customer), the Look-alike method is recommended. 

### Look-alike customer

After data analysis, the target customer characteristics are known. It assumes that customers, having characteristics look-alike the target customer, are potential having demand with the products and services.

<img src="https://github.com/carfirst125/portfolio/blob/main/Business%20Information%20System%20-%20Customer%20Orientation/figures/BIS_MLAI_lookalike.png" width="80%" height="80%" alt="lookalike"/>

With the same amount of customer reaching, reaching look-alike customers have higher conversion rate than less strict targeting group. 

Figure 6 illustrates Look-alike in deployment flow. The effective of Look-alike method depends dramatically on how well we understand our customers, how correct we target them.

_Figure 6: Look-alike in Flow and Ads Controller_

<img src="https://github.com/carfirst125/portfolio/blob/main/Business%20Information%20System%20-%20Customer%20Orientation/figures/BIS_MLAI_lookalike2.png" width="80%" height="80%" alt="lookalike in system"/>

The Ads Controller or Campaign controller implements small A/B test on several channel, then assesses effective of those. Based on the result of A/B and assessment, the Ads Controllers re-distribute budget for ads run among omni-channels. The feedback data from channels still be followed and budget adjustment by time session.

### Social Listening

Beside the operation of trying to analyzing customer data to understanding them, listening directly, what they are talking about us, is the fantastic way.

The Figure 7 shows what we can get from the social listening.

_Figure 7: Social Listening_

<img src="https://github.com/carfirst125/portfolio/blob/main/Business%20Information%20System%20-%20Customer%20Orientation/figures/BIS_MLAI_social_lis.png" width="80%" height="80%" alt="social listening"/>

Social listening records the direct voice of customer to us. It supports in increasing customer relationship, getting what customer need, and opening the big chance for product and service adjustment to adapt the customers’ demand. Relevant product and service is important, because customer even has the demand, they also choose the competitors’ product by its appropriation.

Beyond listening to customer helps us in managing potential risk. The early recognize in trend change, and customer response are advantageous for suitable adjustment.
Social listening is not only for our customer, but also competitors. We therefore can know which points that customer satisfy from competitors’ service and what they still complain.

Understand customer, adjust the product and service in the right way, which adapt customer need, is the great way to create foundation or increase sell.

### CLV (Customer Lifetime Value), Retention Rate and Churn

Below is the CLV formula:

<img src="https://github.com/carfirst125/portfolio/blob/main/Business%20Information%20System%20-%20Customer%20Orientation/figures/BIS_MLAI_CLVFomula.png" width="60%" height="60%" alt="CLV formula"/>


