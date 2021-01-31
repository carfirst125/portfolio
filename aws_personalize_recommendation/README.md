## AWS Personalize Recommendation (User Personalization, Personalized Ranking, SIMS)

AWS Personalize provides various recommendation methods for different questions. This project three recommendation method are discussed including **User Personalization**, **Personalized Ranking** and **SIMS**. Those are mentioned further as below.

AWS Personalize uses HRNN model to implement this application.

#### User Personalization

Input is USER_ID. The model returns list of ITEM_IDs which have highest probability in purchasing at the next transaction.

#### Personalized Ranking (Reranking)

Input is list of ITEM_IDs and USER_ID. The model computes and re-ranks ITEM_ID in list. So we could know which ITEM_ID is most potential to the customer.

#### SIMS

Input is ITEM_ID. The model returns list of ITEM_IDs which are highest correlation with the input ITEM_ID (no User behavior reference)

## General Block Diagram

![alt text](https://github.com/carfirst125/portfolio/blob/main/aws_personalize_recommendation/images/aws_personalize_recommendation_BlockDiagram.png?raw=true)

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

![alt text](https://lh3.googleusercontent.com/-v7F0d6rKVUk/WbIOqqTcdfI/AAAAAAAAA7E/exAPR6fqRoEJPHjHmGHp5tSwVLBS8butgCEwYBhgLKtQDAL1Ocqyrgsaxht0kDTeaghYHEalI5hMjnbpBlqoZu3JdBJ7RJDW5BcqwJSw5TxfKh7BimSoPaBuC4JEGgaMncJJU49TXx515GY28GgOkGl5z-sqtmZqwYxDWAFPrarqx-Ru3JLMVtmvpBPrphm8kJ10mBmP_94Z6wHdRT9qaVhO-NJ_WgpjaMmlhpaAHraMtiPDnJTAJLrsrKC6_APReMt3-NSgFdXBPRU0NOxJ-EncuR3A1oV_TMallXqtama_UF9V9hhmSiGOS8o4gzPcGSJV-AXcBmUC6EJ-Jemm5MuFzgU1dgYHDtgD3u7MFqJf4thVxtXcXp-d8V72mMHYhfgecN-kyLFkR0_8Yyas10vvLmkCkPPHnxvcjX526hmofDncehNRa0R7WqBhpWXtYN3NuU5_eFwJCfplwnHYgUlZqJ1iPRFmHweSwKCoFfFLSEG6wk3y17W8ncgzNF1NdiZ_D2Wgr-ouj21J_oc4zY-3C9YK_cdq-PppEcM5zmLikIRgAiLLLcqI-oJpFM4fB786NUP0JNmp-WHXV0av2JvYhpC54qc_1asMmK1HsdUZic-BsG161DohPKtpHcp_Uhu9UNX_mDERYRiGpL6M8GS2GcC2kMLHG2YAG/w140-h139-p/Nhan.png?raw=true)

| I am text to the left  | ![avatar](/https://lh3.googleusercontent.com/-v7F0d6rKVUk/WbIOqqTcdfI/AAAAAAAAA7E/exAPR6fqRoEJPHjHmGHp5tSwVLBS8butgCEwYBhgLKtQDAL1Ocqyrgsaxht0kDTeaghYHEalI5hMjnbpBlqoZu3JdBJ7RJDW5BcqwJSw5TxfKh7BimSoPaBuC4JEGgaMncJJU49TXx515GY28GgOkGl5z-sqtmZqwYxDWAFPrarqx-Ru3JLMVtmvpBPrphm8kJ10mBmP_94Z6wHdRT9qaVhO-NJ_WgpjaMmlhpaAHraMtiPDnJTAJLrsrKC6_APReMt3-NSgFdXBPRU0NOxJ-EncuR3A1oV_TMallXqtama_UF9V9hhmSiGOS8o4gzPcGSJV-AXcBmUC6EJ-Jemm5MuFzgU1dgYHDtgD3u7MFqJf4thVxtXcXp-d8V72mMHYhfgecN-kyLFkR0_8Yyas10vvLmkCkPPHnxvcjX526hmofDncehNRa0R7WqBhpWXtYN3NuU5_eFwJCfplwnHYgUlZqJ1iPRFmHweSwKCoFfFLSEG6wk3y17W8ncgzNF1NdiZ_D2Wgr-ouj21J_oc4zY-3C9YK_cdq-PppEcM5zmLikIRgAiLLLcqI-oJpFM4fB786NUP0JNmp-WHXV0av2JvYhpC54qc_1asMmK1HsdUZic-BsG161DohPKtpHcp_Uhu9UNX_mDERYRiGpL6M8GS2GcC2kMLHG2YAG/w140-h139-p/Nhan.png?raw=true) |

- - - - - 
Feeling Free to contact me if you have any question around.

    - Nhan Thanh Ngo
    - Email: ngothanhnhan125@gmail.com
    - Skype: ngothanhnhan125
    - Phone: (+84) 938005052

