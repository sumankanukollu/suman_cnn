# IMDB Movie Review (Sentiment Analysis - Neural Word Embeddings)

* Useful Blog source : https://gaurav4664.medium.com/how-to-speed-up-aws-lambda-deployment-on-serverless-framework-by-leveraging-lambda-layers-623f7c742af4

### AWS deployment end point

* https://l75gx6g2d9.execute-api.ap-south-1.amazonaws.com/dev/predict

### AWS Spacy + Torch Text + torch Layer:

* **<u>ARN</u>** : arn:aws:lambda:ap-south-1:936131757702:layer:gp-torchtext-spacy-pytorch-layer:2 

## Code:

* [Spacy TEXT vocab file](https://drive.google.com/file/d/1-8uwkwMq8KMW4yv78YrrN5_c3ZqmANz2/view?usp=sharing)
* CNN model notebook file: https://github.com/sumankanukollu/suman_cnn/blob/master/sessions/s9_sentimentAnalysis/trainedModel/4-Convolutional%20Sentiment%20Analysis.ipynb
* For AWS inference : https://github.com/sumankanukollu/suman_cnn/blob/master/sessions/s9_sentimentAnalysis/trainedModel/loadModelForDeployment_v3.ipynb

# Results:

##### Positive Review:

![positive](https://github.com/sumankanukollu/suman_cnn/blob/master/sessions/s9_sentimentAnalysis/snippets/s9_possitiveReview.JPG)



###### Negative Review:

![negative](https://github.com/sumankanukollu/suman_cnn/blob/master/sessions/s9_sentimentAnalysis/snippets/s9_negativeReview.JPG)