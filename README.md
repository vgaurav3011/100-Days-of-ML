# 100-Days-of-ML

The aim is to comprehensively revise essential statistics, machine learning algorithms, state of art models and learn to develop end-to-end products using machine learning and deep learning 

### Day 1 : Binary Classification
- Studied Classification and various approaches to achieve enhanced results 
- Implemented a basic network for Binary Classification using Tensorflow
- Learnt to compress Tensorflow Models to TfLite Model for Binary Classification on Dogs vs Cats Dataset
- Worked on developing an end-to-end Flutter Application which can perform Cats vs Dogs Classification with TfLite as backend
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%201/catsvsdogs.ipynb">Link</a> <br/>
<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%201/cat%20vs%20dog%20app%20flutter/cat.jpeg" width="250" height="700"></div>
 <div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%201/cat%20vs%20dog%20app%20flutter/dog.jpeg" width="250" height="700"></div>

### Day 2  : Regularization
- Worked on learning different techniques to handle overfitting of a model
- Studied L1, L2 regularization and dropout regularization from Andrew NG deeplearning.ai course
- Completed the assignment on regularization in Week 1 of the course
- Studied visual intuition of regularization: <a href="https://towardsdatascience.com/a-visual-intuition-for-regularization-in-deep-learning-fe904987abbb">Link</a><br/>
- Studied Tensorflow implementation of Dropout from Lei Mao's Blog: <a href="https://leimao.github.io/blog/Dropout-Explained/">Link</a>
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%202/Regularization.ipynb">Link</a><br/>

### Day 3 : Generative Adversarial Networks
- Studied basics of Generative Adversarial Networks (GANs) and their applications
- Worked on implementing a simple DCGAN to reconstruct MNIST images
- Will be working on studying GANs in depth
- Reference: <a href="https://blog.floydhub.com/gans-story-so-far/">Link</a>
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%203/DCGAN.ipynb">Link</a> <br/>

<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/images/dcgan.gif"></div><br/>

### Day 4  : Neural Style Transfer
- Studied the concepts of convolution neural networks and their working process
- Implemented a simple style transfer network using pretrained weights and VGG-16 
- Studied the concept of loss function and hyperparameter tuning from Andrew NG course
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Neural_Style_Transfer.ipynb">Link</a><br/>
<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/images/1.jpg" width="200" height="200"></div>
<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/images/2.jpg" width="200" height="200"></div>
<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/images/download.png" width="200" height="200"></div><br/>

### Day 5  : Supervised Learning: Regression
- Studied the concepts of Regression and Classification from Introduction to Machine Learning by Udacity (GaTech)
- Implemented feature engineering, preprocessing and scaling on Black Friday Data in dataset-1 folder 
- Studied ElasticNet, Lasso, Ridge, Linear, AdaBoost, Random Forest Regression, Gradient Boosting and applied the same on the dataset
- Studied the basics of XGBoosting and its applications
- Implemented a Black Friday Sales Web App for prediction using the models with Flask Framework and Pickle in backend
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%205/Black-Friday.ipynb">Link</a><br/>
<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/images/output_flask.gif"></div><br/>

### Day 6  : Supervised Learning: Linear Regression and Gradient Descent
- Studied the concept of Linear Regression in depth from multiple blogs
- Read the Stanford UFDL notes on Gradient Descent and Stochastic Gradient Descent at <a href="http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/">Link</a> <br/>
- Completed implementation of Gradient Descent with Linear Regression from scratch using only mathematics through Numpy in Python.
- Studied the background mathematics of Gradient Descent and Regression
- Model:<a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%206/linear_model.py">Link</a><br/>

### Day 7  : Optimization Algorithms and Hyperparameter Tuning
- Studied and implemented the concept of Mini Batch Gradient Descent
- Implemented the concept of gradient checking and how it boosts training time
- Studied the use of exponential weighted averages and use of momentum to speed up training processes
- Studied the various alternative optimization algorithms and learnt to implement ADAM and RMSProp from scratch
- Finished the deep learning course 2 by Andrew NG Stanford and studied the theoretical details of the concepts.
- Models Folder: <a href="https://github.com/vgaurav3011/100-Days-of-ML/tree/master/Day%207">Link</a><br/>
<div style="text-align:center;"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/images/certy-1.png" width="700" height="400"></div><br/>

### Day 8 (31-03-20) : Structuring Machine Learning Projects and Basics of Tensorflow
- Studied the fine tuning of networks
- Studied train/dev/test set distribution techniques
- Studied sampling techniques and finished the third course of Andrew NG
- Read use of dockers for deployment of models
- Studied the blog on real time deployment of models at FloydHub: <a href="https://blog.floydhub.com/structuring-and-planning-your-machine-learning-project/">Link</a> <br/>
<img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/images/certy-2.png" width="700" height="400"><br/> <br/>

### Day 9 (01-04-20) : UNet for Biomedical Image Segmentation
- Studied the UNet Architecture from the state of art paper: <a href="https://arxiv.org/pdf/1505.04597.pdf">Link</a>
- Studied the concepts of upsampling, encoder-decoder architecture integration into UNet
- Read and made notes of the blog by Ayyuce Kizrak: <a href="https://heartbeat.fritz.ai/deep-learning-for-image-segmentation-u-net-architecture-ff17f6e4c1cf">Link</a>
- Designed a custom data generator for Nuclei Segmentation from Biomedical Images
- Dataset Link: <a href="https://www.kaggle.com/c/data-science-bowl-2018/rules">Link</a>
- The data was part of the Data Science Bowl 2018
- Implemented a UNet architecture after analysing the paper
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%209/model.ipynb">Link</a><br/>

<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%209/input.png"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%209/output.png"></div><br/></div><br/>

### Day 10 (02-04-20) : Basics of RNN and LSTM
- Studied the idea of Gated Recurrent Units (GRU)
- Studied RNN and LSTM
- Started the Sequence Models Course by Andrew NG<br/>
- Worked on implementing Music Generation by LSTM<br/>
- Watched MIT Deep Learning Lab Video on LSTM<br/>
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Music_Generator.ipynb">Link</a><br/>

### Day 11 (03-04-20) : Autoencoders
- Studied the concept of Encoder and Decoder architectures
- Studied the basics of Autoencoders
- Implemented a Simple Autoencoder on MNIST data using Tensorflow
- Studied the official Keras blog on autoencoders <a href="https://blog.keras.io/building-autoencoders-in-keras.html">Link</a>
- Studied about Sparse Autoencoders: <a href="https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html">Link</a>
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2011/VAE.ipynb">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2011/download.png" width="800" height="500"></div><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2011/download%20(1).png" width="500" height="500"></div><br/>

### Day 12 (04-04-20) : Variational Autoencoder
- Studied the idea of introducing variations in the autoencoding architecture
- Studied the concepts of VAE: <a href="https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/">Link</a>
- Finished the Week 3 of the CNN Course by Andrew NG
- Model:<a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2012/VAE_Fashion_MNIST.ipynb">Link</a><br/>

### Day 13 (05-04-20) : Basics of PyTorch
- Studied the basics of implementing neural networks in PyTorch
- Studied the major differences between PyTorch and Tensorflow
- Implemented a neural network for MNIST using PyTorch
- Implemented a neural network for Fashion-MNIST using PyTorch 
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2013/Fashion-MNIST.ipynb">Link</a><br/>
<div style="text-align=center"><img src="https://pbs.twimg.com/media/EU_ol5iUEAIzK-q?format=png&name=360x360"><br/><img src="https://pbs.twimg.com/media/EU_ol6FUcAAzmCW?format=png&name=small"></div>
<div style="text-align=center"><img src="https://pbs.twimg.com/media/EU_ol6EUMAEMf0_?format=png&name=small"></div>
<br/>

### Day 14 (06-04-20) : Building Models in PyTorch
- Finished the course on Building PyTorch Deep Learning Models by Janana Ravi on Pluralsight
- Studied the architectural framework of PyTorch
- Implemented AutoGrad, Text Classification and Image Classification Models using PyTorch
- Models: <a href="https://github.com/vgaurav3011/100-Days-of-ML/tree/master/Day%2014">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2013/certy.png"></div><br/>

### Day 15 (07-04-20) : Open Computer Vision: Image Kernels and Thresholding
- Studied the basics of image kernels: <a href="https://setosa.io/ev/image-kernels/">Link</a>
- Studied different types of image thresholding techniques
- Studied the watershed segmentation algorithm and its implementation in Open CV
-Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2015/preprocessing.py">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2015/adaptive_thresholding.png"></div><br/>

### Day 16 (08-04-20) : Open Computer Vision: Image Filters and Edge Detection
- Studied the basics of gradients and edge detection: <a href="https://medium.com/themlblog/image-operations-gradients-and-edge-detection-c4279049a3ad/">Link</a>
- Studied the mathematical aspects of image filters especially Sobel Filters
- Studied the canny edge detection algorithm and its implementation in Open CV
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2016/static/watershed.py">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2016/static/Figure_1.png"></div><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2016/static/fig1.png"></div><br/>

### Day 17 (09-04-20) : Linear Regression End-to-End Application
- Studied the interview based reasoning questions on Linear Regression
- Implemented a regression model on the car price prediction
- Constructed an end-to-end Flask Application for predicting Car Prices from the data
- To run the app we have the following commands
```console
$ sudo docker-compose build
```
```console
$ sudo docker-compose up
```
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/images/output_flask1.gif" width="800" height="500"></div><br/>

### Day 18 (10-04-20) : Natural Languages using BERT
- Studied the basics of Logistic Regression
- Implemented a sentiment analysis classifier using BERT
- Worked on analysing DistilBERT
- Studied BioBERT
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2018/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb">Link</a> <br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2018/bert.png"></div> <br/>

### Day 19 (11-04-20) : Basic Data Visualisations
- Learned implementation of histogram, bar grpahs, stacked graphs and pie charts
- Implemented the probability distributions in seaborn and matplotlib
- Completed the assignment exercise for visualisation given by the IITB course 
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2019/Basic_Visualizations.ipynb">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2019/plot.png"></div><br/>

### Day 20 (12-04-20) : Mathematics for Machine Learning
- Started my own blog on Machine Learning and Programming
- Revisited all the necessary mathematical concepts
- Revised Linear Algebra
- Revised Statistics
- Revised Probability Distributions
- Revised Multivariate Calculus
- Link to the Blog: <a href="https://vgaurav3011.github.io/2020/04/15/first-maths-in-ml.html">Link</a> </br>
<div style="text-align=center"><img src="https://www.whatissixsigma.net/wp-content/uploads/2014/03/Basic-Statistics-Descriptive-Statistics.png"></div><br/>

### Day 21 (13-04-20) : Outlier Detection
- Learned about the concepts of Outliers
- Revisited all the necessary mathematical concepts behind outlier detection
- Implemented a small program regarding outlier detection with visualisation
- Studying more in-depth reasoning about outliers 
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2021/outlier_pyod.py">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2021/Figure_1.png"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2021/Figure_2.png"></div><br/>

### Day 22 (14-04-20) : Concept of Probability and Probability Distributions
- Started 7 Days of Statistics Challenge
- Learned about probability concepts in-depth
- Studied all probability distributions
- Implemented basic programs on normal distribution, poisson distribution and binomal distribution 
- Models: <a href="https://github.com/vgaurav3011/100-Days-of-ML/tree/master/Day%2022">Link</a> <br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2022/prob.jpeg"></div><br/>

### Day 23 (15-04-20) : Descriptive Statistics
- Completed the course on Descriptive Statistics for Data Scientists by Dr Abhinanda Sarkar
- Finished Visualisation techniques
- Explored the dataset on predicting the sales of cardiofitness equipments as a real time case study
- Visualised and studied feature extraction in machine learning 
- Models: <a href="https://github.com/vgaurav3011/100-Days-of-ML/tree/master/Day%2023"></a><br/>

<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2023/index.png"></div><br/>

### Day 24 (16-04-20) : Anomaly Detection
- Studied the idea of anomaly detection
- Studied SMOTE to overcome overfitting
- Studied the problem of imbalanced classes
- Implemented outlier detection on Big Market Dataset
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2024/Anomaly_Detection.ipynb">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2024/final.png"></div><br/>

### Day 25 (17-04-20) : Credit Card Fraud Detection-I
- Implemented machine learning models to handle imbalanced class problems 
- Implemented a basic network to detect outliers in credit card fraud detection dataset
- Studied PyOD Library by CMU
- Implemented PyOD on Credit Card Fraud Detection
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2025/fraud_credit.py">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2025/Figure_3.png"></div><br/>

### Day 26 (18-04-20) : Credit Card Fraud Detection-II
- Implemented Autoencoder for credit card fraud detection
- Studied the difference in implementation of autoencoder for imbalanced classes 
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2026/CreditCardFraud.ipynb">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2026/auto.png"></div><br/>

### Day 27 (19-04-20) : Complete PyOD Implementation
- Worked on outlier detection on credit card fraud detection dataset
- Studied outlier and anomaly detection
- Worked on maths behind anomaly detection
- Implemented five algorithms under PyOD
- Completed a blog on outlier detection: <a href="https://vgaurav3011.github.io/2020/04/23/outlier-detection.html">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2027/blog.gif"></div><br/>

### Day 28 (20-04-20) : Neural Network From Scratch
- Learned to implement a neural network from scratch
- Studied the mathematical aspects of forward and backward propagation
- Tried to implement a neural network from scratch for gender classification 
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2028/neural_network.py">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2028/loss.png"></div><br/>

### Day 29 (21-04-20) : Decision Trees
- Studied the concept of decision tree classification
- Studied the CART, C4.5 and ID3 algorithms
- Solved Mathematics behind Gini Index, Information Gain, and Entropy
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2029/Decision_Tree_From_Scratch.ipynb">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2029/decision.jpeg"></div><br/>

### Day 30 (22-04-20) : Basic Classification Revised
- Studied the concept of binary classification
- Worked with revising multiple classifiers
- Studied the concept of decision boundary
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2030/ML_ASST_II.ipynb">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2030/final.png"></div><br/>

### Day 31 (23-04-20) : Hypothesis Testing
- Worked on Hypothesis Testing in Statistics
- Studied the concepts of ANOVA, Chi-Square Distribution and Test of Proportions and Variance
- Understood One-Tailed and Two-Tailed Testing
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/tree/master/Day%2031">Link</a><br/>
<div style="text-align=center"><img src="https://miro.medium.com/max/862/1*VXxdieFiYCgR6v7nUaq01g.jpeg"></div><br/>

### Day 32 (24-04-20) : Statistics Final Project
- Finished the course on Statistics for Machine Learning by Dr. Abhinanda Sarkar of Great Learning Academy
- Completed the capstone project on predicting medical insurance claims for smokers and non-smokers
- Studied the in-depth implementation of statistics on real life use case
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2032/Statistics_Complete.ipynb">Link</a><br/>
<div style="text-align=center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2032/index.png"></div><br/>

### Day 33 (25-04-20) : Bagging and Random Forest Algorithm
- Studied the concept of bagging
- Started the course of Ensemble Learning Models on AnalyticsVidhya
- Learned the concept of Random Forest: <a href="https://victorzhou.com/blog/intro-to-random-forests/">Link</a>
- Revised the concept of Gini Impurity: <a href="https://victorzhou.com/blog/gini-impurity/">Link</a>
- Model: <a href="">Link</a><br/>
<div style="text-align=center;"><img src=""></div>
<br/>

### Day 34 (26-04-20) : XGBoost
- Studied the concept of boosting
- Implemented XGBoost Classifier using Python
- Studied Feature Importance
- Finished the course on Ensemble Learning on AnalyticsVidhya
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2034/XGBoost_Demo.ipynb">Link</a><br/>
<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2034/xg.png"></div><br/>

### Day 35 (27-04-20) : TedX Dataset Analysis
- Studied the Udacity Data Scientist Nanodegree Program
- Worked on assignment of TedX Talk Dataset using basic pandas
- Data Exploration and insights on analysis
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2035/tutorial.ipynb">Link</a><br/>
 
<div style="text-align-center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2035/tedx.jpg" height="500" width="500"></div><br/>

### Day 36 (28-04-20) : Assignment-1 Udacity Data Scientist Program

- Studied the basics of Crisp DM Approach
- Studied Software Engineering
- Read about code modularity
- Finished working on Credit Card Fraud Detection with Crisp DM Approach
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2036/Model_Final.ipynb">Link</a><br/>
- Blog on Medium: <a href="https://medium.com/@vgaurav3011/credit-card-fraud-detection-you-have-to-be-odd-to-be-number-1-e158ceaf62f2">Link</a>

<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2036/credit-card.jpg"></div><br/>


### Day 37 (29-04-20) : Support Vector Machines Intutive Understanding

- Studied the basics of decision boundaries
- Studied concept of Maximal Margin Classifier
- Studied the concepts of differentiating between SVM and Logistic Regression
- Studied Hyperparameter Tuning of SVM with C and gamma values
- Studied the influence of outliers and feature scaling on SVM
- Implemented a basic SVM model on Iris Data
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2037/SVM_Concepts.ipynb">Link</a><br/>

<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2037/large_margin.jpeg"></div><br/>

### Day 38 (30-04-20) : Support Vector Machines Regression

- Studied the concept of fitting maximum points on the decision boundary
- Studied concept of decision functions
- Studied convex constraint optimizations
- Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2038/SVM_Regression.ipynb">Link</a><br/>

<div style="text-align:center"><img src="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2038/sv.png"></div><br/>

### Day 39 (01-05-20) : SQL for Data Science

- Studied the concepts of entity relationships and introduction to databases
- Started the course by UC Davis for SQL for Data Science Specialization
- Studied the differences between NoSQL and SQL
- Introduction to Data Modelling

### Day 40 (02-05-20) : SQL for Data Science

- Studied the idea of data pipelines
- Studied the basics of filtering datasets using SQL
- Studied Aggregation using MIN(), MAX(), SUM() and AVG()
- Continued with the Week 1 of SQL for Data Science Specialization by UC Davis

### Day 41 (03-05-20) : SQL for Data Science

- Studied the fundamentals of Data Modelling and Entity Relationship Diagrams
- Draw Entity Relationship Diagrams for multiple databases to practice
- Studied the ChinookDatabase ER Diagram
- Started learning queries on databases

### Day 42 (04-05-20) : SQL for Data Science

- Studied the idea of wildcards in SQL using %
- Intuitively understood math operations in SQL
- Completed Week 1 of SQL for Data Science by UC Davis and started Week 2

### Day 43 (05-05-20) : SQL for Data Science

- Studied the idea behind group-by
- Identified similarities and differences between group by in pandas and SQL
- Worked on implementing Group By command on Chinook Database
- Completed Week 2 of the SQL course by UC Davis

### Day 44 (06-05-20) : SQL for Data Science

- Studied the idea of subqueries
- Learned to evaluate nested queries
- Started writing complex queries in SQL
- Started Week 3 of SQL course by UC Davis

### Day 45 (07-05-20) : SQL for Data Science

- Studied the fundamentals of Data Modelling and Entity Relationship Diagrams
- Intuitively understood the concept of merging datasets
- Learned different varieties of JOINS in SQL
- Finished Week 3 of SQL Course by UC Davis

### Day 46 (08-05-20) : SQL for Data Science

- Finished the course on SQL for Data Science by UC Davis
- Practiced SQL problems on HackerRank

### Day 47 (09-05-20) : SQL for Data Science

- Earned a gold star for SQL on HackerRank
- Started practicing a few selected problems on LeetCode
- Learned about Dense_Rank() and Substr()

### Day 48 (10-05-20) : SQL for Data Science

- Completed the Easy Set of SQL Practice Problems on LeetCode
- Learned about CONCAT, DATETIME, and a few advanced SQL commands
- Practiced a few set of interview questions on SQL

### Day 49 (11-05-20) : Graph Databases for Data Science

- Studied the fundamentals of Graph DB
- Started the Neo4js course to understand Graph DB
- Studied the architecture of working for GraphDBs

### Day 50 (12-05-2020): Reinforcement Learning

- Studied the basic concepts of RL
- Started watching DeepLizard videos on RL
- Finished a Frozen Lake Game using open ai gym library based on RL
- Stepwise Frozen Lake Beginning to Goal using Dynamic Programming

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2050/main.py"> Link </a>

Output: 
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Up)
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Up)
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Up)
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Up)
SFFF
FHFH
FFFH
HFFG
  (Up)
SFFF
FHFH
FFFH
HFFG
  (Down)
SFFF
FHFH
FFFH
HFFG
  (Right)
SFFF
FHFH
FFFH
HFFG
  (Right)
SFFF
FHFH
FFFH
HFFG
  (Right)
SFFF
FHFH
FFFH
HFFG
  (Down)
SFFF
FHFH
FFFH
HFFG
  (Down)
SFFF
FHFH
FFFH
HFFG
  (Down)
SFFF
FHFH
FFFH
HFFG
  (Down)
SFFF
FHFH
FFFH
HFFG
  (Right)
SFFF
FHFH
FFFH
HFFG
  (Down)
SFFF
FHFH
FFFH
HFFG
  (Left)
SFFF
FHFH
FFFH
HFFG
  (Down)
SFFF
FHFH
FFFH
HFFG
  (Down)
SFFF
FHFH
FFFH
HFFG
You reached the goal!!

### Day 51:  Q-Learning Reinforcement Learning

- Visualised the Q-learning game using Open AI Gym
- Studied reward state, and process to optimize the steps towards the reward state.

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2052/Message_Passing.ipynb">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2051/final.gif)


### Day 52: Pageranking using RL

- Learned about visualising graphs and social connections
- Created a simple page ranking algorithm using DP and RL

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2052/Message_Passing.ipynb">Link</a>

![alt-text](data:image/png;base64,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%0A)

### Day 53: Visualising Convolutions

- Finished the DeepLizard RL series
- Started intuition of convolutions pretty well.
- Finished up convolution videos on Deep Learning Udacity
- Visualized convolution and maxpooling on images

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2053/conv_layer.gif)

### Day 54:  Autoencoders from Scratch

- Finished working on autoencoders from scratch
- Humble Coding Experience doing manual mathematics on paper too.

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2054/Autoencoders.ipynb">Link</a>


![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2054/autoencoder.png)

### Day 55: Recurrent Neural Networks

- Continued with Udacity Deep Learning Course
- Finished working on RNN and its concepts
- Finished the TV Script Generation Project

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2055/dlnd_tv_script_generation.ipynb">Link</a>

### Day 56: LSTM

- Read about shortcomings of RNN
- Finished working on LSTM
- Wrapped up final project on TV Script Generation using LSTMs

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2056/dlnd_tv_script_generation.ipynb">Link</a>

### Day 57: Revision- Image Augmentation Techniques

- A quick revision capsule of the data augmentation concepts

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2057/Image_Classification_Part_1_Image_Augmentation.ipynb">Link</a>

### Day 58: K-Nearest Neighbors on CIFAR-10

- With the udacity deep learning course, also started revising CS231n notes for more clarity
- Learned about different nearest neighbor approaches

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2058/Nearest_Neighbours_Classifier.ipynb">Link</a>

### Day 59: Principal Component Analysis

- Understanding first and second principal components
- Understanding its uses on image datasets
- Visualized the CIFAR-10 using PCA

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2059/K_Fold_Cross_Validation_and_Dimensionality_Reduction.ipynb">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2059/PCA.png)

### Day 60: Optimizer Visualizations

- SGD Optimization study
- Regression Line Fitting

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2060/optimizers_visualisation.py">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2060/gd.gif)

### Day 61: Visualizing Gradient Descent

- Gradient Descent revision capsule
- Visualized GD to understand better
- More intuitive understanding of GD

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2061/Derivation_GD_Terrain_3D_Plotting.ipynb">Link</a>

![alt-text](data:image/png;base64,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%0A)

### Day 62: Customer Churn Prediction

- Worked on churn prediction problem
- Kaggle Notebook designing for EDA Churn
- Customer Churn Analysis

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2062/model.ipynb">Link</a>

### Day 63: Complete Statistics Revision Part-1

- Watched Krish Naik Playlist on Statistics for Data Science
- Made Handwritten Notes for the entire course

Notes: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2063/Statistics%20by%20Krish%20Naik%20Notes.pdf">Link</a>

### Day 64: Complete Statistics Revision Part-2

- Watched Krish Naik Playlist on Statistics for Data Science
- Made Handwritten Notes for the entire course

Notes: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2064/measures%20of%20tendency.pdf">Link</a>

### Day 65: Learning Matplotlib Animations

- Learned about fractals
- Tried to code fractals in Python
- Matplotlib visualization of fractals

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2065/fractal.py">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2065/mandelbrot.gif)

### Day 66: Stock Price Scraping 

- Web Scraping of stocks from Yahoo Website
- Finished video by Sentdex on stock visualization
- Tried my own hand on it

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2066/stock.py">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2066/stonks.gif)

### Day 67: Time Series Forecasting 

- Started learning Time Series related problems
- Watched vidoes by Srivatsan Srinivasan

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2067/Time_Series_Forecasting_Introduction.ipynb">Link</a>

### Day 68: Moving Averages in Time Series

- Started Moving Averages working in time series
- Concept of windows 
- Seasonality in series

Model: <a href="https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2068/index.png">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2068/index.png)

### Day 69: Time Series Decomposition

- Decomposition of time series analysis
- Seasonality Studies
- Trends and Prediction

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2069/Time_Series_Decomposition.ipynb">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2069/time_decomp.png)

### Day 70: Modelling Time Series Functions

- Concept of rolling in time series
- Lag Concept
- Studied relation between lag and rolling over windows

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2070/Time_Series_Modelling_%26_Functions.ipynb">Link</a>

### Day 71: Holt Linearity in Time Series

- Different concepts of time series 
- Holt Linearity Concepts
- Autoregressive Integrated Moving Average

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2071/Practice_TimeSeries.ipynb">Link</a>


![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2070/time_functions.png)

### Day 72: ARIMA

- Time Series ARIMA implementation
- Regressive Average Concepts

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2072/ARIMA-II.ipynb">Link</a>

### Day 73: Analyticsvidhya Janatahack

- Worked on Machine Learning Problem of Pesticide Prediction
- Finished baseline Model for same at rank 22 public leaderboard

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2073/Model-1.py">Link</a>

### Day 74: Machine Learning Pesticide Prediction

- Finished 128 in Jantahack, however worked hard to make it to Rank 15 on Private Leaderboard
- Final model EDA and analysis

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2074/Model_2.ipynb">Link</a>

### Day 75: Final Submission Agriculture Prediction

- Finished EDA
- PrvLB 22 Model for Agriculture Prediction

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2075/correlation.png)

Model: <a href="https://github.com/vgaurav3011/Statistics-for-Machine-Learning/blob/master/notebooks/004-Visualization-Seaborn.ipynb">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2075/correlation.png)

### Day 76: Sorting Visualization

- Finished working on Matplotlib Tutorial
- Wrote an interesting python code for animated sorting visualizations

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2076/sort.py">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2076/merge.gif)

### Day 77: AIEngineering Time Series

- Finished the AI Engineering Playlist on Time Series.
- Made notes for each video
- Will be revising it again.
- There are multiple notebooks, check the repository for them.

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2077/final_AITime.png)

### Day 78: Big Market Sales Analysis-1

- Started working on making applied ML notebooks on Analyticsvidhya
- Finished a baseline regression model for Big Market Sales data
- Learned feature aggregation

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2078/005-Regression.ipynb">Link</a>


![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2078/images/Item%20Outlet%20Sales/Outlet_Type.png)

### Day 79: Big Market Sales Analysis-2

- Worked on Exploratory Data Analysis for Big Market Sales Data
- Created a notebook with detailed explanations for each characteristic with visuals.

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2079/Exploratory-Data-Analysis.ipynb">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2078/images/Item%20MRP/Visibility.png)

### Day 80: Big Market Sales Analysis-3

- Final Insights to the model
- Simple Regression techniques experimented
- Feature Engineering on the data

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2080/Model-3.ipynb">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2078/images/Item%20Type/Weight.png)

### Day 81: Loan Prediction Analysis-1

- Started the basic classification problem of loan prediction on Analyticsvidhya
- Finished a baseline model for the same

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2081/Model.ipynb">Link</a>

![alt-text](https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2081/feature-importance.png?raw=true)

### Day 82: Loan Prediction Analysis-2

- Finished EDA on Loan Prediction data.
- Crafted a well explained notebook for the same.

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2082/Exploratory-Data-Analysis.ipynb">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2082/final.png)

### Day 83: Loan Prediction Analysis-3

- Finished a simple model well explained with basic classification
- Crafted a well explained notebook for the same.

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2083/Model-Final.ipynb">Link</a>

### Day 84: Bike Prediction Analysis-1

- Started working on neural network for classification.
- Made a detailed notebook for bike price prediction.
- Made a streamlit app for EDA

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2084/model.ipynb">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2084/dashboard.gif)

### Day 85: Bike Prediction Analysis-2

- Deployed model as a streamlit app for prediction
- Completed End-to-End Machine Learning Project
- Detailed repository below

Model: <a href="https://github.com/vgaurav3011/Bike-Sharing-Demand-Analysis">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2084/visual.gif)

### Day 86: Car Value Prediction-1

- Started working on categorical data ideas.
- Learning end to end deployment for the same 
- Finished an app for deployment.

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2086/assets/dashboard.gif)

### Day 87: Car Prediction-2

- Final Modelling of Car Value Data
- Final Deployment as Heroku App
- Finished end to end machine learning project with streamlit and Heroku.

Heroku Web App Link: <a href="https://aqueous-falls-45593.herokuapp.com/">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2086/assets/car_eval.gif)

### Day 88: Cross Selling Prediction-1

- Participated in the Janatahack Cross Sell 
- Baseline Model came at rank 81
- Made a complete evaluative model to predict cross selling prices

Model Link: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2088/Baseline_Model.ipynb">Link</a>

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2089/target.png)


### Day 89: Cross Selling Prediction-2

- Finished at Public LB rank 42
- Exploratory Analysis Completed
- Well explained notebook made with love in Python

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2089/Exploratory_Data_Analysis.ipynb">Link</a>


![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2089/premium.png)

### Day 90: Cross-Sell Prediction-3

- Final Model for Janatahack
- Feature Engineering
- Private Leaderboard Rank-10

Models: <a href="https://github.com/vgaurav3011/100-Days-of-ML/tree/master/Day%2090">Link</a>

### Day 91: Bank Upsell Prediction-1

- Participated in Hacklive Jantahack Analyticsvidhya
- Identified major parameters and feature engineering
- Worked on EDA and baseline

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2091/Initial.ipynb">Link</a>

### Day 92: Bank Upsell Prediction-2

- Finished at LB 42 again
- Final Modelling and feature engineering

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/tree/master/Day%2092">Link</a>

### Day 93: Youtube Statistical Analysis-1

- Prediction of number of likes on videos given statistics
- Worked on the regression concepts
- Finished a baseline model at PB Rank 7

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2093/Model.ipynb">Link</a>

### Day 94: Youtube Statistical Analysis-2

- Finished at PB Rank 10, a major improvement in ranks
- Worked on feature engineering
- Tuning of Model using LGBM could have been better

Model: <a href="https://github.com/vgaurav3011/100-Days-of-ML/blob/master/Day%2094/Final_Model_HackLive2.ipynb">Link</a>

### Day 96: R-Programming for Statistics and Data Science

- Finished the 6.5 hours course on Udemy on R-Programming
- Learned Hypothesis Testing in R and basic linear regression techniques in R
- Made notes on Data Visualisation strategies in R and learned ggplot2
- Learned data cleaning using dplyr, tidyverse and tidyr
- Finished reading the O-Reilly book on R-programming

![alt-text](https://raw.githubusercontent.com/vgaurav3011/100-Days-of-ML/master/Day%2096/UC-f14f6177-1ab4-422a-a74b-e9225add08e9.jpg)
