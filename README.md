# DataScienceResources
my notes and resources of data science and other fun!

********************************************
MISSING DATA

********************************************
USEFULL TIPS IN PRACTICE

********************************************
FEATURE ENGINEERING

* feature selection: https://people.eecs.berkeley.edu/~jordan/courses/294-fall09/lectures/feature/slides.pdf

* <b>my strategy</b>
    * extract features based on understanding of the data and goal
    * evaluate features by precision rate, coverage area and complexity
    * how to get and store the features
    * clean outliers/weighted sampling/data unbalance sampling
  * <b>single feature</b>: regularization, discretization, dummy coding
    * missing value imputation,logarithm transfer, exponential transfer, Box-Cox
  * <b>multiple features</b>: PCA, LDA
    * Filter: correlation with the dependent variable. Pearson’s Correlation, LDA, ANOVA, Chi-Square, information gain.
    * <b>Wrapper Methods</b>: iteration to generate subset of features to train model. then using AUC/MSE, BIC, AIC etc. to decide to add or remove features based on the previous model, such as forward/backward/recursive feature elimination.
        * full search, heuristic search, stochastic search(GA, SA).
        * usually computationally very expensive. 
    * <b>Embedded Methods</b>, implemented by algorithms that have their own built-in feature selection methods. LASSO(L1)/RIDGE(L2) regression, regularized trees, random multinomial logit, deep learning.
   * should be pay attention to the quality and weight of features by monitor the algorithms

********************************************
DATA PREPROCESSING

This article has useful compare of the pros and cons of several methods dealing imbalanced data. Random Over-Sampling, Cluster-Based Over Sampling,Random Under-Sampling, MSMOTE, Algorithmic Ensemble Techniques, bagging, boosting tree etc.
handle imbalanced classification data: https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650724464&idx=1&sn=1f34358862bacfb4c7ea17c864d8c44d

Basic practical code and evaluation metrics when dealing with imbalanced data
https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/?utm_content=buffer929f7&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer

Setting AUC Threshold when use xgboost is inspiringly useful!
https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

********************************************
MACHINE LEARNING

********************************************
HADOOP DISTRIBUTED FILE SYSTEM (HDFS)
MapReduce, Pig, Hive, Hbase pseudocode and example https://www.ee.columbia.edu/~cylin/course/bigdata/EECS6893-BigDataAnalytics-Lecture2.pdf

********************************************
COMPUTATIONAL COST & EFFICIENT


********************************************
STATISTICAL METHODS

Comparison of parametric (Z-test) and non-parametric (chi-squared) methods : https://www.r-bloggers.com/comparison-of-two-proportions-parametric-z-test-and-non-parametric-chi-squared-methods/
Use Z-test with two assumptions: the probability of common success is approximate 0.5, and the number of samples is very high

********************************************
DATA VISUALIZATION

Plotly, an interactive visualization tool, it can be used with python, js, R spark, d3 etc. I like plotly's default color scale and it's strong interactive ability. But they really need to make their api documentation more readable and sometimes the version update of plotly will bring several version compatible troubles.  
https://plot.ly/python/apache-spark/
https://plot.ly/~morpheos/367/forecast-time-series-spark/

Using plotly, python, sqlite to explore the 3.9G csv from NYC open data. Using sqlite is a lightweight and good choose especially when dealing GB level data and when the data finally used for visualization is not that big. Can do all the summarize and aggregation in database. https://plot.ly/python/big-data-analytics-with-pandas-and-sqlite/


d3 tutorial (basic but useful) https://github.com/uwdata/d3-tutorials


Learning Perceptual Kernels for Visualization Design: https://github.com/uwdata/perceptual-kernels


