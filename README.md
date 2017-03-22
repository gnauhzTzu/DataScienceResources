# DataScienceResources
my notes and resources of data science and other fun!

********************************************
MISSING DATA

********************************************
TIPS IN PRACTICE

As far as I know, `fread()` in `data.table` has the best performance in importing (big) csv files. It also has a lot convenience features:

https://www.r-bloggers.com/efficiency-of-importing-large-csv-files-in-r/
https://github.com/Rdatatable/data.table/wiki/Convenience-features-of-fread

Certain algorithms in sklearn, XGBoost can only have numerical values as their predictor variables. Hence Label Encoding or One Hot Encoding becomes necessary. But in H2O say Distributed Random Forest, you can use categorical variables in input data frame, H2O will deal with it.

how to properly **cross-validate** when we have **imbalanced** data: 
 
 * Oversampling the minority class can result in over-fitting problems if we oversample before cross-validating
 * Undersampling could solve the class imbalance issue and increased the sensitivity of our models, but results could be poor because the more imbalanced the dataset the more samples will be discarded when undersampling, therefore throwing away potentially useful information. 
 * I found deep neural network models could deal with the imbalanced dataset better when having at least more than 30 features.
 * Perform feature selection before we go into cross-validation.
 * oversampling must be part of the cross-validation and not done before. 
    * Inside the cross-validation loop, get a sample out and do not use it for anything related to features selection, oversampling or model building.
    * Oversample your minority class, without the sample you already excluded.
    * Use the excluded sample for validation, and the oversampled minority class + the majority class, to create the model.
    * Repeat n times, where n is your number of samples (if doing leave one participant out cross-validation).

http://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation

********************************************
FEATURE ENGINEERING

* feature selection: https://people.eecs.berkeley.edu/~jordan/courses/294-fall09/lectures/feature/slides.pdf

* **my strategy**
    * extract features based on understanding of the data and goal
    * evaluate features by precision rate, coverage area and complexity
    * how to get and store the features
    * clean outliers/weighted sampling/data unbalance sampling
  * **single feature**: regularization, discretization, dummy coding
    * missing value imputation,logarithm transfer, exponential transfer, Box-Cox
  * **multiple features**: PCA, LDA
    * Filter: correlation with the dependent variable. Pearson’s Correlation, LDA, ANOVA, Chi-Square, information gain.
    * **Wrapper Methods**: iteration to generate subset of features to train model. then using AUC/MSE, BIC, AIC etc. to decide to add or remove features based on the previous model, such as forward/backward/recursive feature elimination.
        * full search, heuristic search, stochastic search(GA, SA).
        * usually computationally very expensive. 
    * **Embedded Methods**, implemented by algorithms that have their own built-in feature selection methods. LASSO(L1)/RIDGE(L2) regression, regularized trees, random multinomial logit, deep learning.
   * should be pay attention to the quality and weight of features by monitor the algorithms

********************************************
DATA PREPROCESSING

Interactive Python Regular Expression cheat sheet:
http://www.pyregex.com/

Exploratory Data Analysis in R.
http://sux13.github.io/DataScienceSpCourseNotes/4_EXDATA/Exploratory_Data_Analysis_Course_Notes.html
Numerical/graphical summaries, binary, nominal, ordinal, discrete, continuous.

Mean, weighted means, trimmed, geometric, harmonic means. Median, Variance, standard deviation,

This article has useful compare of the pros and cons of several methods dealing **imbalanced data**. Random Over-Sampling, Cluster-Based Over Sampling,Random Under-Sampling, MSMOTE, Algorithmic Ensemble Techniques, bagging, boosting tree etc.
handle imbalanced classification data: https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650724464&idx=1&sn=1f34358862bacfb4c7ea17c864d8c44d

Basic practical code and evaluation metrics when dealing with imbalanced data
https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/?utm_content=buffer929f7&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer

Setting AUC Threshold when use xgboost to deal with imbalanced dataset:
https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

**Sparse Matrix Storage Formats**: DOK, COO, CSR, CSC etc.
these format are all available in `scipy.sparse`
http://www.scipy-lectures.org/advanced/scipy_sparse/storage_schemes.html
http://flyxu.github.io/2016/05/30/2016-5-30/


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

****************
Languages

R basic tutorial https://www.coursera.org/learn/r-programming
My code for R basic, data preprocess, data wrangling:
R machine learning package library https://cran.r-project.org/web/views/MachineLearning.html


