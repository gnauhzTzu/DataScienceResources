# DataScienceResources
my notes and resources of data science and other fun!

********************************************
MISSING DATA

********************************************
TIPS IN PRACTICE

a single node H20 server with 12GB RAM can parse and train around 24 GB csv file, while R in the same machine can't handle the file with same size. 

As far as I know, `fread()` in `data.table` has the best performance in importing (big) csv files. It also has a lot convenience features:

https://www.r-bloggers.com/efficiency-of-importing-large-csv-files-in-r/
https://github.com/Rdatatable/data.table/wiki/Convenience-features-of-fread

Most of the versions update (say 3.10.0.8 and 3.10.0.10) in h2o will break the existing model, which means, the model trained on old versions cannot be imported by new versions of h2o.
tips on h2o troubleshooting
https://support.h2o.ai/support/solutions/articles/17000012114-h2o-troubleshooting-tips

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

**Sparse Matrix**

https://projects.ncsu.edu/hpc/Courses/6sparse.html
http://math.ucla.edu/~dakuang/cse6040/lectures/6040_lecture8.pdf
- Advantages of the CSR format
    - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
    - efficient row slicing
    - fast matrix vector products
- Disadvantages of the CSR format
    - slow column slicing operations (consider CSC)
    - changes to the sparsity structure are expensive (consider LIL or DOK)

Storage efficient of COO,CSR,CSC,DIA,ELL,HYB etc.
http://www.cnblogs.com/xbinworld/p/4273506.html


when it's **not easy/expensive to know the real dimension of the feature vector**. For example, the **bag-of-word** representation in document classification problem requires scanning entire dataset to know how many words we have, i.e. the dimension of the feature vector. We can use [Feature hashing](https://en.wikipedia.org/wiki/Feature_hashing): 

http://amunategui.github.io/feature-hashing/
http://scikit-learn.org/stable/auto_examples/text/hashing_vs_dict_vectorizer.html#sphx-glr-auto-examples-text-hashing-vs-dict-vectorizer-py




Encoding methods of categorical variables: Ordinal, One-Hot, Binary, Helmert Contrast, Sum Contrast, Polynomial Contrast, Backward Difference Contrast, Hashing, BaseN: https://github.com/scikit-learn-contrib/categorical-encoding

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
    * **filter**: correlation with the dependent variable. Pearson’s Correlation, LDA, ANOVA, Chi-Square, information gain.
    * **wrapper methods**: iteration to generate subset of features to train model. then using AUC/MSE, BIC, AIC etc. to decide to add or remove features based on the previous model, such as forward/backward/recursive feature elimination.
        * full search, heuristic search, stochastic search(GA, SA).
        * usually computationally very expensive. 
    * **embedded methods**, implemented by algorithms that have their own built-in feature selection methods. LASSO(L1)/RIDGE(L2) regression, regularized trees, random multinomial logit, deep learning.
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
Text Mining

Extracted the top words of images using TF-IDF:
https://openvisconf.com/2016/#videos


********************************************
HADOOP DISTRIBUTED FILE SYSTEM (HDFS)
MapReduce, Pig, Hive, Hbase pseudocode and example https://www.ee.columbia.edu/~cylin/course/bigdata/EECS6893-BigDataAnalytics-Lecture2.pdf

Debug hadoop map reduce work locally on Eclipse before submitting the jar over to claster
You cannot debug your mappers and reducer in a distributed mode, you can only debug them in local mode
http://let-them-c.blogspot.com/2011/07/running-hadoop-locally-on-eclipse.html

Tutorial on how to debugging MapReduce Programs With MRUnit
http://blog.cloudera.com/blog/2009/07/debugging-mapreduce-programs-with-mrunit/


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

Fastest way to check if any NaN in data frame is `df.isnull().values.any()`
`df.isnull().values.sum()` is a bit slower, but show the number of NaNs.
http://stackoverflow.com/questions/29530232/python-pandas-check-if-any-value-is-nan-in-dataframe

Find same query, filter, sample, sort, group, slice, aggregate, group by, match, data format transfer methods in R and in pandas:  
http://pandas.pydata.org/pandas-docs/stable/comparison_with_r.html

Almost all R basic knowledge; https://www.coursera.org/learn/r-programming
My code for R basic, data preprocess, data wrangling:
R machine learning package library https://cran.r-project.org/web/views/MachineLearning.html
As far as I know, fread() in data.table is the most fastest way to import (big) csv files. It also has a lot convenience features
https://www.r-bloggers.com/efficiency-of-importing-large-csv-files-in-r/
https://github.com/Rdatatable/data.table/wiki/Convenience-features-of-fread

Chained assignment should be avoid in pandas data frame selecting and indexing:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

****************
Spark

- Use Spark Transformations and Actions wherever possible (Search DataFrame reference API)
- Never use collect() in production, instead use take(n) 
- cache() DataFrames that you reuse a lot

The ultimate place of mine to collect all the nuts and bolts of using Spark, including useful tricks and tips
https://www.gitbook.com/book/jaceklaskowski/mastering-apache-spark/details
https://github.com/jaceklaskowski/spark-workshop

- PyPy is astonishingly faster than CPython (about 5x) as its Just-in-Time compiler.
- PySpark can now run on PyPy to speed up the Python code execution.
- In Python Spark, your logic will be split between the Scala/JVM implementation of the core logic and the Python implementation of your logic and parts of the PySpark API
https://databricks.com/blog/2015/04/24/recent-performance-improvements-in-apache-spark-sql-python-dataframes-and-more.html

Compare the running time of Scala, Scala SQL, Python and Python SQL with load, join, map, reduce, sort
http://emptypipes.org/2015/01/17/python-vs-scala-vs-spark/

DataFrame Operations in PySpark. Creating DataFrame from RDD/CSV. Pandas vs PySpark DataFrame
https://www.analyticsvidhya.com/blog/2016/10/spark-dataframe-and-operations/

Incremental shortest distance algorithm for Spark:
https://blog.insightdatascience.com/computing-shortest-distances-incrementally-with-spark-1a280064a0b9


********************************************
RECOMMENDATION SYSTEM

Recommendations should:

- Simultaneously ease and encourage rather than replace social processes, should make it easy to participate while leaving in hooks for people to pursue more personal relationships if they wish
- Be for sets of people not just individuals...multi-person recommending is often important, for example, when two or more people want to choose a video to watch together
- Be from people not a black box machine or so-called "agent"
- Tell how much confidence to place in them, in other words they should include indications of how accurate they are

Overview of recommendation algorithms: 

**Content Based**

Main idea: Recommend items to customer **C** similar to previous items rated highly by **C**

Cons:
- Finding the appropriate features to create the item profile.
e.g., images, movies, music
- Overspecialization
 Never recommends items outside user’s content profile and people might have multiple interests
- Recommendations for new users

**Memory-based (user-based) Collaborative Filtering** ([Breese et al, UAI98](https://www.microsoft.com/en-us/research/publication/empirical-analysis-of-predictive-algorithms-for-collaborative-filtering/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fheckerman%2Fbhk98uai.pdf))

Main idea: First Consider user **C**, Find set **D** of other users whose ratings are “similar” to **C**’s ratings. Estimate user’s ratings based on ratings of users in **D**.

 - CF Based on vector similarity methods:
    - K-nearest neighbor
    - Pearson correlation coefficient 
    - Cosine distance 
    - Cosine with "inverse user frequency"
 - Evaluation:
    - Root-mean-square error (RMSE)
        - RMSE might penalize a method that does well for high ratings and badly for others, but in practice, we care only to predict high ratings.
    - Coverage: Number of items/users for which system can make predictions
    - Precision: Accuracy of predictions
    - Receiver operating characteristic (ROC): Tradeoff curve between false positives and false negatives

Pros: Works for any kind of item, No feature selection needed
Cons: 

- Cold Start: Need enough users in the system to find a match   
- The user/rating matrix is sparse
- So expensive to finding k most similar customers, O(|U|) sparse matrix U.
- New items/users?
- Popularity  bias: Tends to recommend popular items
- Can use clustering, partitioning as alternatives, but quality degrades


**Model-based (item-based) CF**

Main idea: For item **S**, find other similar items, estimate rating for item based on ratings for similar items. Can use same similarity metrics and
prediction functions as in user-based model

In practice, it has been observed that item-item often works better than user-user, because similarity between items is more static, the similarity between users can change if only a few ratings are changing (the overlap
between users profiles is small)

**Hybrid**

- Implement two or more different recommenders and combine predictions, perhaps using a linear model
- Add content-based methods to collaborative filtering
    - item profiles for new item problem
    - demographics to deal with new user problem

Key problem: matrix is sparse as most people have not rated most items


https://lagunita.stanford.edu/courses/course-v1:ComputerScience+MMDS+SelfPaced/info
http://web.stanford.edu/class/cs345a/handouts.html
http://infolab.stanford.edu/~ullman/mmds/ch9.pdf




*****************
basic text mining method
https://www.analyticsvidhya.com/blog/2015/12/kaggle-solution-cooking-text-mining-competition/
