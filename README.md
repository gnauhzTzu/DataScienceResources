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

********************************************
MACHINE LEARNING

********************************************
CLOUD PLATFORM

********************************************
COMPUTATIONAL COST & EFFICIENT


********************************************
STATISTICAL METHODS
