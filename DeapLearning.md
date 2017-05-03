Main ideas of unsupervised feature learning and deep learning

* Artifical neural network
* Convolutional neural network
* Recurrent neural network

**Autoencoder**: 
 * Basically just a multilayer perceptron (MLP)
 * Popular for pre-training a network on unlabeled data
 * Output size is equal to input size

**Deep Autoencoder**:
- Reconstruct image from learned low dimensional code
- Weights are tied
- Learned features are often useful for classification
- Can add noise to input image to prevent overfitting

From **MLP** to **CNN**
- So far no notion of neighborhood
- Invariant to permutation of input
- **Convolutional neural networks** preserve neighborhood

CNN Advantages
- neighborhood preserved
- translation invariant
- tied weights

(Deep Neural Networks) DNNs are hard to train
- backpropagation – gradient descent
- many local minima
- prone to overfitting
- many parameters to tune
- SLOW

Unsupervised feature learners:
– RBMs
– Auto-encoder variants
– Sparse coding variants

Dropout
- Helps with overfitting
- Typically used with random initialization
- Training is slower than without dropout

Deep Learning for Sequences (as MLPs and CNNs have fixed input size)
Sequences Example: Complete a sentence
–…
–are …
–How are …

**Tips in Parameter Tuning**:

1. Number of Layers / Size of Layers
- If data is unlimited larger, then deeper should be better
- Larger networks can overfit more easily
- Take computational cost into account

2. Learning Rate
- One of the most important parameters
- If network diverges most probably learning rate is too large
- Smaller works better
- Can slowly decay over time
- Can have one learning rate per layer

3. Momentum
(http://proceedings.mlr.press/v28/sutskever13.pdf)
- Helps to escape local minima
- Crucial to achieve high performance

4. Convergence
Monitor validation error
- Stop when it doesn’t improve within n iterations
- If learning rate decays you might want to adjust number of iterations

5. Initialization of W
- Need randomization to break symmetry
- Bad initializations are untrainable
- Most heuristics depend on the number of input (and output) units
- Sometimes W is rescaled during training
–Weight decay (L2 regularization)
–Normalization

6. Data Preprocess
(http://deeplearning.stanford.edu/wiki/index.php/Data_Preprocessing)

7. Non-Linear Activation Function
- Sigmoid
    - Traditional choice
- Tanh
    - Symmetric around the origin
    - Better gradient propagation than Sigmoid
- Rectified Linear
    - max(x,0)
    - State of the art
    - Good gradient propagation
    - Can “die”

8. L1 and L2 Regularization
- Most pictures of nice filters involve some regularization
- L2 regularization corresponds to weight decay
- L2 and early stopping have similar effects
- L1 leads to sparsity
- Might not be needed anymore (more data, dropout)

9. Monitoring Training
- Monitor training and validation performance
- Can monitor hidden units
- Good: Uncorrelated and high variance

10. Stochastic gradient descent (SGD)

Stochastic back-propagation method is an instance of a more general technique called stochastic gradient descent (SGD)

**Recommendations** for using stochastic gradient algorithms:
- Use stochastic gradient descent when training time is the bottleneck

- Randomly shuffle the training examples and Choose Examples with Maximum Information Content, as Networks learn the fastest from the most unexpected sample
    - Shuffle the training set so that successive training examples never (rarely) belong to the same class.
    - Present input examples that produce a large error more frequently than examples that produce a small error. (outliers will destroy it)
- Shift inputs so the average of each input variable over the training set should be close to zero.
- Scale input variables so that their covariances are about the same.
- Input variables should be uncorrelated if possible.
- decorrelate the inputs such as PCA, can decrease the size of the network

**Monitor** both the training cost and the validation error

1. Zip once through the shuffled training set and perform the stochastic gradient
descent updates.
2. With an additional loop over the training set, compute the training cost.
Training cost here means the criterion that the algorithm seeks to optimize.
3. With an additional loop over the validation set, to compute the validation
set error. Error here means the performance measure of interest, such as
the classification error.

These steps means a significant computation effort but it worth.

**Check the gradients** using finite differences
http://leon.bottou.org/publications/pdf/tricks-2012.pdf
