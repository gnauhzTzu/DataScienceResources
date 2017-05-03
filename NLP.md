********************************************
NEURAL NETWORK

Fine-tuning the parameters of a neural network

* Adapting the learning rate with error

    * In order to reduce training time while avoiding local minimums. Start with a small learning rate
    * Increase it exponentially if two epochs in a row reduce the error
    * Decrease it rapidly if a significant error increase occurs.
    
* Adapting the learning rate with gradient direction
    
    * Start with a small learning rate
    * Increase it if two epochs in a row have a gradient direction pretty similar
    * Decrease it drastically if the direction differs greatly.

* Pick a separate learning rate for each weight, and then modify the rate for each weight as it trains. 
    * If after successive runs the weight moves in the same direction, go ahead and increase the learning rate. 
    * If the weight moves in the opposite direction, lower the learning rate for that particular weight.
*  Depth and width of the neural network
    * If adding a new layer does not provide a significant decrease in the validation error, then there most likely is no need to add more layers. Same goes for the neurons in each layer
*  When a certain training error is stop there, monitor the network as it trains. Once the training error decreasing but the validation error increasing consistently, going further might cause over fitting. The training error right before this started occurring is typically the best.

https://www.quora.com/How-should-one-select-parameters-when-fine-tuning-neural-networks



********************************************