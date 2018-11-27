# House Price Prediction Using a Deep Feed-Forward Neural Network

The goal of this mini-project is to construct a model to predict a home’s current market value and score
a test data set with this model. The best result is achieved using a feed-forward neural network with
two hidden layers. This network achieves the mean absolute error value of less than 0.2 on the validation
set. TensorFlow’s Python API is used to create the model.

## Dependencies:
- Python (3.6 preferably; also works fine with python 2)
- NumPy 1.15.3
- [Tensorflow](https://github.com/tensorflow/tensorflow)>=1.11
- Matplotlib (for saving images) 3.0.1

## Instruction
All the parameters and Hyper-parameters are accessible through the ```config.py ``` file.
 You can run the code in Terminal using the following commands:

- For training on the data: ```python main.py ```
- For training with a different batch size: ```python main.py --batch_size=100```

The rest of the parameters can also be changed in the same way.
After training the model, it can be tested using the following command:

- To test the trained model: ```python main.py --mode='test --reload_Epoch=232```

where ```reload_Epoch``` determines which model to be loaded. 
The above command will load the model which is trained for 232 epochs.


