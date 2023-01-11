# Conditional Mutual Information Neural Estimator

## Introduction
In this repository you may find the method explained in [[1]](https://arxiv.org/abs/2006.07225) to estimate conditional mutual information.
This technique is based on variational lower bounds for relative entropy known as Donsker-Varadhan bound **(DV bound)** and **NWJ bound**. Below is the DV bound for conditional mutual information: 

<img src="DV.png" width=400>

We use the k- nearest neighbor technique to help us design a neural classifier that is the basis of our estimation.

The model that we used in our simulations is a Gaussian model:

<img src="model.png" width=150>


The MI-Diff directory contains my independent implementation of the method proposed in the [[2]](http://proceedings.mlr.press/v115/mukherjee20a.html), where the authors estimate conditional mutual information as the difference of two mutual information terms. Among several methods in [[2]](http://proceedings.mlr.press/v115/mukherjee20a.html), the MI-Diff method performs better according to the provided simulation results. So we compare our results with this approach.


## Implementation
The neural network is implemented with **PyTorch**. For k-nearest neighbor, we use **Scikit-learn** library. 

## How to run
To run the code and reproduce the results in the paper use the help below:

### estimate I(X;Y|Z)
python main.py --d 5 --k 20 --n 80000 --scenario 0 --seed 123

### estimate I(X;Z|Y)
python main.py --d 3 --k 10 --n 80000 --scenario 1 --seed 123

### test DPI and additivity
python main.py --d 5 --k 10 --n 80000 --scenario 2 --seed 123

### Run MI-Diff method
python MIDiff.py --d 5 --k 10 --n 80000 --scenario 0 --seed 123

python MIDiff.py --d 3 --k 10 --n 80000 --scenario 1 --seed 123

## Visualization
The provided notebook shows how to load and visualize the data

## References
[1] Sina Molavipour, Germán Bassi, Mikael Skoglund, 'On Neural Estimators for Conditional Mutual Information Using Nearest Neighbors Sampling,' arXiv preprint arXiv:2006.07225, 2020.

[2] Sudipto Mukherjee, Himanshu Asnani, Sreeram Kannan, 'CCMI : Classifier based Conditional Mutual Information Estimation,' Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, PMLR 115:1083-1093, 2020.

