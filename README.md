# PopNet

The source code for *PopNet: Real-Time Population-Level Disease Prediction with Data Latency*

## Requirements

* Install python, pytorch. We use Python 3.8, Pytorch 1.1.
* Install dgl.
* If you plan to use GPU computation, install CUDA.

## Synthetic dataset

We provide the synthetic dataset in ```data``` directory. The synthetic dataset have 1015 locations and the sequence length is 63. We also provide the generated location graph in ```data/g```. Use following codes to load the dataset

```python
import pickle
FILE = pickle.load(open('./data/FILENAME','rb'))
```

## Test PopNet

You provide the testing code in the ```train-popnet.ipynb``` notebook. We also provide a trained model in the ```save``` directory. You can run the code to get the performance of PopNet on the synthetic dataset.

## Train PopNet
We provide the training code for PopNet in the notebook. You can use them to train our model on other dataset.