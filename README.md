# Signature Verification
This is a custom-built siamese, one-shot learning network for signature verification

## Set Up

 1. Ensure you have Keras 2.3+ and TensorFlow 2.0+  
	 - `pip install keras`
	 - `pip install tensorflow`

 2. Download the clean dataset from [here](https://www.dropbox.com/s/mlzce0xen9sv5of/data.zip?dl=1) (Original dataset from [here](https://www.kaggle.com/robinreni/signature-verification-dataset))
	 - `wget https://www.dropbox.com/s/mlzce0xen9sv5of/data.zip?dl=1`
 3. Unzip the dataset
	 - `unzip data.zip?dl=1 -d data`

## How To Run
Navigate to correct directory and run
`python train.py`

## Customization
|Paramater| Location |
|--|--|
| Epochs | train.py |
| Optimization Function | train.py |
| Batch Size | train.py |
| Hard Triplets | train.py |
| Random Triplets | train.py |
| Input Shape | helper_functions.py |
| Network | helper_functions.py (create_network) |
| Triplet Loss Alpha | helper_functions.py (create_model) |
