# Lead score prediction

This repository contains code to train a machine learning model capable of predicting lead scores.

## Requirements:
Just need to install the required packages in the file `requirements.txt`, you can do that like this:

`$ pip install -r requirements.txt`

## Usage:

### Recommended method (train and predict separately:

###### To train a machine learning model:
		To train a new predictive pipeline (a model) just need to follow the instructions on the file `train.py`

###### To generate predictions for new leads:
		To generate lead score predictions for a new leads file, just need to follow the instructions in the file `predict.py`



### Alternative method:

###### To train the model and generate predictions all at once:
		To train the model and generate new lead score predictions all at once (using one command)
		- pros: 
			Save the computer storage space, since the trained pipeline, which takes around 800 MB, will not be saved during the process.
		- cons: 
			Every time we want to predict new lead scores, the model needs to be retrained, which means that we need to wait for 15-20 min each time.
