# Direct-Style-Transfer

Code for "On Learning Text Style Transfer with Direct Rewards", NAACL 2021.

## Libraries

We use PyTorch 1.2.0 for our experiments. The dependencies are specified in requirements.txt

## Data

### Yelp and Amazon

Please download the data at [Yelp](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data/yelp) and [Amazon](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data/amazon) and put the downloaded data at `./yelp` and `./amazon`.

### IMDb

Please download the data at [IMDb](https://github.com/fastnlp/nlp-dataset/tree/master/text%20style%20transfer) and put the downloaded data at `./imdb`.

### GYAMC

Please download the data at [GYAMC](https://github.com/raosudha89/GYAFC-corpus) and put the downloaded data at `./formality_family`. (We use Family & Relationships category for our experiments)

## Preprocessing

Please rename the downloaded files following the format `{sentiment, formality}.{train, dev, test}.{0, 1}.txt`.

For example, `sentiment.test.0.txt` contains the negative samples in the test set.

## Code

We use huggingface [Transformers](https://github.com/huggingface/transformers) in our experiments.

`bootstrap.py` - First Stage Training

To run the experiment, you may specify the hyperparameters in `config` dict, and run
```
python bootstrap.py --cuda --gpuid [GPUID] -l 
```


`main.py` - Second Stage Training

To run the experiment, you may specify the hyperparameters in `config` dict, and run
```
python main.py --cuda --gpuid [GPUID] -l -s -r -p -u
```

`evaluate.py` - Evaluation Functions

To run the experiment, please run
```
python evaluate.py --cuda --gpuid [GPUID] --file [OUTPUT_FILE_NAME] --dataset [DATASET_NAME] --model_pt [MODEL_CHECKPOINT]
```

`classifier.py` - classifiers for training

`Dataloader.py` - dataloaders

`gpt_utils.py` - modified transformers function to enable approximate word indexes

`ref_sim.py`, `sim_models.py`, `sim_utils.py` - code for SIM model

`utils.py` - utility functions

## Results

The results of our models can be found in the `./output` directory.

Each line of the files contains the source, reference and model output, seperated by `\t` (source and model output only for IMDb dataset).

Due to licensing restrictions, we only provide the model outputs for GYAMC dataset. 




