# datamining

Uses OpenRec, a modular recommender framework build on Tensorflow.

## Project Structure

`dataset/` contains raw input data

`data_extraction/` contains csv parsing and javascript data wrangling scripts to process and format the data sets

`model-*, checkpoint` OpenRec saved artefacts (in the form of tensorflow sessions)

`*.py` Various scripts to save, load, combine and serve/evaluate recommender systems
