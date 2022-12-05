# pj_fdu_sjtu
## Run the code

### Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.4.0.
* Tensorboard

### Instructions
1. Put the data folder(data_fx) inside the root folder.
2. Create a log folder(para1) inside the root folder.
3. python Main.py [-opt.parameter]

### Data Set
1. train_one_1000.pkl: **Input training set.** Training set for one town(totally 13 towns) with a lookback window 100000 seconds.
2. train.pkl: Training set for 13 towns, without lookbback window. **Don't use it as the inputs.**
3. To view the dataset, run: pd.read_pickle("filename")

# pj_fdu_sjtu_without_qx
It's a indenpent version of model. The weather information is not considered in this version of model(pj_fdu_sjtu_without_qx), which preforms better for now.
