# GAP4SR: Graph Contrastive Learning with Attention and PolyLoss for Sequential Recommendation
## Requirements

This implementation is based on PyTorch Geometric. To run the code, you will need the following dependencies:

- python 3
- torch
- torch-geometric
- tqdm
- pickle
- scipy

## Datasets

### Format Description

Taking the Home dataset as an example:

#### `home.txt`

Each line represents a user and their interaction sequence:

```
0 1,2,3,4,5,6,7,8
1 5,9,10,11,12
...
```

#### `all_train_seq.txt`

Same format as `home.txt`, but with the last and the second last interaction items removed:

```
0 1,2,3,4,5,6
1 5,9,10
...
```

#### `train.pkl`

This file contains a tuple of four lists:

```python
(
 [0, 0, 0, 0, 0, 1, ...], 
 [[1, 2, 3, 4, 5],
  [1, 2, 3, 4],
  [1, 2, 3],
  [1, 2],
  [1],
  [5, 9],
  ...],
 [6, 5, 4, 3, 2, 10, ...],
 [5, 4, 3, 2, 1, 2, ...]
)
```

#### `test.pkl` and `valid.pkl`

Have the same format as `train.pkl`.

### Build Weighted Item Transition Graph

To build a weighted item transition graph using all observed data in `all_train_seq.txt`, run:

```bash
python build_witg.py
```

Figure 1 in the paper shows an example of the transition graph without edge weight normalization.

## Usage

To run GCL4SR on the Home dataset, execute:

```bash
python runner.py --data_name='home'
```

To see the full list of configurable hyperparameters and their explanations:

```bash
python runner.py -h
```

## Notes

* Ensure the dataset files are placed in the correct directories.
* If you encounter issues, please check the dependencies or dataset format.
