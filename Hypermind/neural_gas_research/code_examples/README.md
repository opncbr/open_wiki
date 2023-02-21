# Neural Gas Experiments

## Dependencies

Python `3.10.8` or higher.

[Pytest](https://docs.pytest.org/en/7.1.x/getting-started.html):

```
pip3 install pytest
```

[Pytorch](https://pytorch.org/):

```
pip3 install torch torchvision torchaudio
```

[PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse)

```
pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu116`, or `cu117` depending on your PyTorch installation.

## Unit tests

To run unit tests, run:

```
python3 -m pytest tests/
```