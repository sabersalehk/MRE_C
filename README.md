# Multi-Resolution Estimator for Convex Loss Functions (MRE-C Algorithm)

This is an implementation of the following papers:

[1] Salehkaleybar, S., Sharifnassab, A., & Golestani, S. J. (2021). [One-Shot Federated Learning: Theoretical Limits and Algorithms to Achieve Them](https://arxiv.org/abs/1905.04634) (Journal of Machine Learning Research).

[2] Sharifnassab, A., Salehkaleybar, S., & Golestani, S. J. (2019). [Order Optimal One-Shot Distributed Learning](https://arxiv.org/abs/1911.00731) (NeurIPS 2019).


If you find this code useful, please cite the following papers:
```
@article{sharifnassab2019order,
  title={Order Optimal One-Shot Distributed Learning},
  author={Sharifnassab, Arsalan and Salehkaleybar, Saber and Golestani, S Jamaloddin},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  pages={2168--2177},
  year={2019}
}

```

```
@article{salehkaleybar2019one,
  title={One-shot federated learning: theoretical limits and algorithms to achieve them},
  author={Salehkaleybar, Saber and Sharifnassab, Arsalan and Golestani, S Jamaloddin},
  journal={arXiv preprint arXiv:1905.04634},
  year={2019}
}
```

## Requirements

- Python 3.5+
- `numpy`
- `scipy`
- `multiprocessing`

## Running an example

In the current implementation, it is assumed that the number of samples per machine (n) is equal to one. In order to run instances of ridge regression problems
for m=100000 number of machines, execute the following code:

$python MRE_C.py

The output is the average error on estimating the true parameters.
