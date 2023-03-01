# OGDA_for_weak_Minty

Supplementary code (in Julia 1.7) for the [paper](https://arxiv.org/abs/2201.12247):
```
@article{bohm2022solving,
  title={Solving nonconvex-nonconcave min-max problems exhibiting weak minty solutions},
  author={B{\"o}hm, Axel},
  journal={arXiv preprint arXiv:2201.12247},
  year={2022}
}
```

Collects popular algorithms for variational inequalities + test problems. Some of them monotone. Some of them weak Minty.

## Usage

The `problems/` folder contains all problem instances.

All algorithms can be found in `utils/methods.jl`.

To reproduce the experiments from the paper, simply run `run_files/plots_for_ogda_paper.jl`.
