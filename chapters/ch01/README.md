# Chapter 01: Introduction

Reproductions for figures in Chapter 1 of Christopher M. Bishop's _Pattern Recognition and Machine Learning_ (PRML).

## Overview

- **Configurations**: YAML configs located in `configs/ch01/`
- **Outputs**: Generated plots saved to `outputs/figures/`
- **Execution**: Run commands as modules from the repository root directory

```bash
python -m chapters.ch01 1
python -m chapters.ch01 2
python -m chapters.ch01 all
```

| Command                       | Target Figure | Description                                                     | Config File                    | Output File                      |
| :---------------------------- | :------------ | :-------------------------------------------------------------- | :----------------------------- | :------------------------------- |
| `python -m chapters.ch01 1`   | Figure 1.2    | Training data generated from $\sin(2\pi x)$ with Gaussian noise | `configs/ch01/figure_1_2.yaml` | `outputs/figures/figure_1_2.png` |
| `python -m chapters.ch01 2`   | Figure 1.4    | Polynomial curve fitting comparison ($M = 0, 1, 3, 9$)          | `configs/ch01/figure_1_4.yaml` | `outputs/figures/figure_1_4.png` |
| `python -m chapters.ch01 all` | All Figures   | Executes all Chapter 1 experiments sequentially                 | —                              | `outputs/figures/`               |
