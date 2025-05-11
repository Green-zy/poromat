# Poromodel Benchmark

A benchmarking framework for predicting stress-strain behavior of porous titanium alloys using hybrid methods:
- Traditional interpolation + ensemble learning
- Machine Learning (LightGBM)
- Bayesian Inference (MCMC with hierarchical priors)

## Features
- Modular code for training and evaluating models
- Includes physical model priors (Z-A, Drucker-Prager)
- Statistical hypothesis testing to select best model
- Easy export of prediction results for dashboard use

## Goals
This repo serves as the modeling core behind `poromodel-app`.

## Structure
- `models/`: Modeling algorithms
- `pipeline/`: Train and evaluate
- `evaluation/`: Compare and validate models
- `utils/`: Data processing and plotting
