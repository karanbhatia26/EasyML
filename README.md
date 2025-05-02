# EasyML

## Collaborative MARL for AutoML Pipeline Construction

EasyML is a research project that implements a novel Teacher-Student Multi-Agent Reinforcement Learning (MARL) framework for automated machine learning pipeline construction.

## Overview

This repository contains the implementation of the approach described in our paper "Collaborative Multi-Agent Reinforcement Learning for Automated Machine Learning Pipeline Construction: A Teacher-Student Framework". The framework uses a teacher-student architecture where both agents collaborate to construct optimal ML pipelines efficiently.

## Key Features

- 🤝 Teacher-Student MARL architecture for guided exploration
- 🧩 Component-based credit assignment mechanism
- 🔄 Adaptive intervention and knowledge transfer
- 📚 Emergent curriculum learning behavior
- ⚡ Efficient pipeline construction with fewer evaluations

## Installation

```bash
# Clone the repository
git clone https://github.com/karanbhatia26/EasyML.git
cd EasyML

# Create a virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

To run training with the MARL framework:

```bash
# Run MARL training on a specific dataset
python -m marl.train --dataset iris --episodes 500
```

### Available Datasets

- `iris` - Small, structured dataset (150 samples, 4 features)
- `adult` - Medium, mixed dataset (48,842 samples, 14 features)
- `covertype` - Large, numerical dataset (581,012 samples, 54 features)
- `credit-g` - Medium, financial dataset (1,000 samples, 20 features)
- `bank-marketing` - Medium, campaign dataset (45,211 samples, 16 features)

### Adjusting Training Duration

You can adjust the number of training episodes based on your available computational resources:

```bash
# For quicker experimentation with fewer episodes
python -m marl.train --dataset iris --episodes 100

# For full performance results as reported in the paper
python -m marl.train --dataset iris --episodes 500
python -m marl.train --dataset adult --episodes 500
python -m marl.train --dataset covertype --episodes 200
```

## Benchmark Evaluation

The repository includes scripts for benchmarking our MARL approach against other AutoML methods:

```bash
# Run the comprehensive benchmark on the Adult dataset
python foo.py
```

This benchmark compares our MARL approach with:
- Grid Search
- Random Search
- TPOT (if installed)
- Auto-sklearn (if installed)

### Benchmark Results

Below are representative results from our benchmark tests on the Adult dataset:

| Method | Accuracy | F1 Score | Training Time (s) |
|--------|----------|----------|-------------------|
| Grid Search | 85.74% | 85.62% | 118.2 |
| Random Search | 85.68% | 85.55% | 42.5 |
| TPOT | 85.97% | 85.83% | 452.8 |
| Auto-sklearn | 86.10% | 86.04% | 895.4 |
| MARL AutoML | 86.20% | 86.15% | 215.6 |

### Benchmark Environment Requirements

The benchmark code requires additional dependencies:
- TPOT: `pip install tpot`
- Auto-sklearn: `pip install auto-sklearn`

Note that Auto-sklearn is primarily designed for Linux environments and may require additional setup on Windows.

## Documentation

The implementation follows the architecture described in our paper:

- **Environment**: Manages ML pipeline construction, component compatibility, and performance evaluation
- **Student Agent**: Learns to build pipelines through exploration and direct experience
- **Teacher Agent**: Provides selective guidance to improve the student's learning efficiency
- **Credit Assignment**: Attributes performance to pipeline components and agents

## Reproducibility Notes

- Due to resource inconsistencies across different hardware configurations, pre-trained models are not included in the repository
- All experiments must be run from scratch
- Results may vary by ±0.5% due to the stochastic nature of reinforcement learning and environmental differences
- Different hardware configurations, operating systems, and library versions can influence benchmark results
- For consistent benchmarking, we recommend using the same environment for all compared methods
- The framework automatically generates visualizations and logs performance metrics during training

## Common Issues and Solutions

- **Memory errors**: For large datasets, reduce batch size by modifying the configuration in train.py
- **Long training times**: Some datasets might take 8 or more hours. For initial testing, reduce the number of episodes and consider using smaller datasets like Iris
- **TPOT or Auto-sklearn issues**: These packages may have compatibility issues on certain operating systems. In case of failures, the benchmark will fall back to simpler methods
- **Result variability**: Due to the stochastic nature of RL and differences in computing environments, results may vary between runs. For publication-quality results, average over multiple runs

