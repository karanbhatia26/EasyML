# EasyML

## Collaborative MARL for AutoML Pipeline Construction

EasyML is a research project that implements a novel Teacher-Student Multi-Agent Reinforcement Learning (MARL) framework for automated machine learning pipeline construction.

## Overview

This repository contains the implementation of the approach described in our paper "Collaborative Multi-Agent Reinforcement Learning for Automated Machine Learning Pipeline Construction: A Teacher-Student Framework". The framework uses a teacher-student architecture where both agents collaborate to construct optimal ML pipelines efficiently.

## Key Features

- Teacher-Student MARL architecture for guided exploration
- Component-based credit assignment mechanism
- Adaptive intervention and knowledge transfer
- Emergent curriculum learning behavior
- Efficient pipeline construction with fewer evaluations

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

Available datasets include:
- iris
- adult
- covertype
- credit-g
- bank-marketing

You can adjust the number of training episodes based on your available computational resources:

```bash
# For quicker experimentation with fewer episodes
python -m marl.train --dataset iris --episodes 100

# For full performance results as reported in the paper
python -m marl.train --dataset iris --episodes 500
python -m marl.train --dataset adult --episodes 500
python -m marl.train --dataset covertype --episodes 200
```

## Reproducibility Notes

- Due to resource inconsistencies across different hardware configurations, pre-trained models are not included in the repository
- All experiments must be run from scratch
- Results may vary slightly due to the stochastic nature of reinforcement learning
- The framework automatically generates visualizations and logs performance metrics during training

## Common Issues and Solutions

- **Memory errors**: For large datasets, reduce batch size by modifying the configuration in `marl/train.py`
- **Long training times**: Some datasets might take 8 or more hours. For initial testing, reduce the number of episodes and consider using smaller datasets like Iris

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{bhatia2024collaborative,
  title={Collaborative Multi-Agent Reinforcement Learning for Automated Machine Learning Pipeline Construction: A Teacher-Student Framework},
  author={Bhatia, Karan and Joseph, Richard},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
