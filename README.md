# FractalFedSumUp 2.0

FractalFedSumUp 2.0 is an advanced federated learning framework with data compression and visualization capabilities. It extends traditional federated learning approaches with fractal-based data summarization techniques.

## Features

- **Multiple Federated Learning Algorithms**: Supports FedAvg, FedProx, FedAdam, FedSD2C, FedSumUp, FedSum, FedDM, and CollabDM
- **Data Compression**: Implements fractal-based data compression for efficient communication
- **Privacy Protection**: Includes differential privacy mechanisms (Laplace, Gaussian)
- **Batch Experiment Management**: Automated batch experiment execution with Excel-based configuration
- **Visualization Tools**: Comprehensive visualization of federated learning processes and results
- **GPU Monitoring**: Real-time GPU usage monitoring for efficient resource allocation
- **Non-IID Data Handling**: Support for various non-IID data distributions

## Project Structure

```
FractalFedSumUp_2.0/
├── __init__.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── run_batch_experiments/
    ├── __init__.py
    ├── run_batch_experiments.py          # Main batch experiment runner
    ├── experiment_runner.py              # Individual experiment execution
    ├── gpu_monitor.py                    # GPU resource monitoring
    ├── draw_linear_plot.py               # Visualization utilities
    ├── extract_result_by_color.py        # Result extraction tools
    ├── generate_template_prompt.py       # Template generation
    ├── get_batch_to_experiment_id_map.py # Experiment mapping
    ├── xlsx_tools.py                     # Excel file manipulation
    ├── template_prompt.txt               # Experiment templates
    ├── *.xlsx                            # Experiment configuration files
    ├── experiment_image_result/          # Generated experiment images
    ├── experiment_logger/                # Experiment logging
    ├── fedsumup_visualizations/          # Visualization outputs
    ├── generate_config_for_one_experiment/ # Configuration generation
    ├── plot/                             # Plotting utilities
    └── run_one_experiment/               # Single experiment execution
        └── run_and_get_results/
            └── config_data/
                ├── config.py             # Main configuration
                ├── data.py               # Dataset configuration
                ├── federated.py          # Federated learning algorithms
                └── schedule.py           # Training schedule
```

## Installation



## Usage

### Running Batch Experiments

The main entry point for running experiments is `run_batch_experiments.py`:

```python
python run_batch_experiments/run_batch_experiments.py
```

### Configuration

Experiments are configured using Excel files in the `run_batch_experiments/` directory:

- `FedSumUp-baselines-new.xlsx`: Baseline experiments
- `FedSumUp-hyperparameter-size.xlsx`: Hyperparameter tuning
- `FedSumUp-FedAvg-DP.xlsx`: Differential privacy experiments
- `FedSumUp-NonIID.xlsx`: Non-IID data experiments
- `FedSumUp-clientnum.xlsx`: Client number experiments
- `FedSumUp-visualize.xlsx`: Visualization experiments

### Key Parameters

- **Federated Learning Config**: Algorithm selection (FedAvg, FedProx, etc.)
- **Dataset**: MNIST, FashionMNIST, CIFAR10
- **Client Number**: Number of clients in federated learning
- **Compressed Image Size**: Size of compressed data representations
- **DP Mechanism**: Differential privacy mechanism (laplace, gaussian, no_dp)
- **DP Epsilon**: Privacy budget for differential privacy

## Supported Algorithms

1. **FedAvg**: Federated Averaging (baseline)
2. **FedProx**: Federated learning with proximal term
3. **FedAdam**: Adaptive optimization for federated learning
4. **FedSD2C**: Federated learning with synthetic data and compression
5. **FedSumUp**: Fractal-based data summarization for federated learning
6. **FedSum**: Training-free federated summarization
7. **FedDM**: Federated learning with diffusion models
8. **CollabDM**: Collaborative diffusion models

## Visualization

The framework includes comprehensive visualization tools:

- Training progress plots
- Client-server communication visualization
- Data compression visualization
- Model performance comparison
- Resource usage monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use FractalFedSumUp 2.0 in your research, please cite:

```bibtex


## Contact

For questions and support, please open an issue on GitHub or contact [your email].

## Acknowledgments

- This work builds upon traditional federated learning frameworks
- Inspired by fractal compression techniques for efficient data representation
- Thanks to the open-source community for various tools and libraries
