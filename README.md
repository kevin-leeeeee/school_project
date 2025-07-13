# 0612_ppt_ana_refactored

This project focuses on the analysis of neural network quantization and pruning, including hardware simulation and visualization of weights and activations.

## Project Overview

This repository contains Python scripts designed to:
*   **Quantize** neural network weights and input activations.
*   **Simulate** the cycle-by-cycle behavior of a custom hardware accelerator for quantized neural networks under various configurations (Segments, Containers).
*   **Analyze** the performance (total cycles) and hardware utilization of different configurations.
*   **Visualize** neuron weights and simulation results.

## Key Features

*   **Weight Quantization & Pruning Analysis**: Tools to process and analyze quantized neural network weights.
*   **Hardware Simulation**: Detailed cycle-accurate simulation of a custom neural network accelerator.
*   **Performance Analysis**: Scripts to analyze simulation logs and summarize performance metrics.
*   **Visualization**: Generate heatmaps and histograms of weights, and 3D plots of simulation cycles.
*   **Configurable Experiments**: `main.py` provides a framework to run experiments with different configurations.

## Project Structure

```
.
├── main.py                 # Main entry point for running experiments
├── requirements.txt        # Python dependencies
├── archive/                # Contains historical data, simulation outputs, and pre-processed data
│   ├── ...                 # (e.g., json_0623_k4, analysis_output_0622, quantized_input_act)
├── config/
│   └── base_config.json    # Base configuration for experiments
├── data/                   # Raw datasets (e.g., MNIST, CIFAR-10)
├── experiments/            # Output directory for experiment results
├── scripts/                # (Placeholder for future utility scripts)
└── src/                    # Core Python scripts for analysis, simulation, and visualization
    ├── ana_channel_util.py
    ├── ana_cycle_range.py
    ├── combine_0618.py     # Hardware simulation (version 0618)
    ├── combine_0623.py     # Hardware simulation (version 0623)
    ├── combine_0624.py     # Hardware simulation (version 0624 - most flexible)
    ├── inputact_quantize_0624.py # Input activation quantization
    └── weight_visual.py    # Weight visualization
```

## Getting Started

### Prerequisites

*   Python 3.x
*   Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running Simulations

The core simulation logic is implemented in the `src/combine_*.py` files. `combine_0624.py` is the most recent and flexible version.

To run a simulation for a specific layer and neuron with defined Segments and Containers:

```bash
python src/combine_0624.py --json archive/analysis_output_0622/kmeans_4_quantized_network_prune.json --out archive/test_simulation_output
```

**Note**: The `LAYERS_TO_RUN` variable inside `src/combine_0624.py` (and other `combine_*.py` files) controls which layers and neurons are processed. You might need to adjust it for your specific needs.

### Running Experiments

The `main.py` script provides a structured way to run experiments based on a configuration file.

```bash
python main.py --config config/base_config.json
```

This will create a new directory under `experiments/` with a timestamp, copy the config file, and run the defined experiment steps.

### Analyzing Results

*   **`src/ana_channel_util.py`**: Analyzes `en_acc` signal utilization from simulation logs.
*   **`src/ana_cycle_range.py`**: Summarizes total cycles across different Segments/Containers configurations.

### Visualizing Weights

*   **`src/weight_visual.py`**: Generates heatmaps and histograms for neuron weights.

## Contributing

If you wish to contribute to this project, please follow these steps:

1.  **Fork** this repository.
2.  **Clone** your forked repository to your local machine.
3.  **Create a new branch** for your feature or bug fix:
    `git checkout -b feature/your-feature-name`
4.  **Make your changes** and commit them with clear, concise messages.
5.  **Push your branch** to your forked repository:
    `git push origin feature/your-feature-name`
6.  **Open a Pull Request** to the `main` branch of the original repository, describing your changes in detail.

---