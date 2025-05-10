# Secure Robust Federated Learning


FedVer is a robust and efficient federated learning framework that implements various aggregation strategies and defense mechanisms against potential attacks. The framework is designed to achieve high accuracy while maintaining privacy and security in distributed learning scenarios.

## Features

- **Robust Aggregation Methods**
  - Adaptive Filtering
  - Coordinate-wise Median
  - Krum
  - Trimmed Mean
  - Momentum-based Aggregation

- **Advanced Training Features**
  - Adaptive Learning Rate
  - Performance-based Client Weighting
  - Momentum-based Updates
  - Robust Defense Mechanisms

- **Supported Models**
  - LeNet5
  - ResNet8
  - ResNet18
  - TRESNet18
  - TRESNet20

- **Dataset Support**
  - MNIST
  - CIFAR-10
  - CIFAR-100

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FedVer.git
cd FedVer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
from flearn.trainers.fedavg import Server

# Initialize server with parameters
params = {
    'model': 'resnet18',
    'dataset': 'cifar10',
    'num_rounds': 100,
    'clients_per_round': 10,
    'num_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.01,
    'robust_aggregation': 'adaptive_filtering'
}

# Create and train server
server = Server(params)
server.train()
```

### Advanced Configuration

```python
params = {
    # Model and Dataset
    'model': 'resnet18',
    'dataset': 'cifar10',
    'dataset_type': 'iid',  # or 'non-iid'
    'n_class': 10,
    
    # Training Parameters
    'num_rounds': 100,
    'clients_per_round': 10,
    'num_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.01,
    
    # Aggregation Settings
    'robust_aggregation': 'adaptive_filtering',
    'momentum': 0.9,
    
    # Defense Parameters
    'attack': None,  # or specify attack type
    'mal_clients': 0,
    'attack_scale': 1.0
}
```

## Key Features

### 1. Robust Aggregation
- Implements multiple aggregation strategies
- Adaptive filtering for outlier detection
- Performance-based client weighting
- Momentum-based updates for better convergence

### 2. Adaptive Learning
- Dynamic learning rate adjustment
- Performance-based client selection
- Momentum-based optimization
- Automatic hyperparameter tuning

### 3. Security Features
- Defense against model poisoning attacks
- Robust aggregation methods
- Client performance monitoring
- Anomaly detection

## Performance

The framework achieves high accuracy while maintaining robustness:
- MNIST: >98% accuracy
- CIFAR-10: >85% accuracy
- CIFAR-100: >70% accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fedver2024,
  author = Ambika,
  title = Secure Robust Federated Laerning,
  year = 2025,
  publisher = GitHub,
  url = https://github.com/ambikad04/Secure-Robust-Federated-Learning
}
```

## Acknowledgments

- Thanks to all contributors who have helped improve this project
- Inspired by various federated learning research papers
- Built with PyTorch and other open-source libraries

## Contact

For questions, please open an issue or contact [ambikadas0412@gmail.com](mailto:your-email@example.com)

