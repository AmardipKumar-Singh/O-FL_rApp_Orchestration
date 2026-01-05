Basic Installation

bash
# Clone the repository
git clone https://github.com/yourusername/O-FL_rApp.git
cd O-FL_rApp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

With Gurobi (Recommended)

For optimal MILP solving performance:

bash
# Install Gurobi
pip install gurobipy

# Obtain license (academic users)
# Visit: https://www.gurobi.com/academia/

Development Installation

bash
# Install with development dependencies
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install



🏛️ Architecture

text
O-FL_rApp/
├── core/                          # Abstract base classes and interfaces
│   ├── __init__.py
│   └── base.py                    # ITask, INetworkTopology, IOrchestrator, etc.
│
├── models/                        # Domain models
│   ├── __init__.py
│   ├── task.py                    # Task implementations (eMBB, uRLLC, Mixed)
│   └── network.py                 # Network topology (O-RAN)
│
├── solvers/                       # Optimization solvers
│   ├── __init__.py
│   ├── tsa_solver.py             # Task and Slice Assignment (Algorithm 1)
│   └── rar_solver.py             # Resource Allocation and Routing (MILP)
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   └── utilities.py              # Utility functions, QoS models, EMA estimator
│
├── data/                          # O-RAN Datasets
│   ├── __init__.py
│   ├── base_dataset.py           # Abstract dataset interface
│   ├── network_traffic_dataset.py # Network Traffic QoS
│   └── cell_load_dataset.py      # Cell Load Balancing
│
├── training/                      # Federated Learning & MARL
│   ├── __init__.py
│   ├── fl_trainer.py             # FL trainer with FedAvg
│   ├── marl_base.py              # MARL abstract interfaces (NEW!)
│   ├── ppo_networks.py           # Actor-Critic networks (NEW!)
│   └── mappo_trainer.py          # MAPPO implementation (NEW!)
│
├── environments/                  # MARL Environments (NEW!)
│   ├── __init__.py
│   └── oran_environment.py       # O-RAN MARL environment
│
├── baselines/                     # Baseline algorithms for comparison
│   ├── __init__.py
│   └── baseline_algorithms.py    # Independent FL, Static, Auction, Priority
│
├── orchestrator.py               # Base O-FL rApp orchestrator
├── orchestrator_with_training.py # Integrated orchestrator with FL
├── main.py                       # Entry point (simulation-based)
├── main_with_datasets.py         # Entry point with FL training
├── main_with_marl.py             # Entry point with MAPPO training (NEW!)
├── config.py                     # System configuration
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── pytest.ini                    # Testing configuration
└── README.md                     # This file