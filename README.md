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
