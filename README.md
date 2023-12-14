# DESDEO2: restructuring and refactoring of the DESDEO framework
Everything in this repository is currently subject to change. Stay tuned for updates.


## Installation instructions

### Prerequisites
- Python 3.12 or newer
- Poetry (https://python-poetry.org/docs/#installation)

### Getting and running DESDEO
- Clone this repository (`git clone https://github.com/industrial-optimization-group/DESDEO.git`)
- Create a virtual environment for DESDEO (For example, with `python -m venv .venv`)
- Activate the virtual environment (For example, with `source .venv/bin/activate`)
- Install DESDEO with Poetry (`poetry install -E standard` or `poetry install -E legacy`)
    - Note that the `standard` extra includes the `polars` dependency which may not work on older CPUs or in an Apple
        Silicon environment under Rosseta 2. In this case, use the `legacy` extra instead.
- Run tests? Run Notebooks? Profit? 


