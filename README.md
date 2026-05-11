# Mathematical Optimization
## Implementation of the paper: Robust location for quarantine facilities under decentralized room assignment: A bi-level mixed-integer programming approach

<img src="/images/for_readme.png" style="width:100%; height:auto;">



### .xlsx files description

The data I have is contained in Excel files (1.xlsx, 2.xlsx, ..., 16.xlsx), and each file has three different sheets. Each sheet contains demand, capacity, cost, revenue, and penalty. 
Each sheet is organized as follows:
- DEMAND: xy matrix (users are the columns, airports or arrival points are the rows)
- CAPACITY: xy matrix (room types are the columns, hotels are the rows)
- COST: xy matrix (airports are the columns, hotels are the rows)
- PRICE: xy matrix (room types are the columns, hotels are the rows)
- REVENUE: a list (one entry per hotel)
- PENALTY: single value, different for each Excel sheet

<!-- ### Description


### Structure -->

### ILS

#### Usage

```bash
# Full run on all 16 instances × 3 sheets, no time limit (default)
python ils.py

# Full run with 2-hour time limit
python ils.py --time-limit

# Test mode: runs a small subset of instances (edit the test block in ils.py to customize)
python ils.py --test
python ils.py -t

# Test mode with time limit
python ils.py -t --time-limit
```

All outputs (CSV results and plots) are saved in the `results/` directory.

A project of Eva Fumo and Gabriele Tomai



### MIBS

#### Install mibs (https://github.com/coin-or/MibS)
```bash
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew
./coinbrew fetch MibS@master
./coinbrew build MibS
```

You also need the following programs:
```bash
sudo apt-get install -y \
  make \
  git \
  wget \
  tar \
  patch \
  dos2unix \
  pkg-config \
  g++ \
  gfortran \
  libblas-dev \
  liblapack-dev \
  libreadline-dev
```

Export the command
```bash

echo 'export LD_LIBRARY_PATH=$HOME/mibs-install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=$HOME/mibs-install/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Run the command
```bash
cd src
python3 run_mibs.py
```