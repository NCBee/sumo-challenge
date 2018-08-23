# API for the SUMO challenge

For additional details, please see the [SUMO challenge website](http://sumochallenge.org)

This API is written using a combination of C++, Python, and Cython.  The user-facing interface is
Python only.

## Requirements

This library was tested with Python 3.5.2 on Ubuntu 16.04.

## Installation Instructions

### Install OS dependencies (Ubuntu 16.04)

```
sudo apt-get install git cmake python3-dev python3.5-venv \
libeigen3-dev libboost-python-dev libopencv-dev python-opencv \
libgmp-dev libcgal-qt5-dev swig
```

### Clone repository

```
git clone https://github.com/facebookresearch/sumo-challenge.git
```

### Setup Virtual Environment

Create a virtual evironment where the required packages will be installed in isolation.
```
cd libsumo
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Build Cython modules

```
python setup.py build_ext --inplace
```

## Post-installation

### Unit Testing


To run all unit tests:

`python -m unittest discover -p "*tests.py" -v`

To run all unit tests under a specific directory:

```
python -m unittest discover -s <path> -p "*tests.py" -v
```
For example:
```
python -m unittest discover -s sumo/threedee -p "*tests.py" -v 
```

### Clean files from build (when needed)

```
python setup.py cleanall
```