# Installation
Follow the steps given below to install the `JellyBamm` Python package. The package must be installed to run the included examples. It is recommended to create a virtual environment for the installation, in order not to alter any distribution python files.

## Create a virtual environment

### Using virtualenv
To create a virtual environment `env` within your current directory type:

```bash
# Create a virtual env
virtualenv env

# Activate the environment
source env/bin/activate
```

Now all the calls to pip described below will install `JellyBamm` and its dependencies into the environment `env`. When you are ready to exit the environment and go back to your original system, just type:

```bash
deactivate
```

### Using conda
Alternatively, use Conda to create a virtual environment then install the `JellyBamm` package.

```bash
# Create a Conda virtual environment
conda create -n JellyBamm python=3.8

# Activate the conda environment
conda activate JellyBamm
```

Now all the calls to pip described below will install `JellyBamm` and its dependencies into the environment `env`. When you are ready to exit the environment and go back to your original system, just type:

```bash
conda deactivate
```

## Using pip
Execute the following command to install `JellyBamm` with pip:

```bash
pip install JellyBamm
```

## Install from source (developer install)
This section describes the build and installation of `JellyBamm` from the source code, available on GitHub. Note that this is not the recommended approach for most users and should be reserved to people wanting to participate in the development of `JellyBamm`, or people who really need to use bleeding-edge feature(s) not yet available in the latest released version. If you do not fall in the two previous categories, you would be better off installing `JellyBamm` using pip.

Run the following command to install the newest version from the Github repository:
To obtain the `JellyBamm` source code, clone the GitHub repository.

```bash
git clone https://github.com/pybamm-team/JellyBamm.git
```
From the `JellyBamm/` directory, you can install `JellyBamm` using -
```bash
# Install the JellyBamm package from within the repository
$ pip install -e .
```
