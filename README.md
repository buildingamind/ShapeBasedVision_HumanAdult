# Setup

### Step 1: Check python version. 
Your python version should be 3.11.5 or greater to successfully install all the required libraries.

### Step 2: Creating a virtual Environment.
You need a virtual environment to install all the libraries required to run our models. A virutal environment can be created either using 'conda' or 'venv'. If you system has an older version of python then use conda otherwise use venv.

```python
# Option1: create a virtual environment using conda
conda create --name myenv python=3.11.5 
# replace myenv with your desired env name
```

```python
# Option2: create a virtual environment using venv
python3 -m venv myenv 
# replace myenv with your desired env name
```

### Step 3: Activate virtual environment.
After creating your virtual environment, activate it to install libraries inside that env.

<b>Note:</b> If you used option1 to create virtual env, then activate using - 

```
conda activate myenv
```

<b>Note: </b> If you used option2 to create virtual env, then navigate to the dir where the virtual environment is created and activate using - 

```
source myenv/bin/activate
```

