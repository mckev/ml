# Machine Learning
My introduction to Machine Learning

## Installation

 1. Install Python 3 from https://www.python.org/downloads/.
 
 2. Upgrade Python Package Manager:
       ```
       python -m pip install --upgrade pip
       pip install --upgrade setuptools
       ```

 3. Install dependencies:
       ```
       pip install --upgrade --requirement requirements.txt
       ```

 4. Run MNIST test:
       ```
       cd ./test/ml/
       python -m unittest test_sgd_mnist.py
       ```

    Output:
       ```
       Reading MNIST file ../../classes/mnist/mnist.raw
       Total 70000 records
       Training...
       Testing...
       Correct 18263 out of 20000 (91.3% correct)
       ```
