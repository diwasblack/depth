DEPTH
=====

A DEep learning PyTHon library written using numpy only.


Usage Instruction
-----------------
- Clone the git repo using
    ```
    git clone https://github.com/diwasblack/depth.git
    ```
- Activate the virtual environment if needed
- Install library from the root folder in editable mode using
    ```
    pip install -e .
    ```

Features
-----------------
- Dense layer with activations:
    - Sigmoid
    - Tanh
    - ReLu
    - LeakyRelu
    - Softmax
    - Linear
- Optimizers
    - Stochastic Gradient Descent
    - ADAM
- Loss functions
    - Mean Squared Error
    - Cross entropy loss
- Regularizer
    - L2 Regularizer

Examples
--------
A sequential network with 2 hidden layer and a softmax layers as final layer with cross entropy loss.

```python
nn_object = Sequential()
nn_object.add_layer(DenseLayer(
    units=32, activation="tanh", input_dimension=10))
nn_object.add_layer(DenseLayer(
    units=64, activation="tanh"))
nn_object.add_layer(DenseLayer(units=10, activation="softmax"))
nn_object.compile(loss="cross_entropy", error_threshold=0.001)
```

See the [examples folder](examples) for more.
