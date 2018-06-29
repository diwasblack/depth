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


Examples
--------
A sequential network with 2 hidden layer and a softmax layers as final layer with cross entropy loss.

```python
nn_object = Sequential()
nn_object.add_layer(units=32, activation_function="tanh", input_dimension=10)
nn_object.add_layer(units=64, activation_function="tanh")
nn_object.add_layer(units=output_data_dimension, activation_function="softmax")
nn_object.compile(loss="cross_entropy", error_threshold=0.001)
```

See the [examples folder](examples) for more examples.
