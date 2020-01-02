# Feed Forward Neural Network

Implement a bit like a keras library. 

Only for machine learning

### Test
run.py file is an implementation to resolve xor.

Here is a construction of the model
```python
# network
net = Network()
net.add(Layer(in_neurons=2, out_neurons=3))
net.add(ActivationLayer(gauss, gauss_prime))
net.add(Layer(in_neurons=3, out_neurons=1))
net.add(ActivationLayer(tanh, tanh_prime))
```
