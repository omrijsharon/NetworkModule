# NetworkModule
An easy was to create a fully connected network with pytorch. Contains additional module functions that can be added to Sequential.

**How to use the module?**

The module gets a list of layers and a list of activation functions.
In the layer's list, each element corresponds to the number of nodes in the layer, and the length of the list is the number of layers in the network.

i.e. 
L = [16, * 2 * [8] , 4]

activation_func = [*(len(L)-2) * [functional.SeLU()], functional.Identity()]

will create a network with:

1st layer - input: 16 nodes

2nd layer - hidden: 8 nodes, with SeLU as activation function 

3rd layer - hidden: 8 nodes, with SeLU as activation function 

4th layer - output: 4 nodes, with a Identity activation function

notice that Identity() is a linear activation function. It is exacly like not puting any activation function, yet it is necessary that each layer which is not the 1st layer will have an activation function in pytorch sequential. In other words, when no activation function is needed, use functional.Identity().

Also notice that the length of the activation_func list is always smaller by 1 than the layers' list, because the 1st layer never gets an activation function.



example:


from NetworkModule import Network

from NetworkModule import functional as functional


input_dim = 16

output_dim = 4

hidden_layers = 2*[8]

L = [input_dim, *hidden_layers, output_dim]

activation_func = [*len(hidden_layers) * [functional.SeLU()], functional.Identity()]

net = Network(L, activation_func, dropout=0.5)
