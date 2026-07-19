import math
import random
from graphviz import Digraph

# Value class

class Value:
    def __init__(self, d, _children=(), lb=''):
        self._backward = lambda: None
        self._prev = set(_children) # done for efficiency (??? but you iterate over it so really who gaf) 
        self.data = d
        self.grad = 0.0
        self._op = ''
        self.label = lb

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backwards():
            # if f = a + b, df/da = 1 and df/db = 1, and * out.grad because chain rule (d(something else)/da = d(something else)/df * df/da)
            # += and not = because what if f = a + a? want df/da to be 2, not 1, and in the non-duplicate case += only happens once so equivalent to =
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backwards
        out._op = '+'
        return out

    def __neg__(self):
        out = Value(self.data *-1, (self,))
        def _backwards():
            # if f = a - b, df/da = 1 and df/db = - 1
            self.grad += -1.0 * out.grad
        out._backward = _backwards
        out._op = '-'
        return out

    def __sub__(self, other):
        return self + other.__neg__()

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backwards():
            # if f = a*b, df/da = b and df/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backwards
        out._op = '*'
        return out
    
    # when we try to do 2.__add__(a) it will freak out and check if a has rmul to call so this is for that
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __radd__(self, other): # other + self
        return self.__add__(other)

    def tanh(self):
        x = self.data
        invtan = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) # shoutout wikipedia!!
        out = Value(invtan, (self, ))
        out._op = 'tanh' # oops
        def _backward():
            self.grad += (1 - invtan**2) * out.grad

        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ))
        out._op = 'exp'
        def _backwards():
            self.grad += out.data * out.grad # derivative of e^x is just e^x so data = grad

        out._backward = _backwards
        return out

    def backwards(self):
        def toposort(g):
            # how do you toposort again??
            # start by dfs-ing
            # this will visit all deepest nodes first i think??
            # so we like that
            # "reverse dfs postorder" hmmm
            # so what this means is if i visit some node, i need to visit all its children first
            # and eventually once ive visited all its
            ordered = []
            seen = set()
            def visit(v):
                if v not in seen:
                    seen.add(v)
                    for child in v._prev:
                        visit(child)
                    ordered.append(v)
            visit(g)
            return ordered
        
        ts = toposort(self)
        self.grad = 1.0
        for value in reversed(ts):
            value._backward()
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,))
    
        def _backwards():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        
        out._backward = _backwards; 
        out._op = f'**{other}'
        return out
    
    def relu(self):
        # derivative of relu is just 0 if before the bend, 1 if after
        out = Value(0.0 if self.data < 0 else self.data, (self,))

        def _backwards():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backwards; out._op = 'ReLU'
        return out
    
    def zero_grad(self):
        # just set every previous value's .grad to 0
        zeroed = set()
        def zero_all_prev(v):
            if v not in zeroed:
                v.grad = 0
                zeroed.add(v)
                for c in v._prev:
                    zero_all_prev(c)
        zero_all_prev(self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    
    def __rsub__(self, other):     return (-self) + other      # for  2 - a
    def __rtruediv__(self, other): return (self**-1) * other    # for  2 / a

# just following along w/ my goat andrej
class Neuron:
    def __init__(self, nin, nonlin=True):
        # make nin weights and one bias 
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # expression is theta_0 + theta_1(x_1) + theta_2(x_2)...
        return act.relu() if self.nonlin else act # and then nonlinearity this thing! chose relu from UDL
    def parameters(self):
        return self.w + [self.b] # just package all the tunable things together
 
class Layer:
    def __init__(self, nin, nout, **kw):
        # keep track of all the neurons in a layer, each with nin inputs :x:>o 
        self.neurons = [Neuron(nin, **kw) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons] # get the outputs for each of the neurons
        return outs[0] if len(outs) == 1 else outs # return an array if more than 1 val, if one just unbox it for convenience
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()] # all the tunable things for the whole layer!
 
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!= len(nouts) - 1)) for i in range(len(nouts))] # wow now we stack layers!!
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0