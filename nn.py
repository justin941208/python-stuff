import math as m

class node(object):
    def __init__(self):
        self.graph = []
        self.is_var = False
        self.is_const = False

class constant(node):
    def __init__(self, value):
        node.__init__(self)
        self.value = value
        self.graph.append(self)
        self.is_const = True

    def backprop(self):
        pass

class variable(node):
    def __init__(self, name, value):
        node.__init__(self)
        self.name = name
        self.value = value
        self.graph.append(self)
        self.is_var = True
        self.grad = 0

    def backprop(self):
        pass

class add(node):
    def __init__(self, x, y):
        node.__init__(self)
        self.x = x
        self.y = y
        self.graph = self.x.graph + self.y.graph + [self]
        self.value = self.x.value + self.y.value

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.grad
        if not self.y.is_const: self.y.grad += self.grad

class mult(node):
    def __init__(self, x, y):
        node.__init__(self)
        self.x = x
        self.y = y
        self.graph = self.x.graph + self.y.graph + [self]
        self.value = self.x.value * self.y.value

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.y.value * self.grad
        if not self.y.is_const: self.y.grad += self.x.value * self.grad

class power(node):
    def __init__(self, base, power):
        node.__init__(self)
        assert base.value > 0, 'Base must be positive.'
        self.base = base
        self.power = power
        self.graph = self.base.graph + self.power.graph + [self]
        self.value = self.base.value ** self.power.value

    def backprop(self):
        if not self.base.is_const:
            self.base.grad += self.power.value * (self.base.value ** \
                (self.power.value - 1))
        if not self.power.is_const:
            self.power.grad += self.value * m.log(self.base.value)

class ComputationalGraph(object):
    def __init__(self, loss):
        self.nodes = loss.graph
        names = []
        for node in self.nodes:
            if node.is_var:
                if node.name in names:
                    raise Exception('The name {} was used multiple times.'.format(node.name))
                else:
                    names.append(node.name)
        self.values = [node.value for node in self.nodes]

    def backprop(self):
        self.nodes[-1].grad = 1
        out = dict()
        for node in reversed(self.nodes):
            node.backprop()
            if node.is_var:
                out['d{}'.format(node.name)] = node.grad
        return out

if __name__=='__main__':
    def get_loss(x, y, z):
        x = variable('x', x)
        y = variable('y', y)
        z = variable('z', z)
        return power(x, y)
    
    graph = ComputationalGraph(get_loss(5, 3, 6))
    loss = graph.values[-1]
    grads = graph.backprop()
    print('Loss: ', loss)
    print('Gradients: ', grads)
