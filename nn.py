'''
Just for fun: a simple module for automatically calculating the exact gradient
of a SCALAR function.

Supported operations:

Addition
Subtraction
Multiplication
Division
Exponentiation
Logarithm
Power

'''

import math as m

NUMS = ['float', 'int']

class node(object):
    def __init__(self):
        self.graph = []
        self.is_var = False
        self.is_const = False
        self.grad = 0

    def forward(self):
        pass

    def backprop(self):
        pass

class constant(node):
    def __init__(self, value):
        node.__init__(self)
        self.graph.append(self)
        self.value = value
        self.is_const = True

class variable(node):
    def __init__(self, name):
        node.__init__(self)
        self.name = name
        self.graph.append(self)
        self.is_var = True

class neg(node):
    '''
    Inputs: x
    Returns: a node with value -x

    '''
    def __init__(self, x):
        node.__init__(self)
        if type(x).__name__ in NUMS:
            self.x = constant(x)
        elif isinstance(x, node):
            self.x = x
        else:
            raise TypeError('Inputs must be of node, float, or int type. Got {}'.format(
                type(var).__name__
            ))
        self.graph = self.x.graph + [self]

    def forward(self):
        self.value = -self.x.value

    def backprop(self):
        if not self.x.is_const: self.x.grad -= self.grad

class add(node):
    '''
    Inputs: x, y
    Returns: a node with value x + y

    '''
    def __init__(self, x, y):
        node.__init__(self)
        for var, varname in zip([x, y], ['x', 'y']):
            if type(var).__name__ in NUMS:
                exec('self.{} = constant({})'.format(varname, varname))
            elif isinstance(var, node):
                exec('self.{} = {}'.format(varname, varname))
            else:
                raise TypeError('Inputs must be of node, float, or int type. Got {}'.format(
                    type(var).__name__
                ))
        self.graph = self.x.graph + self.y.graph + [self]

    def forward(self):
        self.value = self.x.value + self.y.value

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.grad
        if not self.y.is_const: self.y.grad += self.grad

class sub(node):
    '''
    Inputs: x, y
    Returns: a node with value x - y

    '''
    def __init__(self, x, y):
        node.__init__(self)
        for var, varname in zip([x, y], ['x', 'y']):
            if type(var).__name__ in NUMS:
                exec('self.{} = constant({})'.format(varname, varname))
            elif isinstance(var, node):
                exec('self.{} = {}'.format(varname, varname))
            else:
                raise TypeError('Inputs must be of node, float, or int type. Got {}'.format(
                    type(var).__name__
                ))
        self.graph = self.x.graph + self.y.graph + [self]

    def forward(self):
        self.value = self.x.value - self.y.value

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.grad
        if not self.y.is_const: self.y.grad -= self.grad

class mult(node):
    '''
    Inputs: x, y
    Returns: a node with value x * y

    '''
    def __init__(self, x, y):
        node.__init__(self)
        for var, varname in zip([x, y], ['x', 'y']):
            if type(var).__name__ in NUMS:
                exec('self.{} = constant({})'.format(varname, varname))
            elif isinstance(var, node):
                exec('self.{} = {}'.format(varname, varname))
            else:
                raise TypeError('Inputs must be of node, float, or int type. Got {}'.format(
                    type(var).__name__
                ))
        self.graph = self.x.graph + self.y.graph + [self]

    def forward(self):
        self.value = self.x.value * self.y.value

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.y.value * self.grad
        if not self.y.is_const: self.y.grad += self.x.value * self.grad

class div(node):
    '''
    Inputs: x, y
    Returns: a node with value x / y

    Raises ValueError if y is 0

    '''
    def __init__(self, x, y):
        node.__init__(self)
        for var, varname in zip([x, y], ['x', 'y']):
            if type(var).__name__ in NUMS:
                exec('self.{} = constant({})'.format(varname, varname))
            elif isinstance(var, node):
                exec('self.{} = {}'.format(varname, varname))
            else:
                raise TypeError('Inputs must be of node, float, or int type. Got {}'.format(
                    type(var).__name__
                ))
        self.graph = self.x.graph + self.y.graph + [self]

    def forward(self):
        if self.y.value == 0:
            raise ValueError('Divison by zero.')
        self.value = self.x.value / self.y.value

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.grad / self.y.value
        if not self.y.is_const: self.y.grad -= self.x.value * self.grad / (self.y.value ** 2)

class exp(node):
    '''
    Inputs: x
    Returns: a node with value exp(x)

    '''
    def __init__(self, x):
        node.__init__(self)
        if type(x).__name__ in NUMS:
            self.x = constant(x)
        elif isinstance(x, node):
            self.x = x
        else:
            raise TypeError('Inputs must be of node, float, or int type. Got {}'.format(
                type(var).__name__
            ))
        self.graph = self.x.graph + [self]

    def forward(self):
        self.value = m.exp(self.x.value)

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.value * self.grad

class log(node):
    '''
    Inputs: x
    Returns: a node with value ln(x)

    Raises ValueError if x <= 0

    '''
    def __init__(self, x):
        node.__init__(self)
        if type(x).__name__ in NUMS:
            self.x = constant(x)
        elif isinstance(x, node):
            self.x = x
        else:
            raise TypeError('Inputs must be of node, float, or int type. Got {}'.format(
                type(var).__name__
            ))
        self.graph = self.x.graph + [self]

    def forward(self):
        if self.x.value <= 0:
            raise ValueError('Logarithm only takes positive values. Got {}'.format(
                self.x.value
            ))
        self.value = m.log(self.x.value)

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.grad / self.x.value

class pwr(node):
    '''
    Inputs: base, power
    Returns: a node with value base ** power

    Raises ValueError if base <= 0

    '''
    def __init__(self, base, power):
        node.__init__(self)
        for var, varname in zip([base, power], ['base', 'power']):
            if type(var).__name__ in NUMS:
                exec('self.{} = constant({})'.format(varname, varname))
            elif isinstance(var, node):
                exec('self.{} = {}'.format(varname, varname))
            else:
                raise TypeError('Inputs must be of node, float, or int type. Got {}'.format(
                    type(var).__name__
                ))
        self.graph = self.base.graph + self.power.graph + [self]

    def forward(self):
        if self.base.value <= 0:
            raise ValueError('Base must be positive. Got {}'.format(self.base.value))
        self.value = self.base.value ** self.power.value

    def backprop(self):
        if not self.base.is_const:
            self.base.grad += self.power.value * (self.base.value ** \
                (self.power.value - 1)) * self.grad
        if not self.power.is_const:
            self.power.grad += self.value * m.log(self.base.value) * self.grad

class ComputationalGraph(object):
    '''
    Summarises the function in a computational graph.

    Inputs: a function

    '''
    def __init__(self, loss):
        self.nodes = loss.graph
        self.vars = dict()
        for node in self.nodes:
            node.grad = 0
            if node.is_var:
                if node.name in self.vars and node is not self.vars[node.name]:
                    raise Exception(
                        'Variable name {} is used for different variables.'.format(
                            node.name
                        )
                    )
                else:
                    self.vars[node.name] = node

    def forward(self, variables):
        '''
        Inputs: variables and their values
        Returns: final value of the function

        '''
        for var in self.vars:
            if var not in variables:
                raise Exception('Value for the variable {} is not given.'.format(var))
        for node in self.nodes:
            if node.is_var:
                node.value = variables[node.name]
            node.forward()
        return self.nodes[-1].value

    def backprop(self):
        '''
        Returns: the gradient of the function

        '''
        self.nodes[-1].grad = 1
        self.vars_grads = dict()
        for node in reversed(self.nodes):
            node.backprop()
            if node.is_var:
                self.vars_grads['d{}'.format(node.name)] = node.grad
        return self.vars_grads

def evaluate_and_gradient(function, variables=None):
    '''
    Inputs: a function, a dictionary of the names of the required variables and their values
    Returns: the final value of the function and its gradient at the given variable values

    '''
    if type(function).__name__ in NUMS:
        function = constant(function)
    assert isinstance(function, node), 'Loss function must be a node.'
    if variables is None:
        variables = dict()
    assert type(variables).__name__ == 'dict', \
        'Variables must be put in a dictionary of the form: \n {{variable_name: value, ...}}'
    g = ComputationalGraph(function)
    value = g.forward(variables)
    return value, g.backprop()

def print_output(f, variables):
    val, grad = evaluate_and_gradient(f, variables)
    print('=' * 30 + '\nFunction Value: {}\nGradient: {}\n'.format(val, grad) + '=' * 30)

if __name__=='__main__':
    '''
    Test the code here

    Step 1: Define the variables, with unique names
    '''
    x = variable('x')
    y = variable('y')

    '''
    Step 2: Define the function
    '''
    f = div(1, add(1, exp(neg(x)))) # f = 1 / (1 + exp(-x))

    '''
    Step 3: Define the variables dictionary
    '''
    variables = {'x': 3, 'y': 2}

    '''
    No need to change this part
    '''
    print_output(f, variables)
