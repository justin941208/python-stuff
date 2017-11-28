'''
Just for fun: a simple module for automatically calculating the exact gradient
of a real SCALAR function.

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
import sys

NUMS = ['float', 'int']

def message(in_str, n=30):
    out = (
        '=' * n + \
        '\n' * 2 + \
        in_str + \
        '\n' * 2 + \
        '=' * n
        )
    return out

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

def check_type_and_add(sel, input_dict):
    '''
    A utility function that checks whether an input is valid, and adds inputs to
    instance if all inputs are valid.

    '''
    node_name = type(sel).__name__
    for varname in input_dict:
        var = input_dict[varname]
        try:
            if type(var).__name__ in NUMS:
                exec('sel.{} = constant(input_dict["{}"])'.format(varname, varname))
            elif isinstance(var, node):
                exec('sel.{} = input_dict["{}"]'.format(varname, varname))
            else:
                raise TypeError
        except TypeError:
            msg = (
                'Error: Inputs of "{}" node must be of "node", "float", or ' + \
                '"int" type. Got "{}" instead.'
                ).format(node_name, type(var).__name__)
            print(message(msg))
            sys.exit()

class neg(node):
    '''
    Inputs: x
    Returns: a node with value -x

    '''
    def __init__(self, x):
        node.__init__(self)
        input_dict = {'x': x}
        check_type_and_add(self, input_dict)
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
        input_dict = {'x': x, 'y': y}
        check_type_and_add(self, input_dict)
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
        input_dict = {'x': x, 'y': y}
        check_type_and_add(self, input_dict)
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
        input_dict = {'x': x, 'y': y}
        check_type_and_add(self, input_dict)
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

    Raises error if y is 0

    '''
    def __init__(self, x, y):
        node.__init__(self)
        input_dict = {'x': x, 'y': y}
        check_type_and_add(self, input_dict)
        self.graph = self.x.graph + self.y.graph + [self]

    def forward(self):
        try:
            self.value = self.x.value / self.y.value
        except ZeroDivisionError:
            print(message('Error: Cannot divide by zero!'))
            sys.exit()

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
        input_dict = {'x': x}
        check_type_and_add(self, input_dict)
        self.graph = self.x.graph + [self]

    def forward(self):
        self.value = m.exp(self.x.value)

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.value * self.grad

class log(node):
    '''
    Inputs: x
    Returns: a node with value ln(x)

    Raises error if x <= 0

    '''
    def __init__(self, x):
        node.__init__(self)
        input_dict = {'x': x}
        check_type_and_add(self, input_dict)
        self.graph = self.x.graph + [self]

    def forward(self):
        try:
            self.value = m.log(self.x.value)
        except ValueError:
            print(message('Error: Logarithm only takes positive values!'))
            sys.exit()

    def backprop(self):
        if not self.x.is_const: self.x.grad += self.grad / self.x.value

class pwr(node):
    '''
    Inputs: base, power
    Returns: a node with value base ** power

    Raises error if base <= 0

    '''
    def __init__(self, base, power):
        node.__init__(self)
        input_dict = {'base': base, 'power': power}
        check_type_and_add(self, input_dict)
        self.graph = self.base.graph + self.power.graph + [self]

    def forward(self):
        try:
            if self.base.value == 0:
                assert self.power.value > 0
            if not float(self.power.value).is_integer():
                if self.base.value < 0:
                    raise ValueError
            self.value = self.base.value ** self.power.value
        except AssertionError:
            print(message('Error: Power must be positive if base is zero!'))
            sys.exit()
        except ValueError:
            print(message('Error: Base must be non-negative unless power is an integer!'))
            sys.exit()

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
                try:
                    if node.name in self.vars and node is not self.vars[node.name]:
                        raise RuntimeError
                    self.vars[node.name] = node
                except RuntimeError:
                    msg = message(
                        'Error: Variable name "{}" used for different variables!'
                        ).format(node.name)
                    print(msg)
                    sys.exit()

    def forward(self, variables):
        '''
        Inputs: variables and their values
        Returns: final value of the function

        '''
        for var in self.vars:
            try:
                if var not in variables:
                    raise RuntimeError
                self.vars[var].value = variables[var]
            except RuntimeError:
                print(message('Error: Value for variable {} is not given!').format(
                    var
                    ))
                sys.exit()
        for node in self.nodes:
            node.forward()
        return self.nodes[-1].value

    def backprop(self, variables):
        '''
        Returns: the gradient of the function

        '''
        self.vars_grads = dict()
        if len(self.vars) != 0:
            self.nodes[-1].grad = 1
            for node in reversed(self.nodes):
                node.backprop()
                if node.is_var:
                    self.vars_grads['d{}'.format(node.name)] = node.grad
        for var in variables:
            if var not in self.vars:
                self.vars_grads['d{}'.format(var)] = 0
        return self.vars_grads

def evaluate_and_gradient(function, variables=None):
    '''
    Inputs: a function, a dictionary of the names of the required variables and their values
    Returns: the final value of the function and its gradient at the given variable values

    '''
    try:
        if type(function).__name__ in NUMS:
            function = constant(function)
        assert isinstance(function, node)
    except AssertionError:
        print(message('Function must be a node!'))
        sys.exit()
    try:
        if variables is None:
            variables = dict()
        assert type(variables).__name__ == 'dict'
    except AssertionError:
        msg = ('Variables must be put in a dictionary of the form: ' + \
            '\n{variable_name: value, ...}')
        print(message(msg))
        sys.exit()
    g = ComputationalGraph(function)
    value = g.forward(variables)
    return value, g.backprop(variables)

def print_output(f, variables):
    val, grad = evaluate_and_gradient(f, variables)
    print(message('Function Value: {}\nGradient: {}').format(val, grad))

if __name__=='__main__':
    '''
    Test the code here

    Step 1: Define the variables, with unique names
    Example:
        x = variable('x')
        y = variable('y')


    Step 2: Define the function
    Example: f = div(1, add(1, exp(neg(x)))) <=== This corresponds to f = 1 / (1 + exp(-x))


    Step 3: Define the variables dictionary
    Example: variables = {'x': 3, 'y': 2}

    '''
    # Code below
    x = variable('x')
    y = variable('y')
    f = pwr(x, 2)
    variables = {'x': 0, 'y': 2}
    # Code above
    print_output(f, variables)
