import numpy as np
import tensorflow as tf
import math

class FeedForwardModel(object):
    def __init__(self, input_shape, output_shape, components):
        self.input_shape = input_shape and \
            (input_shape if (type(input_shape) is list or type(input_shape) is tuple) \
             else [input_shape])
        self.output_shape = output_shape and \
            (output_shape if (type(output_shape) is list or type(output_shape) is tuple) \
             else [output_shape])
        self.components = components
        self.configure()
        
    def configure(self):
        inferred_input_indices = set()
        inferred_output_indices = set()
        inferred_input_shape = None
        given_input_shape = self.output_shape
        for i in reversed(range(len(self.components))):
            assert self.components[i].output_shape is None or \
                (inferred_input_shape or given_input_shape) is None, \
                'output_shape of ' + type(self).__name__ + ' at index ' + str(i) + \
                    ' can be inferred, and should not be specified'
            self.components[i].output_shape = self.components[i].output_shape or \
                inferred_input_shape or given_input_shape
            if self.components[i].output_shape:
                inferred_input_indices.add(i)
            given_input_shape = self.components[i].input_shape
            inferred_input_shape = self.components[i].output_shape and \
                self.components[i].infer_input_shape(self.components[i].output_shape)
            if i > 0:
                assert given_input_shape is None or inferred_input_shape is None, \
                    'input_shape of ' + type(self).__name__ + ' at index ' + str(i) + \
                        ' can be inferred, and should not be specified'
            if i == 0 and inferred_input_shape:
                assert given_input_shape == inferred_input_shape, \
                    'Model input and output shapes are incompatible given the specified components'
            self.components[i].input_shape = given_input_shape or inferred_input_shape
            if self.components[i].input_shape:
                inferred_output_indices.add(i)
        inferred_output_shape = None
        given_output_shape = self.input_shape
        for i in range(len(self.components)):
            assert self.components[i].input_shape is None or \
                (inferred_output_shape or given_output_shape) is None or \
                i in inferred_input_indices, \
                'input_shape of ' + type(self).__name__ + ' at index ' + str(i) + \
                    ' can be inferred, and should not be specified'
            self.components[i].input_shape = self.components[i].input_shape or \
                inferred_output_shape or given_output_shape
            given_output_shape = self.components[i].output_shape
            inferred_output_shape = self.components[i].input_shape and \
                self.components[i].infer_output_shape(self.components[i].input_shape)
            if i < len(self.components) - 1:
                assert given_output_shape is None or inferred_output_shape is None or \
                    i in inferred_output_indices, \
                    'output_shape of ' + type(self).__name__ + ' at index ' + str(i) + \
                        ' can be inferred, and should not be specified'
            if i == len(self.components) - 1 and inferred_output_shape:
                assert given_output_shape == inferred_output_shape, \
                    'Model input and output shapes are incompatible given the specified components'
            self.components[i].output_shape = given_output_shape or inferred_output_shape
                    
        for i in range(len(self.components)):
            assert self.components[i].input_shape, 'input_shape of ' + \
                type(self).__name__ + ' at index ' + str(i) + \
                ' cannot be inferred, and should be specified'
            assert self.components[i].output_shape, 'output_shape of ' + \
                type(self).__name__ + ' at index ' + str(i) + \
                ' cannot be inferred, and should be specified'
            self.components[i].input_size = np.prod(self.components[i].input_shape)
            self.components[i].output_size = np.prod(self.components[i].output_shape)
                
    def build(self):
        x = tf.placeholder(tf.float32, [None] + self.input_shape)
        out = x
        for component in self.components:
            component.initialize()
            out = component.apply(out)
        return x, out
        
    def __repr__(self):
        return '\n'.join(map(str, self.components))
        

class Component(object):
    def __init__(self, params={}, **kwargs):
        self.params = params
        self.kwargs = kwargs
        self.input_shape = self.input_shape and \
            (self.input_shape if type(self.input_shape) is list else [self.input_shape])
        self.output_shape = self.output_shape and \
            (self.output_shape if type(self.output_shape) is list else [self.output_shape])
    
    def initialize(self):
        pass
        
    def infer_output_shape(self, input_shape):
        return None
    def infer_input_shape(self, output_shape):
        return None
    
    def __repr__(self):
        return type(self).__name__ + '(' + \
            'input_shape=' + str(self.input_shape) + \
            ', output_shape=' + str(self.output_shape) + \
            ', '.join([''] + map(lambda k: k + '=' + str(self.params[k]), self.params)) + \
            ', '.join([''] + map(lambda k: k + '=' + str(self.kwargs[k]), self.kwargs)) + \
            ')'
    
    
class FullyConnectedComponent(Component):
    def __init__(self, output_shape=None):
        self.input_shape = None
        self.output_shape = output_shape
        Component.__init__(self)
    
    def initialize(self):
        self.weights = tf.Variable(
            tf.random_normal([self.input_size, self.output_size],
                             stddev=np.sqrt(2./self.input_size)))
        self.biases = tf.Variable(tf.zeros(self.output_size))
        
    def apply(self, x):
        x = tf.reshape(x, [-1, self.input_size])
        x = tf.add(tf.matmul(x, self.weights), self.biases)
        x = tf.reshape(x, [-1] + self.output_shape)
        return x
    
    
class Convolutional1DComponent(Component):
    def __init__(self, filter_width,
                 stride=1, padding='SAME', num_kernels=1, **kwargs):
        self.filter_width = filter_width
        self.stride = stride
        self.padding = padding
        self.num_kernels = num_kernels
        self.input_shape = None
        self.output_shape = None
        Component.__init__(self, {'filter_width': filter_width,
                                  'stride': stride,
                                  'padding': padding,
                                  'num_kernels': num_kernels},
                           **kwargs)
    
    def initialize(self):
        in_channels = self.input_shape[1] if len(self.input_shape) == 2 else 1
        self.weights = tf.Variable(
            tf.random_normal([self.filter_width, in_channels, self.num_kernels],
                             stddev=np.sqrt(2./(self.filter_width*in_channels*self.num_kernels))))
        self.biases = tf.Variable(tf.zeros(self.num_kernels))
        
    # Calculate output as a function of stride and padding type
    def infer_output_shape(self, input_shape):
        assert len(self.input_shape) >= 1 and len(self.input_shape) <= 2, \
            'input_shape of ' + type(self).__name__ + ' must be 1D or 2D (channel dimension)'
        includes_channel_dimension = len(self.input_shape) == 2
        if self.padding == 'SAME':
            return [int(math.ceil(input_shape[0] / float(self.stride)))] + \
                ([self.num_kernels] if self.num_kernels > 1 else [])
        else:
            return [int(math.ceil((input_shape[0] - self.filter_width + 1)/float(self.stride)))] + \
                ([self.num_kernels] if self.num_kernels > 1 else [])
        
    def apply(self, x):
        includes_channel_dimension = len(self.input_shape) == 2
        if not includes_channel_dimension:
            x = tf.expand_dims(x, 2)
        x = tf.nn.conv1d(x, self.weights, stride=self.stride, padding=self.padding,
                         **self.kwargs)
        x = tf.nn.bias_add(x, self.biases)
        if not includes_channel_dimension and self.num_kernels == 1:
            x = tf.squeeze(x, 2)
        return x
    
    
class Convolutional2DComponent(Component):
    def __init__(self, filter_size,
                 stride=None, strides=[1, 1], padding='SAME', num_kernels=1, **kwargs):
        if type(filter_size) is not list and type(filter_size) is not tuple:
            filter_size = [filter_size] * 2
        self.filter_width = filter_size[0]
        self.filter_height = filter_size[1]
        if stride is not None:
            strides = [stride] * 2
        assert len(strides) == 2
        self.strides = strides
        self.padding = padding
        self.num_kernels = num_kernels
        self.input_shape = None
        self.output_shape = None
        Component.__init__(self, {'filter_size': filter_size,
                                  'strides': strides,
                                  'padding': padding,
                                  'num_kernels': num_kernels},
                           **kwargs)
    
    def initialize(self):
        in_channels = self.input_shape[2] if len(self.input_shape) == 3 else 1
        self.weights = tf.Variable(
            tf.random_normal(
                [self.filter_width, self.filter_height, in_channels, self.num_kernels],
                stddev=np.sqrt(
                    2./(self.filter_width*self.filter_height*in_channels*self.num_kernels))))
        self.biases = tf.Variable(tf.zeros(self.num_kernels))
        
    # Calculate output as a function of strides and padding type
    def infer_output_shape(self, input_shape):
        assert len(self.input_shape) >= 2 and len(self.input_shape) <= 3, \
            'input_shape of ' + type(self).__name__ + ' must be 2D or 3D (channel dimension)'
        includes_channel_dimension = len(self.input_shape) == 3
        if self.padding == 'SAME':
            return [int(math.ceil(input_shape[0] / float(self.strides[0]))),
                    int(math.ceil(input_shape[1] / float(self.strides[1])))] + \
                ([self.num_kernels] if self.num_kernels > 1 else [])
        else:
            return [int(math.ceil(
                        (input_shape[0] - self.filter_width + 1)/float(self.strides[0]))),
                    int(math.ceil(
                        (input_shape[1] - self.filter_height + 1)/float(self.strides[1])))] + \
                    ([self.num_kernels] if self.num_kernels > 1 else [])
        
    def apply(self, x):
        includes_channel_dimension = len(self.input_shape) == 3
        if not includes_channel_dimension:
            x = tf.expand_dims(x, 3)
        x = tf.nn.conv2d(x, self.weights,
                         strides=([1]+self.strides+[1]), padding=self.padding,
                         **self.kwargs)
        x = tf.nn.bias_add(x, self.biases)
        if not includes_channel_dimension and self.num_kernels == 1:
            x = tf.squeeze(x, 3)
        return x
    
    
class DropoutComponent(Component):
    def __init__(self, keep_prob, **kwargs):
        self.keep_prob = keep_prob
        self.input_shape = None
        self.output_shape = None
        Component.__init__(self, {'keep_prob': keep_prob},
                           **kwargs)
        
    # Identical input and output shapes
    def infer_output_shape(self, input_shape):
        return input_shape
    def infer_input_shape(self, output_shape):
        return output_shape
        
    def apply(self, x):
        x = tf.nn.dropout(x, self.keep_prob, **self.kwargs)
        return x
    
    
class ReshapeComponent(Component):
    def __init__(self, output_shape, **kwargs):
        self.input_shape = None
        self.output_shape = output_shape
        Component.__init__(self, **kwargs)
        
    def initialize(self):
        assert self.input_size == self.output_size, 'Cannot reshape ' + \
            str(self.input_shape) + ' to ' + str(self.output_shape)
        
    def apply(self, x):
        x = tf.reshape(x, [-1] + self.output_shape, **self.kwargs)
        return x
    
    
# Could be implemented using a CustomComponent, but kept as an alias
class ActivationComponent(Component):
    def __init__(self, activation_fn, **kwargs):
        self.activation_fn = activation_fn
        self.input_shape = None
        self.output_shape = None
        Component.__init__(self, {'activation_fn': activation_fn.__name__},
                           **kwargs)
        
    # Activation functions have identical input and output shapes
    def infer_output_shape(self, input_shape):
        return input_shape
    def infer_input_shape(self, output_shape):
        return output_shape
        
    def apply(self, x):
        x = self.activation_fn(x)
        return x
    
    
# CustomComponents must retain the shape of the input
class CustomComponent(Component):
    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.input_shape = None
        self.output_shape = None
        Component.__init__(self, **kwargs)
        
    # Must have identical input and output shapes, else inference is impossible
    def infer_output_shape(self, input_shape):
        return input_shape
    def infer_input_shape(self, output_shape):
        return output_shape
        
    def apply(self, x):
        x = self.fn(x, **self.kwargs)
        return x
    

