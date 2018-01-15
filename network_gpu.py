# -*- coding: UTF-8 -*-
##--------------------------------------
# Autor: Èric Mor
# Institut Salvador Vilaseca
# Curs 2017-2018
# 
# Aquest arxiu requereix les següents llibreries:
#  - theano
#  - numpy
#
# Aquest arxiu permet crear xarxes neuronals que funcionin amb la GPU. La xarxa funciona
# encara que l'ordinador no tingui suport per executar el codi en la targeta gràfica. 
# Es millor utilitzar aquest arxiu que network.py, ja que aquest té varies funcionalitats més.
# Té una classe principal anomenada 'NeuralNetwork' que serveix per guardar l'estructura de la xarxa.
# Es poden utilitzar diversos tipus de capes:
#  - Convolutional
#  - Pooling
#  - FullyConnected
#  - Flatten (per convertir una imatge en una cadena de valors)
#  - Dropout
#
# Les capes Convolutional i FullyConnected accepten un argument on es pot especificar amb quina funció
# s'inicialitzen els paràmetres. Les possibles funcions són:
#  - glorot_initializer
#  - he_initializer
#  - random_initializer
#
# Les capes s'afegeixen a una NeuralNetwork amb el mètode add_layer. Un cop s'han afegit
# totes les capes necessàries, s'executa el mètode 'build()', passant com a argument les dimensions de les dades.
# Per exemple:
# 
# import network_gpu as nn
#
# xarxa = nn.NeuralNetwork()
# xarxa.add_layer(nn.Convolutional('RELU', 64, 3))
# xarxa.add_layer(nn.Flatten())
# xarxa.add_layer(nn.FullyConnected('SIGMOID', 100))
# xarxa.add_layer(nn.FullyConnected('SIGMOID', 10))
# xarxa.build((1, 28, 28))
# 
##--------------------------------------

import theano
import theano.tensor as T

import numpy as np
import math
import struct

from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d

# Les diferents funcions d'activació

def sigmoid(z):
    return 1 / (1 + T.exp(-z))
    
def tanh(z):
    return (2 / (1 + T.exp(-2*z))) - 1

def ReLU(z):
    return 0.5 * (z + abs(z))

def ELU(z, alpha=1):
    return T.switch(z >= 0, z, alpha * T.exp(z) - 1)
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def glorot_initializer(shape, neurons_in, neurons_out):
    n = math.sqrt(2 / (neurons_in + neurons_out))
    return np.random.normal(scale=n, size=shape)
    #return np.random.uniform(-n, n, size=s)
    
def he_initializer(shape, neurons_in, neurons_out):
    n = math.sqrt(2 / neurons_in)
    return np.random.normal(scale=n, size=shape)

def random_initializer(shape, neurons_in, neurons_out):
    return np.random.randn(*shape)
            
# Un diccionari que associa cada nom a la funció corresponent.            
FUNCTIONS = {
    'SIGMOID': sigmoid,
    'TANH': tanh,
    'RELU': ReLU,
    'ELU': ELU,
    'SOFTMAX': softmax
    }


class Layer:
    """
    La base de totes les capes, no s'ha d'utilitzar directament.
    
    Les subclasses tenen aquests mètodes principals:
    
    - build(x, input_shape, for_training=True):
        Inicialitza els paràmetres de la capa i estableix l'operació de theano que representa el que calcula la capa.
        Ha de tornar la variable de theano que surt de la capa i les seves dimensions.
        El paràmetre for_training s'utiltiza per averiguar si s'està construint per a entrenar o per a calcular un valor,
        ja que la capa Dropout actua de manera diferent.
        
    - save(f):
        Guarda els paràmetres de la capa a l'arxiu donat.
        
    - load(f):
        Guarda els paràmetres de la capa a l'arxiu donat.
    """
    def __init__(self):
        self.neural_network = None
        
        # Els paràmetres que poden ser optimitzats. Això ho completa cada classe.
        self.parameters = []
        
        self.name = None
        
        
class NeuralNetwork:
    """
    Una xarxa neuronal que es construeix acumulant capes de diferents tipus.
    """
    def __init__(self):
        self.layers = []
        self.input_data = None
        # Tots els paràmetres que es poden optimitzar.
        self.parameters = []
        # Les dimensions de les dades d'entrada.
        self.input_shape = None
        # La Tensor variable d'entrada.
        self.x = None
        # La Tensor variable de sortida.
        self.y = None
        
        # La funció de 'theano' per a calcular un resultat (feedforward).
        self.f = None
        # La funció de 'theano' per a calcular el cost.
        self.cost_f = None
        # La funció de 'theano' per a entrenar-se amb un batch.
        self.train_batch_f = None
        
    def add_layer(self, layer):
        """
        Afegeix una capa al final d'aquesta xarxa.
        
        Paràmetres
        ----------
        layer : capa
            La capa que s'afegeix.
        """
        self.layers.append(layer)
        layer.neural_network = self
        
    def build(self, input_shape):
        """
        Inicialitza tots els paràmetres de la xarxa i crea totes les funcions necessàries. Pot trigar una mica.
        
        Paràmetres
        ----------
        input_shape : tuple
            La forma de les dades d'entrada. Per exemple, (1, 28, 18).
        """
        
        print("#- Construint model...")
        
        # Hem d'afegir None per a representar el que serà la mida del batch.
        self.input_shape = (None,) + input_shape
        
        # Si l'entrada té una dimensió, la guardem en una matriu de forma (batch_size, input_shape)
        # Si l'entrada té tres dimensions, la guardem en un tensor4 de forma (batch_size, input_shape[0], input_shape[1], input_shape[2])
        if len(input_shape) == 1:   
            self.x = T.matrix('x')
        elif len(input_shape) == 3:
            self.x = T.tensor4('x')
        else:
            raise NameError("La 'shape' d'entrada no es valida.")
        
        # Construim totes les capes necessàries per a entrenar.
        x = self.x
        input_shape = self.input_shape
        for i in range(len(self.layers)):
            self.layers[i].name = "%s_%d" % (self.layers[i].__class__.__name__, i)
            x, input_shape = self.layers[i].build(x, input_shape, for_training=True)
            
        self.train_y = x
        
        # Hem de tornar a construir totes les capes per a calcular el valor, ja que 
        # el Dropout funciona de manera diferent en els dos.
        x = self.x
        input_shape = self.input_shape
        for i in range(len(self.layers)):
            x, input_shape = self.layers[i].build(x, input_shape, for_training=False)
            
        self.y = x
        # Els valors reals, s'utilitza en l'entrenament. Ha de tenir les mateixes dimensions que el valor de sortida de la xarxa.
        self.real_output = T.TensorType(self.y.dtype, self.y.broadcastable)('real_output')
        
        # La funció per calcular els valors de sortida.
        self.f = theano.function([self.x], self.y)
            
        # Posem en una llista tots els paràmetres que es poden optimitzar.
        for layer in self.layers:
            self.parameters.extend(layer.parameters)
            
        # Mean squared error
        # En fem dues, una per a l'entrenament i l'altra per a calcular el cost fora de l'entrenament
        self.train_cost = T.mean(((self.real_output - self.train_y) ** 2))
        self.cost = T.mean(((self.real_output - self.y) ** 2))
        
        # Una funció per calcular el cost.
        self.cost_f = theano.function([self.x, self.real_output], self.cost)
        
        # Variables que encara no sabem: la mida del 'batch' i la taxa d'aprenentatge.
        self.mini_batch_size = T.fscalar()
        self.learning_rate = T.fscalar()
        
        
        print("#- Preparant entrenament...")
        
        # Calculem les derivades per a tots els paràmetres.
        gradients = [T.grad(self.train_cost, parameter) for parameter in self.parameters]
        
        # Això actualitzarà els weights i biases després de processar cada 'batch'.
        updates = [(parameter, parameter - self.learning_rate / self.mini_batch_size * gradient) for parameter, gradient in zip(self.parameters, gradients)]
        
        # La funcionar per entrenar un batch.
        self.train_batch_f = theano.function([self.x, self.real_output, self.learning_rate, self.mini_batch_size], self.train_cost, updates=updates)
        
    def calculate_cost(self, inputs, expected_outputs):
        """
        Calcula el cost mitjà total per a tots els valors donats.
        """
        return self.cost_f(inputs, expected_outputs)
        
    def save(self, path):
        """
        Guarda els paràmetres de la xarxa a l'arxiu especificat.
        """
        with open(path, 'w+b') as f:
            for layer in self.layers:
                layer.save(f)
                
    def load(self, path):
        """
        Carrega els paràmetres de la xarxa desde l'arxiu especificat.
        """
        with open(path, 'r+b') as f:
            for layer in self.layers:
                layer.load(f)
        
    def feedforward(self, input_data):
        """ 
        Calcula els valors de sortida per als valors d'entrada donats. En pot calcular molts a la vegada.
        
        Paràmetres
        ----------
        input_shape : array de numpy
            Les dades d'entrada.
        """
        return self.f(input_data)
    
    def train(self, inputs, expected_outputs, mini_batch_size, learning_rate, num_examples, example_start = 0):
        """
        Entrena la xarxa amb els exemples donats.
        
        Paràmetres
        ----------
        inputs : array de numpy
            Les dades d'entrada.
        expected_outputs : array de numpy
            Les dades que s'esperen de sortida. Han de tenir el mateix ordre que els 'inputs'.
        mini_batch_size : int
            La mida de cada 'batch' (conjunt d'exemples que s'entrenen alhora).
        learning_rate : float
            La taxa d'aprenentatge.
        num_examples : int
            El nombre d'exemples amb els que s'ha d'entrenar (això es per si no s'entrenen tots de cop).
        example_start : int, opcional
            L'índex del que serà el primer exemple (això es per si no s'entrenen tots de cop).
        """
        
        example_end = example_start + num_examples
        
        for i in range(example_start, example_end, mini_batch_size):
            if (i+mini_batch_size) <= example_end:
                self.train_batch_f(inputs[i : i+mini_batch_size], expected_outputs[i : i+mini_batch_size], learning_rate, mini_batch_size)
            else:
                self.train_batch_f(inputs[i : example_end], expected_outputs[i : example_end], learning_rate, example_end - i)
    
    
# Una capa senzilla que afegim per conveniència: només transforma una matriu (que es com arriben els valors)
# en una array d'una sola dimensió, de manera que pot ser utilitzada per la capa FullyConnected
class Flatten(Layer):
    def __init__(self):
        super().__init__()
    
    def build(self, x, input_shape, for_training=True):
        n = input_shape[1] * input_shape[2] * input_shape[3]
        
        return T.reshape(x, (x.shape[0], n)), (input_shape[0], n)
    
    def save(self, f):
        pass
                
    def load(self, f):
        pass
    
    
class Convolutional(Layer):
    """
    Una capa convolucional. 
    
    Hiperparàmetres
    ----------
    activation_function : string
        La funció d'activació, e.g. 'SIGMOID', 'RELU', None,.. 
    num_kernels : int
        La quantitat de filtres que té la capa. 
    kernel_size: int
        La mida dels filtres de la capa; només hi ha suport per filtres quadrats.
    stride : int, opcional
        També conegut com a 'subsampling', utilitzat per reduir la mida de la imatge.
    initializer : funció, opcional
        El mètode amb el que s'inicialitzen els 'weights'.
    """
    def __init__(self, activation_function, num_kernels, kernel_size, stride=1, initializer=glorot_initializer):
        super().__init__()
        
        self.initializer = initializer
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        
        if activation_function is not None:
            self.activation_function = FUNCTIONS[activation_function]
        else:
            self.activation_function = None
        
        # Els filtres, amb la forma (output channels, input channels, filter rows, filter columns)
        self.w = None
        # Els llindars d'activació
        self.b = None
        
    # La forma d'entrada és (batch size, input channels, input rows, input columns)
    def build(self, x, input_shape, for_training=True):
        # El paràmetre 'filter_shape' en el mètode conv2d és (output channels, input channels, filter rows, filter columns)
        filter_shape = (self.num_kernels, input_shape[1], self.kernel_size, self.kernel_size)
        
        if for_training:
            neurons_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
            neurons_out = filter_shape[0] * filter_shape[2] * filter_shape[3]
            
            self.w = theano.shared(self.initializer(filter_shape, neurons_in, neurons_out).astype(dtype=theano.config.floatX), name='%s_w' % self.name, borrow=True)
            self.b = theano.shared(np.zeros(self.num_kernels, dtype=theano.config.floatX), name='%s_b' % self.name, borrow=True)
            
            # Els paràmetres que es poden optimitzar.
            self.parameters = [self.w, self.b]
        
        # Per a poder afegir el 'bias', l'hem d'extendre per a que sigui un tensor4 amb els valors a la segona dimensió
        output = conv2d(input=x, filters=self.w, 
                   input_shape=input_shape, filter_shape=filter_shape,
                   subsample=(self.stride, self.stride), filter_flip=True,
                   border_mode='half') + self.b.dimshuffle('x', 0, 'x', 'x')
        
        if self.activation_function is not None:
            output = self.activation_function(output)
            
        return output, (input_shape[0], self.num_kernels, int(input_shape[2] / self.stride), int(input_shape[3] / self.stride))
    
    def save(self, f):
        w_shape = self.w.get_value(borrow=True, return_internal_type=True).shape
        b_shape = self.b.get_value(borrow=True, return_internal_type=True).shape
        weights_format = "<" + str(np.array(w_shape).prod()) + "d"
        biases_format = "<" + str(np.array(b_shape).prod()) + "d"  
    
        f.write(struct.pack(weights_format, *self.w.get_value(borrow=True).flat))
        f.write(struct.pack(biases_format, *self.b.get_value(borrow=True).flat))
                
    def load(self, f):
        w_shape = self.w.get_value(borrow=True, return_internal_type=True).shape
        b_shape = self.b.get_value(borrow=True, return_internal_type=True).shape
        
        weights_format = "<" + str(np.array(w_shape).prod()) + "d"
        biases_format = "<" + str(np.array(b_shape).prod()) + "d"  
        weights_size = struct.calcsize(weights_format)
        biases_size = struct.calcsize(biases_format)
        
        self.w.set_value(np.asarray(struct.unpack(weights_format, f.read(weights_size)), 
                                    dtype=theano.config.floatX).reshape(w_shape), borrow=True)  # @UndefinedVariable
        self.b.set_value(np.asarray(struct.unpack(biases_format, f.read(biases_size)), 
                                    dtype=theano.config.floatX).reshape(b_shape), borrow=True)  # @UndefinedVariable
        
        
class FullyConnected(Layer):
    """
    Una capa densa, totalment connectada. 
    
    Hiperparàmetres
    ----------
    activation_function : string
        La funció d'activació, e.g. 'SIGMOID', 'RELU', None,.. 
    num_neurons : int
        La quantitat de neurones de la capa.
    initializer : funció, opcional
        El mètode amb el que s'inicialitzen els 'weights'.
    """
    def __init__(self, activation_function, num_neurons, initializer=glorot_initializer):
        super().__init__()
        
        self.initializer = initializer
        self.num_neurons = num_neurons
        
        # Els pesos que conecten la capa anterior amb aquesta.
        self.w = None
        # Els llindars d'activació de les neurones d'aquesta capa.
        self.b = None
        
        if activation_function is not None:
            self.activation_function = FUNCTIONS[activation_function]
        else:
            self.activation_function = None
        
        
    def build(self, x, input_shape, for_training=True):
        
        if for_training:
            neurons_in = input_shape[1] 
            neurons_out = self.num_neurons
                
            weights = self.initializer((input_shape[1], self.num_neurons), neurons_in, neurons_out).astype(
                dtype=theano.config.floatX)  # @UndefinedVariable    
            biases = np.zeros((self.num_neurons,), dtype=theano.config.floatX)  # @UndefinedVariable
            
            # Enviem els 'weights' i 'biases' a theano per a que es puguin utilitzar a la GPU.
            self.w = theano.shared(weights, name='%s_w' % self.name, borrow=True)
            self.b = theano.shared(biases, name='%s_b' % self.name, borrow=True)
            
            # Els paràmetres que es poden optimitzar
            self.parameters = [self.w, self.b]
        
        output = T.dot(x, self.w) + self.b
        
        if self.activation_function is not None:
            output = self.activation_function(output)
        
        # La sortida és una matriu de dimensions (mida de batch, nombre de neurones)
        return output, (input_shape[0], self.num_neurons)
    
    def save(self, f):
        w_shape = self.w.get_value(borrow=True, return_internal_type=True).shape
        b_shape = self.b.get_value(borrow=True, return_internal_type=True).shape
        weights_format = "<" + str(np.array(w_shape).prod()) + "d"
        biases_format = "<" + str(np.array(b_shape).prod()) + "d"  
    
        f.write(struct.pack(weights_format, *self.w.get_value(borrow=True).flat))
        f.write(struct.pack(biases_format, *self.b.get_value(borrow=True).flat))
                
    def load(self, f):
        w_shape = self.w.get_value(borrow=True, return_internal_type=True).shape
        b_shape = self.b.get_value(borrow=True, return_internal_type=True).shape
        
        weights_format = "<" + str(np.array(w_shape).prod()) + "d"
        biases_format = "<" + str(np.array(b_shape).prod()) + "d"  
        weights_size = struct.calcsize(weights_format)
        biases_size = struct.calcsize(biases_format)
        
        self.w.set_value(np.asarray(struct.unpack(weights_format, f.read(weights_size)), 
                                    dtype=theano.config.floatX).reshape(w_shape), borrow=True)  # @UndefinedVariable
        self.b.set_value(np.asarray(struct.unpack(biases_format, f.read(biases_size)), 
                                    dtype=theano.config.floatX).reshape(b_shape), borrow=True)  # @UndefinedVariable
    
    
# Una capa de 'pooling'    o 'subsampling'.
class Pooling(Layer):
    """
    Una capa d'agurpació (coneguda com a 'pooling' o 'subsampling'). 
    
    Hiperparàmetres
    ----------
    stride : int
        La quantitat que avança la regió d'agrupació cada cop.
    extent : int
        La mida de la regió d'agrupació. Només hi ha suport per regions quadrades.
    """
    def __init__(self, stride, extent):
        super().__init__()
        
        self.stride = stride
        self.extent = extent
        
    def build(self, x, input_shape, for_training=True):
        
        width = (input_shape[2] - self.extent) // self.stride + 1
        height = (input_shape[3] - self.extent) // self.stride + 1
        
        return pool_2d(x, ws=(self.extent, self.extent), stride=(self.stride, self.stride), ignore_border=True), (input_shape[0], input_shape[1], width, height)
    
    def save(self, f):
        pass
                
    def load(self, f):
        pass


class Dropout(Layer):
    """
    Aplica un dropout a les dades que li entren.
    
    Hiperparàmetres
    ----------
    p : float
        La probabilitat que té una neurona de desactivarse, en tant per 1. 
        Es pot entendre com el tant per 1 de neurones que es desactiven (i.e. 0.5 es desactiven la meitat).
    """
    def __init__(self, p):
        super().__init__()
        
        self.p = p
        
    def build(self, x, input_shape, for_training=True):
        
        # Generem un RandomStream amb una 'seed' aleatòria.
        random_stream = T.shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
        
        mask = random_stream.binomial(size=x.shape, n=1, p=(1 - self.p))
        
        # Si no estem entrenant no s'aplica dropout
        if for_training:
            return x * T.cast(mask, theano.config.floatX) / (1 - self.p), input_shape
        else:
            return x, input_shape
    
    def save(self, f):
        pass
                
    def load(self, f):
        pass
    