# -*- coding: UTF-8 -*-

##--------------------------------------
# Autor: Èric Mor
# Institut Salvador Vilaseca
# Curs 2017-2018
# 
# Aquest arxiu requereix les següents llibreries:
#  - scipy
#  - numpy
#
# Aquest arxiu permet crear xarxes neuronals convolucionals. 
# Té una classe principal anomenada 'NeuralNetwork' que serveix per guardar l'estructura de la xarxa.
# Es poden utilitzar diversos tipus de capes:
#  - Convolutional
#  - Pooling
#  - FullyConnected
#  - Flatten (per convertir una imatge en una cadena de valors)
#
# Les capes s'afegeixen a una NeuralNetwork amb el mètode add_layer. Un cop s'han afegit
# totes les capes necessàries, s'executa el mètode 'build()', passant com a argument les dimensions de les dades.
# Per exemple:
# 
# import network as nn
#
# xarxa = nn.NeuralNetwork()
# xarxa.add_layer(nn.Convolutional('RELU', 64, 3))
# xarxa.add_layer(nn.Flatten())
# xarxa.add_layer(nn.FullyConnected('SIGMOID', 100))
# xarxa.add_layer(nn.FullyConnected('SIGMOID', 10))
# xarxa.build((1, 28, 28))
# 
##--------------------------------------

# Per a fer operacions amb matrius, vectors,...
import numpy as np
# Per a la convolució.
from scipy.signal import convolve2d
# Per a guardar/cargar els paràmetres d'un model.
from io import open
import struct
# Per a calcular arrels quadrades
import math


# Les funcions d'activació i les seves derivades.

def sigmoid(z):
	return 1 / (1 + np.exp(-z))
	
def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))
	
def tanh(z):
	return (2 / (1 + np.exp(-2*z))) - 1

def tanh_derivative(z):
	return 1 - (tanh(z)**2)
	
def ReLU(z):
	return np.maximum(z, 0)

def ReLU_derivative(z):
	return np.maximum(z, 0)

def ELU(z):
	x = np.copy(z)
	neg_indices = z < 0
	
	x[neg_indices] = (np.exp(x[neg_indices]) - 1)
	return x
	
def ELU_derivative(z):
	x = np.copy(z)
	x[x<0] = np.exp(x[x<0])
	return x
		
# Un diccionari que associa cada nom a la funció corresponent.	
FUNCTIONS = {
	'SIGMOID': sigmoid,
	'TANH': tanh,
	'RELU': ReLU,
	'ELU': ELU,
	}
	
# Un diccionari que associa cada nom a la funció derivada corresponent.
DERIVATIVES = {
	'SIGMOID': sigmoid_derivative,
	'TANH': tanh_derivative,
	'RELU': ReLU_derivative,
	'ELU': ELU_derivative
	}
	
	
def cost_function(real_output, estimated_output):
	"""
	Calcula el valor de la funció de cost.
	
	Paràmetres
	----------
	real_output : vector
	    Les dades de sortida esperades.
	estimated_output : vector
	    Les dades de sortida generades per la xarxa.
	"""
	return ((real_output - estimated_output) ** 2) / 2


def cost_derivative(real_output, estimated_output):
	"""
	Calcula el la derivada de la funció de cost.
	
	Paràmetres
	----------
	real_output : vector
	    Les dades de sortida esperades.
	estimated_output : vector
	    Les dades de sortida generades per la xarxa.
	"""
	return -(real_output - estimated_output)
	
	
class Layer:
	"""
	La base de totes les capes, no s'ha d'utilitzar directament.
	
	Les subclasses tenen aquests mètodes principals:
	
	- build(input_shape):
		Inicialitza els paràmetres de la capa. Rep les dimensions de les dades que entren i
		ha de tornar les dimensions de les dades que surten.
		
	- feedforward(input_data):
		Calcula el resultat de la capa.
		
	- backpropagate(delta):
		Calcula els gradients per a la senyal d'error que rep. Ha de tornar la senyal d'error propagada.
		
	- apply_gradient(learning_rate):
		Aplica tots els gradients acumulats als paràmetres amb la taxa d'aprenentatge donada.
		
	- save(f):
		Guarda els paràmetres de la capa a l'arxiu donat.
		
	- load(f):
		Guarda els paràmetres de la capa a l'arxiu donat.
	"""
	def __init__(self):
		self.neural_network = None
		# L'activació de la capa l'últim cop que s'ha utilitzat.
		self.activation = None
	
	
class Input(Layer):
	def __init__(self):
		super().__init__()
		self.input_shape = None
	
	def build(self, input_shape):
		self.input_shape = input_shape
		return input_shape
		
	def feedforward(self, input_data):
		return input_data
	
	def backpropagate(self, delta):
		return delta
	
	def apply_gradient(self, learning_rate):
		pass
	
	def save(self, f):
		pass
				
	def load(self, f):
		pass
	
	
class NeuralNetwork:
	"""
	Una xarxa neuronal que es construeix acumulant capes de diferents tipus.
	"""
	def __init__(self):
		self.layers = []
		self.input_data = None
		
	def build(self, input_shape):
		"""
		Inicialitza tots els paràmetres de la xarxa.
		
		Paràmetres
		----------
		input_shape : tuple
		    La forma de les dades d'entrada. Per exemple, (1, 28, 18).
		"""
		for layer in self.layers:
			input_shape = layer.build(input_shape)
		
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
		
	def next_layer(self, layer):
		"""
		Torna la capa que ve després de la capa especificada, o None si
		és la última capa.
		
		Paràmetres
		----------
		layer : capa
		    La capa de la qual es vol saber la capa que ve després.
		"""
		index = self.layers.index(layer)
		
		# Si l'index ja es l'útlima capa
		if index == len(self.layers) - 1:
			return None
		else:
			return self.layers[index + 1]
		
	def previous_layer(self, layer):
		"""
		Torna la capa que ve abans de la capa especificada.
		
		Paràmetres
		----------
		layer : capa
		    La capa de la qual es vol saber la capa que ve abans.
		"""
		
		index = self.layers.index(layer)
		
		# Si l'index es la primera capa, tornem sempre alguna cosa.
		if index == 0:
			layer = Input()
			layer.activation = self.input_data
			return layer
		else:
			return self.layers[index - 1]
		
	def save(self, path):
		"""
		Guarda els paràmetres d'aquesta xarxa en un arxiu.
		
		Paràmetres
		----------
		path : string
		    La ruta al arxiu.
		"""
		with open(path, 'w+b') as f:
			for layer in self.layers:
				layer.save(f)
				
	def load(self, path):
		"""
		Carrega els paràmetres d'aquesta xarxa des d'un arxiu.
		
		Paràmetres
		----------
		path : string
		    La ruta al arxiu.
		"""
		with open(path, 'r+b') as f:
			for layer in self.layers:
				layer.load(f)
				
	def from_gpu(self, net):
		"""
		Carrega els paràmetres d'una xarxa neuronal creada amb 'network_gpu'.
		
		Paràmetres
		----------
		path : network_gpu.NeuralNetwork
		    La xarxa neuronal.
		"""
		for layer, gpu_layer in zip(self.layers, net.layers):
			layer.from_gpu(gpu_layer)
		
	def feedforward(self, input_data):
		"""
		Calcula el resultat de la capa de sortida a partir de les dades d'entrada específicades.
		
		Paràmetres
		----------
		input_data : array de numpy
		    Les dades d'entrada.
		"""
		self.input_data = input_data
		
		for layer in self.layers:

			input_data = layer.feedforward(input_data)
			layer.activation = input_data
			
		return self.layers[-1].activation
	
	def backpropagate(self, estimated_output, real_output):
		"""
		Calcula la quantitat que han de canviar els paràmetres amb l'agoritme de retropropagació.
		Això no aplica els canvis, ja que aquest mètode es pot utilitzar per anar acumulant gradients.
		
		Paràmetres
		----------
		estimated_output : array de numpy
		    Les dades de sortida reals.
		real_output : array de numpy
		    Les dades de sortida generades amb la xarxa.
		"""
		delta = cost_derivative(real_output, estimated_output)
		
		self.initial_delta = np.copy(delta)
		
		for layer in reversed(self.layers):
			
			delta = layer.backpropagate(delta)
			
	def apply_gradient(self, learning_rate):
		"""
		Aplica les gradients de tots els paràmetres de la xarxa.
		
		Paràmetres
		----------
		learning_rate : float
			La taxa d'aprenentatge. Si s'utilitza aprenentatge per 'batch', 
			s'ha de dividir per la mida del 'batch' abans.
		"""
		for layer in self.layers:
			layer.apply_gradient(learning_rate)

	
# Una capa convolucional.
# Aplica un o més filtres (kernels) a l'entrada mitjançant una convolució.
# Hi ha uns hiperparàmetres fixes (per a facilitar l'implementació): padding=1, stride=1
class Convolutional(Layer):
	"""
	Una capa convolucional. 
	
	Hiperparàmetres
	----------
	function : string
		La funció d'activació, e.g. 'SIGMOID', 'RELU', None,.. 
	num_kernels : int
		La quantitat de filtres que té la capa. 
	kernel_size: int
		La mida dels filtres de la capa; només hi ha suport per filtres quadrats.
	stride : int, opcional
		També conegut com a 'subsampling', utilitzat per reduir la mida de la imatge.
	"""
	def __init__(self, function, num_kernels, kernel_size, stride=1):
		super().__init__()
		self.function = FUNCTIONS[function] if function is not None else None
		self.derivative = DERIVATIVES[function] if function is not None else None
		self.num_kernels = num_kernels
		self.kernel_size = kernel_size
		self.stride = stride
		
		self.kernels = None
		self.biases = None
		
		self.kernels_gradient = []
		self.biases_gradient = None
		
		self.input_shape = None
		
		# El valor de la activació abans de passar per la funció d'activació
		self.z = None
		# Els valors d'entrada, s'utilitza per a la backpropagation
		self.input_data = None
		
	def build(self, input_shape):
		n = math.sqrt(2 / (input_shape[1] * input_shape[2]))
		
		# El nombre de filtres es multiplica per a cada canal que té la imatge
		self.kernels = [np.random.random((self.kernel_size, self.kernel_size)) * n for _ in range(self.num_kernels * input_shape[0])]
		
		self.biases = np.zeros(self.num_kernels)
		
		self.biases_gradient = np.zeros(self.biases.shape)
		
		self.input_shape = input_shape
		
		return (self.num_kernels, input_shape[1] // self.stride, input_shape[2] // self.stride)
		#return (13, 13, self.num_kernels)
		
	def feedforward(self, input_data):
		self.input_data = input_data
		
		# La quantitat de canals que té l'entrada.
		channels = input_data.shape[0]
		
		# El volum de sortida és igual que el de l'entrada excepte per la profunditat, que és igual al nombre de filtres assignat.
		output = np.zeros((self.num_kernels, input_data.shape[1], input_data.shape[2]))
		
		# Hem de fer una convolució per cada filtre, però cada filtre té una versió diferent per a cada canal.
		for i in range(self.num_kernels):
			
			for j in range(channels):
				kernel = self.kernels[i * channels + j]
				# Els resultats de la convolució dels diferents canals es suma.
				output[i] += convolve2d(input_data[j], kernel, mode='same', boundary='symm')
				
			output[i] /= channels
				
			# Finalment, sumem el llindar d'activació.
			output[i,:,:] += self.biases[i]
			
		if self.stride != 1:
			
			output = output[:, ::self.stride, ::self.stride]
			
		self.z = output
		
		# Apliquem la funció d'activació (si n'hi ha)
		if self.function is not None:
			output = self.function(output)
							
		return output
	
	def backpropagate(self, delta):
		
		if self.derivative is not None:
			delta *= self.derivative(self.z)
			
		self.delta = delta
			
		num_channels = self.input_shape[0]
		
		new_delta = np.zeros(self.input_shape)
			
		# delta es calcula amb una convolució amb els filtres girats 180º
		for c in range(num_channels):
			for k in range(self.num_kernels):
				new_delta[c,::self.stride,::self.stride] += convolve2d(delta[k,:,:], np.rot90(self.kernels[k * num_channels + c], 2), mode='same', boundary='symm')
				
				
		# Si el stride és 1, el gradient és una convolució
		# Si no, és una convolució amb dilatació.
		
		if self.stride == 1:
			for k in range(self.num_kernels):
				for c in range(num_channels):
					convolve_result = convolve2d(self.delta[k], self.input_data[c], mode='valid', boundary='symm')
					
					self.kernels_gradient.append(convolve_result)
					
		else:
			for k in range(self.num_kernels):
				for c in range(num_channels):
					self.kernels_gradient.append(np.dot(self.delta[k].ravel(), self.input_data[c, ::self.stride, ::self.stride,].ravel()))
				
		self.biases_gradient += np.sum(self.delta, axis=(1, 2))
		
		return new_delta
	
	def apply_gradient(self, learning_rate):
		
		num_channels = self.input_data.shape[0]
		for k in range(self.num_kernels):
			for c in range(num_channels):
				self.kernels[k * num_channels + c] -= learning_rate * self.kernels_gradient[k * num_channels + c]
		
		self.biases -= learning_rate * self.biases_gradient
		
		del self.kernels_gradient[:]
		self.biases_gradient = np.zeros(self.biases.shape)
	
	def save(self, f):
		data_format = "<" + str(self.kernel_size * self.kernel_size) + "d" 
		
		for i in range(len(self.kernels)):
			f.write(struct.pack(data_format, *self.kernels[i].flat))
			
		data_format = "<" + str(self.num_kernels) + "d" 
		f.write(struct.pack(data_format, *self.biases.flat))
				
	def load(self, f):
		data_format = "<" + str(self.kernel_size * self.kernel_size) + "d" 
		size = struct.calcsize(data_format)
		
		for i in range(len(self.kernels)):
			self.kernels[i] = np.array(struct.unpack(data_format, f.read(size))).reshape(self.kernels[i].shape)
			
		biases_format = "<" + str(self.biases.size) + "d"  
		biases_size = struct.calcsize(biases_format)
		
		self.biases = np.array(struct.unpack(biases_format, f.read(biases_size))).reshape(self.biases.shape)
			
	def from_gpu(self, net):
		self.biases = net.b.get_value()
		
		weights = net.w.get_value()
		
		num_channels = self.input_shape[0]
		for k in range(self.num_kernels):
			for c in range(num_channels):
				self.kernels[k * num_channels + c] = weights[k][c]

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
		
		# Per a fer la 'backpropagation' hem de guardar quins indexs contenien els valors màxims.
		self.indices = None
		# També necessitarem el volum original.
		self.input_shape = None
		
	def build(self, input_shape):
		self.input_shape = input_shape
		
		width = (input_shape[1] - self.extent) // self.stride + 1
		height = (input_shape[2] - self.extent) // self.stride + 1
		
		# Aquí guardarem els indexs dels valors màxims
		self.indices = np.zeros((input_shape[0], width, height))
		
		return (input_shape[0], width, height)
		
	def feedforward(self, input_data):
	
		# Calculem el volum de sortida
		extent = self.extent
		stride = self.stride
		
		width = (input_data.shape[1] - extent) // stride + 1
		height = (input_data.shape[2] - extent) // stride + 1
		
		output = np.empty((input_data.shape[0], width, height))
		
		for y in range(height):
			for x in range(width):
				
				for c in range(input_data.shape[0]):
					
					# La regió d'on calculem el valor màxim.
					pooling_data = input_data[c, x*stride : x*stride + extent,  y*stride : y*stride + extent]
					
					# Obtenim el index que conté el valor més gran.
					index = np.argmax(pooling_data)
					self.indices[c][x][y] = index
					
					# El index és para una versió plana de la array, així que utilitzem la funció numpy.unravel_index per a convertir-lo.
					output[c, x, y] = pooling_data[np.unravel_index(index, pooling_data.shape)]
										
		return output
	
	def backpropagate(self, delta):
		
		# L'error és sempre 0 excepte per als valors que eren màxims, ja que són els únics que afecten
		# al resultat final.
		new_delta = np.zeros(self.input_shape)
		
		for y in range(delta.shape[2]):
			for x in range(delta.shape[1]):
				for c in range(delta.shape[0]):
					index = np.unravel_index(int(self.indices[c, x, y]), (1, self.stride, self.stride))
					new_delta[c, index[1] + x * self.stride, index[2] + y * self.stride] = delta[c,x,y]
		
		return new_delta
	
	def apply_gradient(self, learning_rate):
		pass
	
	def save(self, f):
		pass
				
	def load(self, f):
		pass
	
	def from_gpu(self, net):
		pass


class Flatten(Layer):
	"""
	Una capa senzilla que afegim per conveniència: només transforma una matriu (que es com arriben els valors)
	en una array d'una sola dimensió, de manera que pot ser utilitzada per la capa FullyConnected.
	"""
	def __init__(self):
		super().__init__()
		
		# Ho necessitem per a la 'backpropagation'. 
		self.input_shape = None
	
	def build(self, input_shape):
		self.input_shape = input_shape
		return input_shape[0] * input_shape[1] * input_shape[2]
		
	def feedforward(self, input_data):
		return input_data.ravel()
	
	def backpropagate(self, delta):
		next_layer = self.neural_network.next_layer(self)
		delta = np.dot(next_layer.weights.transpose(), delta)
		
		# Tan sols hem de tornar l'error de la capa posterior convertit a les dimensions original.
		return np.array(delta).reshape(self.input_shape)
	
	def apply_gradient(self, learning_rate):
		pass
	
	def save(self, f):
		pass
				
	def load(self, f):
		pass
	
	def from_gpu(self, net):
		pass
		

# Una capa totalment connectada, com una xarxa neuronal artificial normal.
class FullyConnected(Layer):
	"""
	Una capa densa, totalment connectada. 
	
	Hiperparàmetres
	----------
	function : string
		La funció d'activació, e.g. 'SIGMOID', 'RELU', None,.. 
	neurons_count : int
		La quantitat de neurones de la capa.
	"""
	def __init__(self, function, neurons_count):
		super().__init__()
		self.function = FUNCTIONS[function] if function is not None else None
		self.derivative = DERIVATIVES[function] if function is not None else None
		
		self.neurons_count = neurons_count
		self.weights = None
		self.biases = None
		
		self.weights_gradient = None
		self.biases_gradient = None
		
		# El valor de la activació abans de passar per la funció d'activació
		self.z = None
		
	def build(self, input_shape):
		self.weights = np.random.randn(self.neurons_count, input_shape) * math.sqrt(2 / (self.neurons_count * input_shape))
		# self.biases = np.random.randn(self.neurons_count)
		self.biases = np.zeros(self.neurons_count)
		
		self.weights_gradient = np.zeros(self.weights.shape)
		self.biases_gradient = np.zeros(self.biases.shape)
		
		return self.neurons_count
		
	def feedforward(self, input_data):
		
		self.z = np.dot(self.weights, input_data) + self.biases
		output = self.z
		
		if self.function is not None:
			output = self.function(output)
		
		return output
	
	def backpropagate(self, delta):
		
		previous_layer = self.neural_network.previous_layer(self)
		next_layer = self.neural_network.next_layer(self)
		
		if next_layer is not None:
			# Si no es l'última capa, si canviem els pesos canviaran les activacions, que alhora canviaran delta.
			# Per tant, hem de multiplicar per la derivada de delta respecte a les activacions d'aquesta capa.
			delta = np.dot(next_layer.weights.transpose(), delta)
			
		# Si hi ha funció d'activació, hem d'aplicar la seva derivada també.
		if self.derivative is not None:
			delta *= self.derivative(self.z)
		
		self.delta = delta
		
		# Els biases es sumen, per tant la derivada de Z respecte als 'biases' es 1. 
		self.biases_gradient += self.delta
		
		delta_matrix = self.delta.reshape(self.delta.shape[0], 1)
		z_matrix = previous_layer.activation.reshape((previous_layer.activation.shape[0], 1))
													
		# Els pesos es multipliquen per les dades que entren (l'activació de la capa anterior). 
		# Per tant, la derivada d'aquest resultat es l'activació de la capa anterior 
		self.weights_gradient +=np.dot(delta_matrix, z_matrix.transpose())
		
		return delta
	
	def apply_gradient(self, learning_rate):
		
		self.biases -= learning_rate * self.biases_gradient
		
		self.weights -= learning_rate * self.weights_gradient
		
		self.weights_gradient = np.zeros(self.weights.shape)
		self.biases_gradient = np.zeros(self.biases.shape)
	
	def save(self, f):
		weights_format = "<" + str(self.weights.size) + "d"
		biases_format = "<" + str(self.biases.size) + "d"  
	
		f.write(struct.pack(weights_format, *self.weights.flat))
		f.write(struct.pack(biases_format, *self.biases.flat))
				
	def load(self, f):
		weights_format = "<" + str(self.weights.size) + "d"
		biases_format = "<" + str(self.biases.size) + "d"  
		weights_size = struct.calcsize(weights_format)
		biases_size = struct.calcsize(biases_format)
		
		self.weights = np.array(struct.unpack(weights_format, f.read(weights_size))).reshape(self.weights.shape)
		self.biases = np.array(struct.unpack(biases_format, f.read(biases_size))).reshape(self.biases.shape)
		
	def from_gpu(self, net):
		self.weights = net.w.get_value().T
		self.biases = net.b.get_value()
	
