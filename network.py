'''
Autor: João Pedro Melquiades Gomes

Projeto: Construção de uma rede neural
para reconhecer dígitos manuscritos

'''

import numpy as np


'''
A função de ativação faz uma transformação
no potencial de ativação u, transformando 
uma combinação linear em algo não linear,
o que permite utilizar do cálculo de derivadas
para analisar o erro da rede. Nessa rede, 
a função de ativação será a sigmoide, uma 
função que tem imagem [0, 1].
'''
def activation(x):

	return 1.0/(1 + np.exp(-x))

'''
A d_activation calcula a derivada da
função de ativação, isso será necessário
quando o algoritmo de backpropagation
for implementado
'''

def d_activation(x):
	return activation(x)*(1.0 - activation(x))



class Network:

	'''
	O construtor da classe irá receber uma lista com cada índice
	representando um layer, e cada valor representando a quantidade
	de neurônios naquele índice
	'''
	def __init__(self, layers):

		self.B = []
		self.W = []
		self.layers_qnt = len(layers)
		self.layers = layers

		#B vai ser uma lista com arrays. Cada índice da lista
		#contém um array com os valores dos bias de cada nó
		#na partir do segundo layer. O input layer não contém
		# bias porque apenas contém as entradas
		for b in layers[1:]:
			self.B.append(np.random.randn(b,1))
		#W será o uma lista com arrays. Cada índice da lista
		#contém um array bidimensional (x,y), sendo x o número 
		#de nós do layer atual e y o número de nós do layer
		#anterior. Começando do segundo layer. Essa matriz
		#indica que para cada nó da camada anterior, existem
		#n conexões com a camada atual, sendo n o número de nós
		#nela.  
		for x,y in zip(layers[1:], layers[0:]):
			self.W.append(np.random.randn(x, y))


	'''
	O método estimate irá calcular uma saída para uma determinada entrada.
	Essa entrada deve ser um array no formato (n_i, 1), sendo n_i o número
	de nós no input layer
	'''
	def estimate(self, x):
		u = []
		g = x

		for b, w in zip(self.B, self.W):
			u = np.dot(w*g) + b
			g = activation(u)
		return g


	'''
	O próximo método a ser implementado é o método de backpropagation. 
	Esse método irá retornar as derivadas parciais de cada saída referente
	a todas as entradas.
	'''
