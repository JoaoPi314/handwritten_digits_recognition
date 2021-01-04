'''
Autor: João Pedro Melquiades Gomes

Projeto: Construção de uma rede neural
para reconhecer dígitos manuscritos

'''

import numpy as np

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


	