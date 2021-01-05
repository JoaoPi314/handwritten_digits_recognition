'''
Autor: João Pedro Melquiades Gomes

Projeto: Construção de uma rede neural
para reconhecer dígitos manuscritos

'''

import numpy as np
import random


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

	def backpropagation(self, x, y):

		gradiente_w = [np.zeros(w.shape()) for w in self.W]
		gradiente_b = [np.zeros(b.shape()) for b in self.B]
		
		##Primeiro, vamos dar o passo pra frente(feedfoward)

		A = [x]

		a = x

		for  b, w in zip(self.B, self.W):
			u = np.dot(w, a) + b
			a = sigmoide(u)
			A.append(a)

		##Agora, vamos dar a passada para trás (backward)
		


		#Para o último layer, calculamos o delta:
		#delta = derivada do custo * derivada da sigmoide do output layer
		#J(w) = Cross entropy function
		#delta - previsão - valor real (isso pode ser provado matematicamente)
		#fazendo a derivada da função de entropia cruzada

		delta = A[-1] - y

		gradiente_w[-1] = np.dot(delta, A[-2].transpose())

		gradiente_b[-1] = delta


		#Para os layers anteriores, vamos repetir o processo, sendo o
		#penúltimo layer: delta(-2) = delta*W(-2)^T*sigmoide do penúltimo layer
		#antepenúltimo layer: delta(-3) = delta(-2)*W(-3)^T*sigmoide do antepnultimo layer

		for i in range(2, self.num_layers):
			
			delta = np.dot(self.W[-i+1].transpose(), delta) * (A[-i-1]) * (1 - A[-i-1])

			gradiente_w[-i] = np.dot(delta, A[-i-1].transpose())
			gradiente_b[-i] = delta

		return (gradiente_w, gradiente_b)


	'''
	O próximo método será responsável por atualizar os mini_batches com os gradientes 
	calculados no método backpropagation()
	'''
	def update_mb(self, mini_batch, eta):

		sum_grad_w = [np.zeros(w.shape()) for w in self.W]
		sum_grad_b = [np.zeros(b.shape()) for b in self.B]

		#Para cada x,y no mini_batch:
		for x,y in mini_batch:
			delta_grad_w, delta_grad_b = self.backpropagation(x,y)

			sum_grad_w = [w + dw for w, dw in zip(sum_grad_w, delta_grad_w)]
			sum_grad_b = [b + db for b, db in zip(sum_grad_b, delta_grad_b)]

		#Agora, os valores dos pesos e bias serão atualizados na direção oposta
		#do gradiente, ou seja, na direção em que o erro diminui

		self.W = [w - (eta/len(mini_batch))*sw for w, sw in zip(self.W, sum_grad_w)]
		self.B = [b - (eta/len(mini_batch))*sb for b, sb in zip(self.B, sum_grad_b)]


	'''
	O próximo método implementará o algoritmo MBGD (Mini-batch gradient descent)
	Esse método é uma junção do BGD e do SGD, unindo o melhor dos dois algoritmos
	training_data é uma lista de tuplas (x,y)
	epochs é o número de iterações
	eta é o learning rate
	mini_batch_size é o tamanho das divisões do training_data

	'''

	def MBGD(self, training_data, epochs, eta, mini_batch_size):

		n = len(training_data)

		for i in range(epochs):
			#Randomiza o training_data
			random.shuffle(training_data)

			mini_batches = []

			for j in range(0, n, mini_batch_size):
				mini_batches.append(training_data[j, j + mini_batch_size])


			for mini_batch in mini_batches:
				self.update_mb(mini_batch, eta)


			print('Epoch {} ok'.format(i))



	def predict(self, test_data):

		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		accuracy = sum(int(x == y) for (x, y) in test_results)/len(test_results)

		return (test_results, accuracy)





