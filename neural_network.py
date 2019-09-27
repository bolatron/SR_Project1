import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, output_size):
        # Inicializando os atributos da classe
        self.input_size = input_size
        self.output_size = output_size
        # Inicialização das matrizes de pesos com valor aleatório
        self.weight_1 = np.random.random((input_size, input_size))
        self.weight_2 = np.random.random((output_size, input_size))

    def train(self, x, y, epochs, lr):
        for i in range(epochs):
            # Cálculo das camadas
            layer_1 = x.T
            layer_2 = 1. / (1. + np.exp(-1*self.weight_1.dot(layer_1)))
            layer_3 = 1. / (1. + np.exp(-1*self.weight_2.dot(layer_2)))

            # Cálculo do erro gerado pelas camadas
            error_2 = np.subtract(layer_3, y.T)
            w2 = self.weight_2.T
            #error_1 = np.multiply(w2.dot(error_2), np.multiply(layer_2, np.subtract(1, layer_2)))
            error_1 = np.multiply(w2.dot(error_2), (1. / (1 + np.exp(-1*layer_2))))

            # Atualização dos pesos
            self.weight_1 = np.subtract(self.weight_1, np.multiply(lr, error_1.dot(layer_1.T)))
            self.weight_2 = np.subtract(self.weight_2, np.multiply(lr, error_2.dot(layer_2.T)))

    def test(self, x):
        layer_1 = x.T
        layer_2 = 1. / (1. + np.exp(-1*self.weight_1.dot(layer_1)))
        layer_3 = 1. / (1. + np.exp(-1*self.weight_2.dot(layer_2)))
        return layer_3

def main():
    # Dados para treinamento
    entrada = np.array([[0., 2./6., 4./6., 6./6.], [3./9., 5./9., 7./9., 9./9.],
                       [16./22., 18./22., 20./22., 22./22.],
                       [4./10., 6./10., 8./10., 10./10.],
                       [1./7., 3./7., 5./7., 7./7.],
                       [5./11., 7./11., 9./11., 11./11.]])
    # Saida desejada para treinamento
    saida = np.array([8./24., 11./24., 24./24., 12./24., 9./24., 13./24.])
    # Criação da rede neural
    nn = NeuralNetwork(4, 1)
    # Inicio do treinamento
    nn.train(entrada, saida, epochs=10000, lr=1.)
    # Dado para teste
    teste = np.array([3./9., 5./9., 7./9., 9./9.])
    # Resultado do teste
    print(nn.test(teste)*24)

if __name__ == '__main__':
    main()
