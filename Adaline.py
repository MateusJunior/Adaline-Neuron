import matplotlib.pyplot as plt
import numpy as np
import random as rd

dictonary ={
     "1000000": 'A',
     "0100000": 'B',
     "0010000": 'C',
     "0001000": 'D',
     "0000100": 'E',
     "0000010": 'J',
     "0000001": 'K'  
}

def getLetter(param,y):
    value = ''
    for i in range(7):
        if(y[i] == 1): value += '1'
        else: value += '0'

    return dictonary.get(value)

def main():

    matX = np.loadtxt('seeds/amostras.txt')
    (amostras, entradas) = np.shape(matX)

    targetY = np.loadtxt('seeds/targets.txt')
    (classes, targets) = np.shape(targetY)

    lim = 0.0

    alfa = float(input('Informe a TAXA DE APRENDIZAGEM: '))
    errorStop = False if input('Parada através de Ciclos? S p/ sim ') == 'S' else True

    tolerancia = 0.1

    pesos = np.zeros((entradas,classes))
    bias = np.zeros((classes,1))

    # Inicializando matriz de pesos (entradas x classes)
    for i in range(entradas):
        for j in range(classes):
            pesos[i][j] = rd.uniform(-0.1,0.1)

    # Inicializando vetor bias (classes)
    for i in range(classes):
        bias[i] = rd.uniform(-0.1,0.1)

    # Armazenamento de ciclos e épocas
    vetorErro = []
    vetorCiclo = []

    # Inicializando saídas 
    yin = np.zeros((classes,1))
    y = np.zeros((classes,1))

    ciclo = 0
    erro = 10

    while(erro > tolerancia if errorStop else ciclo <= 10):
        ciclo = ciclo + 1
        erro = 0

        # Para cada amostra, multiplicar cada entrada pelos pesos e somar com o bias = saída yin
        for i in range(amostras):
            xaux = matX[i,:]
            for m in range(classes):
                soma = 0
                for n in range (entradas):
                    soma += xaux[n] * pesos[n][m]
                yin[m] = soma + bias[m]

            # Função de ativação
            for j in range(classes):
                if yin[j] >= lim: y[j] = 1.0
                else: y[j] = -1.0

            # Atualização do erro
            for j in range(classes):
                erro += 0.5*((targetY[j][i] - y[j])**2)

            pAnterior = pesos
            # Atualização dos pesos
            for m in range(entradas):
                for n in range(classes):
                    pesos[m][n] = pAnterior[m][n] + (alfa * (targetY[n][i] - y[n]) * xaux[m])

            bAnterior = bias
            # Atualização do bias
            for m in range(classes):
                bias[m] = bAnterior[m] + alfa * (targetY[m][i] - y[m])
    
        # Plotagem
        vetorCiclo.append(ciclo)
        vetorErro.append(erro)


    # plt.scatter(ciclo,erro,marker='*',color='red')
    plt.plot(vetorCiclo,vetorErro,'bo')
    plt.xlabel('Ciclo')
    plt.ylabel('Erro')
    plt.show()
    print('TERMINOU O TREINAMENTO')


    # Início do teste
    num = int(input('Informe a amostra a ser testada: (1 a 21) '))
    xtest = matX[num-1,:]
    for i in range(classes):
        soma = 0
        for j in range(entradas):
            soma += xtest[j] * pesos[j][i]
            yin[i] = soma + bias[i] 

    for j in range(classes):
        if yin[j] >= lim: y[j] = 1.0
        else: y[j] = -1.0

    print(y)
    print(getLetter(classes,y))

    print('TERMINOU')

if __name__ == "__main__":
    main()