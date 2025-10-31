import random as r
import numpy as np
import math

def modulator(word, sigma, mu=0):
    signal = []
    for i in range(len(word)):
        if(word[i] == 0):
            signal.append(1)
        else:
            signal.append(-1)
    signal = np.array(signal, dtype=int)
    signal = gaussian_noise(signal, sigma, mu)
    return signal

def gaussian_noise(signal, sigma, mu=0):
    print(type(signal), type(signal[0]), type(sigma), type(mu))

    for i in range(len(signal)):
        x = r.uniform(-1, 1)  # заменяет r.random(-1,1)
        noise = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        signal[i] = signal[i] + noise
    return signal


def hard_demodulator(signal):
    for i in range(len(signal)):
        if(signal[i] > 0):
            signal[i] = 0
        else:
            signal[i] = 1
    return signal

def soft_demodulator(signal, sigma):
    for i in range(len(signal)):
        signal[i] = signal[i]*2/(sigma**2)
        if(signal[i] > 0):
            signal[i] = 0
        else:
            signal[i] = 1
    return signal

def main():
    input_msg = input("Введите сообщение через пробел: ")
    int_msg = [int(x) for x in input_msg.split()]
    msg = np.array(int_msg, dtype=int)
    sigma = float(input("Введите значение сигма: "))
    signal = modulator(msg, sigma)
    temp = signal.copy()
    print("Переданный сигнал: ",signal)
    print("Результат жесткой демодуляции: ",hard_demodulator(temp))
    print("Результат мягкой демодуляции: ",soft_demodulator(signal, sigma))

main()