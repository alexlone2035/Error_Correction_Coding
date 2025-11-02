import random as r
import numpy as np

def modulator(word, sigma):
    signal = []
    for i in range(len(word)):
        if(word[i] == 0):
            signal.append(1)
        else:
            signal.append(-1)
    signal = np.array(signal, dtype=float)
    return signal

def gaussian_noise(signal, sigma, mu=0):
    for i in range(len(signal)):
        x = r.uniform(0, mu+0.2)
        noise = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))
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
    print("Переданный сигнал: ",signal)
    signal = gaussian_noise(signal, sigma)
    print("Сигнал с шумом", signal)
    temp = signal.copy()
    print("Результат жесткой демодуляции: ",hard_demodulator(temp))
    print("Результат мягкой демодуляции: ",soft_demodulator(signal, sigma))

main()