import random as r
import numpy as np

def modulator(code, sigma):
    signal = []
    for i in range(len(code)):
        if(code[i] == 0):
            signal.append(1)
        else:
            signal.append(-1)
    signal = np.array(signal, dtype=float)
    return signal

def gaussian_noise(signal, sigma):
    i=0
    size = len(signal)
    while(i < size):
        x1 = r.uniform(-1, 1)
        x2 = r.uniform(-1, 1)
        s = x1**2 + x2**2
        while(s>1 or s<=0):
            x1 = r.uniform(-1, 1)
            x2 = r.uniform(-1, 1)
            s = x1**2 + x2**2
        z1 = x1*np.sqrt(-2*np.log(s)/s)
        z2 = x2*np.sqrt(-2*np.log(s)/s)
        signal[i] = signal[i] + sigma*z1
        if(size > i+1):
            signal[i+1] = signal[i+1] + sigma*z2
        i+=2
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