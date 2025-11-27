import numpy as np
import random as r

def get_r(k):
    r=1;
    while True:
        if(2**r>=k+r+1):
            break
        r+=1
    return r

def get_G(k, r):
    G = np.eye(k, dtype=int)
    R = np.ones((k, k), dtype=int)
    i = k - 1
    j = 0
    while (i >= 0):
        R[i, j] -= 1
        i -= 1
        j += 1
    G = np.hstack([G,R])
    return G

def coder(msg, k, G):
    msg = msg.reshape(1,k)
    code = np.matmul(msg, G) % 2
    code = np.array(code, dtype=int)
    return code

def modulator(code):
    signal = []
    for i in range(len(code[0])):
        if(code[0, i] == 0):
            signal.append(1)
        else:
            signal.append(-1)
    signal = np.array(signal, dtype=float)
    return signal

def gaussian_noise(signal, sigma):
    i = r.randint(0, len(signal)-1)
    check = signal[i]
    x1 = r.uniform(-1, 1)
    x2 = r.uniform(-1, 1)
    s = x1 ** 2 + x2 ** 2
    while (s > 1 or s <= 0):
        x1 = r.uniform(-1, 1)
        x2 = r.uniform(-1, 1)
        s = x1 ** 2 + x2 ** 2
    z1 = x1 * np.sqrt(-2 * np.log(s) / s)
    z2 = x2 * np.sqrt(-2 * np.log(s) / s)
    signal[i] = signal[i] + sigma * z1
    if(check == 1):
        if(signal[i] > 0):
            signal[i] = 1
            signal = gaussian_noise(signal, sigma)
    if (check == -1):
        if (signal[i] < 0):
            signal[i] = -1
            signal = gaussian_noise(signal, sigma)
    return signal

def gen_word(k):
    word = []
    for i in range(k):
        word.append(r.choice([0,1]))
    return word

def main():
    size = 0
    with open('dataset.txt', 'w') as f:
        for k in range(4, 20):
            r = get_r(k)
            G = get_G(k, r)
            sigma = 1
            for i in range(300):
                word = gen_word(k)
                for j in range(len(word)):
                    f.write(str(word[j]))
                f.write(' ')
                word = np.array(word, dtype=int)
                code = coder(word, k, G)
                signal = modulator(code)
                signal = gaussian_noise(signal, sigma)
                for j in range(len(signal)):
                    f.write(str(signal[j])+' ')
                f.write('\n')
                size += 1
                sigma += 0.02

main()