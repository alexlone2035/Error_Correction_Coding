import numpy as np
import random as rand

def get_r(k):
    r=1;
    while True:
        if(2**r>=k+r+1):
            break
        r+=1
    return r

def get_H(r,n):
    H = []
    for i in range(r):
        row = []
        for pos in range(1, n + 1):
            if ((pos>>i)&1):
                row.append(1)
            else:
                row.append(0)
        H.append(row)
    H = np.array(H, dtype=int)
    cols = np.where(np.sum(H, axis=0) > 1)[0]
    H = H[:, cols]
    H = np.hstack([H, np.eye(r)])
    return H

def get_G(k, r):
    G = np.eye(k, dtype=int)
    H = get_H(r,k+r)
    R = H[:, :k]
    G = np.hstack([G,R.T])
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

def gaussian_noise(signal, sigma, mode = 1, errors = 1): # mode 1 - normal noise, mode 2 - noisy error bit
    noisy_signal = []
    p = 3 / len(signal)
    state = 1
    error = 0
    for bit in signal:
        cur_sigma = sigma
        rand_num = np.random.random()
        if state == 1:
            if rand_num < p:
                state = 2
        noise = np.random.normal(0, cur_sigma)
        if(mode == 1):
            noisy_bit = bit + noise
        else:
            noisy_bit = bit
        if state == 2:
            noise = np.random.normal(2, cur_sigma)
            noisy_bit = bit + noise
            if (bit * noisy_bit > 0):
                noisy_bit = bit - noise
            state = 3
        if(bit*noisy_bit < 0):
            error += 1
        noisy_signal.append(noisy_bit)
    if(error != errors):
        noisy_signal[0] = 999
    return noisy_signal

def gen_word(k):
    word = []
    for i in range(k):
        word.append(rand.choice([0,1]))
    return word

def get_dataset(sigma1_train, sigma2_train, sigma1_test, sigma2_test, min_word = 5, max_word = 5, dataset_size = 36000, mode = 1, gauss_mode = 1, errors = 1): # mode 1 - sigma = const, mode 2 - sigma from set, mode 3 - sigma from segment
    min_size = min_word
    max_size = max_word
    range_size = dataset_size
    test_range_size = range_size * 0.3
    train_range_size = range_size * 0.7
    min_train_sigma = sigma1_train
    min_test_sigma = sigma1_test
    max_train_sigma = sigma2_train
    max_test_sigma = sigma2_test
    with open('trainset.txt', 'w') as f:
        for k in range(min_size, max_size+1):
            r = get_r(k)
            G = get_G(k, r)
            if (mode == 3):
                sigma = rand.uniform(min_train_sigma, max_train_sigma)
            else:
                sigma = min_train_sigma
            for i in range(int(train_range_size)):
                word = gen_word(k)
                word = np.array(word, dtype=int)
                code = coder(word, k, G)
                signal = modulator(code)
                signal = gaussian_noise(signal, sigma, gauss_mode, errors)
                if(signal[0]==999):
                    continue
                for j in range(len(word)):
                    f.write(str(word[j]))
                f.write(' ')
                for j in range(len(signal)):
                    f.write(str(signal[j])+' ')
                f.write('\n')
                if(mode == 2):
                    sigma += 0.01
                    if sigma > max_train_sigma:
                        sigma = min_train_sigma
                if (mode == 3):
                    sigma = rand.uniform(min_train_sigma, max_train_sigma)
    with open('testset.txt', 'w') as f:
        for k in range(min_size, max_size + 1):
            r = get_r(k)
            G = get_G(k, r)
            if (mode == 3):
                sigma = rand.uniform(min_test_sigma, max_test_sigma)
            else:
                sigma = min_test_sigma
            for i in range(int(test_range_size)):
                word = gen_word(k)
                word = np.array(word, dtype=int)
                code = coder(word, k, G)
                signal = modulator(code)
                signal = gaussian_noise(signal, sigma, gauss_mode, errors)
                if (signal[0] == 999):
                    continue
                for j in range(len(word)):
                    f.write(str(word[j]))
                f.write(' ')
                for j in range(len(signal)):
                    f.write(str(signal[j]) + ' ')
                f.write('\n')
                if (mode == 2):
                    sigma += 0.01
                    if sigma > max_test_sigma:
                        sigma = min_test_sigma
                if (mode == 3):
                    sigma = rand.uniform(min_test_sigma, max_test_sigma)

def main():
    get_dataset(0.1, 0.5, 1, 1, 5, 5, 36000, 3, 2, 1)

main()