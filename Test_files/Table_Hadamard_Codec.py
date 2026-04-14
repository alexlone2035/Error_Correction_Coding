from operator import matmul
from random import randint
import numpy as np

def error(code, k):
    pos = randint(0, k-1)
    if code[0,pos] == 1:
        code[0,pos] = 0
    else:
        code[0,pos] = 1
    return pos

def get_G(k,n):
    G = []
    for i in range(k):
        row = []
        for j in range(1, n+1):
            if ((j>>i)&1):
                row.append(1)
            else:
                row.append(0)
        G.append(row)
    G = np.array(G, dtype=int)
    cols = np.where(np.sum(G, axis=0) > 1)[0]
    G = G[:, cols]
    G = np.hstack([G, np.eye(k)])
    return G

def get_H(k, r):
    H = np.eye(r, dtype=int)
    G = get_G(k,k+r)
    R = G[:, :r]
    H = np.hstack([H,R.T])
    return H

def coder(msg, n, k):
    G = get_G(k, n)
    msg = msg.reshape(1,k)
    code = np.matmul(msg, G) % 2
    return code

def decoder(code, r, k):
    H = get_H(k, r)
    S = np.matmul(code, H.T) % 2
    for i in range(r+k):
        if np.array_equal(S, H.T[i].reshape(1,r)):
            if (code[0,i] == 0):
                code[0,i] = 1
            else:
                code[0,i] = 0
            break
    word = code[:, -k:]
    return word

def main():
    input_msg = input("Введите сообщение через пробел: ")
    int_msg = [int(x) for x in input_msg.split()]
    msg = np.array(int_msg, dtype=int)
    k = len(msg)
    n = 2**k - 1
    r = n - k
    code = coder(msg, n, k)
    print('Код:', code)
    pos = error(code, k)
    print('Код с ошибкой в позиции ', pos + 1, ':', code)
    word = decoder(code, r, k)
    print('Декодированное слово:', word)


main()