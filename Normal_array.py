from operator import matmul
from random import randint
import numpy as np

def error(code, k):
    pos = randint(0, k-1)
    pos1 = randint(0, k - 1)
    if code[0,pos] == 1:
        code[0,pos] = 0
    elif code[0,pos] == 0:
        code[0,pos] = 1
    if code[0,pos1] == 0:
        code[0,pos1] = 1
    elif code[0,pos1] == 0:
        code[0,pos1] = 1
    return [pos, pos1]

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

def get_E(n):
    E = []
    for i in range(n):
        row = []
        for j in range(2**n):
            if ((j >> i) & 1):
                row.append(1)
            else:
                row.append(0)
        E.append(row)
    E = np.array(E, dtype=int)
    cols = np.where(np.sum(E, axis=0) < 3)[0]
    E = E[:, cols]
    E=E.T
    return E

def get_S(n, k, r):
    E = get_E(n)
    size = E.shape[0]
    H = get_H(k, r)
    S = []
    for i in range(size):
        row = matmul(E[i],H.T)%2
        S.append(row)
    S = np.array(S, dtype=int)
    flag = 0
    i=0
    while (i<size-1):
        j=i+1
        while(j<size):
            if np.array_equal(S[i], S[j]):
                flag=1
                E = np.delete(E, j, axis=0)
                S = np.delete(S, j, axis=0)
                size-=1
            j+=1
        if flag:
            cols = np.where(np.sum(E, axis=0) < 2)[0]
            if(i != cols.any()):
                E = np.delete(E, i, axis=0)
                S = np.delete(S, i, axis=0)
                i -= 1
                size-=1
            flag = 0
        i+=1

    return S, E

def decoder(r,n,k, code):
    H = get_H(k, r)
    s = np.matmul(code, H.T) % 2
    S, E = get_S(n, k, r)
    size = S.shape[0]
    G = get_G(k, n)
    for i in range(size):
        if (np.array_equal(S[i].reshape(1,r), s)): ##################################
            code = (code - E[i])%2
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
    print('Код с ошибками в позицииях ', pos[0] + 1, 'и', pos[1] + 1,':', code)
    word = decoder(r, n, k, code)
    print('Декодированное слово:', word)


main()