from random import randint
import numpy as np

def error(code, k):
    pos = randint(0, k-1)
    if code[0,pos] == 1:
        code[0,pos] = 0
    else:
        code[0,pos] = 1
    return pos

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

def coder(msg, r, k):
    G = get_G(k, r)
    msg = msg.reshape(1,k)
    code = np.matmul(msg, G) % 2
    return code

def decoder(code, r, k):
    H = get_H(r, k+r)
    S = np.matmul(code, H.T) % 2
    for i in range(r+k):
        if np.array_equal(S, H.T[i].reshape(1,r)):
            if (code[0,i] == 0):
                code[0,i] = 1
            else:
                code[0,i] = 0
            break
    word = code[:, :k]
    return word


def main():
    input_msg = input("Введите сообщение через пробел: ")
    int_msg = [int(x) for x in input_msg.split()]
    msg = np.array(int_msg, dtype=int)
    k = len(msg)
    r = get_r(k)
    code = coder(msg, r, k)
    print('Код:', code)
    pos = error(code, k)
    print('Код с ошибкой в позиции ', pos + 1, ':', code)
    word = decoder(code,r,k)
    print('Декодированное слово:', word)


main()