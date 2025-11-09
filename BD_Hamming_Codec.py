import numpy as np
from random import randint

def error(code, n):
    pos = randint(0, n)
    pos1 = randint(0, n - 1)
    if code[pos] == 1:
        code[pos] = 0
    elif code[pos] == 0:
        code[pos] = 1
    if code[pos1] == 0:
        code[pos1] = 1
    elif code[pos1] == 0:
        code[pos1] = 1
    return [pos, pos1]

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
    sum=0
    for j in range(r+k):
        sum += code.item(j)
    sum = sum % 2
    code = np.append(code, sum)

    return code

def get_C(k,n,r):
    TT = []
    for i in range(2**k):
        row = []
        for j in range(k):
            if ((i>>j)&1):
                row.append(1)
            else:
                row.append(0)
        TT.append(row)
    G = get_G(k, r)
    TT = np.array(TT, dtype=int)
    C = []
    for i in range(2**k):
        temp = TT[i]
        temp.reshape(1,k)
        code = np.matmul(TT[i], G) % 2
        sum = 0
        for j in range(n-1):
            sum += code[j]
        sum = sum%2
        code = np.append(code, sum)
        C.append(code)
    C = np.array(C, dtype=int)
    return C

def dist(code1, code2):
    return np.sum(code1 != code2)

def decoder(code, C, k):
    distances = [dist(code, C[i]) for i in range(len(C))]
    min_d = min(distances)
    for i in range(len(C)):
        if min_d >= dist(code, C[i]):
            word = C[i][:k]
            return word


def main():
    input_msg = input("Введите сообщение через пробел: ")
    int_msg = [int(x) for x in input_msg.split()]
    msg = np.array(int_msg, dtype=int)
    k = len(msg)
    r = get_r(k)
    code = coder(msg, r, k)
    n = r+k+1
    print('Код:', code)
    pos = error(code, k)
    print('Код с ошибками в позицииях ', pos[0] + 1, 'и', pos[1] + 1,':', code)
    C = get_C(k, n, r)
    word = decoder(code, C, k)
    print('Декодированное слово:', word)

main()