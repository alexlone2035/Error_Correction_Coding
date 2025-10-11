from random import randint

def coder(word):
    r1 = word[0]^word[1]^word[3]
    r2 = word[0]^word[2]^word[3]
    r3 = word[2]^word[1]^word[3]
    code = [r1, r2, word[0], r3, word[1], word[2], word[3]]
    return code

def error(code):
    size = len(code)
    pos = randint(0, size-1)
    if code[pos] == 1:
        code[pos] = 0
    else:
        code[pos] = 1
    return pos

def decoder(code):
    s = [0,0,0]
    s[0] = code[0]^code[2]^code[4]^code[6]
    s[1] = code[1]^code[2]^code[5]^code[6]
    s[2] = code[3]^code[4]^code[5]^code[6]
    pos = s[2]*4 + s[1]*2 + s[0] - 1
    if code[pos] == 1:
        code[pos] = 0
    else:
        code[pos] = 1
    word = [code[2], code[4], code[5], code[6]]
    return word

def main():
    word = [0,1,1,0]
    print('Слово:', word)
    code = coder(word)
    print('Код:', code)
    pos = error(code)
    print('Код с ошибкой в позиции ', pos+1, ':', code)
    word = decoder(code)
    print('Декодированное слово:', word)

main()