import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('log.txt', 'r') as f:
        lines = f.readlines()
        train = [ float(line.split(' ')[2]) for line in lines]
        val = [ float(line.split('VAL: ')[1][:-2]) for line in lines]
        print(train, val)
    plt.plot(train, label='train')
    plt.plot(val, label='val')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.show()
    