import numpy as np
import neurolab as nl

# С В В
target =  [[1,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0],
            [1,1,1,1,1,
            1,0,0,0,0,
            1,0,0,0,0,
            1,0,0,0,0,
            1,1,1,1,1],
            [1,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0]]

chars = ['С', 'В', 'В']
target = np.asfarray(target)
target[target == 0] = -1

# Створити та навчити мережу
net = nl.net.newhop(target)

output = net.sim(target)
print("Тестування на тренувальних прикладах:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

# Тестування на пошкодженій букві С
print("\nТестування на пошкодженій С:")
test_c = np.asfarray([0,1,1,1,0,
                     1,0,0,0,1,
                     1,0,0,0,1,
                     1,0,0,0,1,
                     0,1,1,1,0])
test_c[test_c == 0] = -1
out_c = net.sim([test_c])
print((out_c[0] == target[0]).all(), 'Кількість кроків', len(net.layers[0].outs))
