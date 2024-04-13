from matplotlib import pyplot as plt

X = []
Y = []

with open("data.txt", "r") as f:
    for line in f:
        point = line[:-1].split(" ")
        X.append(int(point[0]))
        Y.append(float(point[1]))

plt.plot(X, Y)
plt.show()
        

