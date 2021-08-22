import madml
from madml.nn import conv2d, linear, maxpool2d, ReLU, Module


class cnn_mnist_model(Module):
    def __init__(self):
        super(cnn_mnist_model, self).__init__()
        self.conv1 = conv2d(1, 32, 3, padding=1)
        self.pool = maxpool2d(2, 2)
        self.conv2 = conv2d(32, 48, 3)
        self.fc1 = linear(48 * 2 * 2, 120)  # (599, 192)
        self.fc2 = linear(120, 84)
        self.fc3 = linear(84, 10)

    def forward(self, X):
        X = self.conv1(X)
        X = ReLU(X)
        X = self.pool(X)  # 32 x 14 x 14
        X = self.conv2(X)  # 46 x 12 x 12
        X = ReLU(X)
        X = X.flatten()
        X = self.fc1(X)
        X = ReLU(X)
        X = self.fc2(X)
        X = ReLU(X)
        X = self.fc3(X)
        return X


x = madml.zeros([76, 76])
model = cnn_mnist_model()
y = model(x)
print(model)

y.backward()

print(y)
