import numpy as np
import matplotlib.pyplot as plt

# reference for # of hidden layer nodes
    #https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# reference for building your own neural net
    #https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

def visualize_samples(x,y):
    plt.scatter(x[:,0], x[:,1])

    for i, label in enumerate(y.tolist()):
        plt.annotate(label, (x[i,0], x[i,1]))
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('XOR of Input 1 and Input 2 and it\'s Annotated Output')
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.show()

def visualize_classification_regions(x,y, nn):
    plt.style.use('ggplot')
    plt.scatter(x[:,0], x[:,1])

    for i, label in enumerate(y.tolist()):
        plt.annotate(label, (x[i,0], x[i,1]))
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('XOR of Input 1 and Input 2 and it\'s Annotated Output')
 
    w1 = nn.w1[0,0]
    w2 = nn.w1[1,0]
    x = np.arange(0,1,0.01).reshape(-1,1)
    y = -(w2 / w1) * x
    plt.plot(x, y)
    y = -(nn.w1[0,1] / nn.w1[1,1]) * x
    plt.plot(x,y)
    y = -(nn.w2[0] / nn.w2[1]) * x
    plt.plot(x,y)
    # plt.xlim((-0.1,1.1))
    # plt.ylim((-0.1,1.1))

    plt.show()



def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

# use squared error loss
def loss(y_pred, y):
    return .5 * np.sum((y_pred - y)**2)

def plot_loss(loss):
    plt.plot(np.arange(loss.shape[0]), loss, linestyle = '--', marker = 'o', color = 'b')
    plt.xlabel('Iterations')
    plt.ylabel('Squared Error Loss')
    plt.show()


# 0 -> 1
# 1 -> -1
def map_nn_output(y):
    return -2 * (y - 0.5)


class neural_net:
    def __init__(self, x, y):
        self.input = x
        self.num_hidden_layer_perceptrons = 2
        # random returns random values between 0 and 1 in a given shape
        self.w1 = np.random.rand(self.input.shape[1],self.num_hidden_layer_perceptrons) 
        self.w2 = np.random.rand(self.num_hidden_layer_perceptrons,1)               
        self.y = y
        self.output = np.zeros(self.y.shape)

    def forward_pass(self, input = None):
        input = input if input is not None else self.input
        self.in_h_layer = np.dot(input,self.w1)
        self.hidden_layer = sigmoid(self.in_h_layer)
        self.in_output = np.dot(self.hidden_layer, self.w2)
        self.output = sigmoid(self.in_output)

    def back_prop(self, alpha):
        delta_0 = (self.output - self.y) * sigmoid_derivative(self.in_output)
        d_w2 = np.dot(self.hidden_layer.T, delta_0)
        d_w1 = np.dot(self.input.T,  (np.dot(delta_0, self.w2.T) * sigmoid_derivative(self.in_h_layer)))

        self.w1 -= alpha * d_w1
        self.w2 -= alpha * d_w2

    def train(self, num_iter, alpha):
        self.loss = np.zeros(num_iter)
        for i in range(num_iter):
            self.forward_pass()
            self.back_prop(alpha)
            self.loss[i] = loss(self.output,self.y)

    def loss(self):
        return loss(self.output, self.y)


def run_neural_net():
    # define neural net inputs
    x = np.array([[1,1],
                  [0,0],
                  [1,0],
                  [0,1]])

    y = np.array([[0],[0],[1],[1]])
    # define neural net
    nn = neural_net(x,y)
    # define neural net training inputs
    num_iter = 100000
    alpha = 1
    # train the net
    nn.train(num_iter, alpha)
    # map outputs from 0 -> 1 and 1 -> -1
    y_pred = map_nn_output(nn.output)
    # print predictions
    print(y_pred)
    # plot loss
    plot_loss(nn.loss)
    # visualize classification regions
    visualize_classification_regions(x,y, nn)



if __name__ == "__main__":
    run_neural_net()
