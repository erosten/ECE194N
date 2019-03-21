import numpy as np
import matplotlib.pyplot as plt


test_loss  = np.loadtxt('test_loss.txt')
test_acc   = np.loadtxt('test_acc.txt')
train_loss = np.loadtxt('train_loss.txt')
train_acc  = np.loadtxt('train_acc.txt')


epochs = np.arange(test_loss.shape[0])

fig = plt.figure(1)
# plot losses
ax = plt.subplot(1,2,1)
plt.plot(epochs, train_loss)
plt.plot(epochs, test_loss)
ax.legend(['Training', 'Test'])
ax.set_title('Training and Test Loss vs Epoch')
ax.set_xlabel('Epoch')
# plot accuracies 
ax = plt.subplot(1,2,2)
plt.plot(epochs, train_acc)
plt.plot(epochs, test_acc)
ax.legend(['Training', 'Test'])
ax.set_title('Training and Test Accuracy vs Epoch')
ax.set_xlabel('Epoch')


fig.tight_layout()
plt.show()


