import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# square footage, number of bedrooms, selling price
def get_data_from_file():
	return np.loadtxt("hw1_housing_prices.txt",dtype='float32',delimiter=',')

def get_train_test_sets(data):
	train_X = data[0:(data.shape[0] - 10),0:2]
	test_X = data[(data.shape[0] - 10):,0:2]
	train_y = data[0:(data.shape[0] - 10),2].reshape(-1,1)
	test_y = data[(data.shape[0] - 10):,2].reshape(-1,1)
	return train_X, test_X, train_y, test_y

def visualize_data(data):
	numInstances = data.shape[0]	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[0:numInstances - 10,0], data[0:numInstances - 10,1], data[0:numInstances - 10,2], c='r', marker='o',label='Training Points')
	ax.set_xlabel('Square Footage')
	ax.set_ylabel('Number of Bedrooms')
	ax.set_zlabel('House Price')
	ax.scatter(data[numInstances - 10:,0], data[numInstances - 10:,1], data[numInstances - 10:,2], c='b', marker='o', label='Testing Points')
	ax.legend()
	plt.show()

# use MSE as the loss function
def compute_loss(train_X, train_y, weights):
	numInstances = train_X.shape[0]
	coeff = 1 / (2 * numInstances)
	loss = np.sum((np.matmul(train_X,weights) - train_y)**2)
	return coeff * loss

def gradientDescent(train_X, train_y, weights, alpha, num_iters):
	numInstances = train_X.shape[0]
	losses = np.zeros((num_iters,))
	for i in range(num_iters):
		# inverse = np.linalg.inv(np.matmul(np.transpose(train_X),train_X))
		# grad = np.matmul(np.transpose(train_y), np.matmul(train_X, inverse))
		grad = np.matmul(np.transpose(train_X), np.matmul(train_X,weights) - train_y)
		weights = weights - (alpha / numInstances) * grad
		loss = compute_loss(train_X, train_y, weights)
		losses[i] = loss
		print('Iteration: {}, Loss: {}'.format(i,loss))
	return weights, losses

def normalize_features(train_X):
	mu = np.mean(train_X, axis = 0)
	sigma = np.std(train_X, axis = 0)
	return (train_X - mu) / sigma

def plotLoss(losses):
	fig = plt.figure()
	ax = plt.axes()
	ax.scatter(range(losses.shape[0]),losses)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Loss')
	plt.show()

def print_results(test_X, weights, test_y):
	predicted_y = np.matmul(test_X, weights)
	print('Weights are {}'.format(weights))
	for i in range(test_X.shape[0]):
		print('Predicted Value: {}, Actual Value: {}'.format(predicted_y[i], test_y[i]))
	coeff = 1 / (2 * test_X.shape[0])
	loss = np.sum((predicted_y - test_y)**2)
	totalMSE = coeff * loss
	print('Total Mean Squared Error: {}'.format(totalMSE))

def append_zeros(X):
	ones = np.ones((X.shape[0],1))
	return np.concatenate((ones,X), axis = 1)

def run_regression():
	# set up and visualize data
	data = get_data_from_file()
	train_X, test_X, train_y, test_y = get_train_test_sets(data)
	visualize_data(data)	
	# train the weights
	weights = np.zeros((3,1))
	learning_rate = 0.01
	num_iter = 1000
	train_X = normalize_features(train_X)
	print(train_X)
	train_X = append_zeros(train_X)
	weights, losses = gradientDescent(train_X, train_y, weights, learning_rate, num_iter)
	# view loss and test results
	plotLoss(losses)
	test_X = normalize_features(test_X)
	test_X = append_zeros(test_X)
	print_results(test_X, weights, test_y)



if __name__ == '__main__':
    run_regression()