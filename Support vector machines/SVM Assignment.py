import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Functions
def compute_p_matrix(inputs, targets, kernel):
    N = len(targets)
    P = numpy.zeros(shape=(N, N))
    result = 0
    for i in range(N):
        for j in range(N):
            P[i, j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])
    return P

def objective(alphas):
    N = len(alphas)
    result = 0
    for i in range(N):
        for j in range(N):
            result += alphas[i] * alphas[j] * P[i, j]
    return 0.5 * result - numpy.sum(alphas)

def zero_function(alphas):
    return numpy.dot(alphas, targets)

def get_support_vectors(inputs, targets, alpha):
    N = len(inputs)
    result = []
    for i in range(N):
        if alpha[i] >= 10**(-5):
            result += [[inputs[i], targets[i], alpha[i]]]
    return result

def calculate_threshold(alphas, targets, support_vectors, inputs, kernel_function):
    N = len(inputs)
    result = 0
    for i in range(N):
        result += alphas[i] * targets[i] * kernel_function(support_vectors[0][0], inputs[i])
    return result - support_vectors[0][1]

def indicator_function(x, y):
    N = len(inputs)
    result = 0
    for i in range(N):
        result += alpha[i] * targets[i] * kernel((x, y), inputs[i])
    return result - b

def linear_kernel(x, y):
    return numpy.dot(x, y)

def polynomial_kernel(x, y):
    return numpy.power((numpy.dot(x, y) + 1), p)

def rbf_kernel(x, y):
    return numpy.exp(-(numpy.dot(numpy.subtract(x,y),numpy.subtract(x,y))/(2*sigma**2)))

# Linear kernel - Example 1
classA = numpy.concatenate(
        (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
            numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = numpy.random.rand(20,2) * 0.2 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
            -numpy.ones(classB.shape[0])))

N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

kernel = linear_kernel
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = None
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Linear Kernel - " + str(N) + " inputs - C=" + str(C))
plt.savefig('linear_kernel_output_1.pdf')
plt.show()

# Linear kernel example 2
classA = numpy.concatenate(
        (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
            numpy.random.randn(10, 2) * 0.2 + [-1.5, -1.5]))
classB = numpy.random.rand(20,2) * 0.2 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
            -numpy.ones(classB.shape[0])))

N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

kernel = linear_kernel
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = None
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Linear Kernel - " + str(N) + " inputs - C=" + str(C))
plt.savefig('linear_kernel_input_2.pdf')
plt.show()

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    print("Couldn't find the linear separation")
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Linear Kernel - " + str(N) + " inputs - C=" + str(C))
plt.savefig('linear_kernel_output_2.pdf')
plt.show()

# Polynomial kernel example 1
kernel = polynomial_kernel
p = 2 
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = None
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Polynomial Kernel - p=" + str(p) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('polynomial_kernel_output_1.pdf')
plt.show()

# RBF kernel - Example 1
kernel = rbf_kernel
sigma = 1
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = None
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("RBF Kernel - sigma=" + str(sigma) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('rbf_kernel_output_1.pdf')
plt.show()

# Polynomial kernel example 2
classA = numpy.concatenate(
        (numpy.random.randn(10, 2) * 0.2 + [0.0, 0.0],
            numpy.random.randn(10, 2) * 0.2 + [-1.0, -1.0],
            numpy.random.randn(10, 2) * 0.2 + [-1.0, 0.0],
            numpy.random.randn(10,2) * 0.2 +  [-0.5, -1.5]))
classB = numpy.concatenate(
        (numpy.random.rand(20,2) * 0.2 + [0.0, -0.5],
            numpy.random.randn(10, 2) * 0.2 + [-0.5, -0.75],
            numpy.random.randn(10, 2) * 0.2 + [0.5, -0.5],
            numpy.random.randn(10, 2) * 0.2 + [0.5, -1.5]))

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
            -numpy.ones(classB.shape[0])))

N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

kernel = polynomial_kernel
p = 2
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = 1
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Polynomial Kernel - p=" + str(p) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('polynomial_kernel_output_2.pdf')
plt.show()

kernel = polynomial_kernel
p = 3
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = 1
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Polynomial Kernel - p=" + str(p) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('polynomial_kernel_output_3.pdf')
plt.show()

kernel = polynomial_kernel
p = 5
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = 1
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Polynomial Kernel - p=" + str(p) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('polynomial_kernel_output_4.pdf')
plt.show()

kernel = polynomial_kernel
p = 10
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = 1
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Polynomial Kernel - p=" + str(p) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('polynomial_kernel_output_5.pdf')
plt.show()

classA = numpy.concatenate(
        (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
            numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = numpy.random.rand(20,2) * 0.2 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
            -numpy.ones(classB.shape[0])))

N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# RBF kernel - Example 2
kernel = rbf_kernel
sigma = 0.5
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = None
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("RBF Kernel - sigma=" + str(sigma) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('rbf_kernel_output_2.pdf')
plt.show()

kernel = rbf_kernel
sigma = 0.75
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = None
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("RBF Kernel - sigma=" + str(sigma) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('rbf_kernel_output_3.pdf')
plt.show()


kernel = rbf_kernel
sigma = 1
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = None
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("RBF Kernel - sigma=" + str(sigma) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('rbf_kernel_output_4.pdf')
plt.show()


kernel = rbf_kernel
sigma = 2
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = None
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    exit(-1)
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("RBF Kernel - sigma=" + str(sigma) + " - " + str(N) + " inputs - C=" + str(C))
plt.savefig('rbf_kernel_output_5.pdf')
plt.show()

# Linear kernel example 2
classA = numpy.concatenate(
        (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
            numpy.random.randn(10, 2) * 0.2 + [-1.5, -1.5]))
classB = numpy.random.rand(20,2) * 0.2 + [0.0, 0.25]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
            -numpy.ones(classB.shape[0])))

N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

kernel = linear_kernel
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = 0.1
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    print("Couldn't find the linear separation")
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Linear Kernel - " + str(N) + " inputs - C=" + str(C))
plt.savefig('linear_kernel_output_3.pdf')
plt.show()


kernel = linear_kernel
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = 1
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    print("Couldn't find the linear separation")
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Linear Kernel - " + str(N) + " inputs - C=" + str(C))
plt.savefig('linear_kernel_output_4.pdf')
plt.show()


kernel = linear_kernel
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = 10
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    print("Couldn't find the linear separation")
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Linear Kernel - " + str(N) + " inputs - C=" + str(C))
plt.savefig('linear_kernel_output_5.pdf')
plt.show()


kernel = linear_kernel
P = compute_p_matrix(inputs, targets, kernel)
start = numpy.zeros(N)
C = 100
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zero_function}

ret = minimize(objective, start, bounds=B, constraints=XC)
if not ret['success']:
    print("Couldn't find the linear separation")
alpha = ret['x']

support_vectors = get_support_vectors(inputs, targets, alpha)
b = calculate_threshold(alpha, targets, support_vectors, inputs, kernel)

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator_function(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.', label = "Class A")

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.', label = "Class B")

plt.plot([p[0][0] for p in support_vectors],
        [p[0][1] for p in support_vectors],
        'yo', label = "Support Vectors")

plt.axis('equal')
plt.legend(loc='upper right', frameon=False)
plt.title("Linear Kernel - " + str(N) + " inputs - C=" + str(C))
plt.savefig('linear_kernel_output_6.pdf')
plt.show()