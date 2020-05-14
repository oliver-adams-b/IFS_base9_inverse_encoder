import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import gumbel_l
from statistics import mean
import pickle

def dim_exp_estimate(edge_matrix):
    n = np.sqrt(len(edge_matrix))
    dimlist = []
    dims = []
    for dim in range(2):
        dimlist.append(max(np.linalg.eigvals(edge_matrix * np.power(1/n, dim))))
        dims.append(dim)
    
    y1, x1 = np.real(dimlist[0]), np.real(dims[0])
    y2, x2 = np.real(dimlist[1]), np.real(dims[1])
    b = np.power(y2/y1, 1/(x2-x1))
    a = y1/np.power(b, x1)
    
    if a == 0:
        a = 0.0000001
        
    return(np.log(1/a)/np.log(b))

    
def spec_rad(edgeMatrix):
    return max(np.abs(np.linalg.eigvals(edgeMatrix)))

def dimension_root(d, A): #returns 0 when d is the dimension for A
        return 1 - spec_rad(A * np.power(1/np.sqrt(np.sqrt(A.size)), d))

def dim_root_estimate(A):
    estimate = 1
    try:
        return optimize.newton(dimension_root, x0 = estimate, args = (A, ))
    except:
        return -1
    
def dim_fit_estimate_9(A):
    x = spec_rad(A)
    a, b, c = [1.00007427e+00, 1.14195706e-04, 1.03869254e-04]
    return inv_power(x, a, b, c)

#working on a method to get an 'even' distribution of dimensions in our dataset 
"""
def bins(n):
    dx = 2/n
    return [[i*dx, (i+1)*dx] for i in range(n)]

def inv_avg_dim_9(dim):
    a = 0.63910036
    b = 1.28623336

    if dim < 2: 
        return [np.power((1/a)*np.log(2-dim/2), (1/b))]
    else: 
        return None

k = 10
l = 1000
set_of_bins = bins(k)
set_of_p = [inv_avg_dim_9(bk[0] + 1/k) for bk in set_of_bins]
set_of_E = [[] for i in range(k)]
set_of_dims = []
i=1 #skip the first bin because it's just all zeroes

while i < len(set_of_bins):
    bk = set_of_bins[i]
    p = set_of_p[i]
    E = abs(np.ceil(np.random.rand(9, 9) - p))
    dim = dim_root_estimate(E)
    
    if dim >= bk[0] and dim <= bk[1]:
        set_of_E[i].append(E)
        set_of_dims.append(dim)
    
    if len(set_of_E[i]) >= l:
        i += 1
        print(bk)
        
 
plt.plot(set_of_dims)


loaded_file = open('base_9_inverse_problem_data/edge_matrices.pkl', 'wb') 
pickle.dump(np.asarray(set_of_E), loaded_file)
loaded_file.close()

loaded_file = open('base_9_inverse_problem_data/dimensions.pkl', 'wb') 
pickle.dump(np.asarray(set_of_dims), loaded_file)
loaded_file.close()
"""



#let's see if there is some kind of relationship between \Phi(A) and \Phi(A_r^\alpha)
def inv_power(x, a, b, c):
    return (1/b)*np.power((x**c-1), 1/a)
"""
#for base 9: 

phi_a = []
phi_ara = []

for i in range(10000):
    A = abs(np.ceil(np.random.rand(9, 9) - np.random.rand(1)[0]))
    dim = dim_root_estimate(A)
    if dim >= 0:
        phi_a.append(spec_rad(A))
        phi_ara.append(dim)
 
p = optimize.curve_fit(inv_power, phi_a, phi_ara)
#plt.scatter(phi_a, [inv_power(x, p[0][0], p[0][1]) for x in phi_a])
plt.scatter(phi_a, [inv_power(x, p[0][0], p[0][1], p[0][2]) for x in phi_a])
plt.scatter(phi_a, phi_ara, s = 0.5)
plt.xlabel("$\Phi(A)$")
plt.ylabel("$\Phi(A_r^a)$")
"""

"""
ole reliable, but slow

number_of_elements_in_soE = 0

while number_of_elements_in_soE < k*l:
    E = abs(np.ceil(np.random.rand(9, 9) - .5))
    dim = abs(dim_root_estimate(E))
    set_of_dims.append(dim)
    
    for index, bk in enumerate(set_of_bins):
        if dim >= bk[0] and dim <= bk[1]:
            set_of_E[index].append(E)
            number_of_elements_in_soE += 1
            print(str(number_of_elements_in_soE) + " out of " + str(k*l))
            
        if len(set_of_E[index]) >= l:
            set_of_bins.remove(bk)
 """        
            

"""   
roughly verifying that the exp estimate of the dimension is approximately 
equal to that of finding it using newton's method for finding the zeros of a function
rootDiffs = []
for i in range(100):
    A = abs(np.ceil(np.random.rand(9, 9) - .1))
    
    estimate = dim_exp_estimate(A)
    root = optimize.newton(dimension_root, x0 = estimate, args = (A, ), maxiter = 1000)
    rootDiffs.append(abs(estimate-root))
    
plt.plot(rootDiffs)
plt.show()
"""

"""
now I want to find some answers to the question: can I find a matrix A such that the dim(F(A)) 
(or by whatever notation) is equal to some a in [0, 2]?  We will use newtons method again! 
TOO HARD ATM LOL FUCK THAT

def desired_dim_root(d, A, a):
    return a - spec_rad(A * np.power(1/np.sqrt(np.sqrt(A.size)), d))

rootDiffs = []
for i in range(10):
    A = A = abs(np.ceil(np.random.rand(9, 9) - .1))
    
    for j in range(2, 40):
    estimate = dim_exp_estimate(A)
    root = optimize.newton(dimension_root, x0 = estimate, args = (A, ), maxiter = 1000000)
    rootDiffs.append(abs(estimate-root))
    
plt.plot(rootDiffs)
plt.title(i)
plt.show()
"""

'''
brute force approach to finding an E such that \Phi(E) = t for some
t in [0, 2]. we fix t and then randomly produce E's, and check to see if we get \Phi(E)
to be relatively (<0.1) close to t. 

t = .78
E = abs(np.ceil(np.random.rand(9, 9) - .5))
attempts = 0

while dim_exp_estimate(E) - t >= 0.01 and attempts < 10000:
    attempts = attempts + 1
    E = abs(np.ceil(np.random.rand(9, 9) - .5))
    

print(E)
print(dim_exp_estimate(E))
'''

def get_dim_spectrum(n, samples, prob):
    set_of_dims = []
    #progress = progressbar(samples, fmt=progressbar.DEFAULT)

    for i in range(samples):
        A = abs(np.ceil(np.random.rand(n*n, n*n) - prob))
         #this matrix represents a fractal in 2 dimensions built with a family
         #of n functions all with scale value 1/n
    
        dimlist = []
        dims = []
        
        for dim in range(2):
            dim = dim/10
            specrad = max(np.linalg.eigvals(A * np.power(1/n, dim)))
            dimlist.append(specrad)
            dims.append(dim)
          
        y1, x1 = dimlist[0], dims[0]
        y2, x2 = dimlist[1], dims[1]
        b = np.power(y2/y1, 1/(x2-x1))
        a = y1/np.power(b, x1)
        d_0 = np.log(1/a)/np.log(b)
        
        set_of_dims.append(d_0)
        
        if(d_0 > 2):
            print('WHAT THE FUCK')
            
    return(set_of_dims)

def generalized_exp(x, a, b):
    return -2*np.exp(a*np.power(x, b))+4
#for each A, characterised by prob p, what is the expected dimension? 
"""   
def get_gumbel_l_distr(p):
    #fit_vals = [dim_root_estimate(abs(np.ceil(np.random.rand(9, 9) - p))) for i in range(500)]
    fit_vals = []
    
    for i in range(1000):
        A = abs(np.ceil(np.random.rand(9, 9) - p))
        if np.max(A) != 0:
            fit_vals.append(dim_root_estimate(A))
    params = gumbel_l.fit(fit_vals)
    return params
       
vals = []
probs = [i/100 for i in range(200)] 

try:
    for i in range(0, 200):
        p = i/100
        vals.append(get_gumbel_l_distr(p)[0])
except:
    print('oops tehe')    

expp = optimize.curve_fit(generalized_exp, probs[:len(vals)], vals)  
"""
def avg_dim_9(P):
    return [generalized_exp(p, 0.63910036, 1.28623336) for p in P]

#plt.scatter(probs[:len(vals)], vals, '-b')
#plt.plot(probs[:len(vals)], avg_dim_9(probs[:len(vals)]), '-b')
#plt.plot(probs[:len(vals)], [inv_avg_dim_9(p) for p in probs[:len(vals)]] , '-r')
#plt.show()

    
"""
for p in range(0, 50):
    prob = p / 100
    dimSpectrum = get_dim_spectrum(16, 4000, prob)
    plt.hist(dimSpectrum, bins = 200, alpha = 0.8, label = str(prob))
    plt.xlabel(prob)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([0, 100])
    plt.show()
"""   
    
    
'''
averages = np.empty(0)

for n in range(2, 12):
    n_dim_spectrum = get_dim_spectrum(n, 50000, 0.5)
    plt.hist(n_dim_spectrum, bins = 200, alpha = 0.8, label = str(n))
    plt.xlabel(n)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    averages = np.append(averages, np.real(np.average(n_dim_spectrum)))

plt.show()
'''

"""
The code below deals with finding the limit of the 'central' maximums (averages-ish) of the 
dimension distribution for random binary nxn matrices as n->inf for a family of n functions acting on the unit square in R^2
We use a curve fitting module to fit the trend to f(x) = (a/(x-c))+b (though this could be changed/optimized), 
where b would approximate the limit of the dimension calculation. The graph of what I am talking about is in the code above. 
# result with 8 samples: 1.7076965499377128
#averages = [ 0, 1.36054057, 1.49834069, 1.5686872 , 1.61289083, 1.64369719, 1.66654948, 1.68450501]
averages = [0, 1.36146502, 1.49791718, 1.56886233, 1.61280861,
       1.64376536, 1.6665913 , 1.6844881 , 1.69892159, 1.71092228]
x_data = np.asarray(range(2, 12))
def test_func(x, a, b, c):
    return (a/(x-c)) + b

params = optimize.curve_fit(test_func, x_data, averages)
print("Dim Limit: params[0][2]")

def cont_frac(list_of_elements):
    result = 0
    for number in reversed(list_of_elements):
        result = 1/(number + result)
        
    return result

approximants = []
for i in range(100):
    sequence = []
    for j in range(i):
        sequence.append(np.math.factorial((j+1)))
    
    approximants.append(cont_frac(sequence))

plt.plot(approximants)
"""
