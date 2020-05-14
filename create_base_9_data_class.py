import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from PIL import Image
from progress_bar import progressbar

class base9data():
    '''
    Here's my attempt to make a class that handles the drawing of
        base 9 matrix data. Takes the 9x9 binary matrics, e0, and outputs the
        fractal encoded by e0. 
    '''
    e0 = []
    size = 700
    res = 3
    
    def get_x0(e0):
        x0 = []
        for i in range(9):
            for j in range(9):
                if e0[i][j] == 1:
                    x0.append(str(i)+str(j))
        return x0   
    
    def get_xn(x0, str_length):
        xn = x0 #initialize Xn
        for loops in range(str_length):
            xnplus1 = []
            for x in xn:
                last = x[-1]
                for y in x0:
                    if y[0] == last:
                        xnplus1.append(x+y[-1])    
            xn = xnplus1       
        return xn
    
    def rgb_to_gray(rgb): 
        rgb = np.asarray(rgb)
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float64)

    def get_points(xn):
        points = []
        for route in xn:
            scale = 2/3
            pos = np.array([0.0, 0.0])
            for j in route:
                dpos = np.empty([0])
                if j == '0':
                    dpos = np.array([-1, -1])
                if j == '1':
                    dpos = np.array([0, -1])
                if j == '2':
                    dpos = np.array([1, -1])
                if j == '3':
                    dpos = np.array([-1, 0])
                if j == '4':
                    dpos = np.array([0, 0])
                if j == '5':
                    dpos = np.array([1, 0])
                if j == '6':
                    dpos = np.array([-1, 1])
                if j == '7':
                    dpos = np.array([0, 1])
                if j == '8':
                    dpos = np.array([1, 1]) 
                dpos = dpos.astype('float64')
                pos += scale*dpos
                scale *= 1/3
            points.append((pos + [1, -1])/2) 
        return np.asarray(points)
    
    def get_im(e0, size, res):
        xn = base9data.get_xn(base9data.get_x0(e0), res)
        points = base9data.get_points(xn) * (size-1)
        im = Image.new("RGB", (size, size), "#FFFFFF")
        pixels = im.load()
        
        for point in points:
            pixels[round(abs(point[0])), round(abs(point[1]))] = (0, 0, 0)
            
        return base9data.rgb_to_gray(im)
    
    def rescale_im(image, size):
        if type(image) == np.ndarray:
            img_data = image
            converted_im = Image.fromarray(img_data).convert("L")
            img_data = np.asarray(converted_im.resize((size, size), Image.ADAPTIVE)) #resizing
            return img_data


"""
rcParams['figure.figsize'] = 16, 8
random = np.int(np.random.rand(1)*1000)
E0 = edge_matrices[-2][random]

path = "/home/oliveradams/Documents/dimension_estimator/base_9_inverse_problem_data/training_data/"

count = 0
for index, BIN in enumerate(edge_matrices):
    #edge matrices was saved in 9 bins, based on the way that the data was saved 
    print("Bin: #"+str(index+1)+"/"+str(len(edge_matrices)))
    progress = progressbar(len(BIN), fmt = progressbar.FULL)
    
    for mat in BIN:
        progress.current += 1
        progress.__call__()
        count += 1

        im = base9data.get_im(mat, 240, 3)
        im = base9data.rescale_im(im, 100)
        pickle.dump([im, mat], open(path+str(count), "wb" ))
    print("\n")
"""

