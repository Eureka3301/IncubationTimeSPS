import numpy as np

import json

def load_prop(prop_filename):
    '''
    loading any json file
    '''
    with open(prop_filename, 'r') as file:
        props = json.load(file)

    return props

def read_csv(filename):
    '''
    reading any two column file
    '''
    xx = []
    yy = []
    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            x, y = tuple(map(float, line.replace(',','.').split(sep=';')))
            xx.append(x)
            yy.append(y*1e6)
            line = file.readline()
    return (np.array(xx), np.array(yy))

def dim(xx, yy):
    return np.min((len(xx), len(yy)))

def rdim(xx, yy):
    return range(np.min((len(xx), len(yy))))

# if data is not in the right sequence it can be sorted automatically
# i assume it is easier to fix minor bugs manually
def connect(xx, yy, start):
    xmax = np.max(xx)/70 # idk why 70
    ymax = np.max(yy)
    points = [np.array([xx[i]/xmax, yy[i]/ymax]) for i in rdim(xx, yy)]
    newpoints = [points.pop(start)]
    while points:
        next = np.argmin([np.linalg.norm(points[i]-newpoints[-1], ord=2) for i in range(len(points))])
        newpoints.append(points.pop(next))
    
    return ([p[0]*xmax for p in newpoints], [p[1]*ymax for p in newpoints])
