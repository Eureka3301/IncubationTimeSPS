import numpy as np

import json

def load_prop(prop_filename):
    with open(prop_filename, 'r') as file:
        props = json.load(file)

    return props

def read_csv(filename):
    xx = []
    yy = []
    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            x, y = tuple(map(float, line.replace(',','.').split(sep=';')))
            xx.append(x)
            yy.append(y)
            line = file.readline()
    return (xx, yy)

def dim(xx, yy):
    return np.min((len(xx), len(yy)))

def rdim(xx, yy):
    return range(np.min((len(xx), len(yy))))

def connect(xx, yy, start):
    xmax = np.max(xx)/70
    ymax = np.max(yy)
    points = [np.array([xx[i]/xmax, yy[i]/ymax]) for i in rdim(xx, yy)]
    newpoints = [points.pop(start)]
    while points:
        next = np.argmin([np.linalg.norm(points[i]-newpoints[-1], ord=2) for i in range(len(points))])
        newpoints.append(points.pop(next))
    
    return ([p[0]*xmax for p in newpoints], [p[1]*ymax for p in newpoints])



# def plot_fldr(pic_fldr):
#     fullfilenames = []
#     for filename in os.listdir(pic_fldr):
#         if filename[-4:] == '.csv':
#             fullfilenames.append(os.path.join(pic_fldr, filename))
    
#     plt.figure(figsize=(2.5, 2.5))
#     for filename in fullfilenames:
#         xx, yy = connect(*read_csv(filename), 0)
#         plt.plot(xx, yy, label = os.path.basename(filename).replace('_','/')[:-4])
#         plt.legend()

# def save_fldr(pic_fldr):
#     fullfilenames = []
#     for filename in os.listdir(pic_fldr):
#         if filename[-4:] == '.csv':
#             fullfilenames.append(os.path.join(pic_fldr, filename))
    
#     plt.figure(figsize=(2.5, 2.5))
#     for filename in fullfilenames:
#         xx, yy = connect(*read_csv(filename), 0)
#         plt.plot(xx, yy, label = os.path.basename(filename).replace('_','/')[:-4])
#         plt.legend(loc = 'upper right', bbox_to_anchor=(1.6, 0.8))

#     plt.savefig(pic_fldr+'.jpg', dpi=300)