import numpy as np
import scipy.io
import tifffile as tf

def load_data():
    img = tf.imread('data/img.tif')
    lc = tf.imread('data/lc.tif')
    nlcd = tf.imread('data/nlcd.tif')
    return img, lc, nlcd

# p(l|c), these statistics are from [17]
nlcd_mu = np.array([[0.00,0.00,0.00,0.00],[0.97,0.01,0.01,0.02],[0.00,0.00,1.00,0.00],[0.00,0.42,0.46,0.11],[0.01,0.31,0.34,0.35],[0.01,0.14,0.21,0.63],[0.01,0.03,0.07,0.89],[0.09,0.13,0.45,0.32],[0.00,0.92,0.06,0.01],[0.00,0.94,0.05,0.01],[0.00,0.92,0.06,0.02],[0.00,0.50,0.50,0.00],[0.00,0.71,0.26,0.03],[0.01,0.38,0.54,0.07],[0.00,0.50,0.50,0.00],[0.00,0.00,1.00,0.00],[0.00,0.00,1.00,0.00],[0.00,0.11,0.86,0.03],[0.00,0.11,0.86,0.03],[0.01,0.90,0.08,0.00],[0.11,0.07,0.81,0.01],[0.00,0.00,0.00,0.00]])

nlcd_cl = [ 0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95, 255 ]
c2i = {cl:i for i,cl in enumerate(nlcd_cl)}

colors = np.array([[0.,0.,1.],[0.,0.5,0.],[0.5,1.,0.5],[0.5,0.5,0.5]])
def vis_lc(r, sparse=False):
    if sparse: r = np.array([(r==i) for i in range(4)])
    z = np.zeros((3,) + r.shape[1:])
    s = r / r.sum(0)
    for c in range(4):
        for ch in range(3):
            z[ch] += colors[c,ch] * s[c]
    return z

ncolors = np.zeros((len(nlcd_cl),3))
ncolors[1:-1] = scipy.io.loadmat('data/nlcd_legend.mat')['nlcd_cmap'][:,1:] / 255.
def vis_nlcd(r, sparse=False):
    if sparse: r = np.array([(r==nlcd_cl[i]) for i in range(22)])
    z = np.zeros((3,) + r.shape[1:])
    s = r / r.sum(0)
    for c in range(22):
        for ch in range(3):
            z[ch] += ncolors[c,ch] * s[c]
    return z