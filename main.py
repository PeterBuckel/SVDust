import re
import cv2
import os
import numpy as np
import scipy.ndimage as ndimage
import time

def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def MRP(img, sz):
    max = ndimage.maximum_filter(img, footprint=np.ones((sz, sz, 3)), mode='nearest')
    return max

def TransmissionEstimate(im, A, sz):
    im_min = ndimage.minimum_filter(im, footprint=np.ones((sz, sz,3)), mode='nearest')
    A_min = ndimage.minimum_filter(A, footprint=np.ones((sz, sz,3)), mode='nearest')
    im_min0 = im_min.min(axis=2)
    A_min0 = A_min.min(axis=2)
    t = 1 - ((im_min0) / (A_min0))
    return t

def TransmissionRefine(im, et, t_r, t_eps):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = t_r
    eps = t_eps
    t = Guidedfilter(gray, et, r, eps)
    return t

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[:, :, ind]) / t + A[:, :, ind]
    return res

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

#Input Folder
folder_with = "Input_folder"
frame = []
for subdirw, dirsw, filesw in os.walk(folder_with):
    filesw.sort(key=natural_keys)

scale = 1
count = 1
w_sz = int(100/scale)
atm_r = int(32/scale)
atm_eps = float(0.01/scale)
t_r = int(200/scale)
t_eps = float(0.01/scale)

for file in filesw:
    start = time.process_time()
    img0 = os.path.join(subdirw, file)
    img = cv2.imread(img0)
    # Calculation of the rough atmospheric light map
    Max_ATM = MRP(img, w_sz)

    I = img.astype('float64') / 255
    # Calculation of the refined atmospheric light map
    newa1 = Guidedfilter(Max_ATM[:, :, 0], I[:, :, 0], atm_r, atm_eps)
    newa2 = Guidedfilter(Max_ATM[:, :, 1], I[:, :, 1],  atm_r, atm_eps)
    newa3 = Guidedfilter(Max_ATM[:, :, 2], I[:, :, 2], atm_r, atm_eps)
    newA = cv2.merge((newa1, newa2, newa3))
    # Calculation of the coarse transmission map
    te = TransmissionEstimate(I, newA, w_sz)
    # Calculation of the refined transmission map
    t = TransmissionRefine(img, te, t_r, t_eps)
    # Recover the dust-free image
    J = Recover(I, t, newA, 0.2)
    # Output folder
    cv2.imwrite('Output_folder/img (' + str(count) + ').png', J*255)
    count += 1

