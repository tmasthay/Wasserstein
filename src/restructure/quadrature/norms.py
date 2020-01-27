import numpy as np

def quad_1D(f, pts, wts, a=-1.0, b=1.0)
    #TODO: assert (len(wts) == len(pts))
    #TODO: a,b not the standard interval
    total = 0.0
    for i in range(0,len(pts)):
        total += wts[i] * f(pts[i])
    return total

def gauss_quad_1D_points(order):
    pts = np.zeros(order)
    wts = np.zeros(order)
    if( order == 1 ):
        pts[0] = 0.0

        wts[0] = 2.0
    elif( order == 2 ):
        pts[0] = -0.57735
        pts[1] = -pts[0]

        wts[0] = 1.0
        wts[1] = 1.0
    elif( order == 3 ):
        pts[0] = 0.0
        pts[1] = -0.774597
        pts[2] = -pts[1]

        wts[0] = 8.0 / 9.0
        wts[1] = 5.0 / 9.0
        wts[2] = 5.0 / 9.0
    elif( order == 4 ):
        pts[0] = -0.339981
        pts[1] = -pts[0]
        pts[2] = -0.861136
        pts[3] = -pts[2]

        wts[0] = 0.652145
        wts[1] = 0.652145
        wts[2] = 0.347855
        wts[3] = 0.347855
    elif( order == 5 ):
        pts[0] = 0.0
        pts[1] = -0.538469
        pts[2] = -pts[1]
        pts[3] = -0.90618
        pts[4] = -pts[3]

        wts[0] = 0.568889
        wts[1] = 0.478629
        wts[2] = wts[1]
        wts[3] = 0.236927
        wts[4] = wts[3]
    else:
        print("WARNING: Gaussian Quadrature exceeding order 5 not supported.")
        return gauss_quad_1D_points(5)
    return pts,wts

    
