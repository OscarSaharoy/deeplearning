#!/usr/bin/env python3

import numpy as np

w = np.random.rand(3)

def predict( obs, weights ):
    res = np.dot( obs, weights )
    return res

obs = [ 1, 2, 3 ]
target = [ .5, .5, .5 ]

lastloss = 1e+8
for iteration in range(int(1e+5)):
    wp = w + np.sign( np.random.rand( 3 ) - .5 ) / 2.
    pred = predict( obs, wp )
    loss = (
        np.linalg.norm( pred - target )
    )
    if loss < lastloss:
        lastloss = loss
        w = wp
        print( iteration, loss )

print( "\n\ntarget =", target )
print( "predict( obs, w ) =", predict( obs, w ) )
print( "w =", w )




