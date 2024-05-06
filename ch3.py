#!/usr/bin/env python3

import numpy as np

w1 = np.array([
    [ 1, 2, 3 ],
    [ 1, 2, 3 ],
    [ 1, 2, 3 ],
])

w2 = np.array([
    [ 1, 2, 3 ],
    [ 1, 2, 3 ],
    [ 1, 2, 3 ],
])

def predict( obs, weights ):
    res = obs
    for w in weights:
        res = res @ w
    return res

obs1 = [ 1, 2, 3 ]
target1 = [ .5, .5, .5 ]

obs2 = [ 6, 5, 4 ]
target2 = [ .5, -.5, 5. ]

lastloss = 1e+8
for iteration in range(int(1e+5)):
    w1p = w1 + np.random.rand( 3, 3 ) - .5
    w2p = w2 + np.random.rand( 3, 3 ) - .5
    w2p = np.identity(3)
    pred1 = predict( obs1, [ w1p, w2p ] )
    pred2 = predict( obs2, [ w1p, w2p ] )
    loss = (
        np.linalg.norm( pred1 - target1 ) +
        np.linalg.norm( pred2 - target2 )
    )
    if loss < lastloss:
        lastloss = loss
        w1 = w1p
        w2 = w2p
        print( iteration, loss )

print( "\n\ntarget1 =", target1 )
print( "predict( obs1, [ w1, w2 ] ) =", predict( obs1, [ w1, w2 ] ) )
print( "\n\ntarget2 =", target2 )
print( "predict( obs2, [ w1, w2 ] ) =", predict( obs2, [ w1, w2 ] ) )
print( "\nw1 =", w1 )
print( "w2 =", w2 )



