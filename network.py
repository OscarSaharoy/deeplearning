#!/usr/bin/env python3

import numpy as np

def relu( x ):
    return ( ( x > 0 ) + ( x < 0 ) * .01 ) * x
def drelu( x ):
    return ( x > 0 ) + ( x < 0 ) * .01
def sigmoid( x ):
    return 1 / ( 1 + np.exp(-x) )
def dsigmoid( x ):
    return sigmoid(x) * ( 1 - sigmoid(x) )
act = relu
dact = drelu
"""
act = sigmoid
dact = dsigmoid
"""

def predict( obs, weights ):
    res = obs
    for w in weights:
        res = act( w @ res )
    return res

def loss( obs, weights, target ):
    pureloss = predict( obs, weights ) - target
    return pureloss.T @ pureloss

def check( obs, weights, target ):
    return np.argmax( predict( obs, weights ) ) == np.argmax( target )

def test_eval( obss, weights, targets ):
    return sum(
        check( obs, weights, target )
        for obs, target in zip(obss, targets)
    ) / targets.shape[0]

def dldw( obs, weights, target ):
    w1, w2 = weights

    z_hid = w1 @ obs
    hid = act( z_hid )
    z_out = w2 @ hid
    pred = act( z_out )

    error_pred = 2 * ( pred - target ) * dact(z_out)
    error_hid = w2.T @ error_pred * dact(z_hid)

    dlossdw2 = np.outer( error_pred, hid )
    dlossdw1 = np.outer( error_hid, obs )

    return dlossdw1, dlossdw2

# load dataset

with np.load( "mnist.npz", allow_pickle=True ) as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]

# training set

obss = x_train[:10000].reshape( 10000, 28*28 ) / 256
targets = np.zeros((10000, 10))
targets[np.arange(10000), y_train[:10000]] = 1

# test set

obss_test = x_test[:1000].reshape( 1000, 28*28 ) / 256
targets_test = np.zeros((1000, 10))
targets_test[np.arange(1000), y_test[:1000]] = 1

# init network

np.random.seed(1)
w1 = np.random.rand(40, 28*28) - .5
w2 = np.random.rand(10, 40) - .5
w1 *= .2
w2 *= .2
weights = [ w1, w2 ]
a = 0.005

# training loop

try:
    for epoch in range(11):
        for i, obs in enumerate(obss):
            target = targets[i]
            dw1, dw2 = dldw( obs, weights, target )
            weights[0] -= a * dw1
            weights[1] -= a * dw2
        print(
            "epoch", epoch,
            "- test accuracy", f"{test_eval( obss_test, weights, targets_test ) * 100:.1f}%"
        )

except KeyboardInterrupt:
    print("ending training!")

# save the weights to a file

np.savez( "weights.npz", w1=weights[0], w2=weights[1] )

# to load the weights

with np.load( "weights.npz", allow_pickle=True ) as f:
    weights = [ f["w1"], f["w2"] ]
