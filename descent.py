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

def predict( obs, weights ):
    res = obs
    for w in weights:
        res = act( w @ res )
    return res

def loss( obs, weights, target ):
    pureloss = predict( obs, weights ) - target
    return pureloss.T @ pureloss

def loss_sum( obss, weights, targets ):
    return sum(
        loss(obs, weights, target)
        for obs, target in zip(obss, targets)
    )

w1 = np.random.rand(40, 3) - .5
w2 = np.random.rand(1, 40) - .5
w1 *= 2
w2 *= 2
weights = [ w1, w2 ]
obss = [
    np.array([[ 1, 1, 1 ]]).T,
    np.array([[ 0, 0, 1 ]]).T,
    np.array([[ 0, 1, 1 ]]).T,
    np.array([[ 1, 0, 1 ]]).T,
]
targets = [
    np.array([[ 0 ]]).T,
    np.array([[ 0 ]]).T,
    np.array([[ 1 ]]).T,
    np.array([[ 1 ]]).T,
]

a = 0.005

def dldw( obs, weights, target ):
    w1, w2 = weights

    act1 = w1 @ obs
    hid = act( act1 )
    act2 = w2 @ hid
    pred = act( act2 )
    _loss = pred.T @ pred - 2 * pred.T @ target + target.T @ target

    dlossdpred = 2 * ( pred - target )
    dpreddact2 = dact( act2 )
    dact2dw2 = np.outer( np.ones(act2.shape), hid )
    dact2dhid = w2
    dhiddact1 = dact( act1 )
    dact1dw1 = np.outer( np.ones(act1.shape), obs )

    dlossdw2 = dlossdpred @ dpreddact2 @ dact2dw2
    dlossdw1 = (dlossdpred @ dpreddact2 @ dact2dhid).T @ dhiddact1.T @ dact1dw1

    return dlossdw1, dlossdw2

for epoch in range(40001):
    for i, obs in enumerate(obss):
        target = targets[i]
        dw1, dw2 = dldw( obs, weights, target )
        weights[0] -= a * dw1
        weights[1] -= a * dw2

    if epoch % 100 == 0:
        print(
            "epoch", epoch,
            "- loss", loss_sum( obss, weights, targets )
        )

print("\n\nresults:")
for obs, target in zip(obss, targets):
    print(
        f"predict( {obs[:, 0]}, weights )",
        predict( obs, weights )
    )

print( *weights )





