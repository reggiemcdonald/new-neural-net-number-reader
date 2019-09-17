package com.reggiemcdonald.neural.feedforward.net;

import java.io.Serializable;

public enum LayerType implements Serializable {

    // FullyConnectedLayer of wrapInput neurons
    INPUT,

    // The middle layers
    HIDDEN,

    // The output layers
    OUTPUT,
}
