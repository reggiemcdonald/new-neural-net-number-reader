package com.reggiemcdonald.neural.feedforward.net;

import java.io.Serializable;

public enum LayerType implements Serializable {

    // CNNLayer of input neurons
    INPUT,

    // The middle layers
    HIDDEN,

    // The output layers
    OUTPUT,
}
