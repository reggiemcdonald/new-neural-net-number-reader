package com.reggiemcdonald.neural.feedforward.net.math;

import java.io.Serializable;

public interface OutputFunction extends Serializable {

    /**
     * Compute the output
     * @return
     */
    double compute ();

    /**
     * Compute the derivative of the output
     * @return
     */
    double derivative (double d);

    /**
     * Return the zed parameter
     * @return
     */
    double z ();


}
