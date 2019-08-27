package com.reggiemcdonald.neural.net.math;

public interface OutputFunction {

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

}
