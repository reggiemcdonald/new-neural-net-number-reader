## New Neural Net Number Reader
#### A neural net for reading handwritten numbers built using nothing more than the Java Standard Libraries. 

This is my second attempt at a neural network for reading handwritten numbers, built without the use of ML libraries. If you would like to see my first attempt (which uses a 3rd party linear algebra library) [see here](https://github.com/reggiemcdonald/neural-net-number-reader).

### What is it? 
This is an artificial neural network that is taught using the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) to read handwritten numbers. For example - given an image such as: 
![](number-image.bmp), the network is expected to signal that the input represents a "5". On average, the network will be able to <strong>read the image correctly 97% of the time</strong>.

### What is an Artifical Neural Network? 
To understand aritifical neural networks, its beneficial to first try and understand their biological inspiration. The architecture of an artifical neural network is based upon organic neural networks such as the human brain. The human brain is a dense collection of specialized cells called neurons, which act as atomic thinking units of the brain. Neurons may form connections to other neurons in the brain/body - biologists call these connections synapses. In a more computer science sense, a neuron may be thought of as a graph node, and each synapse as a directed edge to a neighbouring node; the result is a directed graph!

But what makes neurons unique are their non-binary response to stimuli. When a neuron "fires", it releases signals to neighbouring neurons that it forms a synapse with. These neighbouring neurons can integrate the various stimuli that they receive, producing their own response - but it only fires when the integrated signals are high enough. This response can effectively be modelled by the sigmoid function: ![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/2560px-Logistic-curve.svg.png) In the sigmoid function, there is a critical point in the curve where the rate of change in the output grows rapidly. The curve before the critical point is approximately zero. The curve after the critical point is approximately 1. Analogously - the neuron is not firing when the integrated stimulus is below the critical point, and is firing when the integrated stimulus is past the critical point.

Artificial neural networks are based on this premise. The neural network is a series of disjoint sets of neurons - forming layers. A neuron is a specialized case of a graph node, that carries with it the implementation to integrate the signals incoming from each neuron in the previous layer, to produce a unique sigmoidal output which is propagated along to each neuron in the next layer: ![](nn-img1.jpeg) I've only drawn the edges for the first neurons in each layer to keep the image simple. 

To compute the sigmoidal output, we assign each incoming signal a weighting. This gives us a list of weights, with a corresponding list of inputs. To allow a neuron to be more or less likely to send a "firing" signal (sigma = ~1) we can add a (positive or negative) bias onto our neuron. This is effectively an additional input with a weight of 1.0. Putting the weights and inputs into matrices, we have all we need to compute sigma (&#963;), our sigmoid response:![](nn-img2.jpeg)
Repeat this process for all the layers from input to output! 

### Learning 
#### // TODO



