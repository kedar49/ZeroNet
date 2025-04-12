#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include <vector>
#include "Neuron.hpp"
#include "Matrix.hpp"

using namespace std;

/**
 * @class Layer
 * @brief Represents a layer of neurons in a neural network
 * 
 * This class manages a collection of neurons that form a layer in the neural network.
 * It provides methods to access neuron values and convert them to matrix format for calculations.
 */
class Layer {
public:	
    /**
     * @brief Constructor for Layer
     * @param size Number of neurons in the layer
     */
    Layer(int size); 
    
    /**
     * @brief Sets the value of a specific neuron in the layer
     * @param index Index of the neuron
     * @param value New value for the neuron
     */
    void setNeuronVal(int index, double value); 
    
    /**
     * @brief Gets the raw value of a specific neuron
     * @param index Index of the neuron
     * @return Raw value of the neuron
     */
    double getNeuronVal(int index) const { return this->neurons.at(index)->getVal(); }
    
    /**
     * @brief Gets all neurons in the layer
     * @return Vector of pointers to neurons
     */
    vector<Neuron*> getNeurons();
    
    /**
     * @brief Converts raw neuron values to a matrix
     * @return Matrix containing raw neuron values
     */
    Matrix* matrixifyVals();
    
    /**
     * @brief Converts activated neuron values to a matrix
     * @return Matrix containing activated neuron values
     */
    Matrix* matrixifyActivatedVals();
    
    /**
     * @brief Converts derived neuron values to a matrix
     * @return Matrix containing derived neuron values
     */
    Matrix* matrixifyDerivedVals();
    
    /**
     * @brief Cleans up memory by deleting all neurons
     */
    void cleanup();
    
private:
    int size;                  ///< Number of neurons in the layer
    vector<Neuron*> neurons; ///< Collection of neurons in the layer
};

#endif // _LAYER_HPP_
