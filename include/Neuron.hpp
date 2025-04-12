#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <iostream>

/**
 * @class Neuron
 * @brief Represents a single neuron in a neural network
 * 
 * This class implements a neuron with activation function and its derivative.
 * It uses a fast sigmoid function for activation: f(x) = x / (1 + |x|)
 */
class Neuron {
public:	
    /**
     * @brief Constructor for Neuron
     * @param val Initial value for the neuron
     */
    Neuron(double val);
    
    /**
     * @brief Sets the neuron value and recalculates activation and derivative
     * @param val New value for the neuron
     */
    void setVal(double val);
    
    /**
     * @brief Applies the activation function to the neuron value
     * 
     * Fast Sigmoid Function: f(x) = x / (1 + |x|)
     */
    void activate();

    /**
     * @brief Calculates the derivative of the activation function
     * 
     * Derivative of Sigmoid: f'(x) = f(x) * (1 - f(x))
     */
    void derive();

    // Getters
    /**
     * @brief Gets the raw neuron value
     * @return The raw value of the neuron
     */
    double getVal() const { return this->val; }
    
    /**
     * @brief Gets the activated neuron value
     * @return The activated value of the neuron
     */
    double getActivatedVal() const { return this->activatedVal; }
    
    /**
     * @brief Gets the derivative of the activated neuron value
     * @return The derivative of the activated value
     */
    double getDerivedVal() const { return this->derivedVal; }

private:
    double val;           ///< Raw neuron value
    double activatedVal;  ///< Value after applying activation function
    double derivedVal;    ///< Derivative of the activation function at current value
};

#endif // _NEURON_HPP_
