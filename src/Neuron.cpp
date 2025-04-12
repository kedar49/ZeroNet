#include <cstdlib>
#include "../include/Neuron.hpp"

/**
 * @brief Constructor for Neuron
 * @param val Initial value for the neuron
 */
Neuron::Neuron(double val) {
    this->val = val;
    activate();
    derive();
}

/**
 * @brief Sets the neuron value and recalculates activation and derivative
 * @param val New value for the neuron
 */
void Neuron::setVal(double val) {
    this->val = val;
    activate();
    derive();
}

/**
 * @brief Applies the activation function to the neuron value
 * 
 * Fast Sigmoid Function: f(x) = x / (1 + |x|)
 */
void Neuron::activate() {
    this->activatedVal = this->val / (1 + abs(this->val));
}

/**
 * @brief Calculates the derivative of the activation function
 * 
 * Derivative of Sigmoid: f'(x) = f(x) * (1 - f(x))
 */
void Neuron::derive() {
    this->derivedVal = this->activatedVal * (1 - this->activatedVal);
}

