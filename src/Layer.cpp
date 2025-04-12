#include "../include/Layer.hpp"
#include "../include/Matrix.hpp"

using namespace std;

/**
 * @brief Constructor for Layer
 * @param size Number of neurons in the layer
 */
Layer::Layer(int size) {
    for (int i = 0; i < size; i++) {
        Neuron* n = new Neuron(0.00);
        this->neurons.push_back(n);
    }
    this->size = size;
}

/**
 * @brief Cleans up memory by deleting all neurons
 */
void Layer::cleanup() {
    for (int i = 0; i < this->neurons.size(); i++) {
        delete this->neurons.at(i);
    }
}

/**
 * @brief Sets the value of a specific neuron in the layer
 * @param index Index of the neuron
 * @param value New value for the neuron
 */
void Layer::setNeuronVal(int index, double value) {
    this->neurons.at(index)->setVal(value);
}

/**
 * @brief Gets all neurons in the layer
 * @return Vector of pointers to neurons
 */
vector<Neuron*> Layer::getNeurons() {
    vector<Neuron*> temp;
    for (int i = 0; i < this->neurons.size(); i++) {
        temp.push_back(this->neurons.at(i));
    }

    return temp;
}

/**
 * @brief Converts raw neuron values to a matrix
 * @return Matrix containing raw neuron values
 */
Matrix* Layer::matrixifyVals() {
    Matrix* m = new Matrix(this->neurons.size(), 1, false);
    for (int i = 0; i < this->neurons.size(); i++) {
        m->setVal(i, 0, neurons.at(i)->getVal());
    }
    
    return m;
}

/**
 * @brief Converts activated neuron values to a matrix
 * @return Matrix containing activated neuron values
 */
Matrix* Layer::matrixifyActivatedVals() {
    Matrix* m = new Matrix(this->neurons.size(), 1, false);
    for (int i = 0; i < this->neurons.size(); i++) {
        m->setVal(i, 0, neurons.at(i)->getActivatedVal());
    }
    
    return m;
}

/**
 * @brief Converts derived neuron values to a matrix
 * @return Matrix containing derived neuron values
 */
Matrix* Layer::matrixifyDerivedVals() {
    Matrix* m = new Matrix(this->neurons.size(), 1, false);
    for (int i = 0; i < this->neurons.size(); i++) {
        m->setVal(i, 0, neurons.at(i)->getDerivedVal());
    }
    
    return m;
}
