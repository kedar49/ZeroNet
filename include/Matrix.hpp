
#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <vector>
#include "Neuron.hpp"

/**
 * @class Matrix
 * @brief Represents a matrix for neural network calculations
 * 
 * This class provides matrix operations needed for neural network computations,
 * including matrix multiplication, addition, subtraction, and element-wise operations.
 */
class Matrix {
public:	
    /**
     * @brief Constructor for Matrix
     * @param numRows Number of rows in the matrix
     * @param numCols Number of columns in the matrix
     * @param isRandom Whether to initialize with random values
     */
    Matrix(int numRows, int numCols, bool isRandom);
    
    /**
     * @brief Creates a transposed version of this matrix
     * @return Pointer to the transposed matrix
     */
    Matrix* transpose();
    
    /**
     * @brief Performs element-wise multiplication with another matrix
     * @param m Pointer to the matrix to multiply with
     * @return Pointer to the resulting matrix
     */
    Matrix* elementwiseMultiply(Matrix* m);
    
    /**
     * @brief Multiplies all elements by a scalar value
     * @param scalar The scalar value to multiply by
     */
    void scalarMultiply(double scalar);
    
    /**
     * @brief Converts the matrix to a vector
     * @return Vector containing all matrix elements
     */
    std::vector<double> toVector();
    
    /**
     * @brief Prints the matrix to the console
     */
    void printToConsole();
    
    /**
     * @brief Sets a value at a specific position in the matrix
     * @param row Row index
     * @param col Column index
     * @param val Value to set
     */
    void setVal(int row, int col, double val) { this->values.at(row).at(col) = val; }
    
    /**
     * @brief Generates a random number for matrix initialization
     * @return Random double value
     */
    double getRandNo();
    
    /**
     * @brief Gets a value at a specific position in the matrix
     * @param row Row index
     * @param col Column index
     * @return Value at the specified position
     */
    double getVal(int row, int col) const { return this->values.at(row).at(col); }
    
    /**
     * @brief Gets the number of rows in the matrix
     * @return Number of rows
     */
    int getNumRows() const { return this->numRows; }
    
    /**
     * @brief Gets the number of columns in the matrix
     * @return Number of columns
     */
    int getNumCols() const { return this->numCols; }

    // Operator overloading
    /**
     * @brief Matrix multiplication operator
     * @param b Matrix to multiply with
     * @return Pointer to the resulting matrix
     */
    Matrix* operator*(Matrix& b);
    
    /**
     * @brief Matrix addition operator
     * @param b Matrix to add
     * @return Pointer to the resulting matrix
     */
    Matrix* operator+(Matrix& b);
    
    /**
     * @brief Matrix subtraction operator
     * @param b Matrix to subtract
     * @return Pointer to the resulting matrix
     */
    Matrix* operator-(Matrix& b);
    
private:
    int numRows;                      ///< Number of rows in the matrix
    int numCols;                      ///< Number of columns in the matrix
    std::vector<std::vector<double>> values; ///< Matrix values
};

#endif // _MATRIX_HPP_

