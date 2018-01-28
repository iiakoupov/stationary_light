/*
 * Copyright (c) 2017 Ivan Iakoupov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef THREADED_EIGEN_MATRIX_H
#define THREADED_EIGEN_MATRIX_H

#include <omp.h>
#include <vector>

#include "Eigen/Dense"

class ThreadedEigenMatrix
{
    int m_rows;
    int m_cols;
    int m_threads;
    std::vector<Eigen::MatrixXcd> m_matrix_chunks;
    std::vector<int> m_block_start_indices;
public:
    ThreadedEigenMatrix() :
        m_threads(omp_get_max_threads()),
        m_rows(0),
        m_cols(0)
    {}
    explicit ThreadedEigenMatrix(std::function<std::complex<double>(int,int)> f,
                                 int rows, int cols) :
        m_threads(omp_get_max_threads()),
        m_rows(rows),
        m_cols(cols)
    {
        m_matrix_chunks.resize(m_threads);
        const int normal_chunk_size = m_rows/m_threads;
        std::vector<int> chunk_sizes(m_threads);
        for (int n = 0; n < m_threads; ++n) {
            chunk_sizes[n] = normal_chunk_size;
        }

        // The last chunk size can be different
        const int last_chunk_size = m_rows - (m_threads-1)*normal_chunk_size;
        chunk_sizes[m_threads-1] = last_chunk_size;

        #pragma omp parallel for
        for (int n = 0; n < m_threads; ++n) {
            const int chunk_size = chunk_sizes[n];
            m_matrix_chunks[n] = Eigen::MatrixXcd::Zero(chunk_size, m_cols);
            for (int i = 0; i < chunk_size; ++i) {
                for (int j = 0; j < m_cols; ++j) {
                    m_matrix_chunks[n](i,j) = f(i + n*normal_chunk_size, j);
                }
            }
        }
        m_block_start_indices.resize(m_threads);
        int rowsSum = 0;
        for (int j = 0; j < m_threads; ++j) {
            m_block_start_indices[j] = rowsSum;
            rowsSum += m_matrix_chunks[j].rows();
        }
    }
    explicit ThreadedEigenMatrix(Eigen::MatrixXcd M) :
        m_threads(omp_get_max_threads()),
        m_rows(M.rows()),
        m_cols(M.cols())
    {
        m_matrix_chunks.resize(m_threads);
        const int normal_chunk_size = m_rows/m_threads;
        std::vector<int> chunk_sizes(m_threads);
        for (int n = 0; n < m_threads; ++n) {
            chunk_sizes[n] = normal_chunk_size;
        }

        // The last chunk size can be different
        const int last_chunk_size = m_rows - (m_threads-1)*normal_chunk_size;
        chunk_sizes[m_threads-1] = last_chunk_size;

        #pragma omp parallel for
        for (int n = 0; n < m_threads; ++n) {
            const int chunk_size = chunk_sizes[n];
            m_matrix_chunks[n] = Eigen::MatrixXcd::Zero(chunk_size, m_cols);
            for (int i = 0; i < chunk_size; ++i) {
                for (int j = 0; j < m_cols; ++j) {
                    m_matrix_chunks[n](i,j) = M(i + n*normal_chunk_size, j);
                }
            }
        }
        m_block_start_indices.resize(m_threads);
        int rowsSum = 0;
        for (int j = 0; j < m_threads; ++j) {
            m_block_start_indices[j] = rowsSum;
            rowsSum += m_matrix_chunks[j].rows();
        }
    }
    Eigen::VectorXcd operator*(const Eigen::VectorXcd &v) const
    {
        Eigen::VectorXcd ret(m_rows);
        #pragma omp parallel for
        for (int j = 0; j < m_threads; ++j) {
            const int block_start = m_block_start_indices[j];
            const int block_end = block_start + m_matrix_chunks[j].rows();
            Eigen::VectorXcd ret_i = m_matrix_chunks[j]*v;
            for (int k = block_start; k < block_end; ++k) {
                ret(k) = ret_i(k-block_start);
            }
        }
        return ret;
    }
};

#endif // THREADED_EIGEN_MATRIX_H

