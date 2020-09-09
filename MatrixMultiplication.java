/* 
 * MIT License
 * 
 * Copyright (c) 2020 Lucas Braune
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BinaryOperator;

/**
 * A program that illustrates how optimizing for performance can be 
 * surprisingly easy.
 * 
 * <p>The program multiplies randomly initialized matrices of various sizes. 
 * Each product is computed twice, with two algorithms. The runtimes of the 
 * computations are printed on the standard output stream.
 * 
 * <p>The first matrix multiplication algorithm is the naive one, with three
 * nested loops. The second algorithm replaces the outer loop in the first by a 
 * parallel stream, and switches the order of the inner loops to avoid cache 
 * misses. On a quad-core machine, <b>the second algorithm can run up over 60
 * times faster than the first</b>, which shows the importance of locality of
 * reference.
 * 
 * <p>For much more on optimization of matrix multiplication, see the first
 * video lecture of 
 * <a href="https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-172-performance-engineering-of-software-systems-fall-2018/">this course</a> 
 * at MIT.
 * 
 * <p><b>Reference:</b>
 * 
 * <p>Charles Leiserson and Julian Shun.
 * <i>6.172 Performance Engineering of Software Systems.</i> Fall 2018. 
 * Massachusetts Institute of Technology:
 * MIT OpenCouseWare, <a href="https://ocw.mit.edu/">https://ocw.mit.edu/</a>. 
 * License: Creative Commons BY-NC-SA.
 * 
 * @author  Lucas Braune
 */
class MatrixMultiplication {

  /**
   * Multiplies randomly initialized matrices of various sizes. Each product is 
   * computed twice, with two algorithsms. The runtimes of the computations are 
   * printed to the standard output stream.
   * 
   * @param   args the command line arguments, not used.
   */
  public static void main(String[] args) {
    System.out.printf("Time to multiply two N by N matrices %n%n");

    int[] dimensions = new int[] {512, 1024, 2048, 4096};
    ThreadLocalRandom rng = ThreadLocalRandom.current();

    for (int N : dimensions) {
      System.out.printf("N = %d %n", N);
      
      double[][] A = randomSquareMatrix(N, rng);
      double[][] B = randomSquareMatrix(N, rng);

      double t = timeInSeconds(MatrixMultiplication::optimizedMultiply, A, B);
      System.out.printf("- Optimized: %.3f s %n", t);

      t = timeInSeconds(MatrixMultiplication::naiveMultiply, A, B);
      System.out.printf("- Naive: %.3f s %n%n", t);
    }
  }

  /**
   * Naively computes the product of two matrices.
   * 
   * @param   A the first matrix
   * @param   B the second matrix
   * @return  the product of the two matrices
   * @throws  NullPointerException if either parameter is null
   * @throws  ArrayIndexOutOfBoundsException if one of the matrices has a 
   *          dimension that is smaller than the number of rows of the first
   *          matrix. 
   */
  static double[][] naiveMultiply(double[][] A, double[][] B) {
    int N = A.length;
    double[][] C = new double[N][N];
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
        for (int k=0; k<N; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return C; 
  }

  /**
   * Computes the product of two matrices. Likely faster than the naive 
   * implementation (see
   * {@link #naiveMultiply(double[][], double[][]) naiveMultiply}).
   * 
   * <p>This implementation of matrix multiplication replaces the outer loop
   * in the body of {@link #naiveMultiply(double[][], double[][]) naiveMultiply} with a 
   * parallel stream, and delegates the two inner loops to the method
   * {@link #multiply(double[], double[][]) multiply}.
   * 
   * @param   A the first matrix
   * @param   B the second matrix
   * @return  the product of the two matrices
   * @throws  NullPointerException if either parameter is null
   * @throws  ArrayIndexOutOfBoundsException if one of the matrices has a 
   *          dimension that is smaller than the number of rows of the first
   *          matrix. 
   */
  static double[][] optimizedMultiply(double[][] A, double[][] B) {
    return Arrays.stream(A).parallel()
                  .map(row -> multiply(row, B))
                  .toArray(double[][]::new);
  }

  /**
   * Computes the product of a row vector by a matrix. 
   * 
   * <p>The implementation uses two nested loops. The order of the loops is 
   * chosen so that the inner loop iterates over a row of the matrix, rather 
   * than iterating over a column. This improves locality of reference and 
   * avoids cache misses, because the contents of a row are stored in a 
   * single array, while the contents of a column are spread over many 
   * arrays.
   * 
   * @param   a the row vector
   * @param   B the matrix
   * @return  the product (row vector times matrix)
   * @throws  NullPointerException if either parameter is null
   * @throws  ArrayIndexOutOfBoundsException if the matrix has a dimension that 
   *          is smaller than the number of entries in the vector
   */
  static double[] multiply(double[] a, double[][] B) {
    int N = a.length;
    double[] c = new double[N];
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        c[j] += a[k] * B[k][j];
      }
    }
    return c;
  }

  /**
   * Returns a square matrix of specified number of rows and columns, whose
   * entries are pseudorandom numbers between zero (inclusive) and one 
   * (exclusive). 
   * 
   * @param   N the number of rows and columns of the returned matrix
   * @param   rng the random number generator
   * @return  a random matrix of the specified size
   * @throws  NegativeArraySizeException if the specified size is negative
   */
  static double[][] randomSquareMatrix(int N, ThreadLocalRandom rng) {
    double[][] A = new double[N][N];
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
        A[i][j] = rng.nextDouble();
      }
    }
    return A;
  }

  /**
   * Returns the runtime, in seconds, of the specified binary operator applied 
   * to the specified inputs.
   *
   * @param   <T> the type of the inputs
   * @param   op the binary operator
   * @param   x the first input of the binary operator
   * @param   y the second input of the binary operator
   * @return  the runtime in seconds
   * @throws  RuntimeException if {@code op.apply(x, y)} throws a runtime 
   *          exception
   */
  static <T> double timeInSeconds(BinaryOperator<? super T> op, T x, T y) {
    long start = System.nanoTime();
    op.apply(x, y);
    long end = System.nanoTime();
    return (end - start) / 1e9;
  }

}