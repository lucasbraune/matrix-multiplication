# matrix-multiplication

A program that illustrates how optimizing for performance can be surprisingly easy.

This program compares the performance of two algorithms for matrix multiplication. The first algorithm is the naive one, with three nested loops. Here is the source code, written in Java:

```java
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
```

On my quad-core machine, <b>the second algorithm can over 60 times faster than the first</b>. It differs from the first algorithm in two regards:

- it replaces the outer loop in the body of `naiveMultiply` with a parallel stream, and
- it switches the order of the two inner loops.

Here is the code:

```java
static double[][] optimizedMultiply(double[][] A, double[][] B) {
  return Arrays.stream(A).parallel()
                .map(row -> multiply(row, B))
                .toArray(double[][]::new);
}
```

```java
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
```

The inner-most loop of `multiply` reads from a row of a matrix, whereas the inner-most loop of `naiveMultiply` reads from the columns of a matrix. In both methods, each row of the matrix are stored in single arrays, whereas each column is spread over several arrays. For this reason, `optimizedMultiply` enjoys much better locality of reference than `naiveMultiply`. Looking at the performance measurements below, we see how much [CPU caching](https://en.wikipedia.org/wiki/CPU_cache) can affect performance!

For much more on optimization of matrix multiplication, see the first video lecture of <a href="https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-172-performance-engineering-of-software-systems-fall-2018/">this course</a> at MIT.

<b>Reference:</b>

Charles Leiserson and Julian Shun. <i>6.172 Performance Engineering of Software Systems.</i> Fall 2018. Massachusetts Institute of Technology: MIT OpenCouseWare, <a href="https://ocw.mit.edu/">https://ocw.mit.edu/</a>. License: Creative Commons BY-NC-SA.

# Usage

Requires Java 8 or higher.

From the command line interface, compile the source code with
```
javac MatrixMultiplication.java
```
and run it with
```
java MatrixMultiplication
```
The program will generate pairs of random matrices of various sizes, and will multiply each pair twice, once with each algorithm. As it finishes evaluating each product, it will print the runtime of the computation on the standard output stream.

# Performance measurements

Here is the output of the program run on a 2020 MacBook Air with a quad-core Intel Core i5:

```
Time to multiply two N x N matrices

N = 512 
- Optimized: 0.186 s 
- Naive: 1.250 s 

N = 1024 
- Optimized: 0.457 s 
- Naive: 10.384 s 

N = 2048 
- Optimized: 3.867 s 
- Naive: 172.324 s 

N = 4096 
- Optimized: 31.296 s 
- Naive: 2003.425 s 
```

