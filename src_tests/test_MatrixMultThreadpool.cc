// ██████╗ ███████╗ ██████╗ ██╗   ██╗███████╗███████╗████████╗
// ██╔══██╗██╔════╝██╔═══██╗██║   ██║██╔════╝██╔════╝╚══██╔══╝
// ██████╔╝█████╗  ██║   ██║██║   ██║█████╗  ███████╗   ██║
// ██╔══██╗██╔══╝  ██║▄▄ ██║██║   ██║██╔══╝  ╚════██║   ██║
// ██║  ██║███████╗╚██████╔╝╚██████╔╝███████╗███████║   ██║
// ╚═╝  ╚═╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚══════╝   ╚═╝
//
// Write a C++11 or later program that utilizes the Eigen3 library to perform matrix-matrix multiplication using block partitioning.
// Given matrices A and B, compute the matrix C = A * B. Matrices A and B must be compatible for multiplication. Given the integers N, P, M, partition the matrices as follows:
//  - Matrix A into N x P blocks
//  - Matrix B into P x M blocks
//  - Matrix C into N x M blocks
// Ensure that the partitioning is compatible. If matrices A and B are incompatible, or if the required partitioning (N, P, M) is not possible, throw an exception.
// Each (i, j) block of matrix C must be computed on a separate thread if available, enabling parallel code execution.
// Finally, compare the execution speed of your block partitioning matrix multiplication with the timing of the standard Eigen3 matrix multiplication command.
// Use the proposed ThreadPool to find a better one to perform parallel tasks.

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <random>

#include "Utils_eigen.hh"
#include "Utils_fmt.hh"


using mat     = Eigen::MatrixXd;
using integer = Eigen::Index;

std::random_device rd;  // a seed source for the random number engine
std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
//std::uniform_int_distribution<> distrib(20,30);
std::uniform_int_distribution<> distrib(2,25);

class BlockMult {

  std::mutex mtx;
  Utils::ThreadPool0 Pool0{5};
  Utils::ThreadPool1 Pool1{5};
  Utils::ThreadPool2 Pool2{5};
  Utils::ThreadPool3 Pool3{5};
  Utils::ThreadPool4 Pool4{5};
  Utils::ThreadPool5 Pool5{5};
  Utils::ThreadPool6 Pool6{5};

  std::vector<integer> const * m_i_block;
  std::vector<integer> const * m_j_block;
  std::vector<integer> const * m_k_block;

  // (n x m) * (m x p)
  void
  Compute_C_block(
    mat const & A,
    mat const & B,
    mat       & C,
    integer     i,
    integer     j
  );

public:
  BlockMult() {}

  bool
  multiply(
    integer                    ntp,
    mat const                  & A,
    mat const                  & B,
    mat                        & C,
    std::vector<integer> const & i_block,
    std::vector<integer> const & j_block,
    std::vector<integer> const & k_block
  );

  ~BlockMult() {}

};

//-----------------------------------------------------

void
BlockMult::Compute_C_block(
  mat const & A,
  mat const & B,
  mat       & C,
  integer     i,
  integer     j
) {
  auto II = Eigen::seqN( (*m_i_block)[i-1], (*m_i_block)[i]-(*m_i_block)[i-1] );
  auto JJ = Eigen::seqN( (*m_j_block)[j-1], (*m_j_block)[j]-(*m_j_block)[j-1] );
  for ( size_t k{1}; k < m_k_block->size(); ++k ) {
    auto KK = Eigen::seqN( (*m_k_block)[k-1], (*m_k_block)[k]-(*m_k_block)[k-1] );
    C(II,JJ) += A(II,KK)*B(KK,JJ);
  }
}

//-----------------------------------------------------

bool
BlockMult::multiply(
  integer                    ntp,
  mat const                  & A,
  mat const                  & B,
  mat                        & C,
  std::vector<integer> const & i_block,
  std::vector<integer> const & j_block,
  std::vector<integer> const & k_block
) {

  if ( A.cols() != B.rows())  {
    fmt::print(
      "Invalid matrix multiplication. Found {} x {} Times {} x {} ",
      A.cols(), A.rows(), B.cols(), B.rows()
    );
    return false;
  }

  m_i_block = &i_block;
  m_j_block = &j_block;
  m_k_block = &k_block;

  C.setZero();

  //#define USE_RUN

  switch ( ntp ) {
  case 0:
    for ( integer i{1}; i < integer(i_block.size()); ++i ) {
      for ( integer j{1}; j < integer(j_block.size()); ++j ) {
        #ifdef USE_RUN
        Pool0.run( &BlockMult::Compute_C_block, this, std::ref(A), std::ref(B), std::ref(C), i, j );
        #else
        auto fun = [this, &A, &B, &C, i, j]() -> void { this->Compute_C_block( A, B, C, i, j ); };
        Pool0.exec( fun );
        #endif
      }
    }
    Pool0.wait();
    break;
  case 1:
    for ( integer i{1}; i < integer(i_block.size()); ++i ) {
      for ( integer j{1}; j < integer(j_block.size()); ++j ) {
        #ifdef USE_RUN
        Pool1.run( &BlockMult::Compute_C_block, this, std::ref(A), std::ref(B), std::ref(C), i, j );
        #else
        auto fun = [this, &A, &B, &C, i, j]() -> void { this->Compute_C_block( A, B, C, i, j ); };
        Pool1.exec( fun );
        #endif
      }
    }
    Pool1.wait();
    break;
  case 2:
    for ( integer i{1}; i < integer(i_block.size()); ++i ) {
      for ( integer j{1}; j < integer(j_block.size()); ++j ) {
        #ifdef USE_RUN
        Pool2.run( &BlockMult::Compute_C_block, this, std::ref(A), std::ref(B), std::ref(C), i, j );
        #else
        auto fun = [this, &A, &B, &C, i, j]() -> void { this->Compute_C_block( A, B, C, i, j ); };
        Pool2.exec( fun );
        #endif
      }
    }
    Pool2.wait();
    break;
  case 3:
    for ( integer i{1}; i < integer(i_block.size()); ++i ) {
      for ( integer j{1}; j < integer(j_block.size()); ++j ) {
        #ifdef USE_RUN
        Pool3.run( &BlockMult::Compute_C_block, this, std::ref(A), std::ref(B), std::ref(C), i, j );
        #else
        auto fun = [this, &A, &B, &C, i, j]() -> void { this->Compute_C_block( A, B, C, i, j ); };
        Pool3.exec( fun );
        #endif
      }
    }
    Pool3.wait();
    break;
  case 4:
    for ( integer i{1}; i < integer(i_block.size()); ++i ) {
      for ( integer j{1}; j < integer(j_block.size()); ++j ) {
        #ifdef USE_RUN
        Pool4.run( &BlockMult::Compute_C_block, this, std::ref(A), std::ref(B), std::ref(C), i, j );
        #else
        auto fun = [this, &A, &B, &C, i, j]() -> void { this->Compute_C_block( A, B, C, i, j ); };
        Pool4.exec( fun );
        #endif
      }
    }
    Pool4.wait();
    break;
  case 5:
    for ( integer i{1}; i < integer(i_block.size()); ++i ) {
      for ( integer j{1}; j < integer(j_block.size()); ++j ) {
        #ifdef USE_RUN
        Pool5.run( &BlockMult::Compute_C_block, this, std::ref(A), std::ref(B), std::ref(C), i, j );
        #else
        auto fun = [this, &A, &B, &C, i, j]() -> void { this->Compute_C_block( A, B, C, i, j ); };
        Pool5.exec( fun );
        #endif
      }
    }
    Pool5.wait();
    break;
  case 6:
    for ( integer i{1}; i < integer(i_block.size()); ++i ) {
      for ( integer j{1}; j < integer(j_block.size()); ++j ) {
        #ifdef USE_RUN
        Pool6.run( &BlockMult::Compute_C_block, this, std::ref(A), std::ref(B), std::ref(C), i, j );
        #else
        auto fun = [this, &A, &B, &C, i, j]() -> void { this->Compute_C_block( A, B, C, i, j ); };
        Pool6.exec( fun );
        #endif
      }
    }
    Pool6.wait();
    break;
  default:
    fmt::print("ERROR\n\n\n");
  }
  return true;
}

int
main() {
  Utils::TicToc tm;
  Eigen::initParallel();
  fmt::print("Eigen Test\n");
  double mean   = 0.0;
  double stdDev = 0.0;
  Eigen::MatrixXd M1, M2, M3a, M3b;
  int n_runs = 3;
  Eigen::VectorXd times(n_runs);
  Eigen::VectorXd stdDev_vec(n_runs);
  int N = 800;
  int P = 400;
  int M = 1200;
  int n = N/40;
  int p = P/30;
  int m = M/50;
  M1.resize(N,P);
  M2.resize(P,M);
  M3a.resize(N,M);
  M3b.resize(N,M);

  M1 = Eigen::MatrixXd::Random(N,P);
  M2 = Eigen::MatrixXd::Random(P,M);

  fmt::print("Standard Product\n");
  for ( int i{0}; i < n_runs; i++) {
    tm.tic();
    M3a = M1 * M2;
    tm.toc();
    times(i) = tm.elapsed_ms();
  }
  mean   = times.mean();
  stdDev = (((times.array() - mean) * (times.array() - mean)).sqrt()).sum()/((double)(n_runs-1));
  fmt::print( "time: {:8.4}ms {:8.4}ms (sdev)\n\n\n", mean, stdDev );

  std::vector<integer> i_block;
  std::vector<integer> j_block;
  std::vector<integer> k_block;

  i_block.clear(); i_block.reserve(n+1); i_block.emplace_back(0);
  k_block.clear(); k_block.reserve(p+1); k_block.emplace_back(0);
  j_block.clear(); j_block.reserve(m+1); j_block.emplace_back(0);

  {
    while ( i_block.back() < N ) i_block.emplace_back( i_block.back() + distrib(gen) );
    i_block.back() = N;
  }

  {
    while ( j_block.back() < M ) j_block.emplace_back( j_block.back() + distrib(gen) );
    j_block.back() = M;
  }

  {
    while ( k_block.back() < P ) k_block.emplace_back( k_block.back() + distrib(gen) );
    k_block.back() = P;
  }

  for ( int nptp{0}; nptp <= 5; ++nptp ) {
    BlockMult BM;
    for ( int i{0}; i < n_runs; ++i ) {
      tm.tic();
      BM.multiply( nptp, M1, M2, M3b, i_block, j_block, k_block );
      tm.toc();
      times(i) = tm.elapsed_ms();
    }
    mean   = times.mean();
    stdDev = (((times.array() - mean) * (times.array() - mean)).sqrt()).sum()/((double)(n_runs-1));
    fmt::print(
      "time (#{}): {:8.4}ms {:8.4}ms (sdev) Check M3a - M3b: {:8.4}\n\n",
      nptp, mean, stdDev, (M3a-M3b).norm()
    );
  }

  return 0;
}
