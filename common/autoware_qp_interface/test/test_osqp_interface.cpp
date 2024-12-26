// Copyright 2023 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/qp_interface/osqp_interface.hpp"
#include "gtest/gtest.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using autoware::qp_interface::CSC_Matrix;
using autoware::qp_interface::OSQPInterface;

namespace
{
// Problem taken from https://github.com/osqp/osqp/blob/master/tests/basic_qp/generate_problem.py
//
// min  1/2 * x' * P * x  + q' * x
// s.t. lb <= A * x <= ub
//
// P = [4, 1], q = [1], A = [1, 1], lb = [   1], ub = [1.0]
//     [1, 2]      [1]      [1, 0]       [   0]       [0.7]
//                          [0, 1]       [   0]       [0.7]
//                          [0, 1]       [-inf]       [inf]
//
// The optimal solution is
// x = [0.3, 0.7]'
// y = [-2.9, 0.0, 0.2, 0.0]`
// obj = 1.88

TEST(TestOsqpInterface, BasicQp)
{
  using autoware::qp_interface::calCSCMatrix;
  using autoware::qp_interface::calCSCMatrixTrapezoidal;
  using autoware::qp_interface::CSC_Matrix;

  auto check_result =
    [](const auto & solution, const std::string & status, const int polish_status) {
      EXPECT_EQ(status, "OSQP_SOLVED");
      EXPECT_EQ(polish_status, 1);

      static const auto ep = 1.0e-8;

      ASSERT_EQ(solution.size(), size_t(2));
      EXPECT_NEAR(solution[0], 0.3, ep);
      EXPECT_NEAR(solution[1], 0.7, ep);
    };

  const Eigen::MatrixXd P = (Eigen::MatrixXd(2, 2) << 4, 1, 1, 2).finished();
  const Eigen::MatrixXd A = (Eigen::MatrixXd(4, 2) << 1, 1, 1, 0, 0, 1, 0, 1).finished();
  const std::vector<double> q = {1.0, 1.0};
  const std::vector<double> l = {1.0, 0.0, 0.0, -autoware::qp_interface::OSQP_INF};
  const std::vector<double> u = {1.0, 0.7, 0.7, autoware::qp_interface::OSQP_INF};

  {
    // Define problem during optimization
    autoware::qp_interface::OSQPInterface osqp(false, 4000, 1e-6);
    const auto solution = osqp.QPInterface::optimize(P, A, q, l, u);
    const auto status = osqp.getStatus();
    const auto polish_status = osqp.getPolishStatus();
    check_result(solution, status, polish_status);
  }

  {
    // Define problem during initialization
    autoware::qp_interface::OSQPInterface osqp(false, 4000, 1e-6);
    const auto solution = osqp.QPInterface::optimize(P, A, q, l, u);
    const auto status = osqp.getStatus();
    const auto polish_status = osqp.getPolishStatus();
    check_result(solution, status, polish_status);
  }

  {
    std::tuple<std::vector<double>, std::vector<double>, int, int, int> result;
    // Dummy initial problem
    Eigen::MatrixXd P_ini = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd A_ini = Eigen::MatrixXd::Zero(4, 2);
    std::vector<double> q_ini(2, 0.0);
    std::vector<double> l_ini(4, 0.0);
    std::vector<double> u_ini(4, 0.0);
    autoware::qp_interface::OSQPInterface osqp(false, 4000, 1e-6);
    osqp.QPInterface::optimize(P_ini, A_ini, q_ini, l_ini, u_ini);
  }

  {
    // Define problem during initialization with csc matrix
    CSC_Matrix P_csc = calCSCMatrixTrapezoidal(P);
    CSC_Matrix A_csc = calCSCMatrix(A);
    autoware::qp_interface::OSQPInterface osqp(false, 4000, 1e-6);

    const auto solution = osqp.optimize(P_csc, A_csc, q, l, u);
    const auto status = osqp.getStatus();
    const auto polish_status = osqp.getPolishStatus();
    check_result(solution, status, polish_status);
  }

  {
    std::tuple<std::vector<double>, std::vector<double>, int, int, int> result;
    // Dummy initial problem with csc matrix
    CSC_Matrix P_ini_csc = calCSCMatrixTrapezoidal(Eigen::MatrixXd::Zero(2, 2));
    CSC_Matrix A_ini_csc = calCSCMatrix(Eigen::MatrixXd::Zero(4, 2));
    std::vector<double> q_ini(2, 0.0);
    std::vector<double> l_ini(4, 0.0);
    std::vector<double> u_ini(4, 0.0);
    autoware::qp_interface::OSQPInterface osqp(false, 4000, 1e-6);
    osqp.optimize(P_ini_csc, A_ini_csc, q_ini, l_ini, u_ini);

    // Redefine problem before optimization
    CSC_Matrix P_csc = calCSCMatrixTrapezoidal(P);
    CSC_Matrix A_csc = calCSCMatrix(A);

    const auto solution = osqp.optimize(P_csc, A_csc, q, l, u);
    const auto status = osqp.getStatus();
    const auto polish_status = osqp.getPolishStatus();
    check_result(solution, status, polish_status);
  }

  // add warm startup
  {
    // Dummy initial problem with csc matrix
    CSC_Matrix P_ini_csc = calCSCMatrixTrapezoidal(Eigen::MatrixXd::Zero(2, 2));
    CSC_Matrix A_ini_csc = calCSCMatrix(Eigen::MatrixXd::Zero(4, 2));
    std::vector<double> q_ini(2, 0.0);
    std::vector<double> l_ini(4, 0.0);
    std::vector<double> u_ini(4, 0.0);
    autoware::qp_interface::OSQPInterface osqp(true, 4000, 1e-6);  // enable warm start
    osqp.optimize(P_ini_csc, A_ini_csc, q_ini, l_ini, u_ini);

    // Redefine problem before optimization
    CSC_Matrix P_csc = calCSCMatrixTrapezoidal(P);
    CSC_Matrix A_csc = calCSCMatrix(A);
    {
      const auto solution = osqp.optimize(P_csc, A_csc, q, l, u);
      const auto status = osqp.getStatus();
      const auto polish_status = osqp.getPolishStatus();
      check_result(solution, status, polish_status);

      osqp.updateCheckTermination(1);
      const auto primal_val = solution;
      const auto dual_val = osqp.getDualSolution();
      for (size_t i = 0; i < primal_val.size(); ++i) {
        std::cerr << primal_val.at(i) << std::endl;
      }
      osqp.setWarmStart(primal_val, dual_val);
    }

    {
      const auto solution = osqp.optimize(P_csc, A_csc, q, l, u);
      const auto status = osqp.getStatus();
      const auto polish_status = osqp.getPolishStatus();
      check_result(solution, status, polish_status);
    }

    // NOTE: This should be true, but currently optimize function reset the workspace, which
    // disables warm start.
    // EXPECT_EQ(osqp.getTakenIter(), 1);
  }
}

// Helper function to create a simple problem
void createSimpleProblem(
  Eigen::MatrixXd & P, Eigen::MatrixXd & A, std::vector<double> & q, std::vector<double> & l,
  std::vector<double> & u)
{
  P.resize(2, 2);
  P << 1, 0, 0, 1;
  A.resize(2, 2);
  A << 1, 1, 1, 0;
  q = {0, 0};
  l = {-1, -1};
  u = {1, 1};
}

// Test default constructor
TEST(OSQPInterfaceTest, DefaultConstructor)
{
  OSQPInterface osqp;
  EXPECT_FALSE(osqp.isSolved());
}

// Test constructor with problem setup
TEST(OSQPInterfaceTest, ConstructorWithProblemSetup)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  OSQPInterface osqp(P, A, q, l, u);
  EXPECT_FALSE(osqp.isSolved());  // ?
}

// Test constructor with CSC matrix
TEST(OSQPInterfaceTest, ConstructorWithCSCMatrix)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  CSC_Matrix P_csc = autoware::qp_interface::calCSCMatrixTrapezoidal(P);
  CSC_Matrix A_csc = autoware::qp_interface::calCSCMatrix(A);
  OSQPInterface osqp(P_csc, A_csc, q, l, u);
  EXPECT_FALSE(osqp.isSolved());
}

// Test constructor with invalid input
TEST(OSQPInterfaceTest, ConstructorWithInvalidInput)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  P.resize(3, 3);  // Invalid size
  EXPECT_THROW(OSQPInterface osqp(P, A, q, l, u), std::invalid_argument);
}

// Test optimize method
TEST(OSQPInterfaceTest, OptimizeMethod)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  OSQPInterface osqp(P, A, q, l, u);
  CSC_Matrix P_csc = autoware::qp_interface::calCSCMatrixTrapezoidal(P);
  CSC_Matrix A_csc = autoware::qp_interface::calCSCMatrix(A);
  std::vector<double> result = osqp.optimize(P_csc, A_csc, q, l, u);
  EXPECT_EQ(result.size(), 2);
}

// Test optimize method with invalid input
TEST(OSQPInterfaceDeathTest, OptimizeMethodWithInvalidInput)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  P(0, 1) = 2;  // Invalid input: nonpositive definite matrix
  OSQPInterface osqp(P, A, q, l, u);
  CSC_Matrix P_csc = autoware::qp_interface::calCSCMatrixTrapezoidal(P);
  CSC_Matrix A_csc = autoware::qp_interface::calCSCMatrix(A);
  EXPECT_DEATH(osqp.optimize(P_csc, A_csc, q, l, u), "");
}

// Test update methods
TEST(OSQPInterfaceTest, UpdateMethods)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  OSQPInterface osqp(P, A, q, l, u);
  osqp.updateEpsAbs(1e-6);
  osqp.updateEpsRel(1e-6);
  osqp.updateVerbose(true);
  osqp.updateMaxIter(1000);
  osqp.updateRho(0.1);
  osqp.updateAlpha(1.8);
  osqp.updateScaling(1);
  osqp.updatePolish(true);
  osqp.updatePolishRefinementIteration(5);
  osqp.updateCheckTermination(10);
  osqp.updateP(P);
  osqp.updateA(A);
  osqp.updateQ(q);
  osqp.updateL(l);
  osqp.updateU(u);
  osqp.updateBounds(l, u);
}

// Test update methods with invalid input
TEST(OSQPInterfaceTest, UpdateMethodsWithInvalidInput)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  OSQPInterface osqp(P, A, q, l, u);
  P.resize(3, 3);                    // Invalid size
  EXPECT_NO_THROW(osqp.updateP(P));  // ?
}

// Test get methods
TEST(OSQPInterfaceTest, GetMethods)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  OSQPInterface osqp(P, A, q, l, u);
  EXPECT_EQ(osqp.getIterationNumber(), 0);  // ?
  EXPECT_EQ(osqp.getStatus(), "OSQP_SOLVED");
  EXPECT_FALSE(osqp.isSolved());         // ?
  EXPECT_NE(osqp.getPolishStatus(), 1);  // ?
  std::vector<double> dual_solution = osqp.getDualSolution();
  EXPECT_EQ(dual_solution.size(), 2);
}

// Test get methods with uninitialized state
TEST(OSQPInterfaceDeathTest, GetMethodsUninitialized)
{
  OSQPInterface osqp;
  EXPECT_DEATH(
    {
      osqp.getIterationNumber();
      osqp.getStatus();
      osqp.isSolved();
      osqp.getPolishStatus();
      osqp.getDualSolution();
    },
    "");
}

// Test warm start methods
TEST(OSQPInterfaceTest, WarmStartMethods)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  OSQPInterface osqp(P, A, q, l, u, true);
  std::vector<double> primal_variables = {0.5, 0.5};
  std::vector<double> dual_variables = {0.5, 0.5};
  EXPECT_TRUE(osqp.setWarmStart(primal_variables, dual_variables));
  EXPECT_TRUE(osqp.setPrimalVariables(primal_variables));
  EXPECT_TRUE(osqp.setDualVariables(dual_variables));
}

// Test warm start methods with invalid input
TEST(OSQPInterfaceTest, WarmStartMethodsInvalidInput)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  OSQPInterface osqp(P, A, q, l, u, true);
  std::vector<double> invalid_primal_variables = {0.5};  // Invalid size
  std::vector<double> invalid_dual_variables = {0.5};    // Invalid size
  EXPECT_FALSE(osqp.setWarmStart(invalid_primal_variables, invalid_dual_variables));
  EXPECT_FALSE(osqp.setPrimalVariables(invalid_primal_variables));
  EXPECT_FALSE(osqp.setDualVariables(invalid_dual_variables));
}

// Test boundary condition with large matrix
TEST(OSQPInterfaceTest, BoundaryConditionLargeMatrix)
{
  int n = 100, m = 50;
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(n, n);
  P = P * P.transpose();  // Make P positive semi-definite
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
  std::vector<double> q(n, 0.0);
  std::vector<double> l(m, -1.0);
  std::vector<double> u(m, 1.0);
  OSQPInterface osqp(P, A, q, l, u);
  EXPECT_TRUE(osqp.isSolved());
}

// Test exception handling for memory allocation failure
TEST(OSQPInterfaceTest, MemoryAllocationFailure)
{
  // This test requires a way to simulate memory allocation failure
  // which is not straightforward in C++. For demonstration, we assume
  // a hypothetical function that can simulate this.
  // In practice, this might require using a memory profiler or a custom allocator.
  // For now, we skip this test.
  GTEST_SKIP() << "Memory allocation failure test skipped";
}

// Test branch coverage for update methods
TEST(OSQPInterfaceTest, UpdateMethodsBranchCoverage)
{
  Eigen::MatrixXd P, A;
  std::vector<double> q, l, u;
  createSimpleProblem(P, A, q, l, u);
  OSQPInterface osqp(P, A, q, l, u);

  // Test updateEpsAbs
  osqp.updateEpsAbs(1e-6);
  osqp.updateEpsAbs(-1e-6);  // Invalid value

  // Test updateEpsRel
  osqp.updateEpsRel(1e-6);
  osqp.updateEpsRel(-1e-6);  // Invalid value

  // Test updateVerbose
  osqp.updateVerbose(true);
  osqp.updateVerbose(false);

  // Test updateMaxIter
  osqp.updateMaxIter(1000);
  osqp.updateMaxIter(-1000);  // Invalid value

  // Test updateRho
  osqp.updateRho(0.1);
  osqp.updateRho(-0.1);  // Invalid value

  // Test updateAlpha
  osqp.updateAlpha(1.8);
  osqp.updateAlpha(-1.8);  // Invalid value

  // Test updateScaling
  osqp.updateScaling(1);
  osqp.updateScaling(-1);  // Invalid value

  // Test updatePolish
  osqp.updatePolish(true);
  osqp.updatePolish(false);

  // Test updatePolishRefinementIteration
  osqp.updatePolishRefinementIteration(5);
  osqp.updatePolishRefinementIteration(-5);  // Invalid value

  // Test updateCheckTermination
  osqp.updateCheckTermination(10);
  osqp.updateCheckTermination(-10);  // Invalid value

  // Test updateP
  osqp.updateP(P);
  P.resize(3, 3);  // Invalid size
  EXPECT_THROW(osqp.updateP(P), std::invalid_argument);

  // Test updateCscP
  CSC_Matrix P_csc = autoware::qp_interface::calCSCMatrixTrapezoidal(P);
  osqp.updateCscP(P_csc);

  // Test updateA
  osqp.updateA(A);
  A.resize(3, 3);  // Invalid size
  EXPECT_THROW(osqp.updateA(A), std::invalid_argument);

  // Test updateCscA
  CSC_Matrix A_csc = autoware::qp_interface::calCSCMatrix(A);
  osqp.updateCscA(A_csc);

  // Test updateQ
  osqp.updateQ(q);
  q.resize(3);  // Invalid size
  EXPECT_THROW(osqp.updateQ(q), std::invalid_argument);

  // Test updateL
  osqp.updateL(l);
  l.resize(3);  // Invalid size
  EXPECT_THROW(osqp.updateL(l), std::invalid_argument);

  // Test updateU
  osqp.updateU(u);
  u.resize(3);  // Invalid size
  EXPECT_THROW(osqp.updateU(u), std::invalid_argument);

  // Test updateBounds
  osqp.updateBounds(l, u);
  l.resize(3);  // Invalid size
  EXPECT_THROW(osqp.updateBounds(l, u), std::invalid_argument);
  u.resize(3);  // Invalid size
  EXPECT_THROW(osqp.updateBounds(l, u), std::invalid_argument);
}

}  // namespace
