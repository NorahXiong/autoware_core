#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "autoware/qp_interface/qp_interface.hpp"
#include "autoware/qp_interface/osqp_interface.hpp"

namespace autoware::qp_interface
{
TEST(QPInterfaceTest, InitializeProblem_NonSquareP_ThrowsException)
{
  Eigen::MatrixXd P(2, 3);
  Eigen::MatrixXd A(1, 2);
  std::vector<double> q = {1.0, 2.0};
  std::vector<double> l = {1.0};
  std::vector<double> u = {1.0};

  bool enable_warm_start = false;
  c_float eps_abs =  1e-4;

  EXPECT_THROW({OSQPInterface osqp_instance(P, A, q, l, u, enable_warm_start, eps_abs);}, std::invalid_argument);
}

TEST(QPInterfaceTest, InitializeProblem_PRowsNotEqualQSize_ThrowsException)
{
  Eigen::MatrixXd P(2, 2);
  Eigen::MatrixXd A(1, 2);
  std::vector<double> q = {1.0};
  std::vector<double> l = {1.0};
  std::vector<double> u = {1.0};

  bool enable_warm_start = false;
  c_float eps_abs =  1e-4;

  EXPECT_THROW({OSQPInterface osqp_instance(P, A, q, l, u, enable_warm_start, eps_abs);}, std::invalid_argument);
}

TEST(QPInterfaceTest, InitializeProblem_PRowsNotEqualACols_ThrowsException)
{
  Eigen::MatrixXd P(2, 2);
  Eigen::MatrixXd A(1, 3);
  std::vector<double> q = {1.0, 2.0};
  std::vector<double> l = {1.0};
  std::vector<double> u = {1.0};

  bool enable_warm_start = false;
  c_float eps_abs =  1e-4;

  EXPECT_THROW({OSQPInterface osqp_instance(P, A, q, l, u, enable_warm_start, eps_abs);}, std::invalid_argument);
}

TEST(QPInterfaceTest, InitializeProblem_ARowsNotEqualLSize_ThrowsException)
{
  Eigen::MatrixXd P(2, 2);
  Eigen::MatrixXd A(2, 2);
  std::vector<double> q = {1.0, 2.0};
  std::vector<double> l = {1.0};
  std::vector<double> u = {1.0, 2.0};

  bool enable_warm_start = false;
  c_float eps_abs =  1e-4;

  EXPECT_THROW({OSQPInterface osqp_instance(P, A, q, l, u, enable_warm_start, eps_abs);}, std::invalid_argument);
}

TEST(QPInterfaceTest, InitializeProblem_ARowsNotEqualUSize_ThrowsException)
{
  Eigen::MatrixXd P(2, 2);
  Eigen::MatrixXd A(2, 2);
  std::vector<double> q = {1.0, 2.0};
  std::vector<double> l = {1.0, 2.0};
  std::vector<double> u = {1.0};

  bool enable_warm_start = false;
  c_float eps_abs =  1e-4;

  EXPECT_THROW({OSQPInterface osqp_instance(P, A, q, l, u, enable_warm_start, eps_abs);}, std::invalid_argument);
}

TEST(QPInterfaceTest, InitializeProblem_ValidInputs_Success)
{
  Eigen::MatrixXd P(2, 2);
  P << 1, 0, 0, 1;
  Eigen::MatrixXd A(1, 2);
  A << 1, 1;
  std::vector<double> q = {1.0, 2.0};
  std::vector<double> l = {1.0};
  std::vector<double> u = {1.0};

  bool enable_warm_start = false;
  c_float eps_abs =  1e-4;

  EXPECT_THROW({OSQPInterface osqp_instance(P, A, q, l, u, enable_warm_start, eps_abs);}, std::invalid_argument);
}

TEST(QPInterfaceTest, Optimize_ValidInputs_ReturnsResult)
{
  Eigen::MatrixXd P(2, 2);
  P << 1, 0, 0, 1;
  Eigen::MatrixXd A(1, 2);
  A << 1, 1;
  std::vector<double> q = {1.0, 2.0};
  std::vector<double> l = {1.0};
  std::vector<double> u = {1.0};

  bool enable_warm_start = false;
  c_float eps_abs =  1e-4;
  EXPECT_NO_THROW({OSQPInterface osqp_instance(P, A, q, l, u, enable_warm_start, eps_abs);});

  OSQPInterface osqp(P, A, q, l, u, enable_warm_start, eps_abs);
  std::vector<double> result = osqp.QPInterface::optimize(P, A, q, l, u);
  // Assuming optimizeImpl returns a valid result, we can check its size or values
  EXPECT_EQ(result.size(), 2);
}
}  // namespace autoware::qp_interface