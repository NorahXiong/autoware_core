#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "autoware_vehicle_msgs/msg/velocity_report.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "../src/vehicle_velocity_converter.hpp" // Adjusted path

// Test fixture for the VehicleVelocityConverter tests
class VehicleVelocityConverterTest : public ::testing::Test {
public:
  VehicleVelocityConverterTest() : node_options_() {
    // Suppress logging for cleaner test output
    node_options_.arguments({"--ros-args", "-r", "__node:=test_vehicle_velocity_converter", "--log-level", "warn"});
    rclcpp::init(0, nullptr, node_options_.get_rcl_init_options());
  }

  ~VehicleVelocityConverterTest() override {
    rclcpp::shutdown(node_options_.get_rcl_init_options().context);
  }

protected:
  void SetUp() override {
    // Initialize parameters
    params_ = {
      {"frame_id", rclcpp::ParameterValue("base_link")},
      {"velocity_stddev_xx", rclcpp::ParameterValue(0.1)},
      {"angular_velocity_stddev_zz", rclcpp::ParameterValue(0.05)},
      {"speed_scale_factor", rclcpp::ParameterValue(1.0)},
    };
    node_options_.parameter_overrides(params_);
    converter_node_ = std::make_shared<vehicle_velocity_converter::VehicleVelocityConverter>(node_options_);

    // Set a callback in the test fixture to capture the output
    converter_node_->set_publisher_callback([this](const geometry_msgs::msg::TwistWithCovarianceStamped & msg) {
      this->received_twist_ = msg;
      this->message_received_ = true;
    });
    message_received_ = false; // Reset flag before each test
  }

  void TearDown() override {
    converter_node_.reset();
  }

  // Helper function to create a VelocityReport message
  autoware_vehicle_msgs::msg::VelocityReport create_velocity_report(
    const std::string& frame_id, float lon_vel, float lat_vel, float heading_rate) {
    autoware_vehicle_msgs::msg::VelocityReport msg;
    // Get current time from the node
    if (converter_node_) {
        msg.header.stamp = converter_node_->now();
    } else {
        // Fallback if node is not yet created (e.g. during param setup for a specific test)
        // Or handle error appropriately
        rclcpp::Clock clock(RCL_SYSTEM_TIME);
        msg.header.stamp = clock.now();
    }
    msg.header.frame_id = frame_id;
    msg.longitudinal_velocity = lon_vel;
    msg.lateral_velocity = lat_vel;
    msg.heading_rate = heading_rate;
    return msg;
  }
  
  // Node and parameters
  rclcpp::NodeOptions node_options_;
  std::vector<rclcpp::Parameter> params_;
  std::shared_ptr<vehicle_velocity_converter::VehicleVelocityConverter> converter_node_;
  
  // Captured message
  geometry_msgs::msg::TwistWithCovarianceStamped received_twist_;
  bool message_received_ = false;
  const double COVARIANCE_UNAVAILABLE = 10000.0; // As per node's logic
};

// Test Case 1: Basic Conversion (Matching Frame ID)
TEST_F(VehicleVelocityConverterTest, BasicConversionMatchingFrame) {
  auto report = create_velocity_report("base_link", 10.0f, 0.5f, 0.2f);
  converter_node_->callback_velocity_report(std::make_shared<autoware_vehicle_msgs::msg::VelocityReport>(report));

  ASSERT_TRUE(message_received_);
  EXPECT_EQ(received_twist_.header.frame_id, "base_link");
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.linear.x, 10.0);
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.linear.y, 0.5);
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.linear.z, 0.0); // Assuming Z is not set from report
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.angular.x, 0.0); // Assuming X is not set
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.angular.y, 0.0); // Assuming Y is not set
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.angular.z, 0.2);

  // Covariance checks: cov[row * 6 + col]
  // Vx (index 0)
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[0], 0.1 * 0.1); 
  // Vy (index 7)
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[7], COVARIANCE_UNAVAILABLE);
  // Vz (index 14)
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[14], COVARIANCE_UNAVAILABLE);
  // Wx (index 21)
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[21], COVARIANCE_UNAVAILABLE);
  // Wy (index 28)
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[28], COVARIANCE_UNAVAILABLE);
  // Wz (index 35)
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[35], 0.05 * 0.05);
}

// Test Case 2: Mismatched Frame ID
TEST_F(VehicleVelocityConverterTest, MismatchedFrameId) {
  auto report = create_velocity_report("other_frame", 5.0f, 0.1f, -0.1f);
  converter_node_->callback_velocity_report(std::make_shared<autoware_vehicle_msgs::msg::VelocityReport>(report));
  
  ASSERT_TRUE(message_received_); // Message should still be processed
  EXPECT_EQ(received_twist_.header.frame_id, "base_link"); // Output frame is based on node's param
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.linear.x, 5.0);
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.linear.y, 0.1);
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.angular.z, -0.1);
  // Covariances should still be calculated
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[0], 0.1 * 0.1);
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[35], 0.05 * 0.05);
}

// Test Case 3: Speed Scale Factor
TEST_F(VehicleVelocityConverterTest, SpeedScaleFactor) {
  // Reconfigure node with new speed_scale_factor
  // Store options to re-initialize node
  auto current_options = node_options_; 
  current_options.parameter_overrides({
      {"frame_id", rclcpp::ParameterValue("base_link")},
      {"velocity_stddev_xx", rclcpp::ParameterValue(0.1)},
      {"angular_velocity_stddev_zz", rclcpp::ParameterValue(0.05)},
      {"speed_scale_factor", rclcpp::ParameterValue(0.5)}, // New scale factor
  });
  converter_node_ = std::make_shared<vehicle_velocity_converter::VehicleVelocityConverter>(current_options);
  converter_node_->set_publisher_callback([this](const geometry_msgs::msg::TwistWithCovarianceStamped & msg) {
    this->received_twist_ = msg;
    this->message_received_ = true;
  });
  message_received_ = false;

  auto report = create_velocity_report("base_link", 10.0f, 0.5f, 0.2f);
  converter_node_->callback_velocity_report(std::make_shared<autoware_vehicle_msgs::msg::VelocityReport>(report));

  ASSERT_TRUE(message_received_);
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.linear.x, 10.0 * 0.5); // 5.0
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.linear.y, 0.5); // Lateral velocity is NOT scaled by speed_scale_factor
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.angular.z, 0.2);    // Heading rate not scaled by speed_scale_factor
}

// Test Case 4: Zero Velocities
TEST_F(VehicleVelocityConverterTest, ZeroVelocities) {
  auto report = create_velocity_report("base_link", 0.0f, 0.0f, 0.0f);
  converter_node_->callback_velocity_report(std::make_shared<autoware_vehicle_msgs::msg::VelocityReport>(report));

  ASSERT_TRUE(message_received_);
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(received_twist_.twist.twist.angular.z, 0.0);
  // Covariances should remain as per stddev parameters
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[0], 0.1 * 0.1);
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[35], 0.05 * 0.05);
}

// Test Case 5: Parameter Variation (stddev)
TEST_F(VehicleVelocityConverterTest, ParameterVariationStddev) {
  // Reconfigure node with new stddevs
  // Store options to re-initialize node
  auto current_options = node_options_;
  current_options.parameter_overrides({
      {"frame_id", rclcpp::ParameterValue("base_link")},
      {"velocity_stddev_xx", rclcpp::ParameterValue(0.2)}, // New stddev_xx
      {"angular_velocity_stddev_zz", rclcpp::ParameterValue(0.1)}, // New stddev_zz
      {"speed_scale_factor", rclcpp::ParameterValue(1.0)},
  });
  converter_node_ = std::make_shared<vehicle_velocity_converter::VehicleVelocityConverter>(current_options);
  converter_node_->set_publisher_callback([this](const geometry_msgs::msg::TwistWithCovarianceStamped & msg) {
    this->received_twist_ = msg;
    this->message_received_ = true;
  });
  message_received_ = false;

  auto report = create_velocity_report("base_link", 10.0f, 0.5f, 0.2f);
  converter_node_->callback_velocity_report(std::make_shared<autoware_vehicle_msgs::msg::VelocityReport>(report));

  ASSERT_TRUE(message_received_);
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[0], 0.2 * 0.2); // 0.04
  EXPECT_DOUBLE_EQ(received_twist_.twist.covariance[35], 0.1 * 0.1); // 0.01
}
