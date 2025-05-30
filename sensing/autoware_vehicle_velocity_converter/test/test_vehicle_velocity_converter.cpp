#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "autoware_vehicle_msgs/msg/velocity_report.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "../src/vehicle_velocity_converter.hpp" // Adjusted path

#include <memory> // For std::shared_ptr
#include <chrono> // For std::chrono_literals

// Test fixture for the VehicleVelocityConverter tests (Black-Box Approach)
class VehicleVelocityConverterTest : public ::testing::Test {
public:
  VehicleVelocityConverterTest() {
    rclcpp::init(0, nullptr);
    test_node_ = std::make_shared<rclcpp::Node>("test_node_for_vvc");

    velocity_report_pub_ = test_node_->create_publisher<autoware_vehicle_msgs::msg::VelocityReport>(
      "velocity_status", 10); // Topic VVC subscribes to

    twist_sub_ = test_node_->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      "twist_with_covariance", // Topic VVC publishes to
      10,
      [this](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
        this->last_received_twist_ = msg;
        this->message_received_flag_ = true;
      });
  }

  ~VehicleVelocityConverterTest() override {
    // Ensure nodes are destroyed before rclcpp::shutdown() if they are members
    velocity_report_pub_.reset();
    twist_sub_.reset();
    test_node_.reset();
    // converter_node_ will be reset in each test or if made a member, reset here
    rclcpp::shutdown();
  }

protected:
  // Helper to create VelocityReport
  autoware_vehicle_msgs::msg::VelocityReport create_velocity_report(
    const std::string& frame_id, float lon_vel, float lat_vel, float heading_rate) {
    autoware_vehicle_msgs::msg::VelocityReport msg;
    msg.header.stamp = test_node_->now(); // Use test_node's clock
    msg.header.frame_id = frame_id;
    msg.longitudinal_velocity = lon_vel;
    msg.lateral_velocity = lat_vel;
    msg.heading_rate = heading_rate;
    return msg;
  }

  // Helper to spin until message is received or timeout
  void spin_for_message(
    std::shared_ptr<rclcpp::Node> node_to_spin, 
    std::shared_ptr<rclcpp::Node> test_helper_node,
    int max_tries = 10, std::chrono::milliseconds spin_duration = std::chrono::milliseconds(100)) {
    
    message_received_flag_ = false;
    last_received_twist_ = nullptr;

    rclcpp::WallRate rate(std::chrono::duration_cast<std::chrono::duration<double>>(spin_duration).count()); // e.g. 10Hz if spin_duration is 100ms

    for (int i = 0; i < max_tries && !message_received_flag_; ++i) {
      rclcpp::spin_some(node_to_spin);
      rclcpp::spin_some(test_helper_node); // Spin our test node to process subscription
      // It might be better to spin both nodes in a single executor if possible,
      // or use a rate that allows both callbacks to be processed.
      // For simplicity here, separate spin_some calls are used.
      // A small delay might be needed if spin_some is too fast and pub/sub is inter-process.
      // However, for intra-process, this should be okay.
      std::this_thread::sleep_for(spin_duration / max_tries); // Small delay to prevent busy loop
    }
  }

  std::shared_ptr<rclcpp::Node> test_node_;
  rclcpp::Publisher<autoware_vehicle_msgs::msg::VelocityReport>::SharedPtr velocity_report_pub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr twist_sub_;
  
  geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr last_received_twist_;
  bool message_received_flag_ = false;
  
  const double COVARIANCE_UNAVAILABLE = 10000.0; // As per node's logic
};

// Test Case 1: Basic Conversion (Matching Frame ID)
TEST_F(VehicleVelocityConverterTest, BasicConversionMatchingFrame) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      {"frame_id", "base_link"},
      {"velocity_stddev_xx", 0.1},
      {"angular_velocity_stddev_zz", 0.05},
      {"speed_scale_factor", 1.0}
  });
  auto converter_node = std::make_shared<autoware::vehicle_velocity_converter::VehicleVelocityConverter>(options);

  auto report = create_velocity_report("base_link", 10.0f, 0.5f, 0.2f);
  velocity_report_pub_->publish(report);
  spin_for_message(converter_node, test_node_);

  ASSERT_TRUE(message_received_flag_) << "Did not receive TwistWithCovarianceStamped message";
  ASSERT_NE(last_received_twist_, nullptr);

  EXPECT_EQ(last_received_twist_->header.frame_id, "base_link");
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.linear.x, 10.0);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.linear.y, 0.5);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.angular.z, 0.2);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.covariance[0], 0.1 * 0.1); 
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.covariance[7], COVARIANCE_UNAVAILABLE);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.covariance[35], 0.05 * 0.05);
  converter_node.reset(); // Explicitly destroy node
}

// Test Case 2: Mismatched Frame ID
TEST_F(VehicleVelocityConverterTest, MismatchedFrameId) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      {"frame_id", "base_link"}, // Node's output frame_id
      {"velocity_stddev_xx", 0.1},
      {"angular_velocity_stddev_zz", 0.05},
      {"speed_scale_factor", 1.0}
  });
  auto converter_node = std::make_shared<autoware::vehicle_velocity_converter::VehicleVelocityConverter>(options);

  auto report = create_velocity_report("other_frame", 5.0f, 0.1f, -0.1f); // Input frame_id is different
  velocity_report_pub_->publish(report);
  spin_for_message(converter_node, test_node_);

  ASSERT_TRUE(message_received_flag_) << "Did not receive TwistWithCovarianceStamped message";
  ASSERT_NE(last_received_twist_, nullptr);
  
  EXPECT_EQ(last_received_twist_->header.frame_id, "base_link"); // Output frame should be "base_link"
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.linear.x, 5.0);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.linear.y, 0.1);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.angular.z, -0.1);
  converter_node.reset();
}

// Test Case 3: Speed Scale Factor
TEST_F(VehicleVelocityConverterTest, SpeedScaleFactor) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      {"frame_id", "base_link"},
      {"velocity_stddev_xx", 0.1},
      {"angular_velocity_stddev_zz", 0.05},
      {"speed_scale_factor", 0.5} // Test with 0.5 scale factor
  });
  auto converter_node = std::make_shared<autoware::vehicle_velocity_converter::VehicleVelocityConverter>(options);
  
  auto report = create_velocity_report("base_link", 10.0f, 0.5f, 0.2f);
  velocity_report_pub_->publish(report);
  spin_for_message(converter_node, test_node_);

  ASSERT_TRUE(message_received_flag_) << "Did not receive TwistWithCovarianceStamped message";
  ASSERT_NE(last_received_twist_, nullptr);

  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.linear.x, 10.0 * 0.5); // 5.0
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.linear.y, 0.5); // Lateral velocity NOT scaled
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.angular.z, 0.2); // Angular velocity NOT scaled
  converter_node.reset();
}

// Test Case 4: Zero Velocities
TEST_F(VehicleVelocityConverterTest, ZeroVelocities) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      {"frame_id", "base_link"},
      {"velocity_stddev_xx", 0.1},
      {"angular_velocity_stddev_zz", 0.05},
      {"speed_scale_factor", 1.0}
  });
  auto converter_node = std::make_shared<autoware::vehicle_velocity_converter::VehicleVelocityConverter>(options);

  auto report = create_velocity_report("base_link", 0.0f, 0.0f, 0.0f);
  velocity_report_pub_->publish(report);
  spin_for_message(converter_node, test_node_);

  ASSERT_TRUE(message_received_flag_) << "Did not receive TwistWithCovarianceStamped message";
  ASSERT_NE(last_received_twist_, nullptr);

  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.twist.angular.z, 0.0);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.covariance[0], 0.1 * 0.1);
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.covariance[35], 0.05 * 0.05);
  converter_node.reset();
}

// Test Case 5: Parameter Variation (stddev)
TEST_F(VehicleVelocityConverterTest, ParameterVariationStddev) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      {"frame_id", "base_link"},
      {"velocity_stddev_xx", 0.2}, // New stddev_xx
      {"angular_velocity_stddev_zz", 0.1}, // New stddev_zz
      {"speed_scale_factor", 1.0}
  });
  auto converter_node = std::make_shared<autoware::vehicle_velocity_converter::VehicleVelocityConverter>(options);

  auto report = create_velocity_report("base_link", 10.0f, 0.5f, 0.2f);
  velocity_report_pub_->publish(report);
  spin_for_message(converter_node, test_node_);

  ASSERT_TRUE(message_received_flag_) << "Did not receive TwistWithCovarianceStamped message";
  ASSERT_NE(last_received_twist_, nullptr);

  EXPECT_DOUBLE_EQ(last_received_twist_->twist.covariance[0], 0.2 * 0.2); // 0.04
  EXPECT_DOUBLE_EQ(last_received_twist_->twist.covariance[35], 0.1 * 0.1); // 0.01
  converter_node.reset();
}
// Note: No main function needed as ament_add_gtest handles it.
// int main(int argc, char** argv) {
//   rclcpp::init(argc, argv); // If not done in fixture constructor
//   ::testing::InitGoogleTest(&argc, argv);
//   int ret = RUN_ALL_TESTS();
//   rclcpp::shutdown(); // If not done in fixture destructor
//   return ret;
// }

/*
Potential improvements for spin_for_message:
- Use rclcpp::executors::SingleThreadedExecutor and add both nodes to it, then spin the executor.
  This would be a more robust way to handle callbacks from multiple nodes.
- Add a timeout mechanism to the spin loop to prevent tests from hanging indefinitely.
  The current loop has max_tries which acts as a timeout.
- The small sleep `std::this_thread::sleep_for(spin_duration / max_tries);` might be too short or too long.
  A common pattern is `rclcpp::Event::SharedPtr graph_event = test_node_->get_graph_event(); test_node_->wait_for_graph_change(graph_event, std::chrono::seconds(1));`
  or using condition variables if more complex synchronization is needed.
  For pub/sub, ensuring the publisher has matched before sending, and subscriber has matched before expecting can also make tests more robust.
  e.g. `while(test_node_->count_publishers("topic_name") == 0) { rclcpp::sleep_for(10ms); }`
  `while(test_node_->count_subscribers("topic_name") == 0) { rclcpp::sleep_for(10ms); }`
  However, these are more for ensuring connections are made, not for message processing itself.
  The current spin_some loop with a flag is a common basic approach.
*/
