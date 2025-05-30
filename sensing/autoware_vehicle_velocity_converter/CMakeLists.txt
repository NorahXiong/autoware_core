cmake_minimum_required(VERSION 3.14)
project(autoware_vehicle_velocity_converter)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/vehicle_velocity_converter.cpp
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "autoware::vehicle_velocity_converter::VehicleVelocityConverter"
  EXECUTABLE ${PROJECT_NAME}_node
  EXECUTOR SingleThreadedExecutor
)

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_lint_auto_lint_types test)
  ament_lint_auto_find_test_dependencies()

  add_executable(
    test_vehicle_velocity_converter
    test/test_vehicle_velocity_converter.cpp
  )
  ament_target_dependencies(
    test_vehicle_velocity_converter
    "rclcpp"
    "autoware_vehicle_msgs"
    "geometry_msgs"
    "${PROJECT_NAME}"  # Link against the package's library
    "ament_cmake_gtest"
  )

  install(
    TARGETS test_vehicle_velocity_converter
    DESTINATION lib/${PROJECT_NAME}
  )

  ament_add_gtest(
    test_vehicle_velocity_converter
    test/test_vehicle_velocity_converter.cpp
  )
endif()
