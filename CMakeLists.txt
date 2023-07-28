cmake_minimum_required(VERSION 3.8)
project(cvnode_base)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(kenning_computer_vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(cv_bridge REQUIRED)

include_directories(include)

if(BUILD_TESTING)
    find_package(ament_cmake_clang_format REQUIRED)
    find_package(ament_cmake_clang_tidy REQUIRED)

    ament_clang_format(CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format
        "include"
        "src"
    )

    ament_clang_tidy(CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format
       ${CMAKE_BINARY_DIR}/compile_commands.json
       "include"
       "src"
    )

endif()

#############

add_library(basecvnode SHARED
    src/cvnode_base.cpp
)

ament_target_dependencies(basecvnode
    kenning_computer_vision_msgs
    sensor_msgs
    rclcpp
)

#############

install(TARGETS
    basecvnode
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

ament_python_install_package(${PROJECT_NAME})

ament_package()
