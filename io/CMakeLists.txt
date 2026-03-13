cmake_minimum_required(VERSION 3.16)

find_package(hikcamera REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(
  sp_vision_25_io
  STATIC
  hikrobot/hikrobot.cpp
  mindvision/mindvision.cpp
  usbcamera/usbcamera.cpp
  camera.cpp
)

target_include_directories(
  sp_vision_25_io
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
    ${hikcamera_INCLUDE_DIRS}
    mindvision/include
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(sp_vision_25_io PUBLIC mindvision/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(sp_vision_25_io PUBLIC mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_link_libraries(sp_vision_25_io PUBLIC ${hikcamera_LIBRARIES} MVSDK usb-1.0 yaml-cpp)
