#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <eigen3/Eigen/Dense>
#include <fmt/format.h>
#include <opencv2/highgui.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "io/command.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

namespace sp_vision_25::bridge {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kRadToDeg = 57.3;

Eigen::Vector3d command_to_direction(const io::Command& command) {
    Eigen::Vector3d direction{
        std::cos(command.pitch) * std::cos(command.yaw),
        std::cos(command.pitch) * std::sin(command.yaw),
        -std::sin(command.pitch),
    };
    if (direction.norm() > 1e-9)
        direction.normalize();
    else
        direction.setZero();
    return direction;
}

std::filesystem::path
    resolve_path_parameter(const std::filesystem::path& package_share, const std::string& path) {
    std::filesystem::path result(path);
    if (result.empty())
        return package_share / "configs/standard3.yaml";
    if (result.is_relative())
        result = package_share / result;
    return result.lexically_normal();
}

std::filesystem::path
    resolve_package_asset_path(const std::filesystem::path& package_root, const std::string& path) {
    std::filesystem::path result(path);
    if (result.empty() || result.is_absolute())
        return result;
    return (package_root / result).lexically_normal();
}

std::filesystem::path prepare_runtime_config(
    const std::filesystem::path& config_path, const std::string& component_name) {
    YAML::Node yaml = YAML::LoadFile(config_path.string());

    std::filesystem::path package_root = config_path.parent_path().parent_path();
    if (!std::filesystem::exists(package_root / "assets"))
        package_root = config_path.parent_path();

    constexpr std::array<const char*, 5> path_keys{
        "classify_model", "yolo11_model_path", "yolov8_model_path", "yolov5_model_path", "model",
    };
    for (const char* key : path_keys) {
        if (!yaml[key] || !yaml[key].IsScalar())
            continue;
        yaml[key] = resolve_package_asset_path(package_root, yaml[key].as<std::string>()).string();
    }

    std::filesystem::path runtime_config =
        std::filesystem::temp_directory_path() / (component_name + "_resolved.yaml");
    YAML::Emitter emitter;
    emitter << yaml;

    std::ofstream output(runtime_config);
    if (!output.is_open())
        throw std::runtime_error("Failed to create runtime config: " + runtime_config.string());
    output << emitter.c_str();
    output.close();

    return runtime_config;
}

void draw_debug_frame(
    const cv::Mat& source_frame,
    const std::list<auto_aim::Armor>& armors,
    const std::list<auto_aim::Target>& targets,
    const auto_aim::Solver& solver,
    const auto_aim::Tracker& tracker,
    const auto_aim::Aimer& aimer,
    const io::Command& command,
    double bullet_speed,
    bool fire_control) {
    auto debug_frame = source_frame.clone();

    tools::draw_text(
        debug_frame,
        fmt::format(
            "[{}] control={} fire={} bullet={:.1f} yaw={:.2f} pitch={:.2f}",
            tracker.state(),
            command.control ? 1 : 0,
            fire_control ? 1 : 0,
            bullet_speed,
            command.yaw * kRadToDeg,
            command.pitch * kRadToDeg),
        {10, 30},
        {255, 255, 255});

    for (const auto& armor : armors) {
        auto info = fmt::format(
            "{:.2f} {} {} {}",
            armor.confidence,
            auto_aim::COLORS[armor.color],
            auto_aim::ARMOR_NAMES[armor.name],
            auto_aim::ARMOR_TYPES[armor.type]);
        tools::draw_points(debug_frame, armor.points, {0, 255, 255}, 2);
        tools::draw_text(debug_frame, info, armor.center, {0, 255, 255}, 0.6, 2);
    }

    if (!targets.empty()) {
        const auto& target = targets.front();
        for (const Eigen::Vector4d& xyza : target.armor_xyza_list()) {
            const auto image_points =
                solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
            tools::draw_points(debug_frame, image_points, {0, 255, 0});
        }

        const auto& aim_xyza = aimer.debug_aim_point.xyza;
        const auto image_points =
            solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        if (aimer.debug_aim_point.valid)
            tools::draw_points(debug_frame, image_points, {0, 0, 255});
        else
            tools::draw_points(debug_frame, image_points, {255, 0, 0});
    }

    cv::resize(debug_frame, debug_frame, {}, 0.5, 0.5);
    cv::imshow("reprojection", debug_frame);
    cv::waitKey(1);
}

} // namespace

class SpVisionBridge
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    SpVisionBridge()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {
        const std::string package_share =
            ament_index_cpp::get_package_share_directory("sp_vision_25");

        if (!has_parameter("config_file"))
            declare_parameter<std::string>("config_file", package_share + "/configs/standard3.yaml");
        if (!has_parameter("bullet_speed_fallback"))
            declare_parameter<double>("bullet_speed_fallback", 23.0);
        if (!has_parameter("result_timeout"))
            declare_parameter<double>("result_timeout", 0.2);
        if (!has_parameter("debug"))
            declare_parameter<bool>("debug", false);

        register_input("/tf", tf_);
        register_input("/predefined/timestamp", timestamp_);
        register_input("/referee/shooter/initial_speed", bullet_speed_, false);

        register_output(
            "/gimbal/auto_aim/control_direction", control_direction_, Eigen::Vector3d::Zero());
        register_output("/gimbal/auto_aim/fire_control", fire_control_, false);
        register_output("/gimbal/auto_aim/laser_distance", laser_distance_, 0.0);
    }

    ~SpVisionBridge() override {
        stop_worker_.store(true, std::memory_order_relaxed);
        if (worker_thread_.joinable())
            worker_thread_.join();
        if (debug_)
            cv::destroyWindow("reprojection");
    }

    void before_updating() override {
        bullet_speed_fallback_storage_ =
            static_cast<float>(get_parameter("bullet_speed_fallback").as_double());
        bullet_speed_snapshot_.store(
            static_cast<double>(bullet_speed_fallback_storage_), std::memory_order_relaxed);

        if (!bullet_speed_.ready())
            bullet_speed_.bind_directly(bullet_speed_fallback_storage_);

        result_timeout_ =
            std::chrono::duration<double>(get_parameter("result_timeout").as_double());
        debug_ = get_parameter("debug").as_bool();

        const auto config_path = resolve_path_parameter(
            ament_index_cpp::get_package_share_directory("sp_vision_25"),
            get_parameter("config_file").as_string());
        runtime_config_path_ = prepare_runtime_config(config_path, get_component_name()).string();

        RCLCPP_INFO(
            get_logger(), "Starting sp_vision bridge with config %s", runtime_config_path_.c_str());
        worker_thread_ = std::thread(&SpVisionBridge::worker_main, this, runtime_config_path_);
    }

    void update() override {
        bullet_speed_snapshot_.store(static_cast<double>(*bullet_speed_), std::memory_order_relaxed);
        store_latest_imu_pose(current_imu_pose());
        publish_latest_result(*timestamp_);
    }

private:
    struct VisionResult {
        Clock::time_point timestamp{};
        Eigen::Vector3d direction = Eigen::Vector3d::Zero();
        double laser_distance = 0.0;
        bool fire_control = false;
        bool valid = false;
    };

    Eigen::Quaterniond current_imu_pose() const {
        Eigen::Quaterniond pose =
            tf_->template get_transform<rmcs_description::PitchLink, rmcs_description::OdomImu>()
                .conjugate();
        pose.normalize();
        return pose;
    }

    void store_latest_imu_pose(const Eigen::Quaterniond& pose) {
        std::lock_guard<std::mutex> lock(imu_pose_mutex_);
        latest_imu_pose_ = pose;
        latest_imu_pose_ready_.store(true, std::memory_order_release);
    }

    bool load_latest_imu_pose(Eigen::Quaterniond& pose) {
        if (!latest_imu_pose_ready_.load(std::memory_order_acquire))
            return false;
        std::lock_guard<std::mutex> lock(imu_pose_mutex_);
        pose = latest_imu_pose_;
        return true;
    }

    void store_result(const VisionResult& result) {
        std::lock_guard<std::mutex> lock(result_mutex_);
        latest_result_ = result;
    }

    void publish_latest_result(const Clock::time_point& now) {
        VisionResult result;
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            result = latest_result_;
        }

        const bool fresh = result.valid
                        && std::chrono::duration<double>(now - result.timestamp) <= result_timeout_;
        if (fresh) {
            *control_direction_ = result.direction;
            *fire_control_ = result.fire_control;
            *laser_distance_ = result.laser_distance;
        } else {
            *control_direction_ = Eigen::Vector3d::Zero();
            *fire_control_ = false;
            *laser_distance_ = 0.0;
        }
    }

    void worker_main(std::string runtime_config_path) {
        try {
            io::Camera camera(runtime_config_path);
            auto_aim::YOLO detector(runtime_config_path, false);
            auto_aim::Solver solver(runtime_config_path);
            auto_aim::Tracker tracker(runtime_config_path, solver);
            auto_aim::Aimer aimer(runtime_config_path);
            auto_aim::Shooter shooter(runtime_config_path);

            while (!stop_worker_.load(std::memory_order_relaxed)) {
                cv::Mat frame;
                Clock::time_point frame_timestamp;
                camera.read(frame, frame_timestamp);
                if (frame.empty())
                    continue;

                Eigen::Quaterniond imu_pose;
                if (!load_latest_imu_pose(imu_pose))
                    continue;

                solver.set_R_gimbal2world(imu_pose);

                auto armors = detector.detect(frame);
                auto targets = tracker.track(armors, frame_timestamp);
                const double bullet_speed = bullet_speed_snapshot_.load(std::memory_order_relaxed);
                auto command = aimer.aim(targets, frame_timestamp, bullet_speed);

                VisionResult result;
                result.timestamp = frame_timestamp;
                result.valid = command.control;
                result.direction =
                    result.valid ? command_to_direction(command) : Eigen::Vector3d::Zero();
                result.laser_distance =
                    aimer.debug_aim_point.valid ? aimer.debug_aim_point.xyza.head<3>().norm() : 0.0;

                if (result.valid) {
                    const Eigen::Vector3d gimbal_pos =
                        tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
                    result.fire_control = shooter.shoot(command, aimer, targets, gimbal_pos);
                }

                if (debug_) {
                    draw_debug_frame(
                        frame,
                        armors,
                        targets,
                        solver,
                        tracker,
                        aimer,
                        command,
                        bullet_speed,
                        result.fire_control);
                }

                store_result(result);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "sp_vision bridge worker stopped: %s", e.what());
        }
    }

    InputInterface<rmcs_description::Tf> tf_;
    InputInterface<Clock::time_point> timestamp_;
    InputInterface<float> bullet_speed_;

    OutputInterface<Eigen::Vector3d> control_direction_;
    OutputInterface<bool> fire_control_;
    OutputInterface<double> laser_distance_;

    std::thread worker_thread_;
    std::atomic<bool> stop_worker_{false};

    std::mutex result_mutex_;
    VisionResult latest_result_;

    std::mutex imu_pose_mutex_;
    Eigen::Quaterniond latest_imu_pose_ = Eigen::Quaterniond::Identity();
    std::atomic<bool> latest_imu_pose_ready_{false};

    std::atomic<double> bullet_speed_snapshot_{23.0};
    float bullet_speed_fallback_storage_ = 23.0F;
    std::chrono::duration<double> result_timeout_{0.2};
    bool debug_ = false;
    std::string runtime_config_path_;
};

} // namespace sp_vision_25::bridge

PLUGINLIB_EXPORT_CLASS(sp_vision_25::bridge::SpVisionBridge, rmcs_executor::Component)
