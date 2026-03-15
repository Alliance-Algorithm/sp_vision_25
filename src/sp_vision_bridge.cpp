#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <list>
#include <mutex>
#include <optional>
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
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

namespace sp_vision_25::bridge {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kRadToDeg = 57.3;
constexpr auto kPlannerPeriod = std::chrono::milliseconds(1);

Eigen::Vector3d angles_to_direction(double yaw, double pitch) {
    Eigen::Vector3d direction{
        std::cos(pitch) * std::cos(yaw),
        std::cos(pitch) * std::sin(yaw),
        -std::sin(pitch),
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
    const auto_aim::Plan& plan,
    const Eigen::Vector4d& debug_xyza,
    double bullet_speed,
    double laser_distance,
    bool fire_control) {
    auto debug_frame = source_frame.clone();

    tools::draw_text(
        debug_frame,
        fmt::format(
            "[{}] control={} fire={} bullet={:.1f} yaw={:.2f} pitch={:.2f}",
            tracker.state(),
            plan.control ? 1 : 0,
            fire_control ? 1 : 0,
            bullet_speed,
            plan.yaw * kRadToDeg,
            plan.pitch * kRadToDeg),
        {10, 30},
        {255, 255, 255});
    tools::draw_text(
        debug_frame, fmt::format("laser={:.2f}m", laser_distance), {10, 60}, {255, 255, 255});

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

        const auto image_points = solver.reproject_armor(
            debug_xyza.head(3), debug_xyza[3], target.armor_type, target.name);
        if (plan.control)
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
        register_output("/gimbal/auto_aim/plan_yaw", plan_yaw_, 0.0);
        register_output("/gimbal/auto_aim/plan_pitch", plan_pitch_, 0.0);
        register_output("/gimbal/auto_aim/plan_yaw_velocity", plan_yaw_velocity_, 0.0);
        register_output("/gimbal/auto_aim/plan_yaw_acceleration", plan_yaw_acceleration_, 0.0);
        register_output("/gimbal/auto_aim/plan_pitch_velocity", plan_pitch_velocity_, 0.0);
        register_output(
            "/gimbal/auto_aim/plan_pitch_acceleration", plan_pitch_acceleration_, 0.0);
    }

    ~SpVisionBridge() override {
        stop_worker_.store(true, std::memory_order_relaxed);
        if (worker_thread_.joinable())
            worker_thread_.join();
        if (planner_thread_.joinable())
            planner_thread_.join();
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
        planner_thread_ = std::thread(&SpVisionBridge::planner_main, this, runtime_config_path_);
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
        double plan_yaw = 0.0;
        double plan_pitch = 0.0;
        double plan_yaw_velocity = 0.0;
        double plan_yaw_acceleration = 0.0;
        double plan_pitch_velocity = 0.0;
        double plan_pitch_acceleration = 0.0;
        Eigen::Vector4d debug_xyza = Eigen::Vector4d::Zero();
    };

    struct TargetState {
        std::optional<auto_aim::Target> target;
        Clock::time_point timestamp{};
        bool ready = false;
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

    void store_latest_target(
        std::optional<auto_aim::Target> target, const Clock::time_point& timestamp) {
        std::lock_guard<std::mutex> lock(target_mutex_);
        latest_target_.target = std::move(target);
        latest_target_.timestamp = timestamp;
        latest_target_.ready = true;
    }

    bool load_latest_target(TargetState& target_state) {
        std::lock_guard<std::mutex> lock(target_mutex_);
        if (!latest_target_.ready)
            return false;
        target_state = latest_target_;
        return true;
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
            *plan_yaw_ = result.plan_yaw;
            *plan_pitch_ = result.plan_pitch;
            *plan_yaw_velocity_ = result.plan_yaw_velocity;
            *plan_yaw_acceleration_ = result.plan_yaw_acceleration;
            *plan_pitch_velocity_ = result.plan_pitch_velocity;
            *plan_pitch_acceleration_ = result.plan_pitch_acceleration;
        } else {
            *control_direction_ = Eigen::Vector3d::Zero();
            *fire_control_ = false;
            *laser_distance_ = 0.0;
            *plan_yaw_ = 0.0;
            *plan_pitch_ = 0.0;
            *plan_yaw_velocity_ = 0.0;
            *plan_yaw_acceleration_ = 0.0;
            *plan_pitch_velocity_ = 0.0;
            *plan_pitch_acceleration_ = 0.0;
        }
    }

    void worker_main(std::string runtime_config_path) {
        try {
            io::Camera camera(runtime_config_path);
            auto_aim::YOLO detector(runtime_config_path, false);
            auto_aim::Solver solver(runtime_config_path);
            auto_aim::Tracker tracker(runtime_config_path, solver);

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
                if (!targets.empty())
                    store_latest_target(targets.front(), frame_timestamp);
                else
                    store_latest_target(std::nullopt, frame_timestamp);

                if (debug_) {
                    VisionResult result;
                    {
                        std::lock_guard<std::mutex> lock(result_mutex_);
                        result = latest_result_;
                    }
                    draw_debug_frame(
                        frame,
                        armors,
                        targets,
                        solver,
                        tracker,
                        auto_aim::Plan{
                            result.valid,
                            result.fire_control,
                            static_cast<float>(result.plan_yaw),
                            static_cast<float>(result.plan_pitch),
                            static_cast<float>(result.plan_yaw),
                            static_cast<float>(result.plan_yaw_velocity),
                            static_cast<float>(result.plan_yaw_acceleration),
                            static_cast<float>(result.plan_pitch),
                            static_cast<float>(result.plan_pitch_velocity),
                            static_cast<float>(result.plan_pitch_acceleration),
                        },
                        result.debug_xyza,
                        bullet_speed_snapshot_.load(std::memory_order_relaxed),
                        result.laser_distance,
                        result.fire_control);
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "sp_vision bridge worker stopped: %s", e.what());
        }
    }

    void planner_main(std::string runtime_config_path) {
        try {
            auto_aim::Planner planner(runtime_config_path);
            auto next_iteration_time = Clock::now();

            while (!stop_worker_.load(std::memory_order_relaxed)) {
                next_iteration_time += kPlannerPeriod;

                TargetState target_state;
                if (!load_latest_target(target_state)) {
                    std::this_thread::sleep_until(next_iteration_time);
                    continue;
                }

                const double bullet_speed = bullet_speed_snapshot_.load(std::memory_order_relaxed);
                auto plan = planner.plan(target_state.target, bullet_speed);

                VisionResult result;
                // Keep freshness tied to the last detection update so stale targets time out even
                // when the planner keeps re-planning the same snapshot at 1 kHz.
                result.timestamp = target_state.timestamp;
                result.valid = plan.control;
                result.fire_control = plan.control && plan.fire;
                result.direction =
                    plan.control ? angles_to_direction(plan.yaw, plan.pitch) : Eigen::Vector3d::Zero();
                result.laser_distance =
                    plan.control ? planner.debug_xyza.head<3>().norm() : 0.0;
                result.plan_yaw = plan.yaw;
                result.plan_pitch = plan.pitch;
                result.plan_yaw_velocity = plan.yaw_vel;
                result.plan_yaw_acceleration = plan.yaw_acc;
                result.plan_pitch_velocity = plan.pitch_vel;
                result.plan_pitch_acceleration = plan.pitch_acc;
                result.debug_xyza = planner.debug_xyza;
                store_result(result);

                std::this_thread::sleep_until(next_iteration_time);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "sp_vision bridge planner stopped: %s", e.what());
        }
    }

    InputInterface<rmcs_description::Tf> tf_;
    InputInterface<Clock::time_point> timestamp_;
    InputInterface<float> bullet_speed_;

    OutputInterface<Eigen::Vector3d> control_direction_;
    OutputInterface<bool> fire_control_;
    OutputInterface<double> laser_distance_;
    OutputInterface<double> plan_yaw_;
    OutputInterface<double> plan_pitch_;
    OutputInterface<double> plan_yaw_velocity_;
    OutputInterface<double> plan_yaw_acceleration_;
    OutputInterface<double> plan_pitch_velocity_;
    OutputInterface<double> plan_pitch_acceleration_;

    std::thread worker_thread_;
    std::thread planner_thread_;
    std::atomic<bool> stop_worker_{false};

    std::mutex result_mutex_;
    VisionResult latest_result_;

    std::mutex target_mutex_;
    TargetState latest_target_;

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
