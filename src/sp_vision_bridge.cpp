#include <array>
#include <atomic>
#include <chrono>
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
#include <rmcs_executor/component.hpp>
#include <rmcs_msgs/hard_sync_snapshot.hpp>
#include <rmcs_msgs/target_snapshot.hpp>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/img_tools.hpp"

namespace sp_vision_25::bridge {
namespace {

using Clock = std::chrono::steady_clock;

Eigen::Quaterniond snapshot_to_pose(const rmcs_msgs::HardSyncSnapshot& snapshot) {
    Eigen::Quaterniond pose{snapshot.qw, snapshot.qx, snapshot.qy, snapshot.qz};
    pose.normalize();
    return pose;
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

void draw_reprojected_armor(
    cv::Mat& frame, const auto_aim::Solver& solver, auto_aim::ArmorType armor_type,
    auto_aim::ArmorName armor_name, const Eigen::Vector4d& xyza, const cv::Scalar& color,
    int thickness = 2) {
    const auto image_points = solver.reproject_armor(xyza.head(3), xyza[3], armor_type, armor_name);
    tools::draw_points(frame, image_points, color, thickness);
}

void draw_debug_frame(
    const cv::Mat& source_frame, const std::list<auto_aim::Armor>& armors,
    const std::list<auto_aim::Target>& targets, const auto_aim::Solver& solver,
    const auto_aim::Tracker& tracker) {
    auto debug_frame = source_frame.clone();

    tools::draw_text(
        debug_frame, fmt::format("[{}] targets={}", tracker.state(), targets.size()), {10, 30},
        {255, 255, 255});

    for (const auto& armor : armors) {
        auto info = fmt::format(
            "{:.2f} {} {} {}", armor.confidence, auto_aim::COLORS[armor.color],
            auto_aim::ARMOR_NAMES[armor.name], auto_aim::ARMOR_TYPES[armor.type]);
        tools::draw_points(debug_frame, armor.points, {0, 255, 255}, 2);
        tools::draw_text(debug_frame, info, armor.center, {0, 255, 255}, 0.6, 2);
    }

    if (!targets.empty()) {
        const auto& target = targets.front();
        for (const Eigen::Vector4d& xyza : target.armor_xyza_list()) {
            draw_reprojected_armor(
                debug_frame, solver, target.armor_type, target.name, xyza, {0, 255, 0}, 1);
        }
    }

    cv::resize(debug_frame, debug_frame, {}, 0.5, 0.5);
    cv::imshow("reprojection", debug_frame);
    cv::waitKey(1);
}

rmcs_msgs::TargetSnapshot make_target_snapshot(
    std::optional<auto_aim::Target> target, const Clock::time_point& timestamp) {
    rmcs_msgs::TargetSnapshot snapshot;
    snapshot.timestamp = timestamp;

    if (!target.has_value())
        return snapshot;

    snapshot.valid = true;
    snapshot.converged = target->convergened();
    snapshot.armor_count = static_cast<uint8_t>(target->armor_count());
    snapshot.armor_type = static_cast<rmcs_msgs::TargetSnapshotArmorType>(target->armor_type);
    snapshot.armor_name = static_cast<rmcs_msgs::TargetSnapshotArmorName>(target->name);

    const Eigen::VectorXd state = target->ekf_x();
    for (size_t i = 0; i < snapshot.state.size() && i < static_cast<size_t>(state.size()); ++i)
        snapshot.state[i] = state[static_cast<Eigen::Index>(i)];

    return snapshot;
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
            declare_parameter<std::string>(
                "config_file", package_share + "/configs/standard3.yaml");
        if (!has_parameter("debug"))
            declare_parameter<bool>("debug", false);

        register_input("/gimbal/hard_sync_snapshot", hard_sync_snapshot_);
        register_output(
            "/gimbal/auto_aim/target_snapshot", target_snapshot_, rmcs_msgs::TargetSnapshot{});
    }

    ~SpVisionBridge() override {
        stop_worker_.store(true, std::memory_order_relaxed);
        if (worker_thread_.joinable())
            worker_thread_.join();
        if (debug_)
            cv::destroyWindow("reprojection");
    }

    void before_updating() override {
        debug_ = get_parameter("debug").as_bool();

        const auto config_path = resolve_path_parameter(
            ament_index_cpp::get_package_share_directory("sp_vision_25"),
            get_parameter("config_file").as_string());
        runtime_config_path_ = prepare_runtime_config(config_path, get_component_name()).string();
        store_latest_hard_sync_snapshot(*hard_sync_snapshot_);

        RCLCPP_INFO(
            get_logger(), "Starting sp_vision bridge with config %s", runtime_config_path_.c_str());
        worker_thread_ = std::thread(&SpVisionBridge::worker_main, this, runtime_config_path_);
    }

    void update() override {
        store_latest_hard_sync_snapshot(*hard_sync_snapshot_);
        publish_latest_target_snapshot();
    }

private:
    void store_latest_hard_sync_snapshot(const rmcs_msgs::HardSyncSnapshot& snapshot) {
        std::lock_guard<std::mutex> lock(hard_sync_snapshot_mutex_);
        latest_hard_sync_snapshot_ = snapshot;
    }

    rmcs_msgs::HardSyncSnapshot load_latest_hard_sync_snapshot() {
        std::lock_guard<std::mutex> lock(hard_sync_snapshot_mutex_);
        return latest_hard_sync_snapshot_;
    }

    void store_latest_target_snapshot(const rmcs_msgs::TargetSnapshot& snapshot) {
        std::lock_guard<std::mutex> lock(target_snapshot_mutex_);
        latest_target_snapshot_ = snapshot;
    }

    void publish_latest_target_snapshot() {
        std::lock_guard<std::mutex> lock(target_snapshot_mutex_);
        *target_snapshot_ = latest_target_snapshot_;
    }

    void worker_main(std::string runtime_config_path) {
        try {
            io::Camera camera(runtime_config_path);
            auto_aim::YOLO detector(runtime_config_path, false);
            auto_aim::Solver solver(runtime_config_path);
            auto_aim::Tracker tracker(runtime_config_path, solver);
            size_t consumed_snapshot_count = 0;
            auto next_log_time = Clock::now() + std::chrono::seconds(1);

            while (!stop_worker_.load(std::memory_order_relaxed)) {
                cv::Mat frame;
                Clock::time_point frame_timestamp;
                camera.read(frame, frame_timestamp);
                if (frame.empty())
                    continue;

                const auto snapshot = load_latest_hard_sync_snapshot();
                if (!snapshot.valid)
                    continue;

                frame_timestamp = snapshot.exposure_timestamp;
                solver.set_R_gimbal2world(snapshot_to_pose(snapshot));
                ++consumed_snapshot_count;

                if (Clock::now() >= next_log_time) {
                    RCLCPP_INFO(
                        get_logger(),
                        "[hard sync] bridge consumed %zu snapshots in the last second",
                        consumed_snapshot_count);
                    consumed_snapshot_count = 0;
                    next_log_time = Clock::now() + std::chrono::seconds(1);
                }

                auto armors = detector.detect(frame);
                auto targets = tracker.track(armors, frame_timestamp);
                if (!targets.empty())
                    store_latest_target_snapshot(make_target_snapshot(
                        std::optional<auto_aim::Target>{targets.front()}, frame_timestamp));
                else
                    store_latest_target_snapshot(
                        make_target_snapshot(std::nullopt, frame_timestamp));

                if (debug_)
                    draw_debug_frame(frame, armors, targets, solver, tracker);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "sp_vision bridge worker stopped: %s", e.what());
        }
    }

    InputInterface<rmcs_msgs::HardSyncSnapshot> hard_sync_snapshot_;

    OutputInterface<rmcs_msgs::TargetSnapshot> target_snapshot_;

    std::thread worker_thread_;
    std::atomic<bool> stop_worker_{false};

    std::mutex target_snapshot_mutex_;
    rmcs_msgs::TargetSnapshot latest_target_snapshot_{};

    std::mutex hard_sync_snapshot_mutex_;
    rmcs_msgs::HardSyncSnapshot latest_hard_sync_snapshot_{};

    bool debug_ = false;
    std::string runtime_config_path_;
};

} // namespace sp_vision_25::bridge

PLUGINLIB_EXPORT_CLASS(sp_vision_25::bridge::SpVisionBridge, rmcs_executor::Component)
