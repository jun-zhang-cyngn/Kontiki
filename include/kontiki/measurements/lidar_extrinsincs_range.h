//
// Created by hannes on 2017-11-29.
//

#ifndef KONTIKIV2_LIDAR_ORIENTATION_MEASUREMENT_H
#define KONTIKIV2_LIDAR_ORIENTATION_MEASUREMENT_H

#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#include <kontiki/trajectories/trajectory.h>
#include <kontiki/trajectory_estimator.h>
#include "../sensors/lidar.h"

namespace kontiki {
namespace measurements {

template<typename LiDARModel>
class LiDARExtrinsicsRange {

 public:
  LiDARExtrinsicsRange(std::shared_ptr<LiDARModel> lidar, double t, double translation_norm)
    : lidar_(lidar), t(t), translation_norm_(translation_norm) {
      std::cout << "translation_norm is: " << translation_norm_ << "\n";
    }

  template<typename TrajectoryModel, typename T>
  T Measure(const type::Trajectory<TrajectoryModel, T> &trajectory,
                                 const type::LiDAR<LiDARModel, T> &lidar) const {
    const Eigen::Matrix<T, 3, 1> p_L_I = lidar.relative_position();
    Eigen::Matrix<T, 1, 1> val;
    val << p_L_I.norm() - translation_norm_;
    return val.norm();
  }

  template<typename TrajectoryModel, typename T>
  T Error(const type::Trajectory<TrajectoryModel, T> &trajectory, const type::LiDAR<LiDARModel, T> &lidar) const {
    return Measure<TrajectoryModel, T>(trajectory, lidar);
  }

  // Measurement data
  std::shared_ptr<LiDARModel> lidar_;
  double t; // not used
  double translation_norm_; 

 protected:

  // Residual struct for ceres-solver
  template<typename TrajectoryModel>
  struct Residual {
    Residual(const LiDARExtrinsicsRange<LiDARModel> &m) : measurement(m) {}

    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      size_t offset = 0;
      auto trajectory = entity::Map<TrajectoryModel, T>(&params[offset], trajectory_meta);

      offset += trajectory_meta.NumParameters();
      auto lidar = entity::Map<LiDARModel, T>(&params[offset], lidar_meta);

      residual[0] = measurement.Error<TrajectoryModel, T>(trajectory, lidar);
      return true;
    }

    const LiDARExtrinsicsRange& measurement;
    typename TrajectoryModel::Meta trajectory_meta;
    typename LiDARModel::Meta lidar_meta;
  }; // Residual;

  template<typename TrajectoryModel>
  void AddToEstimator(kontiki::TrajectoryEstimator<TrajectoryModel>& estimator) {
    using ResidualImpl = Residual<TrajectoryModel>;
    auto residual = new ResidualImpl(*this);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<ResidualImpl>(residual);
    std::vector<entity::ParameterInfo<double>> parameter_info;

    // Add trajectory to problem
    const int extrinsics_type = 2;
    // The below code somehow will crash the ceres solver
    // lidar_->AddExtrinsicsCalibrationToProblem(estimator.problem(), extrinsics_type, {{t,t}}, residual->lidar_meta, parameter_info);
    lidar_->AddToProblem(estimator.problem(), {{t,t}}, residual->lidar_meta, parameter_info);


    for (auto& pi : parameter_info) {
      cost_function->AddParameterBlock(pi.size);
    }

    // Add measurement
    cost_function->SetNumResiduals(1);
    // If we had any measurement parameters to set, this would be the place

    // Give residual block to estimator problem
    estimator.problem().AddResidualBlock(cost_function,
                                         nullptr,
                                         entity::ParameterInfo<double>::ToParameterBlocks(parameter_info));
  }

  // TrajectoryEstimator must be a friend to access protected members
  template<template<typename> typename TrajectoryModel>
  friend class kontiki::TrajectoryEstimator;
};

} // namespace measurements
} // namespace kontiki


#endif //KONTIKIV2_LIDAR_ORIENTATION_MEASUREMENT_H
