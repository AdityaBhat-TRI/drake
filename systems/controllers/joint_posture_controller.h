#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/drake_deprecated.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/controllers/state_feedback_controller_interface.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"

namespace drake {
namespace systems {
namespace controllers {

/**
 * Posture controller for a multibody plant.
 *
 * @system
 * name: PostureController
 * input_ports:
 * - estimated_state
 * - desired_state
 * - desired_acceleration
 * output_ports:
 * - force
 * @endsystem
 *
 *
 * @pre The robot is fully actuated, its position and velocity have the same
 * dimension, and it does not have a floating base.
 * This controller was not designed for closed-loop systems: the controller
 * accounts for neither constraint forces nor actuator forces applied at
 * loop constraints. Use on such systems is not recommended.
 *
 * The JointPostureController is similar in function to the
 * InverseDynamicsController. It extends the capabilities of the
 * InverseDynamicsController by incorporating friction compensation.
 * Additionally, the name PostureController is consistent with the Operational
 * Space Control nomenclature.
 *
 * @tparam_default_scalar
 * @ingroup control_systems
 */
template <typename T>
class JointPostureController final
    : public drake::systems::Diagram<T>,
      public drake::systems::controllers::StateFeedbackControllerInterface<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(JointPostureController)

  /**
   * Constructs an posture controller for the given `plant` model.
   * The %JointPostureController holds an internal, non-owned reference to
   * the MultibodyPlant object so you must ensure that `plant` has a longer
   * lifetime than `this` %JointPostureController.
   * @param plant The model of the plant for control.
   * @param kp Proportional gain.
   * @param kd Derivative gain.
   * @pre `plant` has been finalized (plant.is_finalized() returns `true`).
   * @throws std::exception if
   *  - The plant is not finalized (see MultibodyPlant::Finalize()).
   *  - The number of generalized velocities is not equal to the number of
   *    generalized positions.
   *  - The model is not fully actuated.
   *  - Vector kp and kd do not all have the same size equal to the number
   *    of generalized positions.
   */
  JointPostureController(const drake::multibody::MultibodyPlant<T>& plant,
                         const Eigen::Ref<const Eigen::VectorXd>& kp,
                         const Eigen::Ref<const Eigen::VectorXd>& kd);

  /**
   * Constructs an posture controller and takes the ownership of the
   * input `plant`.
   */
  JointPostureController(
      std::unique_ptr<drake::multibody::MultibodyPlant<T>> plant,
      const Eigen::Ref<const Eigen::VectorXd>& kp,
      const Eigen::Ref<const Eigen::VectorXd>& kd);

  // Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit JointPostureController(const JointPostureController<U>& other);

  ~JointPostureController() override;

  /**
   * Returns the input port for the reference acceleration.
   */
  const drake::systems::InputPort<T>& get_input_port_desired_acceleration()
      const {
    DRAKE_DEMAND(input_port_index_desired_acceleration_ >= 0);
    return drake::systems::Diagram<T>::get_input_port(
        input_port_index_desired_acceleration_);
  }

  /**
   * Returns the input port for the estimated state.
   */
  const drake::systems::InputPort<T>& get_input_port_estimated_state()
      const final {
    return this->get_input_port(input_port_index_estimated_state_);
  }

  /**
   * Returns the input port for the desired state.
   */
  const drake::systems::InputPort<T>& get_input_port_desired_state()
      const final {
    return this->get_input_port(input_port_index_desired_state_);
  }

  /**
   * Returns the output port for computed control.
   */
  const drake::systems::OutputPort<T>& get_output_port_control() const final {
    return this->get_output_port(output_port_index_control_);
  }

  /**
   * Returns a constant pointer to the MultibodyPlant used for control.
   */
  const drake::multibody::MultibodyPlant<T>* get_multibody_plant_for_control()
      const {
    return multibody_plant_for_control_;
  }

 private:
  void SetUp(std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant,
             const Eigen::Ref<const Eigen::VectorXd>& kp,
             const Eigen::Ref<const Eigen::VectorXd>& kd);

  const drake::multibody::MultibodyPlant<T>* multibody_plant_for_control_{
      nullptr};
  drake::systems::InputPortIndex input_port_index_estimated_state_;
  drake::systems::InputPortIndex input_port_index_desired_state_;
  drake::systems::InputPortIndex input_port_index_desired_acceleration_;
  drake::systems::OutputPortIndex output_port_index_control_;
};

}  // namespace controllers
}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::controllers::JointPostureController)
