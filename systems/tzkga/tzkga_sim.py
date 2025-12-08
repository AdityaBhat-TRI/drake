import time
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
    Simulator,
    StartMeshcat,
    AddDefaultVisualization,
    HalfSpace,
    CoulombFriction,
    LeafSystem,
    BasicVector,
)
from typing import Sequence, Optional
from pydrake.lcm import DrakeLcm
from pydrake.systems.lcm import LcmSubscriberSystem, LcmInterfaceSystem, LcmPublisherSystem
from drake.lcmt_drake_signal import lcmt_drake_signal
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.tree import JointActuatorIndex


def add_ground(plant):
    X_WG = HalfSpace.MakePose(np.array([0., 0., 1.]), np.array([0., 0., 0.]))
    mu = CoulombFriction(1.0, 1.0)
    plant.RegisterCollisionGeometry(plant.world_body(), X_WG, HalfSpace(), "ground_collision", mu)
    rgba = np.array([0.9, 0.6, 0.6, 1.0])
    plant.RegisterVisualGeometry(plant.world_body(), X_WG, HalfSpace(), "ground_visual", rgba)

# --- Kinematics ---
def Jp_block(h: float, beta: float) -> np.ndarray:
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[1, 0, -h*s], [0, 1, h*c]], dtype=float)

def Jq_block(phi: float, r:float, b:float) ->np.ndarray:
    s, c = np.sin(phi), np.cos(phi)
    return np.array([[-s/b, c/b], [c/r, s/r]], dtype=float)

def J_tilde_i(phi: float, r:float, b:float, h: float, beta: float) -> np.ndarray:
    return Jq_block(phi, r, b) @ Jp_block(h, beta)

def J_tilde_stack(phis, rs, bs, hs, betas) -> np.ndarray:
    Js = [J_tilde_i(phi, r, b, h, beta) for phi, r, b, h, beta in zip(phis, rs, bs, hs, betas)]
    return np.vstack(Js)


def actuator_velocity_indices(plant):
    idx = []
    for i in range(plant.num_actuators()):
        act = plant.get_joint_actuator(JointActuatorIndex(i))  # <-- FIX
        j = act.joint()
        nv_j = j.num_velocities()
        if nv_j != 1:
            raise RuntimeError(
                f"Actuator '{act.name()}' is on joint '{j.name()}' with {nv_j} "
                "velocities. Update the mapping to select which velocity to control."
            )
        idx.append(j.velocity_start())
    return idx

class CartesianActuation(LeafSystem):
    def __init__(self,
                 num_actuators: int,
                 steer_q_indices: Sequence[int],
                 state_dim: int):
        super().__init__()
        self._nu = int(num_actuators)
        self.steer_q_indices = list(steer_q_indices)
        self.rs = np.asarray([0.08, 0.08, 0.08, 0.08], float)
        self.bs = np.asarray([0.02, 0.02, 0.02, 0.02], float)
        self.hs = np.asarray([0.15, 0.15, 0.15, 0.15], float)
        self.phi0 = np.deg2rad([0.0, 0.0, 0.0, 0.0])
        self.betas = np.deg2rad([45.0, 135.0, 225.0, -45.0])
        self.N = len(self.rs)
        assert len(self.bs) == len(self.hs) == len(self.betas) == self.N
        assert len(self.steer_q_indices) == self.N
        self.state_dim = int(state_dim)

        self.DeclareVectorInputPort("xdot_body", BasicVector(3))
        self.DeclareVectorInputPort("plant_state", BasicVector(self.state_dim))
        # Output is desired actuator *velocities* (v_des)
        self.DeclareVectorOutputPort("u", BasicVector(self._nu), self._calc_output)

    def _read_phis(self, context) -> np.ndarray:
        x = self.get_input_port(1).Eval(context)  # plant_state
        return np.array([x[iq] for iq in self.steer_q_indices], dtype=float)

    def _calc_output(self, context, output):
        xdot_body = self.get_input_port(0).Eval(context).reshape(3)
        phis = self._read_phis(context) + self.phi0
        q0dot = J_tilde_stack(phis, self.rs, self.bs, self.hs, self.betas) @ xdot_body  # (2N,)

        u = np.zeros(self._nu, dtype=float)
        # steer φ̇1..4
        u[0] = q0dot[0]
        u[1] = q0dot[2]
        u[2] = q0dot[4]
        u[3] = q0dot[6]
        # drive ρ̇1..4 with signs
        u[4] = 1 * q0dot[1]
        u[5] =  1 * q0dot[3]
        u[6] =  1 * q0dot[5]
        u[7] =  1 * q0dot[7]
        output.SetFromVector(u)

class DrakeSignalNEncoder(LeafSystem):
    def __init__(self, n: int, coord=None):
        super().__init__()
        self.n = int(n)
        self.coord = list(coord) if coord is not None else [f"u{i}" for i in range(self.n)]
        self.DeclareVectorInputPort("u", BasicVector(self.n))
        # Input for timestamp in seconds (sim time)
        self.DeclareVectorInputPort("t_sec", BasicVector(1))
        self.DeclareAbstractOutputPort("lcm_message", lambda: AbstractValue.Make(lcmt_drake_signal()), self._calc)

    def _calc(self, context, output):
        msg = lcmt_drake_signal()
        v = self.get_input_port(0).Eval(context)
        t_sec = float(self.get_input_port(1).Eval(context)[0])
        msg.dim = self.n
        msg.val = [float(x) for x in v]
        msg.coord = self.coord
        msg.timestamp = int(t_sec * 1e6)  # µs
        output.set_value(msg)

class VelocityToTorquePID(LeafSystem):
    def __init__(self, plant, kp: float = 50.0, ki: float = 0.0, kd: float = 0.0,
                 v_indices: Optional[Sequence[int]] = None, dt_est: float = 0.001):
        super().__init__()
        self._plant = plant
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.dt = float(dt_est)
        self.na = plant.num_actuators()
        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()

        self.v_indices = list(range(self.na)) if v_indices is None else list(v_indices)
        assert len(self.v_indices) == self.na

        self.DeclareVectorInputPort("v_des", BasicVector(self.na))
        self.DeclareVectorInputPort("plant_state", BasicVector(self.nq + self.nv))
        self.DeclareVectorOutputPort("tau", BasicVector(self.na), self._calc)

        if self.kd != 0.0:
            self._prev_v = self.DeclareDiscreteState(self.na)
            self.DeclarePerStepDiscreteUpdateEvent(self._upd_prev)
        if self.ki != 0.0:
            self._z = self.DeclareDiscreteState(self.na)
            self.DeclarePerStepDiscreteUpdateEvent(self._upd_int)

    def _extract_v(self, x):
        v_full = x[self.nq:self.nq+self.nv]
        return v_full[self.v_indices]

    def _calc(self, context, output):
        v_des = self.get_input_port(0).Eval(context)
        x = self.get_input_port(1).Eval(context)
        v_meas = self._extract_v(x)
        tau = self.kp * (v_des - v_meas)

        if self.ki != 0.0:
            z = context.get_discrete_state(self._z).get_value()
            tau = tau + self.ki * z
        if self.kd != 0.0:
            v_prev = context.get_discrete_state(self._prev_v).get_value()
            dv = (v_meas - v_prev) / max(self.dt, 1e-6)
            tau = tau - self.kd * dv

        output.SetFromVector(tau)

    def _upd_prev(self, context, state):
        x = self.get_input_port(1).Eval(context)
        v_meas = self._extract_v(x)
        state.get_mutable_discrete_state(self._prev_v).get_mutable_value()[:] = v_meas

    def _upd_int(self, context, state):
        v_des = self.get_input_port(0).Eval(context)
        x = self.get_input_port(1).Eval(context)
        v_meas = self._extract_v(x)
        err = v_des - v_meas
        z = state.get_mutable_discrete_state(self._z).get_mutable_value()
        z[:] = z + self.dt * err

class DrakeSignal3Decoder(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareAbstractInputPort("lcm_message", AbstractValue.Make(lcmt_drake_signal()))
        self.DeclareVectorOutputPort("xdot_body", BasicVector(3), self._calc)
    def _calc(self, context, output):
        msg = self.get_input_port(0).Eval(context)
        vals = np.asarray(msg.val, float)
        v = np.zeros(3); v[:min(3, vals.size)] = vals[:min(3, vals.size)]
        output.SetFromVector(v)

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)

    dmd_path = "systems/tzkga/tzk_base.dmd.yaml"
    directives = LoadModelDirectives(dmd_path)
    ProcessModelDirectives(directives, plant, parser)
    add_ground(plant)
    mi = plant.GetModelInstanceByName("base_chassis")
    base_body = plant.GetBodyByName("chassis_link", mi)
    X_WB = RigidTransform(RollPitchYaw(0.0, 0.0, np.pi/2.0), [0.0, 0.0, 0.0])
    plant.SetDefaultFreeBodyPose(base_body, X_WB)
    plant.Finalize()

    steer_q_indices = [7, 9, 11, 13]
    state_dim = plant.num_positions() + plant.num_velocities()

    cart = builder.AddSystem(CartesianActuation(
        num_actuators=plant.num_actuators(),
        steer_q_indices=steer_q_indices,
        state_dim=state_dim))

    # --- LCM subscription for [vx, vy, omega] ---
    lcm = DrakeLcm()
    lcm_system = builder.AddSystem(LcmInterfaceSystem(lcm=lcm))
    sub = builder.AddSystem(LcmSubscriberSystem.Make(
        channel="XDOT_BODY",
        lcm_type=lcmt_drake_signal,
        lcm=lcm,
        wait_for_message_on_initialization_timeout=0.0,
    ))
    decoder = builder.AddSystem(DrakeSignal3Decoder())
    builder.Connect(sub.get_output_port(0), decoder.get_input_port(0))      # msg -> decoder
    builder.Connect(decoder.get_output_port(0), cart.get_input_port(0))     # xdot_body -> cart

    # Plant state -> controller (to read steer angles)
    builder.Connect(plant.get_state_output_port(), cart.get_input_port(1))

    # --- NEW: Velocity PID -> torques ---
    v_indices = actuator_velocity_indices(plant)  # robust even with floating base
    vel_pid = builder.AddSystem(VelocityToTorquePID(
        plant, kp=2.1, ki=0.0, kd=0.0, v_indices=v_indices, dt_est=0.01))

    # Wire: desired actuator velocities -> velocity PID
    builder.Connect(cart.get_output_port(0), vel_pid.get_input_port(0))
    builder.Connect(plant.get_state_output_port(), vel_pid.get_input_port(1))

    # PID torques -> plant (REPLACES direct cart->plant connection)
    builder.Connect(vel_pid.get_output_port(0), plant.get_actuation_input_port())

    # --- Encode & publish torques over LCM ---
    tau_enc = builder.AddSystem(DrakeSignalNEncoder(
        n=plant.num_actuators(),
        coord=[plant.get_joint_actuator(JointActuatorIndex(i)).name() for i in range(plant.num_actuators())]
    ))
    builder.Connect(vel_pid.get_output_port(0), tau_enc.get_input_port(0))

    # Provide sim time to the encoder
    time_source = builder.AddSystem(LeafSystem())  # tiny helper to output time
    time_source.DeclareVectorOutputPort("t_sec", BasicVector(1),
        lambda ctx, out: out.SetFromVector([ctx.get_time()]))
    builder.Connect(time_source.get_output_port(0), tau_enc.get_input_port(1))

    # Publisher (LCM)
    tau_pub = builder.AddSystem(LcmPublisherSystem.Make(
        channel="TORQUE_CMD",
        lcm_type=lcmt_drake_signal,
        lcm=lcm,
        publish_period=0.001  # 1 kHz
    ))
    builder.Connect(tau_enc.get_output_port(0), tau_pub.get_input_port(0))
    print(f"Model loaded with {plant.num_positions()} positions and {plant.num_velocities()} velocities.")

    nq, nv = plant.num_positions(), plant.num_velocities()
    state_labels = [f"q{i}" for i in range(nq)] + [f"v{i}" for i in range(nv)]
    x_enc = builder.AddSystem(DrakeSignalNEncoder(n=nq+nv, coord=state_labels))
    builder.Connect(plant.get_state_output_port(), x_enc.get_input_port(0))
    builder.Connect(time_source.get_output_port(0), x_enc.get_input_port(1))
    x_pub = builder.AddSystem(LcmPublisherSystem.Make(
        channel="PLANT_STATE",
        lcm_type=lcmt_drake_signal,
        lcm=lcm,
        publish_period=0.001  # 1 kHz, match your control loop or choose slower (e.g., 0.01)
    ))
    builder.Connect(x_enc.get_output_port(0), x_pub.get_input_port(0))

    meshcat = StartMeshcat()
    AddDefaultVisualization(builder, meshcat=meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    q0 = plant.GetPositions(plant_context).copy()
    v0 = plant.GetVelocities(plant_context).copy()
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    print("Starting simulation with Meshcat visualization... (Ctrl+C to stop)")
    try:
        while True:
            t = context.get_time()
            simulator.AdvanceTo(t + 0.001)
            time.sleep(0.0005)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")

if __name__ == "__main__":
    main()
