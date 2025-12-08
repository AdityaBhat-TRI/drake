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
from typing import Sequence
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.lcm import DrakeLcm
from pydrake.systems.lcm import LcmSubscriberSystem, LcmInterfaceSystem
from drake.lcmt_drake_signal import lcmt_drake_signal
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform, RollPitchYaw

def add_ground(plant):
    """Adds a z=0 ground plane (contact + visual) to the world body."""

    # Plane with normal +Z through the origin; solid region is below the plane.
    X_WG = HalfSpace.MakePose(np.array([0., 0., 1.]), np.array([0., 0., 0.]))

    # Contact (collision) geometry with friction.
    mu = CoulombFriction(1.0, 1.0)  # static, dynamic
    plant.RegisterCollisionGeometry(
        plant.world_body(), X_WG, HalfSpace(), "ground_collision", mu
    )

    # Visual geometry (uses the overload that takes an RGBA numpy array).
    rgba = np.array([0.9, 0.6, 0.6, 1.0])
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_WG,
        HalfSpace(),
        "ground_visual",
        rgba,
    )

# Compute kinematics
def Jp_block(h: float, beta: float) -> np.ndarray:
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[1, 0, -h*s], [0, 1, h*c]], dtype=float)

def Jq_block(phi: float, r:float, b:float) ->np.ndarray:
    s, c = np.sin(phi), np.cos(phi)
    return np.array([[-s/b, c/b], [c/r, s/r]], dtype=float)

def J_tilde_i(phi: float, r:float, b:float, h: float, beta: float) -> np.ndarray:
    Jq = Jq_block(phi, r, b)
    Jp = Jp_block(h, beta)
    return Jq @ Jp

def J_tilde_stack(phis, rs, bs, hs, betas) -> np.ndarray:
    Js = [J_tilde_i(phi, r, b, h, beta) for phi, r, b, h, beta in zip(phis, rs, bs, hs, betas)]
    return np.vstack(Js)

class CartesianActuation(LeafSystem):
    """
    Maps desired body twist xdot_body = [vx, vy, omega] (BODY frame)
    to actuator velocity commands q0dot = [phi_dot_1..N, rho_dot_1..N].

    Inputs (by index):
      0: xdot_body (size 3)
      1: plant_state (size nq+nv)  # to read steering angles φ_i

    Output:
      "u": actuator velocities (size = num_actuators). The first 2N entries are
           [phi_dot_1..N, rho_dot_1..N] from J_tilde * xdot_body. Any remaining
           actuators (if num_actuators > 2N) are filled with 0.

    Notes:
      - Set zero_steer=True to force φ_dot = 0 (like your original).
      - Provide steering position indices (into q) so we can read φ_i.
      - Geometry is given by arrays rs, bs, hs, betas (length N).
    """
    def __init__(self,
                 num_actuators: int,
                 steer_q_indices: Sequence[int],        # length N, indices into positions q
                 state_dim: int):
        super().__init__()
        self._nu = int(num_actuators)
        self.steer_q_indices = list(steer_q_indices)
        self.rs = np.asarray([0.08, 0.08, 0.08, 0.08], float)
        self.bs = np.asarray([0.05, 0.05, 0.05, 0.05], float)
        self.hs = np.asarray([0.15, 0.15, 0.15, 0.15], float)
        self.phi0 = np.deg2rad([0.0, 0.0, 0.0, 0.0])
        # self.betas = np.asarray([0.0, np.pi/2, np.pi, -np.pi/2], float)
        self.betas = np.deg2rad([45.0, 135.0, 225.0, -45.0])
        self.N = len(self.rs)
        assert len(self.bs) == len(self.hs) == len(self.betas) == self.N
        assert len(self.steer_q_indices) == self.N
        self.state_dim = int(state_dim)

        # Inputs: xdot_body (3), plant_state (nq+nv)
        self.DeclareVectorInputPort("xdot_body", BasicVector(3))
        self.DeclareVectorInputPort("plant_state", BasicVector(self.state_dim))

        # Output: actuator velocities (size = num_actuators)
        self.DeclareVectorOutputPort("u", BasicVector(self._nu), self._calc_output)

    def _read_phis(self, context) -> np.ndarray:
        x = self.get_input_port(1).Eval(context)  # plant_state
        # positions first in plant state, so indices point into q
        return np.array([x[iq] for iq in self.steer_q_indices], dtype=float)

    def _calc_output(self, context, output):
        xdot_body = self.get_input_port(0).Eval(context).reshape(3)
        phis = self._read_phis(context)
        phis = phis + self.phi0
        # Build J~ and compute actuator rates
        q0dot = J_tilde_stack(phis, self.rs, self.bs, self.hs, self.betas) @ xdot_body  # (2N,)
        print(q0dot)
        # Pack into full actuator vector (pad with zeros if plant has extra actuators)
        u = np.zeros(self._nu, dtype=float)
        u[0] = q0dot[0]
        u[1] = q0dot[2]
        u[2] = q0dot[4]
        u[3] = q0dot[6]

        u[4] = 1 * q0dot[1]
        u[5] = 1 * q0dot[3]
        u[6] = 1 * q0dot[5]
        u[7] = 1 * q0dot[7]
        output.SetFromVector(u)

class DrakeSignal3Decoder(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareAbstractInputPort(
            "lcm_message", AbstractValue.Make(lcmt_drake_signal()))
        self.DeclareVectorOutputPort("xdot_body", BasicVector(3), self._calc)
    def _calc(self, context, output):
        msg = self.get_input_port(0).Eval(context)  # lcmt_drake_signal
        vals = np.asarray(msg.val, float)
        v = np.zeros(3); v[:min(3, vals.size)] = vals[:min(3, vals.size)]
        output.SetFromVector(v)

def main():
    # Build a Diagram so we can add visualization components.
    builder = DiagramBuilder()

    # Create plant + scene graph together (required for visualization).
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)

    # Load the model from directives
    dmd_path = "systems/tzkga/tzk_base.dmd.yaml"  # <-- your path
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
    cart = builder.AddSystem(CartesianActuation(num_actuators=plant.num_actuators(), steer_q_indices=steer_q_indices, state_dim=state_dim))

    lcm = DrakeLcm()
    lcm_system = builder.AddSystem(LcmInterfaceSystem(lcm=lcm))
    sub = builder.AddSystem(
    LcmSubscriberSystem.Make(
        channel="XDOT_BODY",
        lcm_type=lcmt_drake_signal,
        lcm=lcm,
        wait_for_message_on_initialization_timeout=0.0,
        )
    )
    
    decoder = builder.AddSystem(DrakeSignal3Decoder())
    builder.Connect(sub.get_output_port(0), decoder.get_input_port(0))   # abstract msg -> decoder
    builder.Connect(decoder.get_output_port(0), cart.get_input_port(0))  # xdot_body -> input 0

    # 5) Don't forget plant state -> controller input 1 and actuation -> plant
    builder.Connect(plant.get_state_output_port(), cart.get_input_port(1))
    builder.Connect(cart.get_output_port(0), plant.get_actuation_input_port())

    print(f"Model loaded with {plant.num_positions()} positions and {plant.num_velocities()} velocities.")

    # Start Meshcat (the web viewer) and add default visualization to the Diagram.
    meshcat = StartMeshcat()  # prints the URL in your console
    AddDefaultVisualization(builder, meshcat=meshcat)

    # Build the full Diagram and create a Simulator for it.
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    q0 = plant.GetPositions(plant_context).copy()
    v0 = plant.GetVelocities(plant_context).copy()
    # q0[13] = 0.4
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)
    
    print(q0)
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    print("Starting simulation with Meshcat visualization... (Ctrl+C to stop)")

    # Run simulation loop indefinitely
    try:
        while True:
            # advance a small step
            t = context.get_time()
            simulator.AdvanceTo(t + 0.001)
            time.sleep(0.0005)  # reduce CPU usage
    except KeyboardInterrupt:
        print("\nSimulation stopped.")

if __name__ == "__main__":
    main()
