
import jax
from jax import numpy as jp
import numpy as np
from pupperv3_mjx import environment
from brax.io import mjcf, html

def verify_wheels():
    print("Initializing environment...")
    
    # Need to point to the correct XML manually since default path might be different or needed
    # Assuming the environment code uses the path argument correctly.
    # We will pass the path to the XML we just edited.
    xml_path = r"c:\Users\starg\wheeled_pupper\Tund_wheel_V3_Fork\description_Wheels_pupperv3\description\mujoco_xml\Wheel_pupper.xml"
    
    # dummy reward config
    class Config:
        class rewards:
            scales = {}
            tracking_sigma = 0.25
    
    mock_reward_config = Config()
    # Populate keys to avoid errors in init
    for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_orientation",
                "lin_vel_z", "ang_vel_xy", "orientation", "torques",
                "joint_acceleration", "mechanical_work", "action_rate",
                "stand_still", "stand_still_joint_velocity", "abduction_angle",
                "feet_air_time", "foot_slip", "termination", "knee_collision", "body_collision"]:
        mock_reward_config.rewards.scales[key] = 0.1
    
    env = environment.PupperV3Env(
        path=xml_path,
        reward_config=mock_reward_config,
        action_scale=1.0,
        observation_history=1
    )
    
    rng = jax.random.PRNGKey(0)
    print("Resetting environment...")
    state = jax.jit(env.reset)(rng)
    
    # Send a strong forward command
    # Action size is 12. Indices 2, 5, 8, 11 are wheels.
    # Let's set wheel actions to +1.0 (which results in +20 rad/s target due to our scaler)
    # And legs to 0 (default pose)
    action = jp.zeros(12)
    # Depending on signs, +1 might be forward or backward.
    # We just want to see BIG velocity.
    action = action.at[jp.array([2, 5, 8, 11])].set(1.0)
    
    print("Stepping environment with wheel action...")
    jit_step = jax.jit(env.step)
    
    # Run a few steps to let it spin up
    for i in range(20):
        state = jit_step(state, action)
        
        # Check velocities
        # qvel indices: 0-5 root, 6-? joints.
        # joint indices map to qvel shifted by +6 (root)
        # 12 joints.
        # wheel joints are 2, 5, 8, 11 in the JOINT list.
        # so in qvel they are 6 + 2, 6 + 5, etc => 8, 11, 14, 17?
        # Let's check environment inner logic for qvel
        
        qvel = state.pipeline_state.qvel
        # qvel has 6 root dofs + 12 joints = 18.
        # Indices 6 onwards are joints.
        wheel_vels = qvel[jp.array([6+2, 6+5, 6+8, 6+11])]
        
        print(f"Step {i}: Wheel Velocities: {wheel_vels}")
        
    print("\nVerification Complete.")
    final_vels = wheel_vels
    if jp.all(jp.abs(final_vels) > 5.0):
        print("SUCCESS: Wheels are spinning fast!")
    else:
        print("FAILURE: Wheels are moving slowly.")

if __name__ == "__main__":
    verify_wheels()
