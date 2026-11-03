"""Reinforcement learning for Assembly1 in Webots (PPO, residual on top of the existing gait).

The policy does NOT learn from scratch. The walk_control gait (+ ankle balance) is the baseline and the
network learns small corrections to the 12 leg joints. This is far faster and safer than learning raw
joint angles, and a robot that "does nothing" (action 0) still walks like v6.

Setup (once):   pip install gymnasium stable-baselines3 torch
                (install into the Python that Webots uses: Webots > Preferences > Python command)
Train:          open worlds/Assembly1_RL.wbt, set MODE = "train" below, press Run (use fast mode, no rendering).
Watch result:   set MODE = "play" and run again (loads models/ppo_walk.zip).
"""
import os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "walk_control"))      # reuse gait tables + ankle map
from gait_tables import JOINT_NAMES, STAND, N, TABLES
from balance_params import ANKLE_MAP
from controller import Supervisor
import gymnasium as gym
from gymnasium import spaces

# ------------------------------------------------------------------ settings
MODE            = "train"        # "train" | "play"
TOTAL_STEPS     = 500_000        # policy steps for training
MODEL_PATH      = os.path.join(HERE, "models", "ppo_walk")
ACTION_SCALE    = 0.12           # rad, max residual per leg joint
ACTION_REPEAT   = 4              # sim steps (8 ms each) per policy step  -> ~31 Hz control
MAX_EP_STEPS    = 400            # policy steps per episode (~12.8 s)
FALL_TILT       = 0.50           # rad, episode ends beyond this
CYCLE_TIME      = 1.6
STARTUP_TIME    = 3.0
BLEND_TIME      = 1.2
ARM_SWING       = 0.35
MOTOR_SPEED     = 4.0
KP, KD, MAX_CORR = 1.0, 0.10, 0.25   # same ankle balance as walk_control
TARGET_VEL      = 0.20           # m/s: forward speed that earns full reward
# ------------------------------------------------------------------

LEFT_ARM, RIGHT_ARM = "r_sho_rol", "l_sho_rol"
HOLD_AT_ZERO = ["c_wst_yaw", "revolute_1", "revolute_3", "l_sho_pit", "r_sho_pit", "l_elb_pit", "r_elb_pit"]


def smooth(x):
    x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)


def clamp(x, lim):
    return max(-lim, min(lim, x))


def sample(table, phase):
    f = (phase % 1.0) * N; i = int(f) % N; j = (i + 1) % N; a = f - int(f)
    return [table[i][k] + (table[j][k] - table[i][k]) * a for k in range(len(table[i]))]


class WalkEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.robot = Supervisor()
        self.dt = int(self.robot.getBasicTimeStep())
        self.dts = self.dt / 1000.0
        self.me = self.robot.getSelf()
        self.names = list(JOINT_NAMES) + [LEFT_ARM, RIGHT_ARM] + HOLD_AT_ZERO
        self.motors = {n: self.robot.getDevice(n) for n in self.names}
        self.psens = {n: self.robot.getDevice(n + "_sensor") for n in JOINT_NAMES}
        self.imu, self.gyro = self.robot.getDevice("imu"), self.robot.getDevice("gyro")
        self.pads = {f: [self.robot.getDevice("%s_%s" % (f, c)) for c in ("FL", "FR", "BL", "BR")] for f in ("lfoot", "rfoot")}
        self._enable()
        self.n_act = 12
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_act,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (34,), np.float32)
        self.first = True
        self.t = 0.0; self.phase = 0.0; self.alpha = 0.0; self.steps = 0
        self.prev_a = np.zeros(self.n_act, np.float32)
        self.share = {"lfoot": 0.5, "rfoot": 0.5}
        self.loads = {"lfoot": 0.0, "rfoot": 0.0}

    # ---------------------------------------------------------------- helpers
    def _enable(self):
        for n in self.names:
            self.motors[n].setVelocity(MOTOR_SPEED)
        for d in list(self.psens.values()) + [self.imu, self.gyro] + sum(self.pads.values(), []):
            d.enable(self.dt)

    def _foot_load(self, f):
        return sum(math.sqrt(sum(x * x for x in p.getValues()[:3])) for p in self.pads[f])

    def _sense(self):
        r, p, _ = self.imu.getRollPitchYaw()
        wF, wL, _ = self.gyro.getValues()[:3]
        self.roll, self.pitch, self.wF, self.wL = r, p, wF, wL
        self.tilt = math.acos(max(-1.0, min(1.0, math.cos(p) * math.cos(r))))
        for f in self.pads:
            self.loads[f] = self._foot_load(f)
        tot = self.loads["lfoot"] + self.loads["rfoot"]
        for f in self.pads:
            raw = self.loads[f] / tot if tot > 2.0 else 0.5
            self.share[f] += min(1.0, self.dts / 0.05) * (raw - self.share[f])

    def _command(self, residual, walking):
        """Baseline gait + ankle balance + learned residual -> motor targets."""
        t = self.t
        stand_up = smooth(t / STARTUP_TIME)
        target = 1.0 if walking else 0.0
        step = self.dts / BLEND_TIME
        self.alpha = min(target, self.alpha + step) if self.alpha < target else max(target, self.alpha - step)
        if self.alpha > 0.0:
            self.phase = (self.phase + self.dts / CYCLE_TIME) % 1.0
        row = sample(TABLES["fwd"], self.phase)
        legs = [STAND[i] + self.alpha * (row[i] - STAND[i]) for i in range(12)]
        cmd = {n: legs[i] * stand_up for i, n in enumerate(JOINT_NAMES)}
        cmd[LEFT_ARM] = -ARM_SWING * self.alpha * row[12]
        cmd[RIGHT_ARM] = +ARM_SWING * self.alpha * row[13]
        for n in HOLD_AT_ZERO:
            cmd[n] = 0.0
        gain = smooth((t - (STARTUP_TIME - 0.5)) / 1.0)
        dp = clamp(KP * self.pitch + KD * self.wL, MAX_CORR)
        dr = clamp(KP * self.roll + KD * self.wF, MAX_CORR)
        for leg, foot in (("L", "lfoot"), ("R", "rfoot")):
            w = 2.0 * self.share[foot] * gain
            pj, rj, M = ANKLE_MAP[leg]
            cmd[pj] += M[0][0] * w * dp + M[0][1] * w * dr
            cmd[rj] += M[1][0] * w * dp + M[1][1] * w * dr
        if residual is not None and gain > 0.99:
            for i, n in enumerate(JOINT_NAMES):
                cmd[n] += float(residual[i]) * ACTION_SCALE
        for n, v in cmd.items():
            self.motors[n].setPosition(v)

    def _sim_step(self, residual, walking):
        if self.robot.step(self.dt) == -1:
            raise SystemExit
        self.t += self.dts
        self._sense()
        self._command(residual, walking)

    def _obs(self):
        q = np.array([self.psens[n].getValue() for n in JOINT_NAMES]) - np.array(STAND)
        tot = self.loads["lfoot"] + self.loads["rfoot"]
        o = np.concatenate([[self.roll, self.pitch], np.clip([self.wF, self.wL], -10, 10) * 0.1,
                            q, [self.loads["lfoot"] / 20.0, self.loads["rfoot"] / 20.0], [self.share["lfoot"], self.share["rfoot"]],
                            [math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)], self.prev_a])
        return o.astype(np.float32)

    def _forward_vel_and_drift(self):
        v = self.me.getVelocity()                       # world frame: vx vy vz wx wy wz
        yaw = self.imu.getRollPitchYaw()[2]
        fx, fy = math.cos(yaw), math.sin(yaw)
        return v[0] * fx + v[1] * fy, -v[0] * fy + v[1] * fx

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self.first:
            self.robot.simulationReset()
            self.robot.simulationResetPhysics()
            self.robot.step(self.dt)
            self._enable()
        self.first = False
        self.t = 0.0; self.phase = 0.0; self.alpha = 0.0; self.steps = 0
        self.prev_a[:] = 0
        self.share = {"lfoot": 0.5, "rfoot": 0.5}
        self.me.resetPhysics()
        # stand up from the CAD pose with the baseline controller (no learning during this part)
        for _ in range(int((STARTUP_TIME + 0.8) / self.dts)):
            self._sim_step(None, False)
        self.x0 = self.me.getPosition()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        for _ in range(ACTION_REPEAT):
            self._sim_step(a, True)
        self.steps += 1
        vf, vl = self._forward_vel_and_drift()
        fell = self.tilt > FALL_TILT
        reward = (1.0 * min(vf, TARGET_VEL) / TARGET_VEL      # walk forward (capped)
                  - 0.5 * abs(vl) / TARGET_VEL                # don't drift sideways
                  - 2.0 * self.tilt ** 2                      # stay upright
                  - 0.02 * float(np.sum(a ** 2))              # small actions
                  - 0.05 * float(np.sum((a - self.prev_a) ** 2))   # smooth actions
                  + 0.1)                                      # alive bonus
        if fell:
            reward -= 5.0
        self.prev_a = a
        return self._obs(), float(reward), bool(fell), self.steps >= MAX_EP_STEPS, {"vf": vf}


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    env = WalkEnv()
    if MODE == "train":
        if os.path.exists(MODEL_PATH + ".zip"):
            print("continuing from", MODEL_PATH + ".zip")
            model = PPO.load(MODEL_PATH, env=env)
        else:
            model = PPO("MlpPolicy", env, n_steps=2048, batch_size=128, learning_rate=3e-4, gamma=0.99,
                        gae_lambda=0.95, ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128]), verbose=1)
        ckpt = CheckpointCallback(save_freq=20_000, save_path=os.path.dirname(MODEL_PATH), name_prefix="ppo_walk_ckpt")
        model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt)
        model.save(MODEL_PATH)
        print("saved", MODEL_PATH + ".zip  - set MODE = 'play' to watch it")
    else:
        model = PPO.load(MODEL_PATH)
        obs, _ = env.reset()
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            if term or trunc:
                obs, _ = env.reset()


main()