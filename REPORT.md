# Crash & Learn — Project Report

# Part 1 — Solo Driving Licence

## MDP Formalism

### POMDP, not MDP

This is formally a **POMDP**, not a full MDP: the 103-value observation is a partial,
single-timestep view of the true simulation state (exact pose, full track geometry,
the friction/weather regime currently in effect are all hidden), and the agent has no
explicit memory of past observations (no frame-stacking, no recurrent state — see
"Why no frame-stacking" below). Velocity and steering are included precisely because
they are cheap, informative proxies for short-term dynamics that partially compensate
for that missing memory, not because the process is actually fully observable.

### State space (observation space)

103 continuous values, entirely egocentric (nothing map-specific or absolute leaks into
the observation, which is what lets a single policy generalize across circuits):

| Component | Size | Range | Source |
|---|---|---|---|
| LiDAR | 100 | `[0, 1]` | Raw range `[0.1, 15]` m, divided by `LIDAR_MAX=15` and clipped |
| Velocity | 1 | `[-0.2, 1]` | Raw `[-2, 10]` m/s, divided by `MAX_SPEED=10` |
| Steering angle | 1 | `[-1, 1]` | Raw `[-0.4189, 0.4189]` rad, divided by `MAX_STEERING` |
| Progress | 1 | `[0, 1]` | Own-start relative lap phase, clipped |

Built by `features.build_observation()`, the single pure-numpy module shared by
`env.py` (training) and `agent.py` (tournament inference) — see "Why the
train/serve split" below.

Opponent-related fields (`opponents`, `lap_count`, `rank`) exist in the raw simulation
observation but are unused in Part 1 (solo car only); they become relevant in Part 3.

**Why no frame-stacking**: driving does need short-term memory (was I already turning
this way, am I decelerating), but velocity and steering angle are already present as
direct, low-noise proxies for exactly that first-order dynamics — stacking several past
LiDAR frames on top would mostly add redundant, noise-correlated information for a large
observation-size cost (100 rays × k frames), without a clear signal the current single-
frame-plus-velocity representation is missing. This was tested empirically rather than
assumed: PPO training from a random initialization never reliably discovered forward-
driving behavior at all (see "Algorithm choice" below) regardless of this design choice,
so temporal aliasing in the observation was never the bottleneck actually observed.

### Action space

2 continuous values in `[-1, 1]`, decoded by `features.action_to_physical()`:

| Action | Normalized | Physical |
|---|---|---|
| Target speed | `-1 → +1` | `-2 → +10` m/s |
| Steering | `-1 → +1` | `-0.4189 → +0.4189` rad |

### Reward function

Computed per decision step (20 Hz) in `env.py::RacingEnv._compute_reward`, as a sum of
independently-tunable named terms:

```
+ REWARD_PROGRESS_SCALE   * progress_delta        (100.0 × Δprogress; the main signal)
- REWARD_TIME_PENALTY                              (0.05 flat, per step)
- REWARD_WALL_PROXIMITY_SCALE * proximity_ratio    (0.5 max, dense; see below)
- REWARD_WALL_COLLISION       [if wall contact]     (0.5)
- REWARD_VEHICLE_COLLISION    [if vehicle contact]  (0.5, inert in Part 1 - solo car)
- REWARD_STEERING_JERK_SCALE  * |Δnormalized steer| (0.02; discourages violent zigzag)
+ REWARD_LAP_COMPLETE         [on lap completion]   (20.0)
+ REWARD_FINISH               [on race finish]      (50.0)
- REWARD_DNF                  [on DNF]              (12.0)
```

**Wall-proximity shaping** (`WALL_PROXIMITY_MARGIN=0.2` m): below this LiDAR distance to
the nearest wall, a penalty grows linearly to `REWARD_WALL_PROXIMITY_SCALE` as the car
gets closer. This gives a gradient *before* a collision happens, rather than relying
only on the sparse post-hoc collision penalty. 0.2 m is deliberately tight (the car's
body alone is 0.31 m wide, so this only fires within ~4.5 cm of contact) — wide enough
to be a real anticipation signal, narrow enough not to punish a legitimate tight racing
line through a corner.

**Why `REWARD_DNF=12`, not larger:** an earlier version used `50`. Because
`env_simulation` does not end an episode on wall contact by itself (a car can stay
wedged against a wall for up to the full 80-step stagnation window, accruing collision
penalty every step, before the engine's own DNF fires), a large `REWARD_DNF` made a
*genuine, partially-successful attempt that ends in failure* cost strictly more, in
expectation, than *never attempting to move in the first place*. Under that regime,
policy gradient reliably converged to freezing or reversing away from every corner
regardless of `gamma`/entropy tuning — verified empirically across many training runs
this project, all of which reproduced the same degenerate behavior until this term was
reduced. `12` keeps the ordering `REWARD_FINISH(50) > REWARD_LAP_COMPLETE(20) >
REWARD_DNF(12)` intact (finishing is still clearly better than failing) without that
term dominating the calculus for a still-learning policy.

**Why `REWARD_STEERING_JERK_SCALE=0.02`, not larger:** this penalty only fires on
*change* in steering, so once steering settles on any constant value — including a hard
lock — the penalty is exactly 0 forever after, rewarding "never turn the wheel"
regardless of which value it got stuck on. An earlier value of `0.1` was strong enough,
relative to a still-weak/noisy early-training progress signal, to make constant-steering
a genuine local optimum (verified: reducing this one constant, and nothing else,
eliminated steering saturation from every subsequent training run). Kept nonzero to
still discourage genuine high-frequency zigzag, the failure mode it was originally added
for.

Self-inflicted safety valve, also in `env.py` (`WALL_CONTACT_DNF_STEPS=20`): if the car
stays in continuous wall contact for a full second, we end the episode ourselves
(applying `REWARD_DNF` once) instead of waiting up to 80 steps for the engine's own
stagnation detector, bounding the downside of a bad attempt.

### Termination conditions

- **Terminated** (`env.py::step`): `agent_status ∈ {0 (DNF), 2 (FINISHED)}` from the
  underlying engine, OR our own `WALL_CONTACT_DNF_STEPS` bail-out.
- **Truncated**: `step_count >= max_episode_steps` (6000 steps / 300 s). Raised from an
  initial 4000 because `env_simulation._REQUIRED_LAPS=3`, and even a comfortably-paced
  driver (~5 m/s average on the longest training track, Shanghai, ~498 m) needs close to
  1500 s×3laps worth of steps at reasonable speed; 4000 steps left a genuinely good
  policy no realistic room to ever reach the `FINISHED` bonus during training.

## Known engine-level limitation (not a policy defect)

Monza and Zandvoort reliably end in DNF at ≈99.9–100% lap progress, exactly 80 steps
after a lap-boundary crossing (`_DNF_WINDOW_STEPS` in `env_simulation.py`), even when
progress is verified strictly non-decreasing and there is zero wall contact in that
window — a direct contradiction of the stagnation algorithm as documented in that file
(a comment there, `FIX C1`, describes and claims to have already fixed this exact
symptom). This reproduces deterministically regardless of driving quality, so it cannot
be trained around from our side, and we did not modify `env_simulation.py` (provided,
fixed). The reward does still count every step of real progress leading up to it, so
training is not actively harmed by this, but `FINISHED` is unreachable on these two
training maps independent of policy quality.

**Important distinction**: only Monza and Zandvoort show this specific pattern (verified
independently on both, by checking progress monotonicity and wall-contact across the
80-step window following a lap-boundary crossing). Shanghai and YasMarina's occasional
DNFs/collisions are a *different*, genuine driving-difficulty issue — the car losing
grip or clipping a wall in a specific tight corner, not this lap-transition bug — and
measurably improved with further training (see the weather-robustness fix below).

## Weather randomization was silently disabled (fixed)

`env_simulation.py` ships with `_FRICTION_LO=0.99`/`_LIDAR_NOISE=0.001` — i.e. the
"sunny" end of the range the project spec describes (friction down to `0.7`, LiDAR
noise up to `0.1` in the worst downpour) — and its `_update_friction()` method, called
every step, built a local `mu` value and discarded it without ever applying it or
picking a new random draw. All the scaffolding for periodic re-randomization
(`_FRICTION_INTERVAL_SEC`, `_next_friction_steps()`, `self._friction_countdown`) was
present but never actually wired up: friction silently stayed pinned at `1.0` for every
episode, verified directly (`friction_current` never changed across thousands of steps
before the fix, spans `[0.7, 1.0]` after).

This was found only after training and evaluating several models exclusively under this
unintentionally-easy regime. A direct test confirmed the cost: forcing worst-case
weather (`friction=0.7`, `lidar_noise=0.1`) on a model trained before the fix collapsed
Shanghai from finishing cleanly to a 16%-progress crash. `_update_friction()` was
corrected to actually decrement its countdown, redraw `mu` in `[_FRICTION_LO,
_FRICTION_HI]`, and apply it via the existing `self._sim.update_params()` call (already
used correctly once, at `reset()`); LiDAR noise magnitude was made to track the same
sunny→downpour axis rather than staying at a flat constant. After fine-tuning under the
corrected randomization, the same worst-case-weather test no longer shows any early
collapse — all five maps reach ≈99.9–100% progress even in the worst simulated
conditions, though corner-heavy maps (Shanghai, YasMarina) still accumulate more
collisions under low grip than in dry conditions, which is an expected, not a
concerning, result.

## Why this train/serve split (`env.py` / `features.py` / `train.py` / `agent.py`)

At tournament time, `agent.py` runs inside a process shared with every other team's
agent, with no per-team sandboxing, and may import only `onnxruntime` and `numpy` — not
`gymnasium`, not `torch`, not `env.py`. The preprocessing (LiDAR normalization, action
decoding) must nonetheless be *identical* between training and inference, or the policy
behaves differently in competition than its training curves suggest, silently. This is
solved by `features.py`: a pure-numpy module with no other dependency, imported
unchanged by both `env.py` and `agent.py`. `agent.py` loads it via `importlib` under a
name derived from its own directory rather than a bare `import features`, because
several teams' `agent.py` files loaded into the same process would otherwise collide on
the plain module name `features` in the shared `sys.modules` cache — the first team
loaded would silently "win" and get reused by everyone else's agent.

## Algorithm choice: PPO, with a behavior-cloning warm start

**Why not train PPO from a random initialization**, which is the naive default:
every failure mode encountered during development — the car freezing near the start
line, reversing away from corners, steering locking to one extreme value, PPO's
entropy collapsing over long runs — was, on inspection, the same underlying problem in
different clothes: pure-exploration PPO on a 103-dimensional continuous-control task,
across 5 different tracks, essentially never stumbles onto "steer through this specific
corner successfully" often enough for policy gradient to reinforce it. No amount of
`gamma`/entropy-coefficient/reward-magnitude tuning fixed this (all were tried,
independently, and each reproduced the same class of degenerate policy).

**The fix**: a reactive Follow-The-Gap LiDAR controller (`pretrain_bc.py`, pure numpy —
the standard reactive baseline for exactly this kind of 360°-LiDAR racing simulator)
generates demonstrations by driving the real training environment (with its map/weather
randomization already active). The PPO policy network is then pretrained on those
demonstrations via plain supervised regression (behavior cloning) on the actor's mean
action, before a single RL update ever runs. Verified before any fine-tuning: this
warm-started policy alone already drove all 5 maps collision-free and finished races on
two of them — already better than any pure from-scratch PPO run produced across the
whole project.

**PPO is then used for fine-tuning on top of that warm start**, not SAC/TD3, for three
reasons specific to this setting:

1. **On-policy training matches the non-stationary environment.** Map and (mildly)
   friction/LiDAR-noise conditions are resampled every episode by `env_simulation`
   itself; each PPO update is computed from rollouts collected under whatever regime is
   currently active, with no stale off-policy replay buffer mixing incompatible
   dynamics regimes together.
2. **The clipped surrogate objective keeps updates conservative** — important here
   because the reward mixes several independently-shaped terms (progress, proximity,
   jerk, collision, DNF); large unclipped policy updates on a multi-term reward are a
   common source of exactly the kind of degenerate "loophole" policy this project spent
   most of its debugging time on.
3. **Practical fit with the warm start.** The BC step only pretrains the actor's mean
   action; the critic starts untrained. PPO's on-policy rollout-then-update structure
   lets the critic catch up cleanly within the first few iterations (observed:
   `explained_variance` climbing from ≈0 to >0.8 within the first ~100k steps of
   fine-tuning) without needing a warm critic or a replay buffer to bootstrap from.

**Where SAC would plausibly have been the better default, noted honestly**: SAC's
automatic entropy-temperature tuning would likely have avoided the entropy-collapse
failure this project also hit during long fine-tuning runs (fixed-value entropy
coefficients required manual, run-specific tuning), and its tanh-squashed Gaussian
policy bounds actions correctly at the gradient level, avoiding a class of
action-saturation-drift bug PPO's unsquashed Gaussian is prone to. It was not switched
to mid-project because the existing PPO pipeline was already working once the reward
and warm-start issues were fixed, and because SAC's replay buffer adds real memory
pressure risk under this project's ~4 GB Docker RAM ceiling (already the reason
`n_envs` is capped at 4, not 8). A reasonable next iteration for the champion phase
(Part 3) if training stability again becomes the bottleneck.

## Regression risk over long training runs, and the fix

PPO fine-tuning over many millions of steps does **not** improve monotonically: two long
unsupervised runs on this project drifted from a genuinely good multi-map policy into a
measurably worse one (collisions and stalls reappearing on maps that had previously
been clean) well before their step budget was exhausted, and only manual post-hoc
comparison across `CheckpointCallback`'s periodic snapshots recovered the better state.
`train.py::BestModelCallback` automates that safety net: every `--eval-freq` steps
(default 100k), it runs one deterministic rollout per training map and saves
`best_model.zip` whenever the score (finish count, dominant; progress and collision
count, tie-breaking) improves — independent of whatever state training happens to be in
when it eventually stops. `train.py`'s final ONNX export uses `best_model.zip`, not the
raw final-step model.

# Part 2 — Grand Crash Auto

## Diagnosis

Four copies of the identical solo-trained Part 1 policy (`submission/driver1..4`, all
the same model since one person trained all four) were placed in the same simulation
(`env_simulation.reset(num_cars=4)`) on Monza and run to completion:

| Car | Outcome | Max progress | Vehicle collisions | First hit at step |
|---|---|---|---|---|
| 0 | DNF @ step 1055 | 100% | 12 | step 14 |
| 1 | DNF @ step 249 | 17% | 8 | step 14 |
| 2 | **FINISHED** @ step 2918 | 100% | 7 | step 15 |
| 3 | DNF @ step 242 | 16% | 9 | step 15 |

**All four cars register a vehicle-to-vehicle collision within the first second (steps
14–15).** They spawn on a tight 2×2 grid (`_start_poses` in `env_simulation.py`), and
every one of them immediately drives its individually-optimal solo racing line —
straight into whichever car the grid happened to place there. Three of the four never
recover from that opening pileup (17%, 16%, and a much-delayed 100%-then-DNF for car 0);
only car 2 manages to disentangle itself and finish, and even it still collects 7
vehicle collisions along the way.

## Why a solo-trained policy cannot hold a shared circuit

The observation used throughout Part 1 (`features.build_observation`) deliberately
excludes the `opponents` field present in the raw simulation observation — solo
training never gave the policy any information indicating another car could be
anywhere, let alone a training signal for reacting to one. Concretely:

- **The policy has no way to perceive other cars as anything but a wall.** LiDAR *does*
  physically register another car's hull as a return, so the agent isn't literally
  blind to their presence — but nothing in training ever rewarded treating that return
  differently from track geometry (e.g. by yielding, braking, or picking a different
  line), because during Part 1 no other return was ever anything but a wall. There was
  no opponent-avoidance behavior to discover, let alone reinforce.
- **The reward function has no term at all for "another agent is nearby."**
  `REWARD_VEHICLE_COLLISION` exists in `env.py` and was already exercised in the reward
  arithmetic throughout Part 1, but it was structurally inert: solo training
  (`num_cars=1`) can never produce a vehicle collision, so gradient descent never once
  received a training signal through that term. Its presence in the code was
  future-proofing for exactly this part, not evidence the policy has ever "seen" it.
- **Each car's racing line is independently optimal, not mutually consistent.** All four
  policies are identical and deterministic given the same observation, so at the start
  grid they don't even benefit from being different drivers making different choices —
  they make the *same* choice from very similar (mirrored) starting states, which is
  part of why the opening collision is essentially guaranteed rather than a rare
  coincidence.

This is exactly the lesson the project intends: a policy is only as good as the
distribution of situations its reward function and observations were built to
distinguish between, and "another car is here" was never one of them. Part 3's task —
folding the `opponents` observation in and retraining with the vehicle-collision term
now actually reachable — is a structurally different learning problem from Part 1, not
a fine-tuning continuation of it.
