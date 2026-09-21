"""agent.py — Tournament-time inference wrapper.

The only file loaded at tournament time, inside a shared process alongside
every other team's agent.py (no per-team sandboxing — see agent_loader.py).
This imposes two hard constraints:

1. May depend only on the standard library, onnxruntime and numpy — never
   gymnasium, torch, or env.py (training-only code/deps must be completely
   absent here).
2. Must reproduce EXACTLY the same preprocessing as env.py, or the policy
   will silently behave differently at tournament time than during training.

Point 2 is solved by importing features.py (pure-numpy, see its docstring),
the same module env.py uses. It is loaded here via importlib instead of a
plain `import features`: agent_loader.py loads every team's agent.py under a
unique module name, but a bare `import features` would still be resolved
through the process-wide `sys.modules` cache keyed on the plain name
"features" — the first team's copy loaded would silently "win" and get
reused by every other team's agent in the same process. Loading it by file
path under a name derived from this file's own directory avoids that
collision entirely.
"""

import importlib.util
import os

import onnxruntime


def _load_local_features():
    here = os.path.dirname(os.path.abspath(__file__))
    unique_name = "_features_" + here.replace(os.sep, "_").replace(".", "_").replace(" ", "_")
    spec = importlib.util.spec_from_file_location(unique_name, os.path.join(here, "features.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_features = _load_local_features()


class Agent:
    def __init__(self):
        # Resolve the model path relative to THIS file, never the cwd:
        # the tournament loads several agents from different folders in one process.
        here = os.path.dirname(os.path.abspath(__file__))
        self.model = onnxruntime.InferenceSession(os.path.join(here, "model.onnx"))
        self._input_name = self.model.get_inputs()[0].name

    def predict(self, obs: dict, info: dict) -> tuple[float, float]:
        # 1. Preprocess exactly like env.py (features.build_observation)
        observation = _features.build_observation(obs)

        # 2. Run ONNX inference
        raw_action = self.model.run(None, {self._input_name: observation[None, :]})[0][0]

        # 3. Postprocess into physical (target_speed, steering)
        target_speed, steering = _features.action_to_physical(raw_action)

        # Return pure Python floats — NOT np.float32 (rejected by verify_submission.py).
        return float(target_speed), float(steering)
