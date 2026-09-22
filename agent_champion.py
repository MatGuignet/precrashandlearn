"""agent_champion.py — Part 3 champion tournament inference wrapper.

Same constraints and same collision-avoidance rationale as agent.py (Part 1) -
see that file's docstring. The only difference: this one builds the
opponent-augmented 123-dim observation (features.build_observation_multi)
instead of the 103-dim solo one, since the champion policy was trained with
opponent awareness.
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
        here = os.path.dirname(os.path.abspath(__file__))
        self.model = onnxruntime.InferenceSession(os.path.join(here, "model.onnx"))
        self._input_name = self.model.get_inputs()[0].name

    def predict(self, obs: dict, info: dict) -> tuple[float, float]:
        observation = _features.build_observation_multi(obs)
        raw_action = self.model.run(None, {self._input_name: observation[None, :]})[0][0]
        target_speed, steering = _features.action_to_physical(raw_action)
        return float(target_speed), float(steering)
