"""CPU-only frozen inference loading; no optimizer/trainer/logger dependency."""
import torch

from .rwm import FrozenRWM,ModelConfig,make_model
from .service import FrozenPackage


def load_frozen_model(package):
    validated=FrozenPackage.load(package)
    raw=validated.metadata["config"]
    fields=ModelConfig.__dataclass_fields__
    config=ModelConfig(**{k:v for k,v in raw.items() if k in fields})
    if raw!=config.resolved():
        raise ValueError("frozen configuration/source pins/architecture differ from this adapter")
    model=make_model(config)
    record=validated.manifest["artifacts"]["weights"]
    weights=torch.load(validated.root/record["path"],map_location="cpu",weights_only=True)
    model.load_state_dict(weights,strict=True)
    model.reset()
    return validated,FrozenRWM(model,config)
