try:
    from .models import available_model_names, available_models, get_model_description, load
except ModuleNotFoundError as _e:
    # The RLDS/TFDS training-data stack (tensorflow_datasets, dlimp, dm-tree, astunparse, ...)
    # is optional and only needed for dataset loading during training. Evaluation imports
    # `prismatic.extern.hf.*` directly (self-contained HF modeling/processing), so if the
    # dataset deps are missing we skip this eager import instead of failing the whole package.
    import warnings as _w
    _w.warn(f"prismatic: skipping .models import (optional dataset deps missing: {_e})")
