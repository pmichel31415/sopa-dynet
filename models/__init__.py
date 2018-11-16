from .sopa import SoPa
from .bilstm import BiLSTM

model_types = {
    "sopa": SoPa,
    "bilstm": BiLSTM,
}
supported_model_types = list(model_types.keys())


def assert_model_type_supported(model_type):
    if model_type not in model_types:
        raise ValueError(
            f"Unknown model type {model_type}. "
            f"Supported model types are {', '.join(supported_model_types)}"
        )


def add_model_args(model_type, parser):
    """Add model specific arguments to the parser"""
    assert_model_type_supported(model_type)
    model_types[model_type].add_args(parser)


def model_from_args(model_type, dic, num_classes, args):
    """Return a model from command line arguments"""
    assert_model_type_supported(model_type)
    return model_types[model_type].from_args(dic, num_classes, args)
