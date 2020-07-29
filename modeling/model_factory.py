from modeling.base_model import BaseModel
from modeling.torch_model import HydraTorch

def create_model(config, is_train = False) -> BaseModel:
    if config["model_type"] == "pytorch":
        return HydraTorch(config)
    # elif config["model_type"] == "tf":
    #     return HydraTensorFlow(config, is_train, num_gpu)
    else:
        raise NotImplementedError("model type {0} is not supported".format(config["model_type"]))