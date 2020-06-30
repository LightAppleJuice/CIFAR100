import json
import hjson
from bunch import Bunch


def parse_config(json_file):
    """
    Parse config from *.json file

    :param json_file: path to json file

    :return: dict object
    """

    with open(json_file, 'r') as config_file:
        config_dict = hjson.load(config_file)

    config = Bunch(config_dict)

    for key in config:
        config_ = Bunch(config_dict[key])
        for key_ in config_:
            if isinstance(config_dict[key][key_], dict):
                config_[key_] = Bunch(config_dict[key][key_])
        config[key] = config_

    return config