from ._factory import (
    regnet_configs,
    regnet_x_400mf,
    regnet_x_800mf,
    regnet_x_1_6gf,
    regnet_x_3_2gf,
    regnet_x_8gf,
    regnet_x_16gf,
    regnet_x_32gf,
    regnet_y_400mf,
    regnet_y_800mf,
    regnet_y_1_6gf,
    regnet_y_3_2gf,
    regnet_y_8gf,
    regnet_y_16gf,
    regnet_y_32gf,
    regnet_y_128gf
)

__all__ = [
    "REGNET_ENTRYPOINT",
    "REGNET_CONFIG",
]

REGNET_ENTRYPOINT = {
    "regnet_x_400mf": regnet_x_400mf,
    "regnet_x_800mf": regnet_x_800mf,
    "regnet_x_1_6gf": regnet_x_1_6gf,
    "regnet_x_3_2gf": regnet_x_3_2gf,
    "regnet_x_8gf": regnet_x_8gf,
    "regnet_x_16gf": regnet_x_16gf,
    "regnet_x_32gf": regnet_x_32gf,
    "regnet_y_400mf": regnet_y_400mf,
    "regnet_y_800mf": regnet_y_800mf,
    "regnet_y_1_6gf": regnet_y_1_6gf,
    "regnet_y_3_2gf": regnet_y_3_2gf,
    "regnet_y_8gf": regnet_y_8gf,
    "regnet_y_16gf": regnet_y_16gf,
    "regnet_y_32gf": regnet_y_32gf,
    "regnet_y_128gf": regnet_y_128gf,    
}

REGNET_CONFIG = {
    "regnet_x_400mf": regnet_configs["regnet_x_400mf"],
    "regnet_x_800mf": regnet_configs["regnet_x_800mf"],
    "regnet_x_1_6gf": regnet_configs["regnet_x_1_6gf"],
    "regnet_x_3_2gf": regnet_configs["regnet_x_3_2gf"],
    "regnet_x_8gf": regnet_configs["regnet_x_8gf"],
    "regnet_x_16gf": regnet_configs["regnet_x_16gf"],
    "regnet_x_32gf": regnet_configs["regnet_x_32gf"],
    "regnet_y_400mf": regnet_configs["regnet_y_400mf"],
    "regnet_y_800mf": regnet_configs["regnet_y_800mf"],
    "regnet_y_1_6gf": regnet_configs["regnet_y_1_6gf"],
    "regnet_y_3_2gf": regnet_configs["regnet_y_3_2gf"],
    "regnet_y_8gf": regnet_configs["regnet_y_8gf"],
    "regnet_y_16gf": regnet_configs["regnet_y_16gf"],
    "regnet_y_32gf": regnet_configs["regnet_y_32gf"],
    "regnet_y_128gf": regnet_configs["regnet_y_128gf"],
}