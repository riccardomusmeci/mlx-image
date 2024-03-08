from .resnet import *
from .vit import *

MODEL_ENTRYPOINT = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
    "vit_base_patch16_224": vit_base_patch16_224,
    "vit_base_patch16_224.swag_lin": vit_base_patch16_224,
    "vit_base_patch32_224": vit_base_patch32_224,
    "vit_base_patch16_384.swag_e2e": vit_base_patch16_384,
    "vit_large_patch16_224": vit_large_patch16_224,
    "vit_large_patch16_224.swag_lin": vit_large_patch16_224,
    "vit_large_patch16_512.swag_e2e": vit_large_patch16_512,
    "vit_huge_patch14_224.swag_lin": vit_huge_patch14_224,
    "vit_huge_patch14_518.swag_e2e": vit_huge_patch14_518,
}

MODEL_CONFIG = {
    "resnet18": resnet_configs["resnet18"],
    "resnet34": resnet_configs["resnet34"],
    "resnet50": resnet_configs["resnet50"],
    "resnet101": resnet_configs["resnet101"],
    "resnet152": resnet_configs["resnet152"],
    "wide_resnet50_2": resnet_configs["wide_resnet50_2"],
    "wide_resnet101_2": resnet_configs["wide_resnet101_2"],
    "vit_base_patch16_224": vit_configs["vit_base_patch16_224"],
    "vit_base_patch16_224.swag_lin": vit_configs["vit_base_patch16_224.swag_lin"],
    "vit_base_patch32_224": vit_configs["vit_base_patch32_224"],
    "vit_base_patch16_384.swag_e2e": vit_configs["vit_base_patch16_384.swag_e2e"],
    "vit_large_patch16_224": vit_configs["vit_large_patch16_224"],
    "vit_large_patch16_224.swag_lin": vit_configs["vit_large_patch16_224.swag_lin"],
    "vit_large_patch16_512.swag_e2e": vit_configs["vit_large_patch16_512.swag_e2e"],
    "vit_huge_patch14_224.swag_lin": vit_configs["vit_huge_patch14_224.swag_lin"],
    "vit_huge_patch14_518.swag_e2e": vit_configs["vit_huge_patch14_518.swag_e2e"],
}
