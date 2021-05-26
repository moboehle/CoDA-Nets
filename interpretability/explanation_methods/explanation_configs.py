"""
Configurations for the explanations methods defined in explainers
"""

explainer_configs = {
    "Ours": {
        "default": {}
    },
    "RISE": {
        "default": {}
    },
    "GCam": {
        "default": {}
    },
    "Occlusion": {
        "default": {},
        "Occ5": {"ks": 5,
                 "stride":  2},
        "Occ9": {
            "ks": 9,
            "stride": 2
        },
        "Occ9-TI": {
            "ks": 9,
            "stride":  4,
            "batch_size": 1
        },
        "Occ13-TI": {
            "ks": 13,
            "stride": 4,
            "batch_size": 1
        },
    },
    "LIME": {
        "default": {},
        "ks4": {"kernel_size": 4},
    },
    "IxG": {
        "default": {}
    },
    "GB": {
        "default": {}
    },
    "Grad": {
        "default": {}
    },
    "IntGrad": {
        "default": {}
    },
    "DeepLIFT": {
        "default": {}
    }
}
