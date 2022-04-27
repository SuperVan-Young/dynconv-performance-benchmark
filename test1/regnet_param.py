DEBUG = 0  # non-zero will check if the scripts could run

def regnet_parameters(num):
    channels = None
    group = None

    if num == "002":
        channels = [24, 56, 152, 368]
        group = 8
    elif num == "004":
        channels = [32, 64, 160, 384]
        group = 16
    elif num == "006":
        channels = [48, 96, 240, 528]
        group = 24
    elif num == "008":
        channels = [64, 128, 288, 672]
        group = 16
    else:
        raise NotImplementedError

    return channels, group if not DEBUG else ([64], 16)

def get_factors(num):
    factors = []
    if num == 7:
        factors = [1, 7]
    elif num == 14:
        factors = [1, 2, 7]
    elif num == 28:
        factors = [1, 2, 4, 7, 14]
    elif num == 56:
        factors = [1, 2, 4, 7, 8, 14, 28]
    return factors if not DEBUG else [1]