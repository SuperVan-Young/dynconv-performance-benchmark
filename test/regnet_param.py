def regnet_parameters(num):
    widths = [56, 28, 14, 7]
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

    return widths, channels, group

def get_factors(num):
    factors = []
    if num == 7:
        factors = [1]
    elif num == 14:
        factors = [1, 2, 7]
    elif num == 28:
        factors = [1, 2, 4, 7, 14]
    elif num == 56:
        factors = [1, 2, 4, 7, 8, 14, 28]
    return factors

def find_best_sparselen(width, slow, shigh):
    """Find best sparselen given sparsity lower limit and higher limit
    """
    def factor(num):
        ls = [i for i in range(1, num+1) if num % i == 0]
        return len(ls)

    sparselens = range(int(width*width*slow), int(width*width*shigh)+1)
    factors = [factor(i) for i in sparselens]
    idx = factors.index(max(factors))
    return max(sparselens[idx], 1) # minimum 1