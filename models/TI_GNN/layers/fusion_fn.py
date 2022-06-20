import torch


class FusionFn:
    def __init__(self):
        self.function = None

    def __call__(self, arg1, arg2):
        return self.function(arg1, arg2)

    def get_output_dim(self, input_dim1, input_dim2):
        pass


class ArithmeticFusionFn(FusionFn):
    def get_output_dim(self, input_dim1, input_dim2):
        if input_dim1 != input_dim2:
            raise RuntimeError('Dimensions must be equal')
        return input_dim1


class PlusFusionFn(ArithmeticFusionFn):
    def __init__(self):
        super().__init__()
        self.function = lambda x,y: x+y


class MinusFusionFn(ArithmeticFusionFn):
    def __init__(self):
        super().__init__()
        self.function = lambda x,y: x-y


class MultFusionFn(ArithmeticFusionFn):
    def __init__(self):
        super().__init__()
        self.function = lambda x,y: x*y


class ConcatFusionFn(FusionFn):
    def __init__(self):
        super().__init__()
        self.function = lambda x,y: torch.cat([x, y], dim=1)

    def get_output_dim(self, input_dim1, input_dim2):
        return input_dim1 + input_dim2


class FirstFusionFn(FusionFn):
    def __init__(self):
        super().__init__()
        self.function = lambda x,y: x

    def get_output_dim(self, input_dim1, input_dim2):
        return input_dim1


class SecondFusionFn(FusionFn):
    def __init__(self):
        super().__init__()
        self.function = lambda x,y: y

    def get_output_dim(self, input_dim1, input_dim2):
        return input_dim2


def get_fusion_fn(fn_name):
    if fn_name is None:
        return None
    if fn_name == 'plus':
        return PlusFusionFn
    if fn_name == 'minus':
        return MinusFusionFn
    if fn_name == 'mult':
        return MultFusionFn
    if fn_name == 'concat':
        return ConcatFusionFn
    if fn_name == 'first':
        return FirstFusionFn
    if fn_name == 'second':
        return SecondFusionFn

