import torch
from zennit.composites import SpecialFirstLayerMapComposite, layer_map_base, LayerMapComposite
from zennit.layer import Sum
from zennit.rules import ZPlus, Epsilon, Flat, Gamma, Pass
from zennit.types import Convolution, Linear
from zennit.composites import EpsilonPlusFlat
from crp.resnet50_canonizer import PUREResNetCanonizer

resnet50_composite = EpsilonPlusFlat(
    epsilon=1e-6,
    zero_params=["bias"],
    canonizers=[PUREResNetCanonizer()],
)

class ReferenceEpsilonPlusFlat(EpsilonPlusFlat):
    """
    For ResNet-50
    """
    def __init__(self, skip_layer, **kwargs):
        super().__init__(**kwargs)
        self.skip_layer = skip_layer

    def mapping(self, ctx, name, module):
        if name == self.skip_layer:
            return None
        return super().mapping(ctx, name, module)


