import torch
from zennit.composites import SpecialFirstLayerMapComposite, layer_map_base, LayerMapComposite
from zennit.layer import Sum
from zennit.rules import ZPlus, Epsilon, Flat, Gamma, Pass
from zennit.types import Convolution, Linear
from zennit.composites import EpsilonPlusFlat
from resnet50_canonizer import PUREResNetCanonizer

resnet50_composite = EpsilonPlusFlat(
    epsilon=1e-6,
    zero_params=["bias"],
    canonizers=[PUREResNetCanonizer()],
)

class EpsilonPlusFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = layer_map_base + [
            (Convolution, ZPlus()),
            (torch.nn.Linear, Epsilon()),
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)

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


