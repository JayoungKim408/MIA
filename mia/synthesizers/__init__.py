from mia.synthesizers.ctgan import CTGANSynthesizer
from mia.synthesizers.octgan import OCTGANSynthesizer
from mia.synthesizers.identity import IdentitySynthesizer
from mia.synthesizers.tablegan import TableganSynthesizer
from mia.synthesizers.tvae import TVAESynthesizer

__all__ = (
    'IdentitySynthesizer',
    'TableganSynthesizer',
    'CTGANSynthesizer',
    'OCTGANSynthesizer',
    'TVAESynthesizer',
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
