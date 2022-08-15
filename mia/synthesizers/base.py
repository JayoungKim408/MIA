import logging
LOGGER = logging.getLogger(__name__)


class BaseSynthesizer:
    def __init__(self, dataset_name, meta):
        self.dataset_name = dataset_name
        self.meta = meta
        
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        pass

    def sample(self, samples):
        pass

    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple(), args=None):
        LOGGER.info("Fitting %s", self.__class__.__name__)

        self.fit(data, categorical_columns, ordinal_columns, args)
        LOGGER.info("Sampling %s", self.__class__.__name__)
        num = data.shape[0]

        return self.sample(num)
