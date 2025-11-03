from allo_optim.covariance_transformer.covariance_autoencoder import (
    AutoencoderCovarianceTransformer,
)
from allo_optim.covariance_transformer.covariance_transformer import (
    DeNoiserCovarianceTransformer,
    DetoneCovarianceTransformer,
    EllipticEnvelopeShrinkageCovarianceTransformer,
    EmpiricalCovarianceTransformer,
    LedoitWolfCovarianceTransformer,
    MarcenkoPasturCovarianceTransformer,
    OracleCovarianceTransformer,
    PCACovarianceTransformer,
    SimpleShrinkageCovarianceTransformer,
)
from allo_optim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)

TRANSFORMER_LIST: list[type[AbstractCovarianceTransformer]] = [
    AutoencoderCovarianceTransformer,
    SimpleShrinkageCovarianceTransformer,
    EllipticEnvelopeShrinkageCovarianceTransformer,
    EmpiricalCovarianceTransformer,
    OracleCovarianceTransformer,
    LedoitWolfCovarianceTransformer,
    MarcenkoPasturCovarianceTransformer,
    PCACovarianceTransformer,
    DeNoiserCovarianceTransformer,
    DetoneCovarianceTransformer,
]
