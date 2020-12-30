# © 2019 University of Illinois Board of Trustees.  All rights reserved
# configDict = {
#     "read_conv0": "architectures.read_convolver",
#     "compressor0": "architectures.compressor_conv_small",
#     "xattn0": "architectures.xattn_subtract",
#     "weight_norm": True,
# }

import architectures.read_convolver
import architectures.compressor_conv_small
import architectures.xattn_subtract

for module in [architectures.read_convolver, architectures.compressor_conv_small, architectures.xattn_subtract]:
    # module.norm_type = "LayerNormModule"
    module.norm_type = "Noop"
    module.activation = "Softplus"
    module.gen_config()

configDict = {
    "read_conv0": architectures.read_convolver.config,
    "compressor0": architectures.compressor_conv_small.config,
    "xattn0": architectures.xattn_subtract.config,
}
