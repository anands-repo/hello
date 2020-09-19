import architectures.read_convolver as rc
import architectures.compressor_conv_small as cc
import architectures.xattn_subtract as xs
import architectures.conv_combiner as cm
import architectures.meta_convolver_ref as mc

for module in [rc, cc, xs, cm, mc]:
    module.weight_norm = True
    module.gen_config()

# This module doesn't use xattn2 and no combiners, and uses
# the reference-based meta-expert
configDict = {
    "read_conv0": rc.config,
    "read_conv1": rc.config,
    "compressor0": cc.config,
    "compressor1": cc.config,
    "xattn0": xs.config,
    "xattn1": xs.config,
    "meta": mc.config,
}
