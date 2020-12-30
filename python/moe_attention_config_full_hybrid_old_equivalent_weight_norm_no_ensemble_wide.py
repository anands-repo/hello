# © 2019 University of Illinois Board of Trustees.  All rights reserved
import architectures.read_convolver_wide as rc
import architectures.compressor_conv_small_wide as cc
import architectures.xattn_subtract_wide as xs
import architectures.conv_combiner_wide as cm

for module in [rc, cc, xs, cm]:
    module.weight_norm = True
    module.gen_config()

# This module doesn't use xattn0, xattn1 or meta experts,
# but a binary classifier for hybrid calling
configDict = {
    "read_conv0": rc.config,
    "read_conv1": rc.config,
    "compressor0": cc.config,
    "compressor1": cc.config,
    "combiner0": cm.config,
    "combiner1": cm.config,
    "xattn2": xs.config,
}
