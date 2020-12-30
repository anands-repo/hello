# © 2019 University of Illinois Board of Trustees.  All rights reserved
import architectures.read_convolver_addendum as rc
import architectures.compressor_conv_small_addendum as cc
import architectures.xattn_subtract_addendum as xs

for module in [rc, cc, xs]:
    module.weight_norm = True
    module.gen_config()

# This module doesn't use xattn0, xattn1 or meta experts,
# but a binary classifier for hybrid calling
configDict = {
    "read_convolver0_addendum": rc.config,
    "read_convolver1_addendum": rc.config,
    "compressor0_addendum": cc.config,
    "compressor1_addendum": cc.config,
    "xattn2_addendum": xs.config,
}
