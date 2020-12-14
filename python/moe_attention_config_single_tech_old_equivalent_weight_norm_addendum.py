import architectures.read_convolver_addendum as rc
import architectures.compressor_conv_small_addendum as cc
import architectures.xattn_subtract_addendum as xs

for module in [rc, cc, xs]:
    module.weight_norm = True
    module.gen_config()

configDict = {
    "read_convolver0_addendum": rc.config,
    "compressor0_addendum": cc.config,
    "xattn0_addendum": xs.config,
}
