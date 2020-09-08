import architectures.read_convolver as rc
import architectures.compressor_conv_small as cc
import architectures.xattn_subtract as xs

for module in [rc, cc, xs]:
    module.weight_norm = True
    module.gen_config()

configDict = {
    "read_conv0": rc.config,
    "compressor0": cc.config,
    "xattn0": xs.config,
}
