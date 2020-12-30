# HELLO: Haplotype Elucidation through neural-network supported Log Likelihood Optimization

The repository contains files for HELLO - a small variant caller that is designed for running standalone and hybrid small variant calling.

The following Docker image may be used with the tool. These images may not be final, and we will update here when the images are updated.

`docker pull oddjobs/hello_image.x86_64`

To build the tool, please run

```
cmake .
make -j 12
```
