from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="bika",
    version="0.1.3",
    description="BiKA layers: Binarized KAN with CUDA kernels",
    packages=find_packages(),                 # picks up the 'bika' package
    ext_modules=[
        CUDAExtension(
            name="bika._C",                   # compiled extension becomes bika._C
            sources=[
                "bika_binding.cpp",
                "bika_linear.cu",
                "bika_conv2d.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
