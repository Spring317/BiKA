from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="bika",
    version="0.1.4",
    description="BiKA layers: Binarized KAN with CUDA kernels",

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    ext_modules=[
        CUDAExtension(
            name="bika._C",
            sources=[
                "src/bika/csrc/bika_binding.cpp",
                "src/bika/csrc/bika_linear.cu",
                "src/bika/csrc/bika_conv2d.cu",
                "src/bika/csrc/bika_conv2d_cpu.cpp",
                "src/bika/csrc/bika_conv2d_cpu_v2.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp", "-mavx2", "-msse4.2", "-mpopcnt", "-mbmi2", "-ffast-math"],
                "nvcc": ["-O3", "-lineinfo", "--use_fast_math",
                         "-gencode=arch=compute_75,code=sm_75"],
            },
        )
    ],

    cmdclass={
        "build_ext": BuildExtension
    },

    zip_safe=False,
)