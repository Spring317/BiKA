from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bika_cuda',
    ext_modules=[
        CUDAExtension(
            name='bika_cuda',
            sources=[
                'bika_linear.cpp',
                'bika_linear_cuda.cu',
                'bika_conv2d.cpp',
                'bika_conv2d_cuda.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
