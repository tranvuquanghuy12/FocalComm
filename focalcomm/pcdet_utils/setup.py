import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, sources):
    nvcc_args = ['-O3']
    if 'TORCH_CUDA_ARCH_LIST' in os.environ:
        pass  # PyTorch will use TORCH_CUDA_ARCH_LIST automatically
    else:
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                nvcc_args.append(f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}')
            else:
                nvcc_args.append('-gencode=arch=compute_75,code=sm_75')
        except Exception:
            nvcc_args.append('-gencode=arch=compute_75,code=sm_75')

    return CUDAExtension(
        name=name,
        sources=sources,
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': nvcc_args
        },
        include_dirs=[os.path.abspath('iou3d_nms/src')]
    )


setup(
    name='iou3d_nms_cuda',
    ext_modules=[
        make_cuda_ext(
                name='iou3d_nms_cuda',
                # module='pcdet.ops.iou3d_nms',
                sources=[
                    'iou3d_nms/src/iou3d_cpu.cpp',
                    'iou3d_nms/src/iou3d_nms_api.cpp',
                    'iou3d_nms/src/iou3d_nms.cpp',
                    'iou3d_nms/src/iou3d_nms_kernel.cu',
                ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)  # Disable ninja build
    }
)