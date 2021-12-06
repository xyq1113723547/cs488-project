# cs488-project
## Overview
My project is about GPU accelerated ray tracing demostrations written in C++ and OpenCL, targeting multi-platform use.  

## Files structures
There are two versions for my ray tracer, one is CPU-based, without OpenGL version.  

## implemented technical aspects
1. recursive ray tracer for CPU
2. [Phong reflection model](https://en.wikipedia.org/wiki/Phong_reflection_model)
3. [Lambertian reflectance](https://en.wikipedia.org/wiki/Lambertian_reflectance)
3. specular recursive reflection for mirror effect
4. iterative ray tracer for OpenCL (since OpenCL language do not allow recursion)

## compile instructions
- On Windows 10, I use MSVC with Visual Studio and AMD RX560 GPU for my OpenCL version.
  - For AMD GPUs, download [AMD APP SDK](https://en.wikipedia.org/wiki/AMD_APP_SDK), which is somewhat abandoned and unoffically hosted [here](https://www.softpedia.com/get/Programming/SDK-DDK/ATI-Stream-SDK.shtml), 
  but contains all the libraries needed to compile OpenCL code, plus code samples and documentation.
  - For Intel GPUs, download Intel SDK for OpenCL [here](https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/overview.html).  
  - For Intel CPUs, download OpenCL Runtimes [here](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html).
  - For Nvidia GPUs, download R465 and later drivers [here](www.nvidia.com/drivers)
- On Linux, there could be some difficulties based on what distribution you use.  
  - For Ubuntu, Red Hat, and SUSE, propiretary SDKs and drivers can be found and installed on the download links above.
  - For Arch Linux, check [here](https://wiki.archlinux.org/title/GPGPU#OpenCL)
  - For Gentoo, check [here](https://wiki.gentoo.org/wiki/OpenCL)
- On MacOS, please
  

## Q&A
1. Why choose OpenCL?  
I choose it because the API supports multiple GPU vendor and operating systems. It compiles just-in-time, or offline if you wish.  
Besides, it is a royalty-free framework for parallel programming using GPUs.
2. Why I did not use Vulkan which I written in proposal?  
The implementation for ray tracing extension in Vulkan is vendor specific. 
Old GPUs do not have the ray tracing extension. Only NVIDIA RTX GPUs or AMD RDMA2 GPUs, which I do not have, support it.
I could dig deeper to use Vulkan for some parallel mathematical operations, but it lacks documentation and requires some low-level hardware understanding for GPU.
3. 

## objective obstacles

1. OpenCL is somewhat abandoned, at least for AMD and Apple, since there are not providing 
