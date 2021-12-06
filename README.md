# cs488-project
## Overview
My project is about GPU accelerated ray tracing demostrations written in C++ and OpenCL, targeting multi-platform use.  
![cl_image](https://user-images.githubusercontent.com/53088781/144907254-13927407-f7b6-4231-b749-6b99e39a31d7.png)


## Files structures
There are two versions for my ray tracer:
- one is CPU-based, without OpenGL version, the output is `nohier.png` file.
- Another is GPU-based, with OpenCL source code and invokes OpenGL using glew and glut library, the output is a static OpenGL window.
- On Windows 10, the OpenCL version can be run by double click `Project1.exe` with `opencl_kernel.cl` `glew64.dll` and `glut64.dll` in the same directory.

## implemented technical aspects
1. recursive ray tracer for CPU
2. [Phong reflection model](https://en.wikipedia.org/wiki/Phong_reflection_model)
3. [Lambertian reflectance](https://en.wikipedia.org/wiki/Lambertian_reflectance)
3. specular recursive reflection for mirror effect
4. iterative ray tracer for OpenCL (since OpenCL language do not allow recursion)

## Source Code explanation
1. Since OpenCL do not allow recursion, I need to transform my recursive ray-tracer to a iterative one. Luckily, by adding accumulator for color, and change ray direction after each iteration, it is sufficient for this transformation.
2. The OpenCL source code is adapted from Ray Tracey's blog [here](https://github.com/straaljager/OpenCL-path-tracing-tutorial-3-Part-1). I change the OpenCL kernel source code by add my ray tracer and ray-sphere intersetion functions, though keep the OpenCL and OpenGL inter-operation header, API invocations, and helper functions as the same as the origin one.
3. The non-OpenCL version is based on my CS488 A4, all written by me.

## OpenCL environment preparation
- On Windows 10
  - For AMD GPUs, download [AMD APP SDK](https://en.wikipedia.org/wiki/AMD_APP_SDK), which is abandoned and unoffically hosted [here](https://www.softpedia.com/get/Programming/SDK-DDK/ATI-Stream-SDK.shtml), 
  but contains all the libraries needed to compile OpenCL code, plus code samples and documentation.
  - For Intel GPUs, download Intel SDK for OpenCL [here](https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/overview.html).  
  - For Intel CPUs, download OpenCL Runtimes [here](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html).
  - For Nvidia GPUs, download R465 and later drivers [here](www.nvidia.com/drivers)
- On Linux, there could be some difficulties based on what distribution you use.  
  - For Ubuntu, Red Hat, and SUSE, propiretary SDKs and drivers can be found and installed on the download links above.
  - For Arch Linux, check [here](https://wiki.archlinux.org/title/GPGPU#OpenCL)
  - For Gentoo, check [here](https://wiki.gentoo.org/wiki/OpenCL)
- On MacOS, please see https://developer.apple.com/opencl/ for support list and code samples.

## compile instruction
- On Windows 10 for OpenCL version
  I use MSVC with Visual Studio and AMD RX560 GPU for my OpenCL version and I installed AMD APP SDK 3.0.
  - add `C:\Program Files (x86)\AMD APP SDK\3.0\include` to "Additional Include Directories" in Visual Studio.
  - add `C:\Program Files (x86)\AMD APP SDK\3.0\lib\x86_64\` to "Additional Library Directories" in linker->input.
  - Then press F5 for debug.
- On Linux for the non-OpenCL version, run in A4 folder `make clean all` to compile, enter Asset folder and run by program by `../A4`

## Q&A
1. Why choose OpenCL?  
I choose it because the API supports multiple GPU vendor and operating systems. It compiles just-in-time, or offline if you wish.  
Besides, it is a royalty-free framework for parallel programming using GPUs.
2. Why I did not use Vulkan which I written in proposal?  
The implementation for ray tracing extension in Vulkan is vendor specific. 
Old GPUs do not have the ray tracing extension. Only NVIDIA RTX GPUs or AMD RDMA2 GPUs support it, unfortunately which I do not have.
I could dig deeper to use Vulkan for some parallel mathematical operations, but it lacks documentation and requires some low-level hardware understanding for GPU.

## Credits
- [smallpt](http://www.kevinbeason.com/smallpt/)
- [Business Card Ray Tracer](http://eastfarthing.com/blog/2016-01-12-card/)
- [Ray Tracey's blog](https://raytracey.blogspot.com/2017/01/opencl-path-tracing-tutorial-3-opengl.html)
- [Ray Tracing in One Weekend](https://raytracing.github.io/)

## objective obstacles

1. OpenCL is somewhat abandoned, at least for AMD and Apple, since they are not providing drivers, libraries and documentations anymore.
2. OpenCL is very hard to debug, although OpenCL C language is a subset of C99 with some extensions, it do *NOT* allow for comment using `//`, `size_t double goto` is missing for some implementations. The OpenCL source code is passed to source code as string then compile just-in-time by invoke API.
3. Implementation for OpenCL on Linux is not really hardware agonostic, because it contains bugs for detecting `libopencl.so` by `opencl-icd`. Proprietary implementation did not work well on my GPU and Arch Linux, it will flickering with strange colors and go black which requiring me to reboot.
