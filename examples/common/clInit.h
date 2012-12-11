//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//
#ifndef OSD_EXAMPLE_CL_INIT_H
#define OSD_EXAMPLE_CL_INIT_H

#if defined(_WIN32)
    #include <windows.h>
    #include <CL/opencl.h>
#elif defined(__APPLE__)
    #include <OpenGL/OpenGL.h>
    #include <OpenCL/opencl.h>
#else
    #include <GL/glx.h>
    #include <CL/opencl.h>
#endif

#include <cstdio>

static bool initCL(cl_context *clContext, cl_command_queue *clQueue)
{
    cl_int ciErrNum;

    cl_platform_id cpPlatform = 0;
    cl_uint num_platforms;
    ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error %d in clGetPlatformIDs call.\n", ciErrNum);
        return false;
    }
    if (num_platforms == 0) {
        printf("No OpenCL platform found.\n");
        return false;
    }
    cl_platform_id *clPlatformIDs;
    clPlatformIDs = new cl_platform_id[num_platforms];
    ciErrNum = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
    char chBuffer[1024];
    for (cl_uint i = 0; i < num_platforms; ++i) {
        ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 1024, chBuffer,NULL);
        if (ciErrNum == CL_SUCCESS) {
            cpPlatform = clPlatformIDs[i];
        }
    }
    // -------------
    cl_device_id clDevice;
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL);

#if defined(_WIN32)
    cl_context_properties props[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
        0
    };
#elif defined(__APPLE__)
    CGLContextObj kCGLContext = CGLGetCurrentContext();
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
    cl_context_properties props[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
        0
    };
#else
    cl_context_properties props[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
        0
    };
#endif
    delete[] clPlatformIDs;

    // XXX context creation should be moved to client code
    *clContext = clCreateContext(props, 1, &clDevice, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error %d in clCreateContext\n", ciErrNum);
        return false;
    }

    *clQueue = clCreateCommandQueue(*clContext, clDevice, 0, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error %d in clCreateCommandQueue\n", ciErrNum);
        return false;
    }
    return true;
}

static void uninitCL(cl_context clContext, cl_command_queue clQueue)
{
    clReleaseCommandQueue(clQueue);
    clReleaseContext(clContext);
}

#endif // OSD_EXAMPLE_CL_INIT_H
