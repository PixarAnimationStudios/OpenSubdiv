//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#ifndef OSD_EXAMPLE_CL_INIT_H
#define OSD_EXAMPLE_CL_INIT_H

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <OpenGL/OpenGL.h>
#else
    #include <GL/glx.h>
#endif

#include "osd/opencl.h"

#include <cstdio>

static bool HAS_CL_VERSION_1_1 () {
#ifdef OPENSUBDIV_HAS_OPENCL
     #ifdef OPENSUBDIV_HAS_CLEW
        static bool clewInitialized = false;
        static bool clewLoadSuccess;
        if (not clewInitialized) {
            clewInitialized = true;
            clewLoadSuccess = clewInit() == CLEW_SUCCESS;
            if (not clewLoadSuccess) {
                fprintf(stderr, "Loading OpenCL failed.\n");
            }
        }
        return clewLoadSuccess;
    #endif
    return true;
#else
    return false;
#endif
}
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
    cl_platform_id *clPlatformIDs = new cl_platform_id[num_platforms];
    ciErrNum = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
    char chBuffer[1024];
    for (cl_uint i = 0; i < num_platforms; ++i) {
        ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 1024, chBuffer,NULL);
        if (ciErrNum == CL_SUCCESS) {
            cpPlatform = clPlatformIDs[i];
        }
    }

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

    int clDeviceUsed = 0;

#if defined(__APPLE__)
    *clContext = clCreateContext(props, 0, NULL, clLogMessagesToStdoutAPPLE, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error %d in clCreateContext\n", ciErrNum);
        return false;
    }

    size_t devicesSize = 0;
    clGetGLContextInfoAPPLE(*clContext, kCGLContext, CL_CGL_DEVICES_FOR_SUPPORTED_VIRTUAL_SCREENS_APPLE, 0, NULL, &devicesSize);
    int numDevices = int(devicesSize / sizeof(cl_device_id));
    if (numDevices == 0) {
        printf("No sharable devices.\n");
        return false;
    }
    cl_device_id *clDevices = new cl_device_id[numDevices];
    clGetGLContextInfoAPPLE(*clContext, kCGLContext, CL_CGL_DEVICES_FOR_SUPPORTED_VIRTUAL_SCREENS_APPLE, numDevices * sizeof(cl_device_id), clDevices, NULL);
#else

    // get the number of GPU devices available to the platform
    cl_uint numDevices = 0;
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices == 0) {
        printf("No CL GPU device found.\n");
        return false;
    }

    // create the device list
    cl_device_id *clDevices = new cl_device_id[numDevices];
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, numDevices, clDevices, NULL);

#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"

    // find a device that supports sharing with GL (SLI / X-fire configurations)
    bool sharingSupported=false;
    for (int i=0; i<(int)numDevices; ++i) {

        size_t extensionSize;
        ciErrNum = clGetDeviceInfo(clDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
        if (ciErrNum != CL_SUCCESS) {
            printf("Error %d in clGetDeviceInfo\n", ciErrNum);
            return false;
        }
        
        if (extensionSize>0) {
            char* extensions = (char*)malloc(extensionSize);
            ciErrNum = clGetDeviceInfo(clDevices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error %d in clGetDeviceInfo\n", ciErrNum);
                return false;
            }
            std::string stdDevString(extensions);
            free(extensions);
            size_t szOldPos = 0, szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
            while (szSpacePos != stdDevString.npos) {
                if (strcmp(GL_SHARING_EXTENSION, 
                    stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str())==0) {
                    clDeviceUsed = i;
                    sharingSupported = true;
                    break;
                }
                do {
                    szOldPos = szSpacePos + 1;
                    szSpacePos = stdDevString.find(' ', szOldPos);
                } while (szSpacePos == szOldPos);
            }
        }
    }

    if (not sharingSupported) {
        printf("No device found that supports CL/GL context sharing\n");
        delete[] clDevices;
        return false;
    }

    *clContext = clCreateContext(props, 1, &clDevices[clDeviceUsed], NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error %d in clCreateContext\n", ciErrNum);
        delete[] clDevices;
        return false;
    }
#endif

    *clQueue = clCreateCommandQueue(*clContext, clDevices[clDeviceUsed], 0, &ciErrNum);
    delete[] clDevices;
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
