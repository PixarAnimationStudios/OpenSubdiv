#include <GL/glew.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <algorithm>
#include "../common/cudaInit.h"

void cudaInit()
{
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
}
