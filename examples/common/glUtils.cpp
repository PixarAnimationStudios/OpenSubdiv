//
//   Copyright 2015 Pixar
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

#include "glLoader.h"

#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include "glUtils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb_image_write.h"

#include <GLFW/glfw3.h>

#if _MSC_VER
#define snprintf _snprintf
#endif

namespace GLUtils {

void InitializeGL()
{
    OpenSubdiv::internal::GLLoader::applicationInitializeGL();
}

static void
_argParseBool(bool *ret, const char *lbl, int i, int argc, char **argv)
{
    if (i < argc-1) {
        if (!strcmp(argv[i+1], "on")) {
            *ret = true;
        } else if (!strcmp(argv[i+1], "off")) {
            *ret = false;
        } else {
            fprintf(stderr, "Unknown setting for %s: %s\n", lbl, argv[i+1]);
            exit(1);
        }
    } else {
        fprintf(stderr,
            "Please specify \"on\" or \"off\" for %s.\n", lbl);
        exit(1);
    }
}

void
SetMinimumGLVersion(int argc, char ** argv) {

#if defined(__APPLE__)
    // Here 3.2 is the minimum GL version supported, GLFW will allocate a
    // higher version if possible. This works on OS X, but instead limits
    // the version to 3.2 on Linux. On Linux & Windows, specifying no 
    // version hint should use the highest version available.
    //
    // http://www.glfw.org/faq.html#how-do-i-create-an-opengl-30-context
    // http://www.glfw.org/faq.html#what-versions-of-opengl-are-supported-by-glfw
    bool coreProfile = true;
    bool forwardCompat = true;
    bool versionSet = true;
    int  major = 3;
    int  minor = 2;
#else
    bool coreProfile = false;
    bool forwardCompat = false;
    bool versionSet = false;
    int  major = 4;
    int  minor = 2;
#endif

    for (int i = 1; i < argc; ++i) {

        if (!strcmp(argv[i], "-glCoreProfile")) {
            _argParseBool(&coreProfile, argv[i], i, argc, argv);
        }
        if (!strcmp(argv[i], "-glForwardCompat")) {
            _argParseBool(&forwardCompat, argv[i], i, argc, argv);
        }
        if (!strcmp(argv[i], "-glVersion")) {
            if (i < argc-1) {
                char *versionStr = argv[i+1];
                size_t len = strlen(versionStr);
                if (len == 3 && versionStr[1] == '.' &&
                    versionStr[0] >= '0' && versionStr[0] <= '9' &&
                    versionStr[2] >= '0' && versionStr[2] <= '9') {
                        
                    major = versionStr[0] - '0';
                    minor = versionStr[2] - '0';
                    versionSet = true;

                } else {
                    fprintf(stderr,
                        "Invalid version number: %s, please specify a number "
                        "in the format M.n, e.g., -glVersion 4.2.\n",
                        versionStr);
                    exit(1);
                }
            } else {
                fprintf(stderr,
                    "Please specify a version number for glVersion "
                    "in the form M.n, e.g., -glVersion 4.2.\n");
                exit(1);
            }
        }
    }

    if (coreProfile) {
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    }

    if (forwardCompat) {
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    }

    if (versionSet) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    }
}

void
PrintGLVersion() {
    std::cout << glGetString(GL_VENDOR) << "\n";
    std::cout << glGetString(GL_RENDERER) << "\n";
    std::cout << glGetString(GL_VERSION) << "\n";

    int i = -1;
    std::cout << "Init OpenGL ";
    glGetIntegerv(GL_MAJOR_VERSION, &i);
    std::cout << i << ".";
    glGetIntegerv(GL_MINOR_VERSION, &i);
    std::cout << i << "\n";

    CheckGLErrors("PrintGLVersion");
}

void
CheckGLErrors(std::string const & where) {
    GLuint err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "GL error: "
                  << (where.empty() ? "" : where + " ")
                  << err << "\n";
    }
}

GLuint
CompileShader(GLenum shaderType, const char *source) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLchar emsg[40960];
        glGetShaderInfoLog(shader, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error compiling GLSL shader: %s\n", emsg);
        return 0;
    }

    return shader;
}

void
WriteScreenshot(int width, int height) {

    std::vector<unsigned char> data(width*height*4 /*RGBA*/);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, &data[0]);

    static int counter=0;
    char fname[64];
    snprintf(fname, 64, "screenshot.%d.png", counter++);

    // flip vertical
    stbi_write_png(fname, width, height, 4, &data[width*4*(height-1)], -width*4);

    fprintf(stdout, "Saved %s\n", fname);
}

void GetMajorMinorVersion(int *major, int *minor){
    const GLubyte *ver = glGetString(GL_SHADING_LANGUAGE_VERSION);
    if (!ver){
        *major = -1;
        *minor = -1;
    }
    else{
        std::stringstream ss;
        ss << std::string(ver, ver + 1) << " " << std::string(ver + 2, ver + 3);
        ss >> *major;
        ss >> *minor;
    }
}

std::string
GetShaderVersion(){
    std::string shader_version;
    int major, minor;
    GetMajorMinorVersion(&major, &minor);
    int version_number = major * 10 + minor;
    switch (version_number){
    case 20:
        shader_version = "110";
        break;
    case 21:
        shader_version = "120";
        break;
    case 30:
        shader_version = "130";
        break;
    case 31:
        shader_version = "140";
        break;
    case 32:
        shader_version = "150";
        break;
    default:
        std::stringstream ss;
        ss << version_number;
        shader_version = ss.str() + "0";
        break;
    }
    return shader_version;
}

// Generates the version definition needed by the glsl shaders based on the
// opengl string
std::string GetShaderVersionInclude(){
    return "#version " + GetShaderVersion() + "\n";
}

bool SupportsAdaptiveTessellation() {
#if defined(GL_VERSION_4_0)
    if (OSD_OPENGL_HAS(VERSION_4_0)) {
        return true;
    }
#endif
#if defined(GL_ARB_tessellation_shader)
    if (OSD_OPENGL_HAS(ARB_tessellation_shader)) {
        return true;
    }
#endif
    return false;
}

bool GL_ARBSeparateShaderObjectsOrGL_VERSION_4_1() {
#if defined(GL_VERSION_4_1)
    if (OSD_OPENGL_HAS(VERSION_4_1)) {
        return true;
    }
#endif
#if defined(GL_ARB_separate_shader_objects)
    if (OSD_OPENGL_HAS(ARB_separate_shader_objects)) {
        return true;
    }
#endif
    return false;
}

bool GL_ARBComputeShaderOrGL_VERSION_4_3() {
#if defined(GL_VERSION_4_3)
    if (OSD_OPENGL_HAS(VERSION_4_3)) {
        return true;
    }
#endif
#if defined(GL_ARB_compute_shader)
    if (OSD_OPENGL_HAS(ARB_compute_shader)) {
        return true;
    }
#endif
    return false;
}

#undef IS_SUPPORTED

}   // namespace GLUtils
