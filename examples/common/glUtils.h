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

#ifndef OPENSUBDIV_EXAMPLES_GL_UTILS_H
#define OPENSUBDIV_EXAMPLES_GL_UTILS_H

#include "glLoader.h"

#include <cstdio>
#include <string>
#include <iostream>

namespace GLUtils {

void InitializeGL();

void SetMinimumGLVersion(int argc=0, char **argv=NULL);

void PrintGLVersion();

void CheckGLErrors(std::string const & where = "");

GLuint CompileShader(GLenum shaderType, const char *source);

void WriteScreenshot(int width, int height);

bool SupportsAdaptiveTessellation();

// Helper function that parses the opengl version string, retrieving the
// major and minor version from it.
void GetMajorMinorVersion(int *major, int *minor);

// Gets the shader version based on the current opengl version and returns 
// it in a string form.
std::string GetShaderVersion();

std::string GetShaderVersionInclude();

bool GL_ARBSeparateShaderObjectsOrGL_VERSION_4_1();

bool GL_ARBComputeShaderOrGL_VERSION_4_3();

};

#endif  // OPENSUBDIV_EXAMPLES_GL_UTILS_H


