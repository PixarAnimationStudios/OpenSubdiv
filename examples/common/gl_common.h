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

#ifndef GL_COMMON_H
#define GL_COMMON_H

#include <osd/opengl.h>

#include <cstdio>
#include <string>
#include <iostream>

void checkGLErrors(std::string const & where = "");

GLuint compileShader(GLenum shaderType, const char *source);

void get_major_minor_version(int *major, int *minor);


const std::string &get_shader_version();

const std::string &get_shader_version_include();

bool uses_tesselation_shaders();

bool GL_ARB_separate_shader_objects_or_GL_VERSION_4_1();
#endif // GL_FRAMEBUFFER_H
