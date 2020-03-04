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

#include "glShaderCache.h"
#include "glUtils.h"

#include <vector>
#include <opensubdiv/far/error.h>

GLDrawConfig::GLDrawConfig(const std::string &version)
    : _version(version), _numShaders(0) {
    _program = glCreateProgram();
}


GLDrawConfig::~GLDrawConfig() {
    if (_program)
        glDeleteProgram(_program);
}

bool
GLDrawConfig::CompileAndAttachShader(GLenum shaderType,
const std::string &source) {


	GLuint shader = glCreateShader(shaderType);

#if 0
	const char *sources[2];
	sources[0] = _version.c_str();
	sources[1] = source.c_str();
#endif

	std::string sources = _version + source;

	const char *src = sources.c_str();

    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
        char * infoLog = new char[infoLogLength];
        glGetShaderInfoLog(shader, infoLogLength, NULL, infoLog);
        OpenSubdiv::Far::Error(OpenSubdiv::Far::FAR_RUNTIME_ERROR,
                               "Error compiling GLSL shader: %s\n",
                               infoLog);
        delete[] infoLog;
        return false;
    }

    glAttachShader(_program, shader);
    ++_numShaders;
    return true;
}

bool
GLDrawConfig::Link() {
    glLinkProgram(_program);

    std::vector<GLuint> shaders(_numShaders);
    GLsizei count = 0;
    glGetAttachedShaders(_program, _numShaders, &count, &shaders[0]);
    for (int i = 0; i < (int)count; ++i) {
        glDeleteShader(shaders[i]);
    }

    GLint status;
    glGetProgramiv(_program, GL_LINK_STATUS, &status );
    if (status == GL_FALSE) {
        GLint infoLogLength;
        glGetProgramiv(_program, GL_INFO_LOG_LENGTH, &infoLogLength);
        char * infoLog = new char[infoLogLength];
        glGetProgramInfoLog(_program, infoLogLength, NULL, infoLog);
        OpenSubdiv::Far::Error(OpenSubdiv::Far::FAR_RUNTIME_ERROR,
                   "Error linking GLSL program: %s\n", infoLog);
        delete[] infoLog;
        return false;
    }

    return true;
}
