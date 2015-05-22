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

#ifndef OPENSUBDIV_EXAMPLES_GL_SHADER_CACHE_H
#define OPENSUBDIV_EXAMPLES_GL_SHADER_CACHE_H

#include <osd/opengl.h>
#include <map>
#include <string>
#include "./shaderCache.h"

class GLDrawConfig {
public:
    explicit GLDrawConfig(const std::string &version);
    ~GLDrawConfig();

    bool CompileAndAttachShader(GLenum shaderType, const std::string &source);
    bool Link();

    GLuint GetProgram() const {
        return _program;
    }

private:
    GLuint _program;
    std::string _version;
    int _numShaders;
};

// workaround for template alias
#if 0
template <typename DESC_TYPE>
using GLShaderCache = ShaderCacheT<DESC_TYPE, GLDrawConfig>;
#else
template <typename DESC_TYPE>
class GLShaderCache : public ShaderCacheT<DESC_TYPE, GLDrawConfig> {
};
#endif


#endif  // OPENSUBDIV_EXAMPLES_GL_SHADER_CACHE_H
