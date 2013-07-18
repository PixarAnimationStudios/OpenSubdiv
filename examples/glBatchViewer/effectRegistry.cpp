//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#include "effectRegistry.h"

#include <osd/opengl.h>

static const char *shaderSource =
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #include "shader.inc"
#else
    #include "shader_gl3.inc"
#endif
;


MyEffectRegistry::SourceConfigType *
MyEffectRegistry::_CreateDrawSourceConfig(DescType const & desc) {

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first);
    assert(sconfig);

    EffectDesc effectDesc = desc.second;

    if (effectDesc.GetScreenSpaceTess()) {
        sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
        sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");
    }

    const char *glslVersion = "#version 400\n";
    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS) {
        sconfig->vertexShader.source = shaderSource;
        sconfig->vertexShader.version = glslVersion;
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
    } else {
        sconfig->geometryShader.AddDefine("SMOOTH_NORMALS");
    }

    sconfig->geometryShader.source = shaderSource;
    sconfig->geometryShader.version = glslVersion;
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    sconfig->fragmentShader.source = shaderSource;
    sconfig->fragmentShader.version = glslVersion;
    sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");

    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS) {
        // uniform catmark, bilinear
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else if (desc.first.GetType() == OpenSubdiv::FarPatchTables::TRIANGLES) {
        // uniform loop
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else {
        // adaptive
        sconfig->vertexShader.source = shaderSource + sconfig->vertexShader.source;
        sconfig->tessControlShader.source = shaderSource + sconfig->tessControlShader.source;
        sconfig->tessEvalShader.source = shaderSource + sconfig->tessEvalShader.source;

        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
    }

    int displayStyle = effectDesc.GetDisplayStyle();
    switch (displayStyle) {
    case kWire:
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_WIRE");
        break;
    case kWireShaded:
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_LINE");
        break;
    case kShaded:
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kVaryingColor:
        sconfig->commonShader.AddDefine("VARYING_COLOR");
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kFaceVaryingColor:
        sconfig->commonShader.AddDefine("FACEVARYING_COLOR");
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    }

    return sconfig;

}

MyEffectRegistry::ConfigType *
MyEffectRegistry::_CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig) {

    // XXX: make sure is this downcast correct
    ConfigType * config = (ConfigType *)BaseRegistry::_CreateDrawConfig(desc.first, sconfig);
    assert(config);

    GLuint uboIndex;
    GLuint program = config->program;

    GLuint g_transformBinding = 0,
        g_tessellationBinding = 0,
        g_lightingBinding = 0;

    // XXXdyu can use layout(binding=) with GLSL 4.20 and beyond
    g_transformBinding = 0;
    uboIndex = glGetUniformBlockIndex(program, "Transform");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(program, uboIndex, g_transformBinding);

    g_tessellationBinding = 1;
    uboIndex = glGetUniformBlockIndex(program, "Tessellation");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(program, uboIndex, g_tessellationBinding);

    g_lightingBinding = 3;
    uboIndex = glGetUniformBlockIndex(program, "Lighting");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(program, uboIndex, g_lightingBinding);

// currently, these are used only in conjunction with tessellation shaders
#if defined(GL_EXT_direct_state_access) || defined(GL_VERSION_4_1)
    GLint loc;
    if ((loc = glGetUniformLocation(program, "OsdVertexBuffer")) != -1) {
        glProgramUniform1i(program, loc, 0); // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(program, "OsdValenceBuffer")) != -1) {
        glProgramUniform1i(program, loc, 1); // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(program, "OsdQuadOffsetBuffer")) != -1) {
        glProgramUniform1i(program, loc, 2); // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(program, "OsdPatchParamBuffer")) != -1) {
        glProgramUniform1i(program, loc, 3); // GL_TEXTURE3
    }
    if ((loc = glGetUniformLocation(program, "OsdFVarDataBuffer")) != -1) {
        glProgramUniform1i(program, loc, 4); // GL_TEXTURE4
    }
#endif

    config->diffuseColorUniform = glGetUniformLocation(program, "diffuseColor");

    return config;
}
