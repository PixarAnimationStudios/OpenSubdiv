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

#include "effect.h"

#include "../common/simple_math.h"
#include "../common/patchColors.h"

#include <osd/opengl.h>

#include <stdio.h>
#include <string.h>

GLuint g_transformUB = 0,
       g_tessellationUB = 0,
    g_lightingUB = 0;

    GLuint g_transformBinding = 0;
    GLuint g_tessellationBinding = 1;
    GLuint g_lightingBinding = 3;


void
MyEffect::SetMatrix(const float *modelview, const float *projection) {
    float mvp[16];
    multMatrix(mvp, modelview, projection);
    
    struct Transform {
        float ModelViewMatrix[16];
        float ProjectionMatrix[16];
        float ModelViewProjectionMatrix[16];
    } transformData;
    memcpy(transformData.ModelViewMatrix, modelview, sizeof(float)*16);
    memcpy(transformData.ProjectionMatrix, projection, sizeof(float)*16);
    memcpy(transformData.ModelViewProjectionMatrix, mvp, sizeof(float)*16);

    // set transform
    if (! g_transformUB) {
        glGenBuffers(1, &g_transformUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(transformData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(transformData), &transformData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_transformBinding, g_transformUB);
}

void
MyEffect::SetTessLevel(float tessLevel) {
    
    struct Tessellation {
        float TessLevel;
    } tessellationData;
    tessellationData.TessLevel = tessLevel;

    if (! g_tessellationUB) {
        glGenBuffers(1, &g_tessellationUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_tessellationUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(tessellationData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_tessellationUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(tessellationData), &tessellationData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_tessellationBinding, g_tessellationUB);
}

void
MyEffect::SetLighting() {

    struct Lighting {
        struct Light {
            float position[4];
            float ambient[4];
            float diffuse[4];
            float specular[4];
        } lightSource[2];
    } lightingData = {
       {{  { 0.5,  0.2f, 1.0f, 0.0f },
           { 0.1f, 0.1f, 0.1f, 1.0f },
           { 0.7f, 0.7f, 0.7f, 1.0f },
           { 0.8f, 0.8f, 0.8f, 1.0f } },
 
         { { -0.8f, 0.4f, -1.0f, 0.0f },
           {  0.0f, 0.0f,  0.0f, 1.0f },
           {  0.5f, 0.5f,  0.5f, 1.0f },
           {  0.8f, 0.8f,  0.8f, 1.0f } }}
    };

    if (! g_lightingUB) {
        glGenBuffers(1, &g_lightingUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_lightingUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(lightingData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_lightingUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(lightingData), &lightingData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_lightingBinding, g_lightingUB);
}

void
MyEffect::BindDrawConfig(MyDrawConfig *config, OpenSubdiv::OsdDrawContext::PatchDescriptor desc) {

    // bind uniforms

// currently, these are used only in conjunction with tessellation shaders
#if defined(GL_EXT_direct_state_access) || defined(GL_VERSION_4_1)

    GLint program = config->program;
    GLint diffuseColor = config->diffuseColorUniform;

    if (displayPatchColor) {
        float const * color = getAdaptivePatchColor( desc );
        glProgramUniform4f(program, diffuseColor, color[0], color[1], color[2], color[3]);
    }
#endif
}

