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
#ifndef EFFECT_H
#define EFFECT_H

#include <osd/glDrawRegistry.h>

#include <osd/opengl.h>

enum DisplayTyle { kWire = 0,
                   kShaded,
                   kWireShaded,
                   kVaryingColor,
                   kFaceVaryingColor };

union EffectDesc {
public:
    struct {
        unsigned int displayStyle:3;
        unsigned int screenSpaceTess:1;
    };
    int value;

    bool operator < (const EffectDesc &e) const {
        return value < e.value;
    }

    int GetDisplayStyle() const { return displayStyle; }
    bool GetScreenSpaceTess() const { return screenSpaceTess; }
};

struct MyDrawConfig : public OpenSubdiv::OsdGLDrawConfig {

    MyDrawConfig() : 
        diffuseColorUniform(-1) {}
    virtual ~MyDrawConfig() {}

    GLint diffuseColorUniform;
};

class MyEffect { // formerly EffectHandle
public:
    MyEffect() : displayPatchColor(false), screenSpaceTess(false), displayStyle(kWire) {}

    EffectDesc GetEffectDescriptor() const {
        EffectDesc desc;

        desc.value = 0;
        desc.displayStyle = displayStyle;
        desc.screenSpaceTess = screenSpaceTess;

        return desc;
    }

    void BindDrawConfig(MyDrawConfig *config, OpenSubdiv::OsdDrawContext::PatchDescriptor desc);

    void SetMatrix(const float *modelview, const float *projection);
    void SetTessLevel(float tessLevel);
    void SetLighting();

    bool operator == (const MyEffect &other) const {
        return (displayPatchColor == other.displayPatchColor) and
               (screenSpaceTess == other.screenSpaceTess) and
               (displayStyle == other.displayStyle);
    }
    bool operator != (const MyEffect &other) const {
        return !(*this == other);
    }

    bool displayPatchColor;  // runtime switchable
    bool screenSpaceTess;    // need recompile (should be considered in effect descriptor)
    int displayStyle;        // need recompile
};


#endif  /* EFFECT_H */
