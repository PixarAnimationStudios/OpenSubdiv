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

#ifndef GL_FONT_UTILS_H
#define GL_FONT_UTILS_H

#include "../common/glUtils.h"

#include <vector>

class GLFont {
public:

    GLFont(GLuint fontTexture);

    ~GLFont();

    void Draw(GLuint transforUB);

    void Clear();

    void Print3D(float const pos[3], const char * str, int color=0);
    
    void SetFontScale(float scale);

    struct Char {
        float pos[3];
        float ofs[2];
        float alpha;
        float color;
    };
    
    std::vector<Char> & GetChars() {
        _dirty=true;
        return _chars;
    }
    
    
private:

    void bindProgram();

    std::vector<Char> _chars;
    bool _dirty;

    GLuint _program,
           _transformBinding,
           _attrPosition,
           _attrData,
           _fontTexture,
           _scale,
           _VAO,
           _EAO,
           _VBO;
};

#endif // GL_FONT_UTILS_H
