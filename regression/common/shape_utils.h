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

#ifndef SHAPE_UTILS_H
#define SHAPE_UTILS_H

#include <string>
#include <vector>

//------------------------------------------------------------------------------
enum Scheme {
  kBilinear=0,
  kCatmark,
  kLoop
};

//------------------------------------------------------------------------------

struct Shape {

    struct tag {

        static tag * parseTag( char const * stream );

        std::string genTag() const;

        std::string              name;
        std::vector<int>         intargs;
        std::vector<float>       floatargs;
        std::vector<std::string> stringargs;
    };

    static Shape * parseObj(char const * Shapestr, Scheme schme, int axis=1);

    std::string genShape(char const * name) const;

    std::string genObj() const;

    std::string genRIB() const;

    ~Shape();

    int GetNumVertices() const { return (int)verts.size()/3; }

    int GetNumFaces() const { return (int)nvertsPerFace.size(); }
    
    bool HasUV() const { return not (uvs.empty() or faceuvs.empty()); }
    
    int GetFVarWidth() const { return HasUV() ? 2 : 0; }

    std::vector<float>  verts;
    std::vector<float>  uvs;
    std::vector<float>  normals;
    std::vector<int>    nvertsPerFace;
    std::vector<int>    faceverts;
    std::vector<int>    faceuvs;
    std::vector<int>    facenormals;
    std::vector<tag *>  tags;
    Scheme              scheme;
};

//------------------------------------------------------------------------------

#endif /* SHAPE_UTILS_H */
