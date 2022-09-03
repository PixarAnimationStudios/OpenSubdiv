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
#include <map>

//------------------------------------------------------------------------------

enum Scheme {
  kBilinear=0,
  kCatmark,
  kLoop
};

struct ShapeDesc 
{
    ShapeDesc(char const * iname, std::string const & idata, Scheme ischeme,
        bool iIsLeftHanded = false) :
        name(iname), data(idata), scheme(ischeme), isLeftHanded(iIsLeftHanded) 
	{ }

    std::string name;
    std::string data;
    Scheme      scheme;
    bool        isLeftHanded;
};

//------------------------------------------------------------------------------

struct Shape {
    // full(er) spec here: http://paulbourke.net/dataformats/mtl/
    struct material {

        material();

        std::string name;

        float ka[3],         // ambient
              kd[3],         // diffuse
              ks[3],         // specular
              ns,            // specular exponent
              ni,            // optical density (1.0=no refraction, glass=1.5)
              sharpness,     // reflection sharpness
              tf[3],         // transmission filter
              d;             // dissolve factor (1.0 = opaque)

        int illum;
    };

    struct tag {

        static tag * parseTag(char const * stream);

        std::string genTag() const;

        std::string              name;
        std::vector<int>         intargs;
        std::vector<float>       floatargs;
        std::vector<std::string> stringargs;
    };

    static Shape * parseObj(ShapeDesc const & shapeDesc, bool parsemtl=false);
    static Shape * parseObj(char const * shapeString, Scheme shapeScheme,
                            bool isLeftHanded=false, bool parsemtl=false);

    void parseMtllib(char const * stream);

    std::string genShape(char const * name) const;

    std::string genObj() const;

    std::string genRIB() const;

    Shape() : scheme(kCatmark), isLeftHanded(false) { }

    ~Shape();

    int GetNumVertices() const { return (int)verts.size()/3; }

    int GetNumFaces() const { return (int)nvertsPerFace.size(); }

    bool HasUV() const { return ! (uvs.empty() || faceuvs.empty()); }

    int GetFVarWidth() const { return HasUV() ? 2 : 0; }

    std::vector<float>      verts;
    std::vector<float>      uvs;
    std::vector<float>      normals;
    std::vector<int>        nvertsPerFace;
    std::vector<int>        faceverts;
    std::vector<int>        faceuvs;
    std::vector<int>        facenormals;
    std::vector<tag *>      tags;
    Scheme                  scheme;
    bool                    isLeftHanded;

    char FindMaterial(char const * name) {
        for (int i=0; i<(int)mtls.size(); ++i) {
            if (mtls[i]->name==name) {
                return (char) i;
            }
        }
        return -1;
    }

    std::string                 mtllib;
    std::vector<unsigned short> mtlbind;
    std::vector<material *>     mtls;
};

//------------------------------------------------------------------------------

#endif /* SHAPE_UTILS_H */
