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

#include "objAnim.h"
#include "../../regression/common/shape_utils.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include <sstream>
ObjAnim::ObjAnim() : _shape(0) {
}

ObjAnim::~ObjAnim() {

    delete _shape;
}

void
ObjAnim::InterpolatePositions(float time, float * positions, int stride) const {

    assert(positions);

    if ( _positions.empty() || (! _shape)) {
        //printf("Error: InterpolatePositions on unfit ObjAnim instance\n");
        return;
    }

    int nkeys = GetNumKeyframes(),
        nverts = GetShape()->GetNumVertices();

    assert(nkeys>0);

    if (nkeys==1) {
        // nothing to interpolate - just copy the coarse verts positions
        float const * vert = &_positions[0][0];
        for (int i = 0; i <nverts; ++i) {
             memcpy( positions, vert, sizeof(float)*3);
             positions += stride;
             vert += 3;
        }
        return;
    }

    const float fps = 24.0f;

    float p = fmodf(time * fps, (float)nkeys);

    int key = (int)p;

    float b = p - key;

    for (int i = 0; i <nverts; ++i) {

        for (int j=0; j<3; ++j) {

            float p0 = _positions[ key         ][i*3+j];
            float p1 = _positions[(key+1)%nkeys][i*3+j];

            positions[i*stride + j] = p0*(1-b) + p1*b;
        }
    }
}

ObjAnim const *
ObjAnim::Create(std::vector<char const *> objFiles, Scheme scheme, bool isLeftHanded) {

    ObjAnim * anim=0;

    Shape const * shape = 0;

    if (! objFiles.empty()) {

        anim = new ObjAnim;

        anim->_positions.reserve(objFiles.size());

        for (int i = 0; i < (int)objFiles.size(); ++i) {

            if (! objFiles[i]) {
                continue;
            }

            std::ifstream ifs(objFiles[i]);

            if (ifs) {

                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();

                printf("Reading %s\r", objFiles[i]);
                fflush(stdout);
                std::string str = ss.str();

                shape = Shape::parseObj(str.c_str(), scheme, isLeftHanded);

                if (i==0) {

                    anim->_shape = shape;
                    anim->_positions.push_back(shape->verts);
                } else {

                    if (shape->verts.size() != anim->_shape->verts.size()) {
                        printf("Error: vertex count doesn't match (%s)\n", objFiles[i]);
                        goto error;
                    }

                    if (shape->nvertsPerFace.size() != anim->_shape->nvertsPerFace.size()) {
                        printf("Error: face vertex count array doesn't match (%s)\n", objFiles[i]);
                        goto error;
                    }

                    if (shape->faceverts.size() != anim->_shape->faceverts.size()) {
                        printf("Error: face vertices array doesn't match (%s)\n", objFiles[i]);
                        goto error;
                    }

                    anim->_positions.push_back(shape->verts);
                    delete shape;
                }

            } else {
                printf("Error in reading %s\n", objFiles[i]);
                goto error;
            }
        }
        printf("\n");
    }

    return anim;

error:
    delete shape;
    delete anim;
    return 0;
}
