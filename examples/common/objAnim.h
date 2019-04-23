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

#ifndef OBJ_ANIM_H
#define OBJ_ANIM_H

#include "../../regression/common/shape_utils.h"

#include <vector>

class ObjAnim {

public:

    // Factory function
    static ObjAnim const * Create(std::vector<char const *> objFiles,
                                  Scheme scheme, bool isLeftHanded=false);

    // Destructor
    ~ObjAnim();

    // Populates 'positions' with the interpolated vertex data for a given
    // time.
    void InterpolatePositions(float time, float * positions, int stride) const;

    // Number of key-frames in the animation
    int GetNumKeyframes() const {
        return (int)_positions.size();
    }

    // Returns the full 'Shape'
    Shape const * GetShape() const {
        return _shape;
    }


private:

    ObjAnim();

    Shape const * _shape;

    std::vector<std::vector<float> > _positions;
};

#endif // OBJ_ANIM_H
