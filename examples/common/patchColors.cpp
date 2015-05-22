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

#include "patchColors.h"

static float _colors[7][4] = {{1.0f,  1.0f,  1.0f,  1.0f},   // regular
                              {1.0f,  0.5f,  0.5f,  1.0f},   // single crease
                              {0.8f,  0.0f,  0.0f,  1.0f},   // boundary
                              {0.0f,  1.0f,  0.0f,  1.0f},   // corner
                              {1.0f,  1.0f,  0.0f,  1.0f},   // gregory
                              {1.0f,  0.5f,  0.0f,  1.0f},   // gregory boundary
                              {1.0f,  0.7f,  0.3f,  1.0f}};  // gregory basis

typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

float const *
getAdaptivePatchColor(Descriptor const & desc) {

    return _colors[(int)(desc.GetType()-Descriptor::REGULAR)];
}

