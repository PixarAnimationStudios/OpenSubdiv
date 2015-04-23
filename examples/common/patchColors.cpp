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

static float _colors[5][7][4] = {{{1.0f,  1.0f,  1.0f,  1.0f},   // regular
                                  {1.0f,  0.5f,  0.5f,  1.0f},   // single crease
                                  {0.8f,  0.0f,  0.0f,  1.0f},   // boundary
                                  {0.0f,  1.0f,  0.0f,  1.0f},   // corner
                                  {1.0f,  1.0f,  0.0f,  1.0f},   // gregory
                                  {1.0f,  0.5f,  0.0f,  1.0f},   // gregory boundary
                                  {1.0f,  1.0f,  0.0f,  1.0f}},  // gregory basis

                                 {{0.0f,  1.0f,  1.0f,  1.0f},   // regular pattern 0
                                  {0.0f,  0.5f,  1.0f,  1.0f},   // regular pattern 1
                                  {0.0f,  0.5f,  0.5f,  1.0f},   // regular pattern 2
                                  {0.5f,  0.0f,  1.0f,  1.0f},   // regular pattern 3
                                  {1.0f,  0.5f,  1.0f,  1.0f}},  // regular pattern 4

                                 {{1.0f,  0.7f,  0.6f,  1.0f},   // single crease pattern 0
                                  {1.0f,  0.7f,  0.6f,  1.0f},   // single crease pattern 1
                                  {1.0f,  0.7f,  0.6f,  1.0f},   // single crease pattern 2
                                  {1.0f,  0.7f,  0.6f,  1.0f},   // single crease pattern 3
                                  {1.0f,  0.7f,  0.6f,  1.0f}},  // single crease pattern 4

                                 {{0.0f,  0.0f,  0.75f, 1.0f},   // boundary pattern 0
                                  {0.0f,  0.2f,  0.75f, 1.0f},   // boundary pattern 1
                                  {0.0f,  0.4f,  0.75f, 1.0f},   // boundary pattern 2
                                  {0.0f,  0.6f,  0.75f, 1.0f},   // boundary pattern 3
                                  {0.0f,  0.8f,  0.75f, 1.0f}},  // boundary pattern 4

                                 {{0.25f, 0.25f, 0.25f, 1.0f},   // corner pattern 0
                                  {0.25f, 0.25f, 0.25f, 1.0f},   // corner pattern 1
                                  {0.25f, 0.25f, 0.25f, 1.0f},   // corner pattern 2
                                  {0.25f, 0.25f, 0.25f, 1.0f},   // corner pattern 3
                                  {0.25f, 0.25f, 0.25f, 1.0f}}}; // corner pattern 4

typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

float const *
getAdaptivePatchColor(Descriptor const & desc) {

    if (desc.GetPattern()==Descriptor::NON_TRANSITION) {
        return _colors[0][(int)(desc.GetType()-Descriptor::REGULAR)];
    } else {
        return _colors[(int)(desc.GetType()-Descriptor::REGULAR)+1][(int)desc.GetPattern()-1];
    }
}

float const *
getAdaptivePatchColor(OpenSubdiv::Osd::DrawContext::PatchDescriptor const & desc) {

    if (desc.GetPattern()==Descriptor::NON_TRANSITION) {
        return _colors[0][(int)(desc.GetType()-Descriptor::REGULAR)];
    } else {
        return _colors[(int)(desc.GetType()-Descriptor::REGULAR)+1][(int)desc.GetPattern()-1];
    }
}

