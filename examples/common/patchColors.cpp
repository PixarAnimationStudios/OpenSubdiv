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

#include "patchColors.h"

float const * getAdaptivePatchColor(OpenSubdiv::OsdDrawContext::PatchDescriptor const & desc) {

    static float _colors[4][5][4] = {{{1.0f,  1.0f,  1.0f,  1.0f},   // regular
                                      {0.8f,  0.0f,  0.0f,  1.0f},   // boundary
                                      {0.0f,  1.0f,  0.0f,  1.0f},   // corner
                                      {1.0f,  1.0f,  0.0f,  1.0f},   // gregory
                                      {1.0f,  0.5f,  0.0f,  1.0f}},  // gregory boundary

                                     {{0.0f,  1.0f,  1.0f,  1.0f},   // regular pattern 0
                                      {0.0f,  0.5f,  1.0f,  1.0f},   // regular pattern 1
                                      {0.0f,  0.5f,  0.5f,  1.0f},   // regular pattern 2
                                      {0.5f,  0.0f,  1.0f,  1.0f},   // regular pattern 3
                                      {1.0f,  0.5f,  1.0f,  1.0f}},  // regular pattern 4
 
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

    typedef OpenSubdiv::FarPatchTables FPT;

    if (desc.GetPattern()==FPT::NON_TRANSITION) {
        return _colors[0][(int)(desc.GetType()-FPT::REGULAR)];
    } else {
        return _colors[(int)(desc.GetType()-FPT::REGULAR)+1][(int)desc.GetPattern()-1];
    }
}

