//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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

