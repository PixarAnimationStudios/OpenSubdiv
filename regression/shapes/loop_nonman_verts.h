//
//   Copyright 2019 DreamWorks Animation LLC.
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

static const std::string loop_nonman_verts =
"#\n"
"#   Four shapes ordered left->right and top->bottom in the XZ plane\n"
"#\n"
"#   Shape 1:  top-left\n"
"#\n"
"v -1.25  0.0  0.75\n"
"v -0.75 -0.25 0.75\n"
"v -0.25  0.0  0.75\n"
"v -1.0   0.0  1.25\n"
"v -0.5   0.0  1.25\n"
"v -1.0   0.0  0.25\n"
"v -0.5   0.0  0.25\n"
"\n"
"f 1 2 4\n"
"f 2 5 4\n"
"f 2 3 5\n"
"f 2 6 7\n"
"\n"
"#\n"
"#   Shape 2:  top-right\n"
"#\n"
"v 0.25  0.0  0.875\n"
"v 0.75 -0.25 0.75\n"
"v 1.25  0.0  0.875\n"
"v 0.5   0.0  1.25\n"
"v 1.0   0.0  1.25\n"
"v 0.25  0.0  0.625\n"
"v 0.5   0.0  0.25\n"
"v 1.0   0.0  0.25\n"
"v 1.25  0.0  0.625\n"
"\n"
"f  8  9 11\n"
"f  9 12 11\n"
"f  9 10 12\n"
"f  9 13 14\n"
"f 15  9 14\n"
"f 15 16  9\n"
"\n"
"#\n"
"#   Shape 3:  bottom-left\n"
"#\n"
"v -0.75  0.0      -0.75\n"
"v -0.25  0.0      -1.0\n"
"v -0.5   0.433013 -1.0\n"
"v -1.0   0.433013 -1.0\n"
"v -1.25  0.0      -1.0\n"
"v -1.0  -0.433013 -1.0\n"
"v -0.5  -0.433013 -1.0\n"
"v -1.0   0.0      -0.5\n"
"v -0.5   0.0      -0.5\n"
"\n"
"f 17 18 19\n"
"f 17 19 20\n"
"f 17 20 21\n"
"f 17 21 22\n"
"f 17 22 23\n"
"f 17 23 18\n"
"f 17 25 24\n"
"\n"
"#\n"
"#   Shape 4:  bottom-right\n"
"#\n"
"v 0.75  0.0      -0.75\n"
"v 1.25  0.0      -1.0\n"
"v 1.0   0.433013 -1.0\n"
"v 0.5   0.433013 -1.0\n"
"v 0.25  0.0      -1.0\n"
"v 0.5  -0.433013 -1.0\n"
"v 1.0  -0.433013 -1.0\n"
"v 1.25  0.0      -0.5\n"
"v 1.0   0.433013 -0.5\n"
"v 0.5   0.433013 -0.5\n"
"v 0.25  0.0      -0.5\n"
"v 0.5  -0.433013 -0.5\n"
"v 1.0  -0.433013 -0.5\n"
"\n"
"f 26 27 28\n"
"f 26 28 29\n"
"f 26 29 30\n"
"f 26 30 31\n"
"f 26 31 32\n"
"f 26 32 27\n"
"f 26 34 33\n"
"f 26 35 34\n"
"f 26 36 35\n"
"f 26 37 36\n"
"f 26 38 37\n"
"f 26 33 38\n"
"\n"
"#\n"
"#   Additional 'shape' 5:  isolated non-manifold vertex in center\n"
"#\n"
"v 0 0 0\n"
"\n"
"t interpolateboundary 1/0/0 1\n"
"\n"
;
