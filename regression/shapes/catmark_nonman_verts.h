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

static const std::string catmark_nonman_verts =
"#\n"
"#   Four shapes ordered left->right and top->bottom in the XZ plane\n"
"#\n"
"#   Shape 1:  top-left\n"
"#\n"
"v -1.25   0    0.75\n"
"v -0.75  -0.3  0.75\n"
"v -0.25   0    0.75\n"
"v -1.25   0    1.25\n"
"v -0.75   0    1.25\n"
"v -0.25   0    1.25\n"
"v -1.0    0    0.5 \n"
"v -0.75   0    0.25 \n"
"v -0.5    0    0.5 \n"
"\n"
"f  1  2  5  4\n"
"f  2  3  6  5\n"
"f  2  7  8  9\n"
"\n"
"#\n"
"#   Shape 2:  top-right\n"
"#\n"
"v  0.25   0    0.85\n"
"v  0.75  -0.3  0.75\n"
"v  1.25   0    0.85\n"
"v  0.25   0    1.25\n"
"v  0.75   0    1.25\n"
"v  1.25   0    1.25\n"
"v  0.25   0    0.65\n"
"v  1.25   0    0.65\n"
"v  0.25   0    0.25\n"
"v  0.75   0    0.25\n"
"v  1.25   0    0.25\n"
"\n"
"f 10 11 14 13\n"
"f 11 12 15 14\n"
"f 11 16 18 19\n"
"f 17 11 19 20\n"
"\n"
"#\n"
"#   Shape 3:  bottom-left\n"
"#\n"
"v -1.25  -0.5  -1.25\n"
"v -0.75  -0.5  -1.25\n"
"v -0.25  -0.5  -1.25\n"
"v -1.25   0    -1.25\n"
"v -0.75   0    -0.75\n"
"v -0.25   0    -1.25\n"
"v -1.25   0.5  -1.25\n"
"v -0.75   0.5  -1.25\n"
"v -0.25   0.5  -1.25\n"
"v -1.1    0    -0.25\n"
"v -0.4    0    -0.25\n"
"\n"
"f 21 22 25 24\n"
"f 22 23 26 25\n"
"f 24 25 28 27\n"
"f 25 26 29 28\n"
"f 30 25 31\n"
"\n"
"#\n"
"#   Shape 4:  bottom-right\n"
"#\n"
"v  0.25 -0.5  -1.25\n"
"v  0.75 -0.5  -1.25\n"
"v  1.25 -0.5  -1.25\n"
"v  0.25  0    -1.25\n"
"v  0.75  0    -0.75\n"
"v  1.25  0    -1.25\n"
"v  0.25  0.5  -1.25\n"
"v  0.75  0.5  -1.25\n"
"v  1.25  0.5  -1.25\n"
"v  0.25 -0.5  -0.25\n"
"v  0.75 -0.5  -0.25\n"
"v  1.25 -0.5  -0.25\n"
"v  0.25  0    -0.25\n"
"v  1.25  0    -0.25\n"
"v  0.25  0.5  -0.25\n"
"v  0.75  0.5  -0.25\n"
"v  1.25  0.5  -0.25\n"
"\n"
"f 32 33 36 35\n"
"f 33 34 37 36\n"
"f 35 36 39 38\n"
"f 36 37 40 39\n"
"f 41 44 36 42\n"
"f 42 36 45 43\n"
"f 44 46 47 36\n"
"f 36 47 48 45\n"
"\n"
"#\n"
"#   Additional 'shape' 5:  isolated non-manifold vertex in center\n"
"#\n"
"v 0 0 0\n"
"\n"
"t interpolateboundary 1/0/0 1\n"
"\n"
;
