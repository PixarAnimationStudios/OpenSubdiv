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

static const std::string loop_triangle_edgeonly =
"v  0.0  0.0   0.966\n"
"v -0.5  0.0   0.1\n"
"v  0.5  0.0   0.1\n"
"\n"
"v -0.6  0.0  -0.234\n"
"v -1.1  0.0  -1.1\n"
"v -0.1  0.0  -1.1\n"
"\n"
"v  0.6  0.0  -0.234\n"
"v  0.1  0.0  -1.1\n"
"v  1.1  0.0  -1.1\n"
"\n"
"vt  0.5   0.933\n"
"vt  0.25  0.5\n"
"vt  0.75  0.5\n"
"vt  0.25  0.433\n"
"vt  0.0   0.0\n"
"vt  0.5   0.0\n"
"vt  0.75  0.433\n"
"vt  0.5   0.0\n"
"vt  1.0   0.0\n"
"\n"
"f 1/1 2/2 3/3\n"
"f 4/4 5/5 6/6\n"
"f 7/7 8/8 9/9\n"
"\n"
"t corner 1/1/0 4 10.0\n"
"t corner 1/1/0 5 10.0\n"
"\n"
"t corner 1/1/0 6 10.0\n"
"t corner 1/1/0 7 10.0\n"
"t corner 1/1/0 8 10.0\n"
"\n"
"t interpolateboundary 1/0/0 2\n"
"\n"
;
