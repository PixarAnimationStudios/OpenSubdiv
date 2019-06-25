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

static const std::string loop_tetrahedron =
"#\n"
"#  Tetrahedron oriented within an axially aligned cube:\n"
"#\n"
"v -1 -1  1\n"
"v  1 -1 -1\n"
"v  1  1  1\n"
"v -1  1 -1\n"
"\n"
"vt 0.375 0.217\n"
"vt 0.75  0.0\n"
"vt 0.375 0.65\n"
"vt 0.0   0.0\n"
"vt 0.25  1.0\n"
"vt 0.625 0.35\n"
"vt 1.0   1.0\n"
"\n"
"f 1/1 2/2 3/3\n"
"f 1/1 3/3 4/4\n"
"f 1/1 4/4 2/2\n"
"f 4/5 3/6 2/7\n"
"\n"
;
