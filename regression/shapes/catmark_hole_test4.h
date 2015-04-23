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

static const std::string catmark_hole_test4 =
"# This file uses centimeters as units for non-parametric coordinates.\n"
"\n"
"mtllib xord3.mtl\n"
"g default\n"
"v -1 0 1\n"
"v -1 0 0\n"
"v -0.5 0 0\n"
"v -0.5 0 1\n"
"v 0 0 1\n"
"v 0.5 0 0\n"
"v 1 0 0\n"
"v 1 0 1\n"
"v -0.5 0 -1\n"
"v 0 0 -1\n"
"v 1 0 -1\n"
"vt 0.000000 0.000000\n"
"vn 0.000000 -1.000000 0.000000\n"
"f 1/1/1 2/1/1 3/1/1 4/1/1\n"
"f 4/1/1 3/1/1 6/1/1 5/1/1\n"
"f 5/1/1 6/1/1 7/1/1 8/1/1\n"
"f 3/1/1 9/1/1 10/1/1 6/1/1\n"
"f 6/1/1 10/1/1 11/1/1 7/1/1\n"
"t hole 1/0/0 1\n"
"\n"
;
