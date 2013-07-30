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
static const std::string loop_triangle_edgeonly =
"# This file uses centimeters as units for non-parametric coordinates.\n"
"\n"
"mtllib triangle.mtl\n"
"g default\n"
"v 0.000000 1.500000 0.000000\n"
"v -2.000000 -1.500000 0.000000\n"
"v 2.000000 -1.500000 0.000000\n"
"vt 0.000000 0.000000\n"
"vt 1.000000 0.000000\n"
"vt 0.384615 0.923077\n"
"vn 0.000000 0.000000 1.000000\n"
"vn 0.000000 0.000000 1.000000\n"
"vn 0.000000 0.000000 1.000000\n"
"s off\n"
"g polySurface1\n"
"usemtl initialShadingGroup\n"
"f 1/1/1 2/2/2 3/3/3\n"
"t interpolateboundary 1/0/0 2\n"
"\n"
;
