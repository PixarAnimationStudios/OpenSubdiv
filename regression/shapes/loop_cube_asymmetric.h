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

static const std::string loop_cube_asymmetric =
"v  0.000000  1.414214  1.000000\n"
"v -1.414214  0.000000  1.000000\n"
"v  1.414214 -0.000000  1.000000\n"
"v -0.000000 -1.414214  1.000000\n"
"v  1.414214 -0.000000 -1.000000\n"
"v -0.000000 -1.414214 -1.000000\n"
"v  0.000000  1.414214 -1.000000\n"
"v -1.414214  0.000000 -1.000000\n"
"\n"
"vt 0.375 0.00\n"
"vt 0.625 0.00\n"
"vt 0.375 0.25\n"
"vt 0.625 0.25\n"
"vt 0.375 0.50\n"
"vt 0.625 0.50\n"
"vt 0.375 0.75\n"
"vt 0.625 0.75\n"
"vt 0.375 1.00\n"
"vt 0.625 1.00\n"
"vt 0.875 0.00\n"
"vt 0.875 0.25\n"
"vt 0.125 0.00\n"
"vt 0.125 0.25\n"
"\n"
"f 1/1  2/2  3/3\n"
"f 3/3  2/2  4/4\n"
"f 3/3  4/4  5/5\n"
"f 5/5  4/4  6/6\n"
"f 5/5  6/6  7/7\n"
"f 7/7  6/6  8/8\n"
"f 7/7  8/8  1/9\n"
"f 1/9  8/8  2/10\n"
"f 2/2  8/11 4/4\n"
"f 4/4  8/11 6/12\n"
"f 7/13 1/1  5/14\n"
"f 5/14 1/1  3/3\n"
"\n"
"t interpolateboundary 1/0/0 2\n"
"\n"
;
