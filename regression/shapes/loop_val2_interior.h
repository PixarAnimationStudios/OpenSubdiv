//
//   Copyright 2022 Pixar
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

static const std::string loop_val2_interior =
"#\n"
"#   Shape 1:\n"
"#\n"
"v -1.75  0.0   0.0\n"
"v -0.75  0.0   0.0\n"
"v -1.25  0.0   0.866\n"
"v -1.75 -0.75  0.866\n"
"v -0.75 -0.75  0.866\n"
"v -1.75  0.75  0.866\n"
"v -0.75  0.75  0.866\n"
"\n"
"vt 0.05 0.10\n"
"vt 0.45 0.10\n"
"vt 0.25 0.40\n"
"vt 0.55 0.10\n"
"vt 0.95 0.10\n"
"vt 0.75 0.40\n"
"vt 0.20 0.90\n"
"vt 0.20 0.50\n"
"vt 0.50 0.70\n"
"vt 0.80 0.50\n"
"vt 0.80 0.90\n"
"\n"
"f  1/1   2/2   3/3\n"
"f  2/4   1/5   3/6\n"
"f  3/9   4/7   5/8\n"
"f  3/9   7/10  6/11\n"
"\n"
"#\n"
"#   Shape 2:\n"
"#\n"
"v -0.5   0.0   0.0\n"
"v  0.5   0.0   0.0\n"
"v  0.0   0.0   0.866\n"
"\n"
"vt 1.05 0.30\n"
"vt 1.45 0.30\n"
"vt 1.25 0.60\n"
"vt 1.55 0.30\n"
"vt 1.95 0.30\n"
"vt 1.75 0.60\n"
"\n"
"f  8/12  9/13 10/14\n"
"f  9/15  8/16 10/17\n"
"\n"
"#\n"
"#   Shape 3:\n"
"#\n"
"v  0.75  0.0   0.0\n"
"v  1.75  0.0   0.0\n"
"v  1.25  0.0   0.866\n"
"v  1.25 -0.75  0.0\n"
"v  1.25  0.75  0.0\n"
"\n"
"vt 2.05 0.60\n"
"vt 2.45 0.60\n"
"vt 2.25 0.90\n"
"vt 2.55 0.60\n"
"vt 2.95 0.60\n"
"vt 2.75 0.90\n"
"vt 2.10 0.30\n"
"vt 2.90 0.30\n"
"vt 2.50 0.10\n"
"vt 2.50 0.50\n"
"\n"
"f 11/18 12/19 13/20\n"
"f 12/21 11/22 13/23\n"
"f 14/24 12/26 11/27\n"
"f 15/25 11/27 12/26\n"
"\n"
"t interpolateboundary 1/0/0 1\n"
"\n"
;
