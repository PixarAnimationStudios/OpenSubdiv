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

static const std::string catmark_val2_back2back =
"#\n"
"#   Four shapes ordered left->right and top->bottom in the XZ plane\n"
"#\n"
"#   Shape 1:  top-left\n"
"#\n"
"v -0.90  0  0.10\n"
"v -0.10  0  0.10\n"
"v -0.50  0  0.90\n"
"\n"
"vt -0.90  0.10\n"
"vt -0.10  0.10\n"
"vt -0.50  0.90\n"
"\n"
"f  1/1   2/2   3/3\n"
"f  3/3   2/2   1/1\n"
"\n"
"#\n"
"#   Shape 2:  top-right\n"
"#\n"
"v  0.10  0  0.10\n"
"v  0.90  0  0.10\n"
"v  0.90  0  0.90\n"
"v  0.10  0  0.90\n"
"\n"
"vt  0.10  0.10\n"
"vt  0.90  0.10\n"
"vt  0.90  0.90\n"
"vt  0.10  0.90\n"
"\n"
"f  4/4   5/5   6/6   7/7\n"
"f  7/7   6/6   5/5   4/4\n"
"\n"
"#\n"
"#   Shape 3:  bottom-left\n"
"#\n"
"v -0.70  0 -0.90\n"
"v -0.30  0 -0.90\n"
"v -0.10  0 -0.50\n"
"v -0.50  0 -0.10\n"
"v -0.90  0 -0.50\n"
"\n"
"vt -0.70  -0.90\n"
"vt -0.30  -0.90\n"
"vt -0.10  -0.50\n"
"vt -0.50  -0.10\n"
"vt -0.90  -0.50\n"
"\n"
"f  8/8   9/9  10/10 11/11 12/12\n"
"f 12/12 11/11 10/10  9/9   8/8\n"
"\n"
"#\n"
"#   Shape 4:  bottom-right\n"
"#\n"
"v  0.30  0 -0.90\n"
"v  0.70  0 -0.90\n"
"v  0.90  0 -0.50\n"
"v  0.70  0 -0.10\n"
"v  0.30  0 -0.10\n"
"v  0.10  0 -0.50\n"
"\n"
"vt  0.30  -0.90\n"
"vt  0.70  -0.90\n"
"vt  0.90  -0.50\n"
"vt  0.70  -0.10\n"
"vt  0.30  -0.10\n"
"vt  0.10  -0.50\n"
"\n"
"f 13/13 14/14 15/15 16/16 17/17 18/18\n"
"f 18/18 17/17 16/16 15/15 14/14 13/13\n"
"\n"
"t interpolateboundary 1/0/0 1\n"
"\n"
;
