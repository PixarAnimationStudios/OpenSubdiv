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

static const std::string catmark_val2_nonman =
"#\n"
"#   Four shapes ordered left->right and top->bottom in the XZ plane\n"
"#\n"
"#   Shape 1:  top-left\n"
"#\n"
"v -0.70  0    0.30\n"
"v -0.30  0    0.30\n"
"v -0.30  0    0.70\n"
"v -0.70  0    0.70\n"
"v -0.70  0.2  0.10\n"
"v -0.30  0.2  0.10\n"
"v -0.70 -0.2  0.10\n"
"v -0.30 -0.2  0.10\n"
"\n"
"vt -0.70  0.30\n"
"vt -0.30  0.30\n"
"vt -0.30  0.70\n"
"vt -0.70  0.70\n"
"vt -0.70  0.10\n"
"vt -0.30  0.10\n"
"vt -0.70  0.10\n"
"vt -0.30  0.10\n"
"\n"
"f  1/1   2/2   3/3   4/4\n"
"f  4/4   3/3   2/2   1/1\n"
"f  1/1   2/2   6/6   5/5\n"
"f  2/2   1/1   7/7   8/8\n"
"\n"
"#\n"
"#   Shape 2:  top-right\n"
"#\n"
"v  0.30  0    0.30\n"
"v  0.70  0    0.30\n"
"v  0.50  0    0.70\n"
"v  0.50  0    0.70\n"
"v  0.30  0.2  0.10\n"
"v  0.70  0.2  0.10\n"
"v  0.30 -0.2  0.10\n"
"v  0.70 -0.2  0.10\n"
"\n"
"vt  0.30  0.30\n"
"vt  0.70  0.30\n"
"vt  0.50  0.70\n"
"vt  0.50  0.70\n"
"vt  0.30  0.10\n"
"vt  0.70  0.10\n"
"vt  0.30  0.10\n"
"vt  0.70  0.10\n"
"\n"
"f  9/9  10/10 11/11\n"
"f 11/11 10/10  9/9\n"
"f  9/9  10/10 14/14 13/13\n"
"f 10/10  9/9  15/15 16/16\n"
"\n"
"#\n"
"#   Shape 3:  bottom-left\n"
"#\n"
"v -0.70  0   -0.70\n"
"v -0.30  0   -0.70\n"
"v -0.30  0   -0.30\n"
"v -0.70  0   -0.30\n"
"v -0.70 -0.2 -0.90\n"
"v -0.30 -0.2 -0.90\n"
"\n"
"vt -0.70 -0.70\n"
"vt -0.30 -0.70\n"
"vt -0.30 -0.30\n"
"vt -0.70 -0.30\n"
"vt -0.70 -0.90\n"
"vt -0.30 -0.90\n"
"\n"
"f 17/17 18/18 19/19 20/20\n"
"f 20/20 19/19 18/18 17/17\n"
"f 21/21 22/22 17/17\n"
"\n"
"#\n"
"#   Shape 4:  bottom-right\n"
"#\n"
"v  0.30  0   -0.70\n"
"v  0.70  0   -0.70\n"
"v  0.70  0   -0.30\n"
"v  0.30  0   -0.30\n"
"v  0.30 -0.2 -0.90\n"
"v  0.70 -0.2 -0.90\n"
"\n"
"vt  0.30 -0.70\n"
"vt  0.70 -0.70\n"
"vt  0.70 -0.30\n"
"vt  0.30 -0.30\n"
"vt  0.30 -0.90\n"
"vt  0.70 -0.90\n"
"\n"
"f 23/23 24/24 25/25 26/26\n"
"f 26/26 25/25 24/24 23/23\n"
"f 27/27 28/28 23/23\n"
"f 27/27 28/28 24/24\n"
"\n"
"t interpolateboundary 1/0/0 1\n"
"\n"
;
