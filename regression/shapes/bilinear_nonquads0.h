//
//   Copyright 2018 DreamWorks Animation LLC.
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

static const std::string bilinear_nonquads0 = 
"\n"
"v -2.05  0 -0.88\n"
"v -0.05  0 -0.88\n"
"v -1.05  0  0.85\n"
"v  0.41  0 -0.91\n"
"v  1.59  0 -0.91\n"
"v  1.95  0  0.21\n"
"v  1.00  0  0.90\n"
"v  0.05  0  0.21\n"
"\n"
"vt 0.0  0.0\n"
"vt 1.0  0.0\n"
"vt 0.5  1.0\n"
"vt 0.20 0.05\n"
"vt 0.74 0.05\n"
"vt 0.98 0.58\n"
"vt 0.50 0.95\n"
"vt 0.02 0.58\n"
"\n"
"f 1/1 2/2 3/3\n"
"f 4/4 5/5 6/6 7/7 8/8\n"
"\n"
"t interpolateboundary 1/0/0 1\n"
;
