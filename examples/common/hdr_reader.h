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

#define HDR_X_MAJOR (1 << 0)
#define HDR_Y_MAJOR (1 << 1)
#define HDR_X_DEC   (1 << 2)
#define HDR_Y_DEC   (1 << 3)

struct HdrInfo
{
    char magic[64];
    char format[64];
    double exposure;
    int width;
    int height;
    char flag;
    int scanLength;
    int scanWidth;
};

extern unsigned char *loadHdr(const char *filename, HdrInfo *info, bool convertToFloat);
