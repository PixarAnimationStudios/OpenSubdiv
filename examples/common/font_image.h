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

#ifndef FONT_IMAGE_H
#define FONT_IMAGE_H

#define FONT_TEXTURE_WIDTH 128
#define FONT_TEXTURE_HEIGHT 128
#define FONT_TEXTURE_COLUMNS 16
#define FONT_TEXTURE_ROWS 8
#define FONT_CHAR_WIDTH (FONT_TEXTURE_WIDTH/FONT_TEXTURE_COLUMNS)
#define FONT_CHAR_HEIGHT (FONT_TEXTURE_HEIGHT/FONT_TEXTURE_ROWS)
#define FONT_CHECK_BOX_OFF     0x3
#define FONT_CHECK_BOX_ON      0x4
#define FONT_RADIO_BUTTON_OFF  0x6
#define FONT_RADIO_BUTTON_ON   0x7

extern unsigned char font_image[];

#endif // FONT_IMAGE_H
