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

#ifndef FONT_IMAGE_H
#define FONT_IMAGE_H

#define FONT_TEXTURE_WIDTH 128
#define FONT_TEXTURE_HEIGHT 128
#define FONT_TEXTURE_COLUMNS 16
#define FONT_TEXTURE_ROWS 8
#define FONT_CHAR_WIDTH (FONT_TEXTURE_WIDTH/FONT_TEXTURE_COLUMNS)
#define FONT_CHAR_HEIGHT (FONT_TEXTURE_HEIGHT/FONT_TEXTURE_ROWS)
#define FONT_CHECK_BOX_OFF     0x2
#define FONT_CHECK_BOX_ON      0x3
#define FONT_RADIO_BUTTON_OFF  0x4
#define FONT_RADIO_BUTTON_ON   0x5
#define FONT_SLIDER_LEFT       0x10
#define FONT_SLIDER_MIDDLE     0x11
#define FONT_SLIDER_RIGHT      0x12
#define FONT_SLIDER_CURSOR     0x13
#define FONT_ARROW_RIGHT       0x14
#define FONT_ARROW_DOWN        0x15

extern unsigned char font_image[];

#endif // FONT_IMAGE_H
