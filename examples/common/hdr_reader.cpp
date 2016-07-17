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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "hdr_reader.h"

static bool
readLine(unsigned char *line, int length, FILE *fp)
{
    // 2, 2, width  (this reader only reads newer format)
    if (getc(fp) != 2) return false;
    if (getc(fp) != 2) return false;
    int w0 = getc(fp);
    int w1 = getc(fp);
    if ((w0 << 8 | w1) != length) return false;

    for (int i = 0; i < 4; ++i) {
        for (int x = 0; x < length;) {
            int c = getc(fp);
            if (c > 128) {
                // runlength
                c &= 127;
                unsigned char value = (unsigned char)getc(fp);
                while (c--) {
                    line[x*4+i] = value;
                    x++;
                }
            } else {
                // non- runlength
                while (c--) {
                    unsigned char value = (unsigned char)getc(fp);
                    line[x*4+i] = value;
                    x++;
                }
            }
        }
    }
    return true;
}

unsigned char *loadHdr(const char *filename, HdrInfo *info, bool convertToFloat)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) return NULL;

    memset(info, 0, sizeof(HdrInfo));
    info->exposure = 1.0;

    unsigned char *dst=NULL, *line=NULL, *d=NULL;
    float *fd=NULL;

    const int MAXLINE = 1024;
    char buffer[MAXLINE];

    // read header
    while(true) {
        if (! fgets(buffer, MAXLINE, fp)) goto error;
        if (buffer[0] == '\n') break;
        if (buffer[0] == '\r' && buffer[0] == '\n') break;
        if (strncmp(buffer, "#?", 2) == 0) {
            strncpy(info->magic, buffer+2, 64);
        } else if (strncmp(buffer, "FORMAT=", 7) == 0) {
            strncpy(info->format, buffer+7, 64);
        }
    }
//    if (strncmp(info->magic, "RADIANCE", 8)) goto error;
    if (strncmp(info->format, "32-bit_rle_rgbe", 15)) goto error;

    // resolution
    if (! fgets(buffer, MAXLINE, fp)) goto error;
    {
        int n = (int)strlen(buffer);
        for (int i = 1; i < n; ++i) {
            if (buffer[i] == 'X') {
                if (! (info->flag & HDR_Y_MAJOR)) info->flag |= HDR_X_MAJOR;
                info->flag |= (char)((buffer[i-1] == '-') ? HDR_X_DEC : 0);
                info->width = atoi(&buffer[i+1]);
            } else if (buffer[i] == 'Y') {
                if (! (info->flag & HDR_X_MAJOR)) info->flag |= HDR_Y_MAJOR;
                info->flag |= (char)((buffer[i-1] == '-') ? HDR_Y_DEC : 0);
                info->height = atoi(&buffer[i+1]);
            }
        }
    }
    if (info->width <= 0 || info->height <= 0) goto error;

    if (info->flag & HDR_Y_MAJOR) {
        info->scanLength = info->width;
        info->scanWidth = info->height;
    } else {
        info->scanLength = info->height;
        info->scanWidth = info->width;
    }

    // read body
    if (convertToFloat) {
        dst = (unsigned char *)malloc(info->width * info->height * 4 * sizeof(float));
        fd = (float*)dst;
    } else {
        dst = (unsigned char *)malloc(info->width * info->height * 4);
        d = dst;
    }
    line = (unsigned char *)malloc(info->scanLength*4);

    for (int y = info->scanWidth-1; y >= 0; --y) {
        if (! readLine(line, info->scanLength, fp)) goto error;
        for (int x = 0; x < info->scanLength; ++x) {
            if (convertToFloat) {
                float scale = powf(2.0f, float(line[x*4+3] - 128))/255.0f;
                *fd++ = line[x*4  ]*scale;
                *fd++ = line[x*4+1]*scale;
                *fd++ = line[x*4+2]*scale;
                *fd++ = 1.0;
            } else {
                *d++ = line[x*4  ];
                *d++ = line[x*4+1];
                *d++ = line[x*4+2];
                *d++ = line[x*4+3];
            }
        }
    }
    free(line);
    fclose(fp);

    return dst;

error:
    printf("Error in reading %s\n", filename);
    if(dst) free(dst);
    if(line) free(line);
    fclose(fp);
    return NULL;
}

/*
int main(int argc, char *argv[]) {
    HdrInfo info;
    loadHdr(argv[1], &info);
}
*/
