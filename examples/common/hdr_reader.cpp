//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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

    unsigned char *dst = NULL, *line = NULL, *d;
    float *fd;

    const int MAXLINE = 1024;
    char buffer[MAXLINE];

    // read header
    while(true) {
        if (not fgets(buffer, MAXLINE, fp)) goto error;
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
    if (not fgets(buffer, MAXLINE, fp)) goto error;
    {
        int n = (int)strlen(buffer);
        for (int i = 1; i < n; ++i) {
            if (buffer[i] == 'X') {
                if (not (info->flag & HDR_Y_MAJOR)) info->flag |= HDR_X_MAJOR;
                info->flag |= (buffer[i-1] == '-') ? HDR_X_DEC : 0;
                info->width = atoi(&buffer[i+1]);
            } else if (buffer[i] == 'Y') {
                if (not (info->flag & HDR_X_MAJOR)) info->flag |= HDR_Y_MAJOR;
                info->flag |= (buffer[i-1] == '-') ? HDR_Y_DEC : 0;
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
        if (not readLine(line, info->scanLength, fp)) goto error;
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
