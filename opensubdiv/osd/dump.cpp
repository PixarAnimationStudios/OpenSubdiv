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
#include "../version.h"

#include <GL/glew.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static void
DumpBuffer(GLuint buffer) {

    printf("----------------------------------------------------\n");
    glBindBuffer(GL_ARRAY_BUFFER, buffer);

    int size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);

    float *p = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    if (p)
        for (int i = 0; i < size/16; i++, p+=4)
            printf("%d: %f %f %f %f\n", i, p[0], p[1], p[2], p[3]);

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
static void
DumpBufferInt(GLuint buffer) {

    if (buffer == 0) return;
    
    printf("---------------------------------------------------\n");
    glBindBuffer(GL_ARRAY_BUFFER, buffer);

    int size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);

    unsigned int *p = (unsigned int*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    if (p) {
        for (int i = 0; i < size/4; i++)
            printf("%04x ", *p++);
        printf("\n");
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static void
DebugAttribs(GLuint program) {

    GLint numAttributes = 0;
    glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &numAttributes);
    if (numAttributes == 0) {
        printf("No attributes\n");
        return;
    }

    GLint maxNameLength = 0;
    glGetProgramiv(program, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &maxNameLength);

    char * name = new char[maxNameLength];

    for (int i=0; i<numAttributes; ++i) {
        GLint size;
        GLenum type;
        glGetActiveAttrib(program, i, maxNameLength, NULL, &size, &type, name);
        printf("Attrib %s, size=%d, type=%x\n", name, size, type);
    }
}

static void
DebugProgram(GLuint program) {

    unsigned char buffer[1*1024*1024];
    GLsizei length = 0;
    GLenum format;
    glGetProgramBinary(program, 10*1024*1024, &length, &format, buffer);

    FILE *fp =fopen ("out.bin", "wb");
    if (fp) {
        fwrite(buffer, 1, length, fp);
        fclose(fp);
    }
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
