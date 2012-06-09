#include <GL/glew.h>

static void
DumpBuffer(GLuint buffer)
{
    printf("----------------------------------------------------\n");
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    int size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    float *p = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    if(p){
        for(int i = 0; i < size/16; i++){
            printf("%d: %f %f %f %f\n", i, p[0], p[1], p[2], p[3]);
            p += 4;
        }
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
static void
DumpBufferInt(GLuint buffer)
{
    if(buffer == 0) return;
    
    printf("---------------------------------------------------\n");
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    int size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    unsigned int *p = (unsigned int*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    if(p){
        for(int i = 0; i < size/4; i++){
            printf("%04x ", *p++);
        }
        printf("\n");
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static void
DebugAttribs(GLuint program)
{
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
        glGetActiveAttrib(program, i,
                          maxNameLength, NULL,
                          &size, &type, name);
        printf("Attrib %s, size=%d, type=%x\n", name, size, type);
    }
}

static void
DebugProgram(GLuint program)
{
    unsigned char buffer[1*1024*1024];
    GLsizei length = 0;
    GLenum format;
    glGetProgramBinary(program, 10*1024*1024, &length, &format, buffer);
    FILE *fp =fopen ("out.bin", "wb");
    if(fp){
        fwrite(buffer, 1, length, fp);
        fclose(fp);
    }
}
