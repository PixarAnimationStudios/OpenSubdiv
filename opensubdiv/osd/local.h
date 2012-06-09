#ifndef OSD_LOCAL_H
#define OSD_LOCAL_H

#include <stdio.h>

#define OSD_STRINGIFY(src) #src

#define CHECK_GL_ERROR(...)  \
    if(GLuint err = glGetError()) {   \
    printf("GL error %x :", err); \
    printf(__VA_ARGS__); \
    }

#define OSD_ERROR(...) printf(__VA_ARGS__);

//#define OSD_DEBUG(...) printf(__VA_ARGS__);
#define OSD_DEBUG(...) 
    

#endif // OSD_LOCAL_H
