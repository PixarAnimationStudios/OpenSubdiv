//
// This file contains standard OpenGL API calls that are mostly uninteresting if
// you are purely trying to learn OSD. The following operations are implemented
// here:
//
//  * First time OpenGL state initialization
//  * Compile vertex and fragment shaders
//  * Link shaders into a program
//  * Per-frame drawing state setup
//

#pragma once

// 
// ### OS Compatibility
// The following is to ensure the example runs on Linux, Windows and OS X
//
#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
#else
    #include <stdlib.h>
    #include <GL/glew.h>
    #include <GL/glut.h>
#endif

#include "algebra.h"

#include <iostream>
#include <fstream>
#include <sstream>

//
// Microsoft uses a slightly different snprintf declaration, which we hide
// here by aliasing it as snprintf
//
#if _MSC_VER
    #define snprintf _snprintf
#endif

#if defined(__APPLE__)
#define drawString(x, y, ...); 
/*               \
    GLUT on OSX means gl 2.1, so forgoe the text drawing until we have a better solution
    { char line[1024]; \
      snprintf(line, 1024, __VA_ARGS__); \
      char *p = line; \
      glWindowPos2f(x, y); \
      while(*p) { glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p++); } }
      */
#else
#define drawString(x, y, ...)               \
    { char line[1024]; \
      snprintf(line, 1024, __VA_ARGS__); \
      char *p = line; \
      glWindowPos2i(x, y); \
      while(*p) { glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p++); } }
#endif

//
// These are standard OpenGL shader program handles. One for wire frame and one
// for shaded rendering
//
GLuint g_quadLineProgram = 0;
GLuint g_quadFillProgram = 0;

//
// To avoid reading the shader source from a file, we include it here as a string
// see [shader.glsl](shader.html).
//
static const char *shaderSource =
#include "shader.inc"
;

GLfloat g_modelView[16], g_proj[16], g_mvp[16];

void
bindProgram(GLuint program)
{
    glUseProgram(program);

    // shader uniform setting
    GLint position = glGetUniformLocation(program, "lightSource[0].position");
    GLint ambient = glGetUniformLocation(program, "lightSource[0].ambient");
    GLint diffuse = glGetUniformLocation(program, "lightSource[0].diffuse");
    GLint specular = glGetUniformLocation(program, "lightSource[0].specular");
    GLint position1 = glGetUniformLocation(program, "lightSource[1].position");
    GLint ambient1 = glGetUniformLocation(program, "lightSource[1].ambient");
    GLint diffuse1 = glGetUniformLocation(program, "lightSource[1].diffuse");
    GLint specular1 = glGetUniformLocation(program, "lightSource[1].specular");
    
    glUniform4f(position, 0.5, 0.2f, 1.0f, 0.0f);
    glUniform4f(ambient, 0.1f, 0.1f, 0.1f, 1.0f);
    glUniform4f(diffuse, 0.7f, 0.7f, 0.7f, 1.0f);
    glUniform4f(specular, 0.8f, 0.8f, 0.8f, 1.0f);
    
    glUniform4f(position1, -0.8f, 0.4f, -1.0f, 0.0f);
    glUniform4f(ambient1, 0.0f, 0.0f, 0.0f, 1.0f);
    glUniform4f(diffuse1, 0.5f, 0.5f, 0.5f, 1.0f);
    glUniform4f(specular1, 0.8f, 0.8f, 0.8f, 1.0f);

    GLint otcMatrix = glGetUniformLocation(program, "objectToClipMatrix");
    GLint oteMatrix = glGetUniformLocation(program, "objectToEyeMatrix");
    multMatrix(g_mvp, g_modelView, g_proj);

    glUniformMatrix4fv(otcMatrix, 1, false, g_mvp);
    glUniformMatrix4fv(oteMatrix, 1, false, g_modelView);
}

static GLuint 
compileShader(GLenum shaderType, const char *section, const char *define)
{
    const char *sources[4];
    char sdefine[64];
    sprintf(sdefine, "#define %s\n", section);

    sources[0] = "#version 330\n";
    sources[1] = define;
    sources[2] = sdefine;
    sources[3] = shaderSource;

    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 4, sources, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if( status == GL_FALSE ) {
        GLchar emsg[1024];
        glGetShaderInfoLog(shader, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error compiling GLSL shader (%s): %s\n", section, emsg );
        fprintf(stderr, "Section: %s\n", sdefine);
        fprintf(stderr, "Defines: %s\n", define);
        fprintf(stderr, "Source: %s\n", sources[2]);
        exit(0);
    }

    return shader;
}

GLuint 
linkProgram(const char *define) 
{
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, "VERTEX_SHADER", define);
    GLuint geometryShader = compileShader(GL_GEOMETRY_SHADER, "GEOMETRY_SHADER", define);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, "FRAGMENT_SHADER", define);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, geometryShader);
    glAttachShader(program, fragmentShader);

    glBindAttribLocation(program, 0, "position");
    glBindAttribLocation(program, 1, "normal");

    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(geometryShader);
    glDeleteShader(fragmentShader);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status );
    if( status == GL_FALSE ) {
        GLchar emsg[1024];
        glGetProgramInfoLog(program, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error linking GLSL program : %s\n", emsg );
        fprintf(stderr, "Defines: %s\n", define);
        exit(0);
    }

    return program;
}

void
initGL() 
{
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glDepthFunc(GL_LEQUAL);
    
    g_quadFillProgram = linkProgram("#define PRIM_QUAD\n#define GEOMETRY_OUT_FILL\n");
    g_quadLineProgram = linkProgram("#define PRIM_QUAD\n#define GEOMETRY_OUT_LINE\n");
}

void
setupForDisplay(int width, int height, float size, float* center) 
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, width, height);
    setIdentity(g_proj);
    setIdentity(g_modelView);
    setIdentity(g_mvp);

    // setup the projection
    float aspect = width/(float)height;
    setPersp(45.0f, aspect, 0.01f, 500.0f, g_proj);

    // setup the model view matrix
    translateMatrix(-center[0], -center[1], -center[2], g_modelView);
    rotateMatrix(-90, 1, 0, 0, g_modelView); // z-up model
    translateMatrix(0, 0, -size, g_modelView);
}

