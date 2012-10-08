
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>

extern void initOsd();
extern void updateGeom();
extern void display();
extern int g_width, g_height, g_frame;

//
// ### Misc. GLUT Callback Methods

// Reshape is called when the window is resized, here we need the width and 
// height so that we can correctly adjust the aspect ratio of the projection
// matrix.
//
void
reshape(int width, int height) 
{
    g_width = width;
    g_height = height;
}

// 
// Idle is called between frames, here we advance the frame number and update
// the procedural animation that is being applied to the mesh
//
void
idle() 
{
    g_frame++;
    updateGeom();
    glutPostRedisplay();
}

//
// ### Draw the Mesh 

// Display handles all drawing per frame. We first call the setupForDisplay 
// helper method to setup some uninteresting GL state and then bind the mesh
// using the buffers provided by our OSD objects
//
void
glutDisplay() 
{
    display();
    glutSwapBuffers();
}


int main(int argc, char ** argv) 
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA |GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(1024, 1024);
    glutCreateWindow("CPU Subdivision Example");

    // 
    // Setup glut, glew and some initial GL state
    //
    glutDisplayFunc(glutDisplay);
    glutReshapeFunc(reshape);
    glewInit();

    initOsd();
    
    //
    // Start the main glut drawing loop
    //
    glutIdleFunc(idle);
    glutMainLoop();
}

