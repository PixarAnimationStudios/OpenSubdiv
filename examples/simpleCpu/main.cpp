
#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
    #define GLFW_INCLUDE_GL3
    #define GLFW_NO_GLU
    #include <stdio.h>
#else
    #include <stdlib.h>
    #include <stdio.h>
    #include <GL/glew.h>
    #if defined(WIN32)
        #include <GL/wglew.h>
    #endif
#endif

#if defined(GLFW_VERSION_3)
    #include <GL/glfw3.h>
    GLFWwindow* g_window=0;
    GLFWmonitor* g_primary=0;
#else
    #include <GL/glfw.h>
#endif

extern void initOsd();
extern void updateGeom();
extern void display();
extern int g_width, g_height, g_frame;

static bool g_running = true;

static void
setGLCoreProfile()
{
#if GLFW_VERSION_MAJOR>=3
    #define glfwOpenWindowHint glfwWindowHint
    #define GLFW_OPENGL_VERSION_MAJOR GLFW_CONTEXT_VERSION_MAJOR
    #define GLFW_OPENGL_VERSION_MINOR GLFW_CONTEXT_VERSION_MINOR
#endif

    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if not defined(__APPLE__)
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#endif
    
}

//
// ### Misc. GLFW Callback Methods

// Reshape is called when the window is resized, here we need the width and 
// height so that we can correctly adjust the aspect ratio of the projection
// matrix.
//
void
#if GLFW_VERSION_MAJOR>=3
reshape(GLFWwindow *, int width, int height) {
#else
reshape(int width, int height) {
#endif
    g_width = width;
    g_height = height;
}

#if GLFW_VERSION_MAJOR>=3
int windowClose(GLFWwindow*) {
#else
int windowClose() {
#endif
    g_running = false;
    return GL_TRUE;
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
}


int main(int argc, char ** argv) 
{
    // 
    // Setup GLFW, glew and some initial GL state
    //
    static const char windowTitle[] = "CPU Subdivision Example";

    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }
    
#define CORE_PROFILE
#ifdef CORE_PROFILE
    setGLCoreProfile();
#endif

    #if GLFW_VERSION_MAJOR>=3
        if (not (g_window=glfwCreateWindow(g_width, g_height, windowTitle, NULL, NULL))) {
            printf("Failed to open window.\n");
            glfwTerminate();
            return 1;
        }
        glfwMakeContextCurrent(g_window);
        glfwSetWindowSizeCallback(g_window, reshape);
        glfwSetWindowCloseCallback(g_window, windowClose);
    #else
        if (glfwOpenWindow(g_width, g_height, 8, 8, 8, 8, 24, 8, GLFW_WINDOW) == GL_FALSE) {
            printf("Failed to open window.\n");
            glfwTerminate();
            return 1;
        }
        glfwSetWindowTitle(windowTitle);
        glfwSetWindowSizeCallback(reshape);
        glfwSetWindowCloseCallback(windowClose);
    #endif

    

#if not defined(__APPLE__)
#ifdef CORE_PROFILE
    // this is the only way to initialize glew correctly under core profile context.
    glewExperimental = true;
#endif
    if (GLenum r = glewInit() != GLEW_OK) {
        printf("Failed to initialize glew. Error = %s\n", glewGetErrorString(r));
        exit(1);
    }
#ifdef CORE_PROFILE
    // clear GL errors which was generated during glewInit()
    glGetError();
#endif
#endif

    initOsd();
    
    //
    // Start the main drawing loop
    //
    while (g_running) {
        idle();
        display();
        
#if GLFW_VERSION_MAJOR>=3
        glfwPollEvents();
        glfwSwapBuffers(g_window);
#else
        glfwSwapBuffers();
#endif
        
        glFinish();
    }
}

