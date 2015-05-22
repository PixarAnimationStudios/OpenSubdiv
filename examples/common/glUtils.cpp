//
//   Copyright 2015 Pixar
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

#include <sstream>
#include <string>
#include "glUtils.h"

namespace GLUtils {

void
CheckGLErrors(std::string const & where) {
    GLuint err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "GL error: "
                  << (where.empty() ? "" : where + " ")
                  << err << "\n";
    }
}

GLuint
CompileShader(GLenum shaderType, const char *source) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLchar emsg[40960];
        glGetShaderInfoLog(shader, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error compiling GLSL shader: %s\n", emsg);
        return 0;
    }

    return shader;
}

bool
SupportsAdaptiveTessellation() {
#ifdef OSD_USES_GLEW
    return glewGetExtension("GL_ARB_tessellation_shader") == GL_TRUE;
#else
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    return true;
#else
    return false;
#endif
#endif
}

///Helper function that parses the open gl version string, retrieving the major 
///and minor version from it.
void get_major_minor_version(int *major, int *minor){
	static bool initialized = false;
	int _major = -1, _minor = -1;
	if (!initialized || _major == -1 || _minor == -1){
		const GLubyte *ver = glGetString(GL_SHADING_LANGUAGE_VERSION);
		if (!ver){
			_major = -1;
			_minor = -1;
		}
		else{
			std::string major_str(ver, ver + 1);
			std::string minor_str(ver + 2, ver + 3);
			std::stringstream ss;
			ss << major_str << " " << minor_str;
			ss >> _major;
			ss >> _minor;
		}
		initialized = true;
	}
	*major = _major;
	*minor = _minor;

}

/** Gets the shader version based on the current opengl version and returns 
 * it in a string form */

const std::string &get_shader_version(){
	static bool initialized = false;
	static std::string shader_version;
	if (!initialized){

		int major, minor;
		get_major_minor_version(&major, &minor);
		int version_number = major * 10 + minor;
		switch (version_number){
		case 20:
			shader_version = "110";
			break;
		case 21:
			shader_version = "120";
			break;
		case 30:
			shader_version = "130";
			break;
		case 31:
			shader_version = "140";
			break;
		case 32:
			shader_version = "150";
			break;
		default:
			std::stringstream ss;
			ss << version_number;
			shader_version = ss.str() + "0";
			break;
		}
		initialized = true;
	}
	return shader_version;
}

/* Generates the version defintion needed by the glsl shaders based on the 
 * opengl string
*/
const std::string &get_shader_version_include(){
	static bool initialized = false;
	static std::string include;
	if (!initialized){
		include = "#version " + get_shader_version() + "\n";
		initialized = true;
	}
	return include;
}

bool uses_tesselation_shaders(){

#if defined(OSD_USES_GLEW)
	bool initialized = false, uses = false;
	if (!initialized){
		uses = glewGetExtension("GL_ARB_tessellation_shader") ||
			(GLEW_VERSION_4_0
			&& glewGetExtension("GL_ARB_tessellation_shader"));
		initialized = true;
	}
	return uses;
#else
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
	return true;
#else
	return false;
#endif
#endif
}

bool GL_ARB_separate_shader_objects_or_GL_VERSION_4_1(){
#if defined(OSD_USES_GLEW)
	bool initialized = false, uses = false;
	if (!initialized){
		uses = glewGetExtension("GL_ARB_separate_shader_objects") ||
			(GLEW_VERSION_4_1
			&& glewGetExtension("GL_ARB_tessellation_shader"));
		initialized = true;
	}
	return uses;
#else
#if defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
	return true;
#else
	return false;
#endif
#endif
}


                      
}   // namesapce GLUtils
