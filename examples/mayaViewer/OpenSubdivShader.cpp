//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#if defined(__APPLE__)
    #include <maya/OpenMayaMac.h>
#else
    #include <GL/glew.h>
    #if defined(WIN32)
        #include <GL/wglew.h>
    #endif
#endif


#include "../common/maya_util.h"
#include "OpenSubdivShader.h"
#include "osdMeshData.h"

#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MDrawContext.h>

#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

// Identifiers
MTypeId OpenSubdivShader::id(0x88110);
MString OpenSubdivShader::drawRegistrantId("OpenSubdivShaderPlugin");

// Subdivision Attributes
MObject OpenSubdivShader::aLevel;                   // Subdivision level
MObject OpenSubdivShader::aTessFactor;              // GPU tesselation factor
MObject OpenSubdivShader::aScheme;                  // Catmull-Clark, Loop, Bilinear
MObject OpenSubdivShader::aKernel;                  // Computation (CPU, CUDA, etc)
MObject OpenSubdivShader::aInterpolateBoundary;     // Boundary interpolation method
MObject OpenSubdivShader::aInterpolateUVBoundary;   // Face-varying interpolation
MObject OpenSubdivShader::aAdaptive;                // Feature-adaptive toggle

// Appearance attributes
MObject OpenSubdivShader::aWireframe;               // Wireframe display toggle
MObject OpenSubdivShader::aDiffuse;                 // Material parameters
MObject OpenSubdivShader::aAmbient;
MObject OpenSubdivShader::aSpecular;
MObject OpenSubdivShader::aShininess;

// Texture attributes
MObject OpenSubdivShader::aDiffuseMapFile;          // Input texture filename
MObject OpenSubdivShader::aUVSet;                   // Optional UV set

//      A default shader is compiled into the plug-in.
//      The shaderSource attribute offers the ability to modify or 
//      replace the shader without having to recompile the plug-in.
MObject OpenSubdivShader::aShaderSource;            // Optional shader file

static const char *defaultShaderSource =
#include "shader.inc"
;


// --------------------------------------------------------------------------------------
// ### EffectDrawRegistry 
//      An OpenSubdiv application builds its own draw registry in 
//      order to define parameters needed for its shader.  In our 
//      case we use the attributes from the OpenSubdivShader to set up 
//      parameters and #define directives to pass through subdivision 
//      options and to control draw style.  Client/application must
//      specialize OsdGLDrawRegistry in order to provide an appearance
//      for objects.  Built-in shaders to not contain lighting code.

// Draw styles for EffectDrawRegistry
enum Effect 
{
    kFill = 0,
    kLine = 1,
    kPoint = 2,
};
typedef std::pair<OpenSubdiv::OsdDrawContext::PatchDescriptor, Effect> EffectDesc;

// #### Override of OpenSubdiv::OsdGLDrawRegistry 
//
// At the very least this class needs to define _CreateDrawSourceConfig
// and _CreateDrawConfig in order to define shader content.
//
class EffectDrawRegistry : public OpenSubdiv::OsdGLDrawRegistry<EffectDesc> 
{

public:
    EffectDrawRegistry() : _isAdaptive(true),
                           _diffuseId(0),
                           _shaderSource( defaultShaderSource )
            {}

    // Accessors
    //
    //      When setInternalValueInContext() gets triggered for certain 
    //      attributes it will set dirty flags causing the shaders to 
    //      be rebuilt.
    //
    /* isAdaptive */
    void setIsAdaptive( bool ad ) { resetIfChanged(ad, _isAdaptive); }
    bool getIsAdaptive() const { return _isAdaptive; }

    /* diffuseId */
    void setDiffuseId( GLuint dId ) { resetIfChanged(dId, _diffuseId); }
    GLuint getDiffuseId() const { return _diffuseId; }

    /* shaderSource */
    void setShaderSource( std::string const & src ) { resetIfChanged(src, _shaderSource); }
    std::string const & getShaderSource() const { return _shaderSource; }

protected:
    // Compile and link the shader
    virtual ConfigType * _CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig);

    // Build shader configuration
    virtual SourceConfigType * _CreateDrawSourceConfig(DescType const & desc);

private:

    // Clear the registry if a value has changed, triggering a rebuild
    template< typename T>
    void resetIfChanged( T newVal, T& curVal )
    {
        if ( newVal != curVal ) {
            curVal = newVal;
            Reset();
        }
    }

    // Parameters mirroring attributes that affect shader generation
    bool        _isAdaptive;
    GLuint      _diffuseId;
    std::string _shaderSource;
};


// #### _CreateDrawSourceConfig
//
//      Called by base registry when a draw configuration is requested.
//      Returns a shader source configuration which consists of a 
//      set of shader source code and compile-time configuration 
//      defines.  These are cached by the effect registry for
//      efficient re-use when rebuilding shaders.
//
EffectDrawRegistry::SourceConfigType *
EffectDrawRegistry::_CreateDrawSourceConfig(DescType const & desc) 
{
    Effect effect = desc.second;

    // Create base draw configuration
    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first);

    bool quad = false;
    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS) {
        // Configuration for adaptive refinement
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.source = _shaderSource;
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
        quad = true;
    } else if (desc.first.GetType() == OpenSubdiv::FarPatchTables::TRIANGLES) {
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.source = _shaderSource;
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
    } else {
        // adaptive patches
        sconfig->geometryShader.AddDefine("SMOOTH_NORMALS");
    }
    assert(sconfig);

    // Enable feature-adaptive face-varying UV generation
    if (_isAdaptive) {
        sconfig->geometryShader.AddDefine("FVAR_ADAPTIVE");
    }

    // Enable diffuse texture display if there is a valid texture map
    if (_diffuseId != 0) {
        sconfig->fragmentShader.AddDefine("USE_DIFFUSE_MAP");
    }

    // NUM_ELEMENTS should be set to the same value that is specified 
    // for the "numElements" argument when creating OsdVertexBuffers, 
    // e.g. OsdGLVertexBuffer
    sconfig->vertexShader.AddDefine("NUM_ELEMENTS", "3");

    // Initialize geometry shader
    sconfig->geometryShader.version = "#version 410\n";
    sconfig->geometryShader.source = _shaderSource;
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    // Initialize fragment shader
    sconfig->fragmentShader.version = "#version 410\n";
    sconfig->fragmentShader.source = _shaderSource;
    sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");

    // Set up directives according to draw style
    if (quad) {
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
    } else {
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
    }
    switch (effect) {
    case kFill:
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kLine:
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_LINE");
        break;
    case kPoint:
        sconfig->geometryShader.AddDefine("PRIM_POINT");
        sconfig->fragmentShader.AddDefine("PRIM_POINT");
        break;
    }

    return sconfig;
}

// Global GL buffer IDs and binding points
GLuint g_transformUB = 0,
       g_transformBinding = 0,
       g_tessellationUB = 0,
       g_tessellationBinding = 0,
       g_lightingUB = 0,
       g_lightingBinding = 0;



// #### _CreateDrawConfig
//
//      Called by base registry when a draw config is requested.
//      Returns a compiled and linked shader program corresponding to 
//      a previously created DrawSourceConfig. The effect registry also 
//      caches these for efficient re-use.
// 
EffectDrawRegistry::ConfigType *
EffectDrawRegistry::_CreateDrawConfig(
        DescType const & desc,
        SourceConfigType const * sconfig)
{
    ConfigType * config = BaseRegistry::_CreateDrawConfig(desc.first, sconfig);
    assert(config);

    // Assign binding points to uniform blocks
    //* XXX dyu can use layout(binding=) with GLSL 4.20 and beyond */
    /* struct Transform */
    g_transformBinding = 0;
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Transform"),
        g_transformBinding);

    /* struct Tessellation */
    g_tessellationBinding = 1;
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Tessellation"),
        g_tessellationBinding);

    /* struct Lighting */
    g_lightingBinding = 2;
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Lighting"),
        g_lightingBinding);

    // Specify texture buffer ID uniforms in shader
    GLint loc;
    if ((loc = glGetUniformLocation(config->program, "OsdVertexBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 0);  // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "OsdValenceBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 1);  // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "OsdQuadOffsetBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 2);  // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "OsdPatchParamBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 3);  // GL_TEXTURE3
    }
    if ((loc = glGetUniformLocation(config->program, "OsdFVarDataBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 4);  // GL_TEXTURE4
    }

    CHECK_GL_ERROR("CreateDrawConfig leave\n");

    return config;
}

EffectDrawRegistry g_effectRegistry;



// --------------------------------------------------------------------------------------

// #### Shader construction and initialization

OpenSubdivShader::OpenSubdivShader()
    : _level(3),
      _tessFactor(2),
      _adaptive(true),
      _wireframe(false),
      _scheme(OsdMeshData::kCatmark),
      _kernel(OsdMeshData::kCPU),
      _interpolateBoundary(OsdMeshData::kInterpolateBoundaryNone),
      _interpolateUVBoundary(OsdMeshData::kInterpolateBoundaryNone),
      _shaderSource( defaultShaderSource ),
      _hbrMeshDirty(true),
      _adaptiveDirty(true),
      _diffuseMapDirty(true),
      _shaderSourceDirty(false)
{
}

OpenSubdivShader::~OpenSubdivShader() 
{
}

void *
OpenSubdivShader::creator() 
{
    return new OpenSubdivShader();
}

MStatus
OpenSubdivShader::initialize() 
{
    MFnTypedAttribute typedAttr;
    MFnNumericAttribute numAttr;
    MFnEnumAttribute enumAttr;

    // Subdivision level
    aLevel = numAttr.create("level", "lv", MFnNumericData::kLong, 3);
    numAttr.setInternal(true);
    numAttr.setMin(1);
    numAttr.setSoftMax(5);
    numAttr.setMax(10);
    addAttribute(aLevel);

    // GPU tesselation factor
    aTessFactor = numAttr.create("tessFactor", "tessf", MFnNumericData::kLong, 2);
    numAttr.setInternal(true);
    numAttr.setMin(1);
    numAttr.setMax(10);
    addAttribute(aTessFactor);

    // Subdivision scheme
    aScheme = enumAttr.create("scheme", "sc", OsdMeshData::kCatmark);
    enumAttr.setInternal(true);
    enumAttr.addField("Catmull-Clark",  OsdMeshData::kCatmark);
    enumAttr.addField("Loop",           OsdMeshData::kLoop);
    enumAttr.addField("Bilinear",       OsdMeshData::kBilinear);
    addAttribute(aScheme);

    // Computation kernel
    aKernel = enumAttr.create("kernel", "kn", OsdMeshData::kCPU);
    enumAttr.setInternal(true);
    enumAttr.addField("CPU",    OsdMeshData::kCPU);
#ifdef OPENSUBDIV_HAS_OPENMP
    enumAttr.addField("OpenMP", OsdMeshData::kOPENMP);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    enumAttr.addField("CL",     OsdMeshData::kCL);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    enumAttr.addField("CUDA",   OsdMeshData::kCUDA);
#endif
    addAttribute(aKernel);

    // Boundary interpolation flag
    aInterpolateBoundary = enumAttr.create("interpolateBoundary", "ib", 
                    OsdMeshData::kInterpolateBoundaryNone);
    enumAttr.addField("None",            OsdMeshData::kInterpolateBoundaryNone);
    enumAttr.addField("Edge Only",       OsdMeshData::kInterpolateBoundaryEdgeOnly);
    enumAttr.addField("Edge and Corner", OsdMeshData::kInterpolateBoundaryEdgeAndCorner);
    enumAttr.addField("Always Sharp",    OsdMeshData::kInterpolateBoundaryAlwaysSharp);
    enumAttr.setInternal(true);
    addAttribute(aInterpolateBoundary);

    // Feature-adaptive toggle
    aAdaptive = numAttr.create("adaptive", "adp", MFnNumericData::kBoolean, true);
    numAttr.setInternal(true);
    addAttribute(aAdaptive);

    // Wireframe display toggle
    aWireframe = numAttr.create("wireframe", "wf", MFnNumericData::kBoolean, false);
    addAttribute(aWireframe);

    // Material attributes 
    aDiffuse = numAttr.createColor("diffuse",   "d");
    numAttr.setDefault(0.6f, 0.6f, 0.7f);
    addAttribute(aDiffuse);

    aAmbient = numAttr.createColor("ambient",   "a");
    numAttr.setDefault(0.1f, 0.1f, 0.1f);
    addAttribute(aAmbient);

    aSpecular = numAttr.createColor("specular", "s");
    numAttr.setDefault(0.3f, 0.3f, 0.3f);
    addAttribute(aSpecular);

    aShininess = numAttr.create("shininess", "shin", MFnNumericData::kFloat, 50.0f);
    numAttr.setMin(0);
    numAttr.setSoftMax(128.0f);
    addAttribute(aShininess);

    // Texture attributes
    aDiffuseMapFile = typedAttr.create("diffuseMap", "difmap", MFnData::kString);
    typedAttr.setInternal(true);
    /* don't let maya hold on to string when fileNode is disconnected */
    typedAttr.setDisconnectBehavior(MFnAttribute::kReset);
    addAttribute(aDiffuseMapFile);

    // UV set (defaults to current UV set)
    aUVSet = typedAttr.create("uvSet", "uvs", MFnData::kString);
    typedAttr.setInternal(true);
    addAttribute(aUVSet);

    // Boundary interpolation flag for face-varying data (UVs)
    aInterpolateUVBoundary = enumAttr.create("interpolateUVBoundary", "iuvb", 
                    OsdMeshData::kInterpolateBoundaryNone);
    enumAttr.addField("None",            OsdMeshData::kInterpolateBoundaryNone);
    enumAttr.addField("Edge Only",       OsdMeshData::kInterpolateBoundaryEdgeOnly);
    enumAttr.addField("Edge and Corner", OsdMeshData::kInterpolateBoundaryEdgeAndCorner);
    enumAttr.addField("Always Sharp",    OsdMeshData::kInterpolateBoundaryAlwaysSharp);
    enumAttr.setInternal(true);
    addAttribute(aInterpolateUVBoundary);

    // Optional shader source filename
    aShaderSource = typedAttr.create("shaderSource", "ssrc", MFnData::kString);
    typedAttr.setInternal(true);
    addAttribute(aShaderSource);

    return MS::kSuccess;
}

void
OpenSubdivShader::postConstructor() 
{
    setMPSafe(false);
}

MStatus
OpenSubdivShader::compute(const MPlug &, MDataBlock &) 
{
    return MS::kSuccess;
}

bool
OpenSubdivShader::getInternalValueInContext(const MPlug &plug, MDataHandle &handle, MDGContext &) 
{
    if (plug == aLevel) {
        handle.setInt(_level);
    } else if (plug == aTessFactor) {
        handle.setInt(_tessFactor);
    } else if (plug == aScheme) {
        handle.setShort(_scheme);
    } else if (plug == aKernel) {
        handle.setShort(_kernel);
    } else if (plug == aInterpolateBoundary) {
        handle.setShort(_interpolateBoundary);
    } else if (plug == aAdaptive) {
        handle.setBool(_adaptive);
 
    } else if (plug == aDiffuseMapFile) {
        handle.setString( _diffuseMapFile );
    } else if (plug == aUVSet) {
        handle.setString( _uvSet );
    } else if (plug == aInterpolateUVBoundary) {
        handle.setShort(_interpolateUVBoundary);
 
    } else if (plug == aShaderSource) {
        handle.setString( _shaderSourceFilename );
    }

    return false;
}

bool
OpenSubdivShader::setInternalValueInContext(const MPlug &plug, const MDataHandle &handle, MDGContext &) 
{
    if (plug == aLevel) {
        _hbrMeshDirty = true;
        _level = handle.asLong();
    } else if (plug == aTessFactor) {
        _tessFactor = handle.asLong();
    } else if (plug == aScheme) {
        _hbrMeshDirty = true;
        _scheme = (OsdMeshData::SchemeType)handle.asShort();
    } else if (plug == aKernel) {
        _hbrMeshDirty = true;
        _kernel = (OsdMeshData::KernelType)handle.asShort();
    } else if (plug == aInterpolateBoundary) {
        _hbrMeshDirty = true;
        _interpolateBoundary = (OsdMeshData::InterpolateBoundaryType)handle.asShort();
    } else if (plug == aAdaptive) {
        _hbrMeshDirty = true;
        _adaptiveDirty = true;
        _adaptive = handle.asBool();
 
    } else if (plug == aDiffuseMapFile) {
        _diffuseMapDirty = true;
        _diffuseMapFile = handle.asString();
    } else if (plug == aUVSet) {
        _hbrMeshDirty = true;
        _uvSet = handle.asString();
    } else if (plug == aInterpolateUVBoundary) {
        _hbrMeshDirty = true;
        _interpolateUVBoundary = (OsdMeshData::InterpolateBoundaryType)handle.asShort();

    } else if (plug == aShaderSource) {
        _shaderSourceFilename = handle.asString();
        std::ifstream ifs;
        ifs.open(_shaderSourceFilename.asChar());
        if (ifs.fail()) {
            printf("Using default shader\n");
            _shaderSource.clear();
            _shaderSourceFilename.clear();
        } else {
            printf("Using %s shader\n", _shaderSourceFilename.asChar());
            std::stringstream buffer;
            buffer << ifs.rdbuf();
            _shaderSource = buffer.str();
        }
        ifs.close();
        _shaderSourceDirty = true;
    }

    return false;
}

MStatus
OpenSubdivShader::renderSwatchImage(MImage & image) 
{
    unsigned int width, height;
    image.getSize(width, height);
    unsigned char *p = image.pixels();
    for (unsigned int i = 0; i < width*height; i++) {
        *p++ = 0;
        *p++ = 0;
        *p++ = 0;
        *p++ = 255;
    }
    return MS::kSuccess;
}

// #### updateAttributes
//
//      Called by openSubdivShaderOverride in updateDG.
//      Pulls values for all non-internal attributes.
//
void
OpenSubdivShader::updateAttributes() 
{
    MObject object = thisMObject();
    MFnDependencyNode depFn(object);

    // Retrieve non-internal attributes
    _diffuse = getColor(object, aDiffuse);
    _ambient = getColor(object, aAmbient);
    _specular = getColor(object, aSpecular);

    getAttribute(object, aWireframe, &_wireframe);
    getAttribute(object, aShininess, &_shininess);

    // Pull on any plugs that might be connected
    getAttribute(object, aDiffuseMapFile, &_diffuseMapFile);
}


// #### Main draw method
//
//      Called by OpenSubdivShaderOverride for each renderItem.
//      Binds the vertex buffer and calls glDrawElements for 
//      each patch.
//
void
OpenSubdivShader::draw(const MHWRender::MDrawContext &mDrawContext,
                             OsdMeshData *data)
{
    glPushAttrib(GL_POLYGON_BIT|GL_ENABLE_BIT);

    if (_wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    CHECK_GL_ERROR("draw begin\n");

    // If shader source attribute has changed, update effectRegistry
    updateRegistry();

    // Bind position vertex buffer
    GLuint bPosition = data->bindPositionVBO();
    OpenSubdiv::OsdGLDrawContext *osdDrawContext = data->getDrawContext();

    glBindBuffer(GL_ARRAY_BUFFER, bPosition);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, osdDrawContext->GetPatchIndexBuffer());

    // Get list of patches from OSD
    OpenSubdiv::OsdDrawContext::PatchArrayVector const & patches =
        osdDrawContext->patchArrays;

    // Draw patches
    for (size_t i = 0; i < patches.size(); ++i) {
        OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];

        GLuint program = bindProgram(mDrawContext, osdDrawContext, patch);

        GLuint uniformGregoryQuadOffsetBase =
            glGetUniformLocation(program, "OsdGregoryQuadOffsetBase");
        GLuint uniformPrimitiveIdBase =
            glGetUniformLocation(program, "OsdPrimitiveIdBase");
        glProgramUniform1i(program, uniformGregoryQuadOffsetBase,
                           patch.GetQuadOffsetIndex());
        glProgramUniform1i(program, uniformPrimitiveIdBase,
                           patch.GetPatchIndex());

        GLenum primType = GL_PATCHES;
        if (patch.GetDescriptor().GetType() == OpenSubdiv::FarPatchTables::QUADS) {
            primType = GL_LINES_ADJACENCY;
        } else if (patch.GetDescriptor().GetType() == OpenSubdiv::FarPatchTables::TRIANGLES) {
            primType = GL_TRIANGLES;
        } else if (patch.GetDescriptor().GetType() >= OpenSubdiv::FarPatchTables::REGULAR) {
            glPatchParameteri(GL_PATCH_VERTICES, patch.GetDescriptor().GetNumControlVertices());
        }
        glDrawElements(primType,
                       patch.GetNumIndices(), GL_UNSIGNED_INT,
                       reinterpret_cast<void *>(patch.GetVertIndex() *
                                                sizeof(unsigned int)));

        CHECK_GL_ERROR("post draw\n");
    }

    // Clear state
    glUseProgram(0);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glPopAttrib();

    CHECK_GL_ERROR("draw end\n");
}


// #### bindTexture
//
//      Utility routine which binds a single texture map
//

GLuint 
OpenSubdivShader::bindTexture( const MString& filename, int textureUnit )
{
    GLuint textureId = 0;

    MHWRender::MTexture *mTexture = _theTextureManager->acquireTexture(filename);
    if (mTexture) {
        textureId = *(GLuint*)mTexture->resourceHandle();
        glActiveTexture(GL_TEXTURE0+textureUnit);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
        fprintf(stderr,"Can't read texture file: \"%s\"\n", filename.asChar());
    }
    return textureId;
}

// #### updateRegistry
//
//      Evaluates dirty flags and updates the effect registry if any
//      attributes have changed that require the shader to be rebuilt
//
void
OpenSubdivShader::updateRegistry()
{
    /* If adaptive flag has changed, update the effectRegistry accordingly */
    if (_adaptiveDirty) {
        g_effectRegistry.setIsAdaptive(_adaptive);
        _adaptiveDirty = false;
    }

    MHWRender::MRenderer *theRenderer = MHWRender::MRenderer::theRenderer();
    _theTextureManager = theRenderer->getTextureManager();

    /* If diffuse texture has changed, update the effectRegistry accordingly */
    if (_diffuseMapDirty) {
        GLuint diffMapId = bindTexture( _diffuseMapFile, DIFF_TEXTURE_UNIT );
        g_effectRegistry.setDiffuseId(diffMapId);
        _diffuseMapDirty = false;
    }

    /* If shader source has changed, update the effectRegistry accordingly */
    if (_shaderSourceDirty) {
        if ( _shaderSource.empty() ) {
            if ( g_effectRegistry.getShaderSource() != defaultShaderSource ) {
                g_effectRegistry.setShaderSource(defaultShaderSource);
            }
        } else {
            if ( g_effectRegistry.getShaderSource() != _shaderSource ) {
                g_effectRegistry.setShaderSource(_shaderSource);
            }
        }
        _shaderSourceDirty = false;
    }
}


// #### bindProgram
//
//      Do all the work to build and install shader including
//      set up buffer blocks for uniform variables, set up
//      default lighting parameters, pass material uniforms
//      and bind texture buffers used by texture maps and by
//      OpenSubdiv's built-in shading code.
//
GLuint
OpenSubdivShader::bindProgram(const MHWRender::MDrawContext &     mDrawContext,
                                    OpenSubdiv::OsdGLDrawContext *osdDrawContext,
                              const OpenSubdiv::OsdDrawContext::PatchArray &   patch)
{

    CHECK_GL_ERROR("bindProgram begin\n");

    // Primitives are triangles for Loop subdivision, quads otherwise
    Effect effect = kFill;
    EffectDesc effectDesc( patch.GetDescriptor(), effect );

    // Build shader
    EffectDrawRegistry::ConfigType *
        config = g_effectRegistry.GetDrawConfig(effectDesc);

    // Install shader
    GLuint program = config->program;
    glUseProgram(program);

    // Update and bind transform state
    struct Transform {
        float ModelViewMatrix[16];
        float ProjectionMatrix[16];
        float ModelViewProjectionMatrix[16];
    } transformData;
    setMatrix(mDrawContext.getMatrix(MHWRender::MDrawContext::kWorldViewMtx),
              transformData.ModelViewMatrix);
    setMatrix(mDrawContext.getMatrix(MHWRender::MDrawContext::kProjectionMtx),
              transformData.ProjectionMatrix);
    setMatrix(mDrawContext.getMatrix(MHWRender::MDrawContext::kWorldViewProjMtx),
              transformData.ModelViewProjectionMatrix);

    if (!g_transformUB) {
        glGenBuffers(1, &g_transformUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(transformData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(transformData), &transformData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_transformBinding, g_transformUB);

    // Update and bind tessellation state
    struct Tessellation {
        float TessLevel;
    } tessellationData;

    tessellationData.TessLevel = static_cast<float>(1 << _tessFactor);

    if (!g_tessellationUB) {
        glGenBuffers(1, &g_tessellationUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_tessellationUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(tessellationData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_tessellationUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(tessellationData), &tessellationData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER,
                     g_tessellationBinding,
                     g_tessellationUB);


    // Update and bind lighting state
    int numLights = mDrawContext.numberOfActiveLights();
    struct Lighting {
        struct Light {
            float position[4];
            float diffuse[4];
            float ambient[4];
            float specular[4];
        } lightSource[2];
    } lightingData;
    memset(&lightingData, 0, sizeof(lightingData));

    for (int i = 0; i < numLights && i < 2; ++i) {
        MFloatPointArray positions;
        MFloatVector direction;
        float intensity;
        MColor color;
        bool hasDirection, hasPosition;
        mDrawContext.getLightInformation(i, positions, direction, intensity,
                                    color, hasDirection, hasPosition);

        MMatrix modelView = mDrawContext.getMatrix(MHWRender::MDrawContext::kWorldViewMtx);
        direction = MVector(direction) * modelView;

        Lighting::Light &light = lightingData.lightSource[i];
        if (hasDirection) {
            light.position[0] = -direction[0];
            light.position[1] = -direction[1];
            light.position[2] = -direction[2];

            for (int j = 0; j < 4; ++j) {
                light.diffuse[j] = color[j] * intensity;
                light.ambient[j] = color[j] * intensity;
                light.specular[j] = color[j] * intensity;
            }
        }
    }

    if (!g_lightingUB) {
        glGenBuffers(1, &g_lightingUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_lightingUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(lightingData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_lightingUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(lightingData), &lightingData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_lightingBinding, g_lightingUB);

    // Update other uniforms
    float color[4] = { 0, 0, 0, 1 };
    _diffuse.get(color);
    glProgramUniform4fv(program,
                        glGetUniformLocation(program, "diffuseColor"),
                        1, color);
    _ambient.get(color);
    glProgramUniform4fv(program,
                        glGetUniformLocation(program, "ambientColor"),
                        1, color);
    _specular.get(color);
    glProgramUniform4fv(program,
                        glGetUniformLocation(program, "specularColor"),
                        1, color);
    glProgramUniform1f(program,
                       glGetUniformLocation(program, "shininess"),
                       _shininess);

    // Bind diffuse map
    if (g_effectRegistry.getDiffuseId()!=0) {
        GLint difmap = glGetUniformLocation(program, "diffuseMap");
        glProgramUniform1i(program, difmap, DIFF_TEXTURE_UNIT);
    }

    // Bind all texture buffers
    //      OpenSubdiv's geometric shading code depends on additional 
    //      GL texture buffers. These are managed by the DrawContext 
    //      and must be bound for use by the program in addition to 
    //      any buffers used by the client/application shading code.
    if (osdDrawContext->GetVertexTextureBuffer()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
                      osdDrawContext->GetVertexTextureBuffer());
    }
    if (osdDrawContext->GetVertexValenceTextureBuffer()) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER,
                      osdDrawContext->GetVertexValenceTextureBuffer());
    }
    if (osdDrawContext->GetQuadOffsetsTextureBuffer()) {
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER,
                      osdDrawContext->GetQuadOffsetsTextureBuffer());
    }
    if (osdDrawContext->GetPatchParamTextureBuffer()) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER,
                      osdDrawContext->GetPatchParamTextureBuffer());
    }
    if (osdDrawContext->GetFvarDataTextureBuffer()) {
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER, 
                      osdDrawContext->GetFvarDataTextureBuffer() );
    }

    glActiveTexture(GL_TEXTURE0);

    CHECK_GL_ERROR("bindProgram leave\n");

    return program;
}

