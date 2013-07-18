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


// Consolidated documentation for the source code collectively known
// as `mayaViewer`:
//
//          OpenSubdivShader.h/cpp
//          OpenSubdivShaderOverride.h/cpp
//          osdMeshData.h/cpp
//          hbrUtil.h/cpp
//


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// ### OpenSubdivShader.h
//
// Class definition of custom MPxHwShaderNode for drawing OpenSubdiv patches.
class OpenSubdivShader : public MPxHwShaderNode 
{
    //      Supports Viewport 2.0 rendering only.
    //      geometry/glGeometry, bind/glBind and unbind/glUnbind
    //      are not implemented at this time. 

public:
    /* ... Standard MPxHwShader methods and definitions ... */

private:
    // ##### OSD attributes
    static MObject aLevel;                  // Subdivision level
    static MObject aTessFactor;             // GPU tesselation factor
    static MObject aScheme;                 // Subdivision scheme
    static MObject aAdaptive;               // Feature-adaptive toggle
    static MObject aWireframe;              // Wireframe display toggle
    static MObject aKernel;                 // Computation kernel
    static MObject aInterpolateBoundary;    // Boundary interpolation flag

    // ##### Material attributes
    static MObject aDiffuse;
    static MObject aAmbient;
    static MObject aSpecular;
    static MObject aShininess;

    // ##### Texture attributes
    static MObject aDiffuseMapFile;         // Texture filename
    static MObject aUVSet;                  // UV set (defaults to current UV set)
    static MObject aInterpolateUVBoundary;  // Boundary interpolation flag for face-varying data (UVs)

    // ##### Shader attribute
    static MObject aShaderSource;           // Optional shader file

private:
    /* ... private methods and member variables ... */
};


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// ### OpenSubdivShader.cpp
//
// Custom MPxHwShaderNode for drawing OpenSubdiv patches

/* GLEW needs to be included before OSD and Maya headers */
#include <GL/glew.h>

/* ... System includes ... */
/* ... Maya includes ... */

// `dyu - difference between DrawContext and DrawRegistry?`
//
// `I could take a stab but you can explain this in your sleep right?`
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>


// ##### Attribute declarations
/* MObject OpenSubdivShader::... */

// ##### shaderSource attribute 
//
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
    kQuadFill = 0,
    kQuadLine = 1,
    kTriFill = 2,
    kTriLine = 3,
    kPoint = 4,
};
typedef std::pair<OpenSubdiv::OsdPatchDescriptor, Effect> EffectDesc;

//
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

    // ##### Accessors
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
    // ##### Build shader configuration
    virtual SourceConfigType * _CreateDrawSourceConfig(DescType const & desc);

    // ##### Compile and link the shader
    virtual ConfigType * _CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig);

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


//
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

    if (desc.first.type != OpenSubdiv::kPerVertex) {
        // Per-vertex descriptors are use for uniform refinement
        if (effect == kQuadFill) effect = kTriFill;
        if (effect == kQuadLine) effect = kTriLine;
        sconfig->geometryShader.AddDefine("SMOOTH_NORMALS");

    } else {
        // Configuration for adaptive refinement
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.source = _shaderSource;
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
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
    switch (effect) {
    case kQuadFill:
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kQuadLine:
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_LINE");
        break;
    case kTriFill:
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kTriLine:
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
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

//
// #### _CreateDrawConfig
//
//      Called by base registry when a draw config is requested.
//      Returns a compiled and linked shader program corresponding to 
//      a previously created DrawSourceConfig. The effect registry also 
//      caches these for efficient re-use.
// 
EffectDrawRegistry::ConfigType *
EffectDrawRegistry::_CreateDrawConfig(
        DescType const & desc) 
        SourceConfigType const * sconfig)
{
    ConfigType * config = BaseRegistry::_CreateDrawConfig(desc.first, sconfig);
    assert(config);

    // Assign binding points to uniform blocks
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
    if ((loc = glGetUniformLocation(config->program, "g_VertexBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 0);  // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "g_ValenceBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 1);  // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "g_QuadOffsetBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 2);  // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "g_patchLevelBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 3);  // GL_TEXTURE3
    }
    if ((loc = glGetUniformLocation(config->program, "g_uvFVarBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 4);  // GL_TEXTURE4
    }

    CHECK_GL_ERROR("CreateDrawConfig leave\n");

    return config;
}


// --------------------------------------------------------------------------------------
// ### Shader construction and initialization

OpenSubdivShader::OpenSubdivShader() {...}
OpenSubdivShader::~OpenSubdivShader() {...}
void * OpenSubdivShader::creator() {...}
MStatus OpenSubdivShader::initialize() {...}
void OpenSubdivShader::postConstructor() {...}

MStatus
OpenSubdivShader::compute(const MPlug &, MDataBlock &) 
{
    return MS::kSuccess;
}


//
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


//
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

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, osdDrawContext->patchIndexBuffer);

    // Get list of patches from OSD
    OpenSubdiv::OsdPatchArrayVector const & patches =
        osdDrawContext->patchArrays;

    // Draw patches
    for (size_t i = 0; i < patches.size(); ++i) {
        OpenSubdiv::OsdPatchArray const & patch = patches[i];

        GLint surfaceProgram = bindProgram(mDrawContext, osdDrawContext, patch);

        if (patch.desc.type != OpenSubdiv::kPerVertex) {
            glPatchParameteri(GL_PATCH_VERTICES, patch.patchSize);

            glDrawElements(GL_PATCHES,
                           patch.numIndices, GL_UNSIGNED_INT,
                           reinterpret_cast<void *>(patch.firstIndex *
                                                    sizeof(unsigned int)));
        } else {
            glDrawElements(_scheme == OsdMeshData::kLoop ? GL_TRIANGLES : GL_LINES_ADJACENCY,
                           patch.numIndices, GL_UNSIGNED_INT,
                           reinterpret_cast<void *>(patch.firstIndex *
                                                    sizeof(unsigned int)));
        }
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


//
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

//
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
        _effectRegistry.setIsAdaptive(_adaptive);
        _adaptiveDirty = false;
    }

    MHWRender::MRenderer *theRenderer = MHWRender::MRenderer::theRenderer();
    _theTextureManager = theRenderer->getTextureManager();

    /* If diffuse texture has changed, update the effectRegistry accordingly */
    if (_diffuseMapDirty) {
        GLuint diffMapId = bindTexture( _diffuseMapFile, DIFF_TEXTURE_UNIT );
        _effectRegistry.setDiffuseId(diffMapId);
        _diffuseMapDirty = false;
    }

    /* If shader source has changed, update the effectRegistry accordingly */
    if (_shaderSourceDirty) {
        if ( _shaderSource.empty() ) {
            if ( _effectRegistry.getShaderSource() != defaultShaderSource ) {
                _effectRegistry.setShaderSource(defaultShaderSource);
            }
        } else {
            if ( _effectRegistry.getShaderSource() != _shaderSource ) {
                _effectRegistry.setShaderSource(_shaderSource);
            }
        }
        _shaderSourceDirty = false;
    }
}


//
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
                              const OpenSubdiv::OsdPatchArray &   patch)
{

    CHECK_GL_ERROR("bindProgram begin\n");

    // Primitives are triangles for Loop subdivision, quads otherwise
    Effect effect = (_scheme == OsdMeshData::kLoop) ? kTriFill : kQuadFill;
    EffectDesc effectDesc( patch.desc, effect );

    // Build shader
    EffectDrawRegistry::ConfigType *
        config = _effectRegistry.GetDrawConfig(effectDesc);

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

#define MAX_LEVEL 16

    // Update and bind tessellation state
    struct Tessellation {
        /* XXX: std140 layout requires vec4 aligns for every entry */
        float TessLevelInner[MAX_LEVEL][4];
        float TessLevelOuter[MAX_LEVEL][4];
        int GregoryQuadOffsetBase;
        int PrimitiveIdBase;
    } tessellationData;

    int tessFactor = 2 << _tessFactor;
    for (int i = 0; i < MAX_LEVEL; ++i) {
        tessellationData.TessLevelInner[i][0] = float(std::max(tessFactor >> i, 1));
        tessellationData.TessLevelOuter[i][0] = float(std::max(tessFactor >> i, 1));
    }

    tessellationData.GregoryQuadOffsetBase = patch.GetQuadOffsetIndex();
    tessellationData.PrimitiveIdBase = patch.GetPatchIndex();

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
    if (_effectRegistry.getDiffuseId()!=0) {
        GLint difmap = glGetUniformLocation(program, "diffuseMap");
        glProgramUniform1i(program, difmap, DIFF_TEXTURE_UNIT);
    }

    // Bind all texture buffers
    //
    //      OpenSubdiv's geometric shading code depends on additional 
    //      GL texture buffers. These are managed by the DrawContext 
    //      and must be bound for use by the program in addition to 
    //      any buffers used by the client/application shading code.
    if (osdDrawContext->vertexTextureBuffer) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
                      osdDrawContext->vertexTextureBuffer);
    }
    if (osdDrawContext->vertexValenceTextureBuffer) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER,
                      osdDrawContext->vertexValenceTextureBuffer);
    }
    if (osdDrawContext->quadOffsetTextureBuffer) {
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER,
                      osdDrawContext->quadOffsetTextureBuffer);
    }
    if (osdDrawContext->patchLevelTextureBuffer) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER,
                      osdDrawContext->patchLevelTextureBuffer);
    }
    if (osdDrawContext->fvarDataTextureBuffer) {
        glActiveTexture( GL_TEXTURE4 );
        glBindTexture(  GL_TEXTURE_BUFFER, 
                      osdDrawContext->fvarDataTextureBuffer );
    }

    glActiveTexture(GL_TEXTURE0);

    CHECK_GL_ERROR("bindProgram leave\n");

    return program;
}


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// ### OpenSubdivShaderOverride.h
//
//      Viewport 2.0 override for OpenSubdivShader
//

class OpenSubdivShaderOverride : public MHWRender::MPxShaderOverride 
{
public:
    /* ... Standard MPxShaderOverride methods and definitions ... */
    
private:
    // ##### Pointer to associated shader
    OpenSubdivShader *_shader;

};



// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// ### OpenSubdivShaderOverride.cpp
//
//      Viewport 2.0 override for OpenSubdivShader, implementing
//      custom shading for OpenSubdiv patches.
//

/* Include GLEW before Maya and OSD includes */
#include <GL/glew.h>

/* ... System includes ... */
/* ... Maya includes ... */

// Set up compute controllers for each available compute kernel.
//
// `dyu - what exactly is a compute controller?`
//
#include <osd/cpuComputeController.h>
OpenSubdiv::OsdCpuComputeController *g_cpuComputeController = 0;

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompComputeController.h>
    OpenSubdiv::OsdOmpComputeController *g_ompComputeController = 0;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clComputeController.h>
    cl_context g_clContext;
    cl_command_queue g_clQueue;
    #include "../common/clInit.h"
    OpenSubdiv::OsdCLComputeController *g_clComputeController = 0;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaComputeController.h>
    extern void cudaInit();
    OpenSubdiv::OsdCudaComputeController *g_cudaComputeController = 0;
#endif

OpenSubdivShaderOverride::OpenSubdivShaderOverride(const MObject &obj) {...}
OpenSubdivShaderOverride::~OpenSubdivShaderOverride() {...}
MHWRender::MPxShaderOverride* OpenSubdivShaderOverride::creator(const MObject &obj) {...}


//
// #### attrChangedCB
//
//      Informs us whenever an attribute on the shape node changes.
//      Overkill since we really only want to know if the topology 
//      changes (e.g. an edge crease is added or changed) but Maya 
//      doesn't give us access to a callback that fine-grained. 
//      MMessage::PolyTopologyChangedCallback sounds promising 
//      but only calls back on a single change for any edit 
//      (i.e. not while dragging).
//
/*static*/
void 
OpenSubdivShaderOverride::attrChangedCB(MNodeMessage::AttributeMessage msg, MPlug & plug,
                                        MPlug & otherPlug, void* clientData)
{
    /* We only care if the plug is outMesh and the action is "evaluate" */
    if ( msg & MNodeMessage::kAttributeEval ) {
        OsdMeshData *meshData = (OsdMeshData*)clientData;
        MFnDependencyNode depNodeFn(meshData->getDagPath().node());
        if ( plug == depNodeFn.attribute("outMesh")) {
            meshData->setMeshTopoDirty();
        }
    }
}


//
// #### addTopologyChangedCallbacks
//
//      Add a callback to inform us when topology might be changing 
//      so we can update the HBR mesh accordingly.
//
void
OpenSubdivShaderOverride::addTopologyChangedCallbacks( const MDagPath& dagPath, OsdMeshData *data )
{
    MStatus status = MS::kSuccess;

    // Extract shape node and add callback to let us know when an attribute changes
    MDagPath meshDagPath = dagPath;
    meshDagPath.extendToShape();
    MObject shapeNode = meshDagPath.node();
    MCallbackId id = MNodeMessage::addAttributeChangedCallback(shapeNode, 
                                    attrChangedCB, data, &status );

    // Keep track so we can delete callbacks in destructor
    if ( status ) {
        _callbackIds.append( id );
    } else {
        cerr << "MNodeMessage.addCallback failed" << endl;
    }
}


//
// #### initialize
//
//      Set up vertex buffer descriptors and geometry requirements
//
MString
OpenSubdivShaderOverride::initialize(const MInitContext &initContext,
                                           MInitFeedback &initFeedback)
{
    MString empty;

    // Roundabout way of getting positions pulled into our OsdBufferGenerator
    // where we can manage the VBO memory size.
    //
    // Needs to be re-visited, re-factored, optimized, etc.
    //
    // `dyu - can we add any more explanation about why this is done this way?`
    {
        MHWRender::MVertexBufferDescriptor positionDesc(
            empty,
            MHWRender::MGeometry::kPosition,
            MHWRender::MGeometry::kFloat,
            3);
        addGeometryRequirement(positionDesc);
    }

    {
        MHWRender::MVertexBufferDescriptor positionDesc(
            "osdPosition",
            MHWRender::MGeometry::kTangent,
            MHWRender::MGeometry::kFloat,
            3);
        positionDesc.setSemanticName("osdPosition");
        addGeometryRequirement(positionDesc);
    }

    // Build data object for managing OSD mesh, pass in as custom data
    if (initFeedback.customData == NULL) {
        OsdMeshData *data = new OsdMeshData(initContext.dagPath);
        initFeedback.customData = data;
    }

    // Add a Maya callback so we can rebuild HBR mesh if topology changes
    addTopologyChangedCallbacks( initContext.dagPath, (OsdMeshData*)initFeedback.customData );

    return MString("OpenSubdivShaderOverride");
}

//
// #### updateDG
//
//      Save pointer to shader so we have access down the line.
//      Call shader to update any attributes it needs to.
//
void
OpenSubdivShaderOverride::updateDG(MObject object)
{
    if (object == MObject::kNullObj)
        return;

    // Save pointer to shader for access from draw()
    _shader = static_cast<OpenSubdivShader*>(
        MPxHwShaderNode::getHwShaderNodePtr(object));

    // Get updated attributes from shader
    if (_shader) {
        _shader->updateAttributes();
    }
}

void
OpenSubdivShaderOverride::updateDevice()
{
}

void
OpenSubdivShaderOverride::endUpdate()
{
}

//
// #### Override draw method.  
//
// Setup draw state and call osdMeshData methods to setup 
// and refine geometry.  Call to shader to do actual drawing.
//
bool
OpenSubdivShaderOverride::draw(
    MHWRender::MDrawContext &context,
    const MHWRender::MRenderItemList &renderItemList) const
{
    {
        MHWRender::MStateManager *stateMgr = context.getStateManager();
        static const MDepthStencilState * depthState = NULL;
        if (!depthState) {
            MDepthStencilStateDesc desc;
            depthState = stateMgr->acquireDepthStencilState(desc);
        }
        static const MBlendState *blendState = NULL;
        if (!blendState) {
            MBlendStateDesc desc;

            int ntargets = desc.independentBlendEnable ?
                MHWRender::MBlendState::kMaxTargets : 1;

            for (int i = 0; i < ntargets; ++i) {
                desc.targetBlends[i].blendEnable = false;
            }
            blendState = stateMgr->acquireBlendState(desc);
        }

        stateMgr->setDepthStencilState(depthState);
        stateMgr->setBlendState(blendState);
    }

    for (int i = 0; i < renderItemList.length(); i++) 
    {
        const MHWRender::MRenderItem *renderItem = renderItemList.itemAt(i);
        OsdMeshData *data =
            static_cast<OsdMeshData*>(renderItem->customData());
        if (data == NULL) {
            return false;
        }

        // If attributes or topology have changed which affect 
        // the HBR mesh it will be regenerated here.
        data->rebuildHbrMeshIfNeeded(_shader);

        const MHWRender::MVertexBuffer *position = NULL;
        {
            const MHWRender::MGeometry *geometry = renderItem->geometry();
            for (int i = 0; i < geometry->vertexBufferCount(); i++) {
                const MHWRender::MVertexBuffer *vb = geometry->vertexBuffer(i);
                const MHWRender::MVertexBufferDescriptor &vdesc = vb->descriptor();

                if (vdesc.name() == "osdPosition")
                    position = vb;
            }
        }

        // If HBR mesh was regenerated, rebuild FAR mesh factory
        // and recreate OSD draw context
        data->prepare();

        // Refine geometry
        data->updateGeometry(position);

        // Draw patches
        _shader->draw(context, data);
    }

    return true;
}

// --------------------------------------------------------------------------------------
// #### OsdBufferGenerator
//
// Vertex buffer generator for OpenSubdiv geometry
//

// `dyu - might need some explanation from takahito here`

class OsdBufferGenerator : public MHWRender::MPxVertexBufferGenerator
{
public:
    OsdBufferGenerator() {}
    virtual ~OsdBufferGenerator() {}

    virtual bool getSourceIndexing(
        const MDagPath &dagPath,
        MHWRender::MComponentDataIndexing &sourceIndexing) const
    {
        MFnMesh mesh(dagPath.node());

        MIntArray vertexCount, vertexList;
        mesh.getVertices(vertexCount, vertexList);

        MUintArray &vertices = sourceIndexing.indices();
        for (unsigned int i = 0; i < vertexList.length(); ++i)
            vertices.append((unsigned int)vertexList[i]);

        sourceIndexing.setComponentType(MComponentDataIndexing::kFaceVertex);

        return true;
    }

    virtual bool getSourceStreams(const MDagPath &dagPath,
                                  MStringArray &) const
    {
        return false;
    }

#if MAYA_API_VERSION >= 201350
    virtual void createVertexStream(
        const MDagPath &dagPath, 
              MVertexBuffer &vertexBuffer,
        const MComponentDataIndexing &targetIndexing,
        const MComponentDataIndexing &,
        const MVertexBufferArray &) const
    {
#else
    virtual void createVertexStream(
        const MDagPath &dagPath, MVertexBuffer &vertexBuffer,
        const MComponentDataIndexing &targetIndexing) const
    {
#endif
        const MVertexBufferDescriptor &desc = vertexBuffer.descriptor();

        MFnMesh meshFn(dagPath);
        int nVertices = meshFn.numVertices();
        MFloatPointArray points;
        meshFn.getPoints(points);

#if MAYA_API_VERSION >= 201350
        float *buffer = static_cast<float*>(vertexBuffer.acquire(nVertices, true));
#else
        float *buffer = static_cast<float*>(vertexBuffer.acquire(nVertices));
#endif
        float *dst = buffer;
        for (int i = 0; i < nVertices; ++i) {
            *dst++ = points[i].x;
            *dst++ = points[i].y;
            *dst++ = points[i].z;
        }
        vertexBuffer.commit(buffer);
    }

    static MPxVertexBufferGenerator *positionBufferCreator()
    {
        return new OsdBufferGenerator();
    }
private:
};


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// ### osdMeshData.h
//
// Class definition for OpenSubdiv MUserData used as custom data
// in OpenSubdivShaderOverride
//

// `dyu - hi again, need intelligent descriptions here too!`
//  ` although maybe not if these are described elsewhere`
//  ` getting a bit blurry now`
//

#include <far/meshFactory.h>
#include <osd/glDrawContext.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clGLVertexBuffer.h>
    #include <osd/clComputeContext.h>
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaGLVertexBuffer.h>
    #include <osd/cudaComputeContext.h>
#endif

/* ... Maya includes ... */


class OsdMeshData : public MUserData 
{
...
};


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// ### osdMeshData.cpp
//
//

/* ... Maya includes ... */

// Set up compute controllers for each available compute kernel.
//
// `dyu - what exactly is a compute controller?`
//  ` same decription from above...`
//
#include <osd/cpuDispatcher.h>
#include <osd/cpuComputeController.h>
extern OpenSubdiv::OsdCpuComputeController *g_cpuComputeController;

#ifdef OPENSUBDIV_HAS_OPENMP
#include <osd/ompDispatcher.h>
#include <osd/ompComputeController.h>
extern OpenSubdiv::OsdOmpComputeController *g_ompComputeController;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
#include <osd/clDispatcher.h>
#include <osd/clComputeController.h>
extern cl_context g_clContext;
extern cl_command_queue g_clQueue;
extern OpenSubdiv::OsdCLComputeController *g_clComputeController;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
#include <osd/cudaDispatcher.h>
#include <osd/cudaComputeController.h>
extern OpenSubdiv::OsdCudaComputeController *g_cudaComputeController;
#endif

#include <osd/glDrawContext.h>

//
// #### Constructor
//
//      Initialize all context and buffers to NULL
//
OsdMeshData::OsdMeshData(const MDagPath& meshDagPath) {...}
    : MUserData(false), 
{
    _cpuComputeContext = NULL;
    _cpuPositionBuffer = NULL;

#ifdef OPENSUBDIV_HAS_OPENCL
    _clComputeContext = NULL;
    _clPositionBuffer = NULL;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    _cudaComputeContext = NULL;
    _cudaPositionBuffer = NULL;
#endif
}

//
// #### Destructor
//
//      Delete meshes, clear contexts and buffers
//
OsdMeshData::~OsdMeshData() 
{
    delete _hbrmesh;
    delete _farmesh;
    delete _drawContext;
    delete _fvarDesc;

    clearComputeContextAndVertexBuffer();
}

void
OsdMeshData::clearComputeContextAndVertexBuffer() 
{
    delete _cpuComputeContext;
    _cpuComputeContext = NULL;
    delete _cpuPositionBuffer;
    _cpuPositionBuffer = NULL;

#ifdef OPENSUBDIV_HAS_CUDA
    delete _cudaComputeContext;
    _cudaComputeContext = NULL;
    delete _cudaPositionBuffer;
    _cudaPositionBuffer = NULL;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    delete _clComputeContext;
    _clComputeContext = NULL;
    delete _clPositionBuffer;
    _clPositionBuffer = NULL;
#endif
}


//
// #### buildUVList
//
// Face-varying data expects a list of per-face per-vertex
// floats.  This method reads the UVs from the mesh and 
// concatenates them into such a list.
//
MStatus
OsdMeshData::buildUVList( MFnMesh& meshFn, std::vector<float>& uvList )
{
    MStatus status = MS::kSuccess;

    MItMeshPolygon polyIt( _meshDagPath );

    MFloatArray uArray;
    MFloatArray vArray;

    // If user hasn't given us a UV set, use the current one
    MString *uvSetPtr=NULL;
    if ( _uvSet.numChars() > 0 ) {
        if (uvSetNameIsValid(meshFn, _uvSet)) {
            uvSetPtr = &_uvSet;
        }
        else {
            MGlobal::displayWarning(MString("OpenSubdivShader:  uvSet \""+_uvSet+"\" does not exist."));
        }
    } else {
        uvSetPtr = NULL;
    }

    // Pull UVs from Maya mesh
    status = meshFn.getUVs( uArray, vArray, uvSetPtr );
    MCHECK_RETURN(status, "OpenSubdivShader: Error reading UVs");

    if ( uArray.length() == 0 || vArray.length() == 0 )
    {
        MGlobal::displayWarning("OpenSubdivShader: Mesh has no UVs");
        return MS::kFailure;
    }

    // Initalize list of UV values
    uvList.clear();
    uvList.resize( meshFn.numFaceVertices()*2 );
    int uvListIdx = 0;

    // For each face-vertex copy UVs into list, adjusting for renderman orientation
    for ( polyIt.reset(); !polyIt.isDone(); polyIt.next() ) 
    { 
        int          faceIdx      = polyIt.index(); 
        unsigned int numPolyVerts = polyIt.polygonVertexCount();

        for ( unsigned int faceVertIdx = 0; 
                           faceVertIdx < numPolyVerts; 
                           faceVertIdx++ )
        {
            int uvIdx;
            polyIt.getUVIndex( faceVertIdx, uvIdx, uvSetPtr );
            // convert maya UV orientation to renderman orientation
            uvList[ uvListIdx++ ] = uArray[ uvIdx ];
            uvList[ uvListIdx++ ] = 1.0f - vArray[ uvIdx ];
        }
    }

    return status;
}


//
// #### rebuildHbrMeshIfNeeded
//
//      If the topology of the mesh changes, or any attributes that affect
//      how the mesh is computed the original HBR needs to be rebuilt
//      which will trigger a rebuild of the FAR mesh and subsequent buffers.
//
void
OsdMeshData::rebuildHbrMeshIfNeeded(OpenSubdivShader *shader)
{
    MStatus status = MS::kSuccess;

    if (!_meshTopoDirty && !shader->getHbrMeshDirty())
        return;

    MFnMesh meshFn(_meshDagPath);

    // Cache attribute values
    _level      = shader->getLevel();
    _kernel     = shader->getKernel();
    _adaptive   = shader->isAdaptive();
    _uvSet      = shader->getUVSet();

    int level = (_level < 1) ? 1 : _level;

    // Get Maya vertex topology and crease data
    MIntArray vertexCount;
    MIntArray vertexList;
    meshFn.getVertices(vertexCount, vertexList);

    MUintArray edgeIds;
    MDoubleArray edgeCreaseData;
    meshFn.getCreaseEdges(edgeIds, edgeCreaseData);

    MUintArray vtxIds;
    MDoubleArray vtxCreaseData;
    meshFn.getCreaseVertices(vtxIds, vtxCreaseData);

    if (vertexCount.length() == 0) return;

    // Copy Maya vectors into std::vectors
    std::vector<int> numIndices(&vertexCount[0], &vertexCount[vertexCount.length()]);
    std::vector<int> faceIndices(&vertexList[0], &vertexList[vertexList.length()]);
    std::vector<int> vtxCreaseIndices(&vtxIds[0], &vtxIds[vtxIds.length()]);
    std::vector<double> vtxCreases(&vtxCreaseData[0], &vtxCreaseData[vtxCreaseData.length()]);
    std::vector<double> edgeCreases(&edgeCreaseData[0], &edgeCreaseData[edgeCreaseData.length()]);

    // Edge crease index is stored as pairs of vertex ids
    int nEdgeIds = edgeIds.length();
    std::vector<int> edgeCreaseIndices;
    edgeCreaseIndices.resize(nEdgeIds*2);
    for (int i = 0; i < nEdgeIds; ++i) {
        int2 vertices;
        status = meshFn.getEdgeVertices(edgeIds[i], vertices);
        if (status.error()) {
            MERROR(status, "OpenSubdivShader: Can't get edge vertices");
            continue;
        }
        edgeCreaseIndices[i*2] = vertices[0];
        edgeCreaseIndices[i*2+1] = vertices[1];
    }

    // Convert attribute enums to HBR enums (this is why the enums need to match)
    HbrMeshUtil::SchemeType hbrScheme = (HbrMeshUtil::SchemeType) shader->getScheme();
    OsdHbrMesh::InterpolateBoundaryMethod hbrInterpBoundary = 
            (OsdHbrMesh::InterpolateBoundaryMethod) shader->getInterpolateBoundary();
    OsdHbrMesh::InterpolateBoundaryMethod hbrInterpUVBoundary = 
            (OsdHbrMesh::InterpolateBoundaryMethod) shader->getInterpolateUVBoundary();


    // Clear any existing face-varying descriptor
    if (_fvarDesc) {
        delete _fvarDesc;
        _fvarDesc = NULL;
    }

    // Read UV data from maya and build per-face per-vert list of UVs for HBR face-varying data
    std::vector< float > uvList;
    status = buildUVList( meshFn, uvList );
    if (! status.error()) {
        // Create face-varying data descriptor.  The memory required for indices
        // and widths needs to stay alive as the HBR library only takes in the
        // pointers and assumes the client will maintain the memory so keep _fvarDesc
        // around as long as _hbrmesh is around.
        int fvarIndices[] = { 0, 1 };
        int fvarWidths[] = { 1, 1 };
        _fvarDesc = new FVarDataDesc( 2, fvarIndices, fvarWidths, 2, hbrInterpUVBoundary );
    }

    if (_fvarDesc && hbrScheme != HbrMeshUtil::kCatmark) {
        MGlobal::displayWarning("Face-varying not yet supported for Loop/Bilinear, using Catmull-Clark");
        hbrScheme = HbrMeshUtil::kCatmark;
    }

    // Convert Maya mesh to internal HBR representation
    _hbrmesh = ConvertToHBR(meshFn.numVertices(), numIndices, faceIndices,
                            vtxCreaseIndices, vtxCreases,
                            std::vector<int>(), std::vector<float>(),
                            edgeCreaseIndices, edgeCreases,
                            hbrInterpBoundary, 
                            hbrScheme,
                            false,                      // no ptex
                            _fvarDesc, 
                            _fvarDesc?&uvList:NULL);    // yes fvar (if have UVs)

    /* note: GL function can't be used in prepareForDraw API. */
    // Set dirty flag for FAR mesh factory to regenerate
    _needsInitializeMesh = true;

    // Mesh topology data is up to date
    _meshTopoDirty = false;
    shader->setHbrMeshDirty(false);
}


//
// #### initializeMesh
//
//  `dyu - some juicy details about mesh factory and what a compute context is`
//
void
OsdMeshData::initializeMesh() 
{
    if (!_hbrmesh)
        return;

    // Create FAR mesh
    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex>
        meshFactory(_hbrmesh, _level, _adaptive);

    _farmesh = meshFactory.Create(false,  /* ptex coords */
                                  true);  /* fvar data */

    delete _hbrmesh;
    _hbrmesh = NULL;

    int numTotalVertices = _farmesh->GetNumVertices();

    // Create context and vertex buffer
    clearComputeContextAndVertexBuffer();

    if (_kernel == kCPU) {
        _cpuComputeContext = OpenSubdiv::OsdCpuComputeContext::Create(_farmesh);
        _cpuPositionBuffer = OpenSubdiv::OsdCpuGLVertexBuffer::Create(3, numTotalVertices);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (_kernel == kOPENMP) {
        _cpuComputeContext = OpenSubdiv::OsdCpuComputeContext::Create(_farmesh);
        _cpuPositionBuffer = OpenSubdiv::OsdCpuGLVertexBuffer::Create(3, numTotalVertices);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (_kernel == kCUDA) {
        _cudaComputeContext = OpenSubdiv::OsdCudaComputeContext::Create(_farmesh);
        _cudaPositionBuffer = OpenSubdiv::OsdCudaGLVertexBuffer::Create(3, numTotalVertices);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (_kernel == kCL) {
        _clComputeContext = OpenSubdiv::OsdCLComputeContext::Create(_farmesh, g_clContext);
        _clPositionBuffer = OpenSubdiv::OsdCLGLVertexBuffer::Create(3, numTotalVertices,
                                                                    g_clContext);
#endif
    }

    _needsInitializeMesh = false;

    // Get geometry from maya mesh
    MFnMesh meshFn(_meshDagPath);
    meshFn.getPoints(_pointArray);

    _needsUpdate = true;
}

//
// #### createKernelDrawContext
//
//  `dyu - more juicy details`
//
void
OsdMeshData::createKernelDrawContext() 
{
    delete _drawContext;

    if (_kernel == kCPU) {
        _drawContext = OpenSubdiv::OsdGLDrawContext::Create(_farmesh,
                                                            _cpuPositionBuffer,
                                                            false,  // ptex coords?
                                                            true);  // fvar data?
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (_kernel == kOPENMP) {
        _drawContext = OpenSubdiv::OsdGLDrawContext::Create(_farmesh,
                                                            _cpuPositionBuffer,
                                                            false,  true);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (_kernel == kCUDA) {
        _drawContext = OpenSubdiv::OsdGLDrawContext::Create(_farmesh,
                                                            _cudaPositionBuffer,
                                                            false,  true);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (_kernel == kCL) {
        _drawContext = OpenSubdiv::OsdGLDrawContext::Create(_farmesh,
                                                            _clPositionBuffer,
                                                            false,  true);
#endif
    } else {
        assert(false);
    }
}

//
// #### prepare
//
//  If HBR mesh has been rebuilt the mesh factory and draw context
//  also need to be rebuilt.
//
//  `dyu - it seems like this could be merged with rebuildHBR`
//
void
OsdMeshData::prepare() 
{
    if (_needsInitializeMesh) {
        initializeMesh();
        createKernelDrawContext();
    }
}

//
// #### updateGeometry
//
//  Refine the geometry.
//
//  `dyu - probably need more detail... do they need to know about the bind/subdata copy?`
//
void
OsdMeshData::updateGeometry(const MHWRender::MVertexBuffer *points)
{
    int nCoarsePoints = _pointArray.length();

    GLuint mayaPositionVBO = *static_cast<GLuint*>(points->resourceHandle());
    int size = nCoarsePoints * 3 * sizeof(float);

    if (_kernel == kCPU || _kernel == kOPENMP) {
        float *d_pos = _cpuPositionBuffer->BindCpuBuffer();
        glBindBuffer(GL_ARRAY_BUFFER, mayaPositionVBO);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, size, d_pos);
        g_cpuComputeController->Refine(_cpuComputeContext, _cpuPositionBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

#ifdef OPENSUBDIV_HAS_CUDA
    } else if (_kernel == kCUDA) {
        glBindBuffer(GL_COPY_READ_BUFFER, mayaPositionVBO);
        glBindBuffer(GL_COPY_WRITE_BUFFER, _cudaPositionBuffer->BindVBO());
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                            0, 0, size);
        g_cudaComputeController->Refine(_cudaComputeContext, _cudaPositionBuffer);

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (_kernel == kCL) {
        glBindBuffer(GL_COPY_READ_BUFFER, mayaPositionVBO);
        glBindBuffer(GL_COPY_WRITE_BUFFER, _clPositionBuffer->BindVBO());
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                            0, 0, size);
        g_clComputeController->Refine(_clComputeContext, _clPositionBuffer);

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
#endif
    }

    _needsUpdate = false;
}


//
// #### bindPositionVBO
// 
// Wrapper to bind position the correct vertex buffer
// according to the current compute kernel
//
GLuint
OsdMeshData::bindPositionVBO() 
{
    if (_kernel == kCPU) {
        return _cpuPositionBuffer->BindVBO();
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (_kernel == kOPENMP) {
        return _cpuPositionBuffer->BindVBO();
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (_kernel == kCUDA) {
        return _cudaPositionBuffer->BindVBO();
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (_kernel == kCL) {
        return _clPositionBuffer->BindVBO();
#endif
    }
    return 0;
}


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// ### hbrUtil.h
//

//
// #### Face-varying data descriptor
//
//      Wrapper for basic information needed to request
//      face-varying data allocation from HBR
//
class FVarDataDesc
{
public:

    // Must be instantiated with descriptor information
    FVarDataDesc(      int   count, 
                  const int *indices,   // start index for each face-varying variable
                  const int *widths,    // width for each face-varying variable
                        int  width,
                        OsdHbrMesh::InterpolateBoundaryMethod 
                             boundary=OsdHbrMesh::k_InterpolateBoundaryNone
                        ) 
    {
        _fvarCount      = count; 
        _totalfvarWidth = width;
        _fvarIndices.assign( indices, indices+count );
        _fvarWidths.assign( widths, widths+count );
        _interpBoundary = boundary;
    }

    ~FVarDataDesc() {}


    // ##### Accessors
          int  getCount() const          { return _fvarCount; }
    const int *getIndices() const        { return &_fvarIndices.front(); }
    const int *getWidths() const         { return &_fvarWidths.front(); }
          int  getTotalWidth() const     { return _totalfvarWidth; }
    OsdHbrMesh::InterpolateBoundaryMethod 
               getInterpBoundary() const { return _interpBoundary; }

private:

    // ##### Face-varying data parameters
    int  _fvarCount;                // Number of facevarying datums
    std::vector<int> _fvarIndices;  // Start indices of the facevarying data 
    std::vector<int> _fvarWidths;   // Individual widths of facevarying data
    int  _totalfvarWidth;           // Total widths of the facevarying data

    // ##### Boundary interpolation
    OsdHbrMesh::InterpolateBoundaryMethod _interpBoundary;
};


extern "C" OsdHbrMesh * ConvertToHBR( ... );

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// ### hbrUtil.cpp

// 
// #### ConvertToHBR
//
//      Take Maya mesh topology input and build an HBR mesh.  Optional
//      input parameters direct the method to create Ptex texture
//      coordinates or to copy given face-varying data to the 
//      HBR faces as they are created.
//
OsdHbrMesh * 
ConvertToHBR( int nVertices,
              std::vector<int>   const & faceVertCounts,
              std::vector<int>   const & faceIndices,
              std::vector<int>   const & vtxCreaseIndices,
              std::vector<double> const & vtxCreases,
              std::vector<int>   const & edgeCrease1Indices,  // face index, local edge index
              std::vector<float> const & edgeCreases1,
              std::vector<int>   const & edgeCrease2Indices,  // 2 vertex indices (Maya friendly)
              std::vector<double> const & edgeCreases2,
              OsdHbrMesh::InterpolateBoundaryMethod interpBoundary,
              HbrMeshUtil::SchemeType scheme,
              bool usingPtex,                       // defaults to false
              FVarDataDesc const * fvarDesc,        // defaults to NULL
              std::vector<float> const * fvarData   // defaults to NULL
            )
{
    static OpenSubdiv::HbrBilinearSubdivision<OpenSubdiv::OsdVertex> _bilinear;
    static OpenSubdiv::HbrLoopSubdivision<OpenSubdiv::OsdVertex> _loop;
    static OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex> _catmark;

    // Build HBR mesh with/without face varying data, according to input data.
    // If a face-varying descriptor is passed in its memory needs to stay 
    // alive as long as this hbrMesh is alive (for indices and widths arrays). 
    OsdHbrMesh *hbrMesh;
    if ( fvarDesc )
    {
        if (scheme == HbrMeshUtil::kCatmark)
            hbrMesh = new OsdHbrMesh(&_catmark,  fvarDesc->getCount(), 
                                                 fvarDesc->getIndices(), 
                                                 fvarDesc->getWidths(), 
                                                 fvarDesc->getTotalWidth());
        else if (scheme == HbrMeshUtil::kLoop)
            hbrMesh = new OsdHbrMesh(&_loop,     fvarDesc->getCount(), 
                                                 fvarDesc->getIndices(), 
                                                 fvarDesc->getWidths(), 
                                                 fvarDesc->getTotalWidth());
        else 
            hbrMesh = new OsdHbrMesh(&_bilinear, fvarDesc->getCount(),
                                                 fvarDesc->getIndices(), 
                                                 fvarDesc->getWidths(), 
                                                 fvarDesc->getTotalWidth());
    }
    else
    {
        if (scheme == HbrMeshUtil::kCatmark)
            hbrMesh = new OsdHbrMesh(&_catmark);
        else if (scheme == HbrMeshUtil::kLoop)
            hbrMesh = new OsdHbrMesh(&_loop);
        else
            hbrMesh = new OsdHbrMesh(&_bilinear);
    }


    // Create empty verts: actual vertices initialized in UpdatePoints();
    OpenSubdiv::OsdVertex v;
    for (int i = 0; i < nVertices; ++i) {
        hbrMesh->NewVertex(i, v);
    }

    std::vector<int> vIndex;
    int nFaces = (int)faceVertCounts.size();
    int fvcOffset = 0;          // face-vertex count offset
    int ptxIdx = 0;

    // Collect vertex indices for each face and create HBR face
    for (int fi = 0; fi < nFaces; ++fi) 
    {
        int nFaceVerts = faceVertCounts[fi];
        vIndex.resize(nFaceVerts);

        bool valid = true;
        for (int fvi = 0; fvi < nFaceVerts; ++fvi) 
        {
            vIndex[fvi] = faceIndices[fvi + fvcOffset];
            int vNextIndex = faceIndices[(fvi+1) % nFaceVerts + fvcOffset];

            // Check for non-manifold face
            OsdHbrVertex * origin = hbrMesh->GetVertex(vIndex[fvi]);
            OsdHbrVertex * destination = hbrMesh->GetVertex(vNextIndex);
            if (!origin || !destination) {
                OSD_ERROR("ERROR : An edge was specified that connected a nonexistent vertex");
                valid = false;
            }

            if (origin == destination) {
                OSD_ERROR("ERROR : An edge was specified that connected a vertex to itself");
                valid = false;
            }

            OsdHbrHalfedge * opposite = destination->GetEdge(origin);
            if (opposite && opposite->GetOpposite()) {
                OSD_ERROR("ERROR : A non-manifold edge incident to more than 2 faces was found");
                valid = false;
            }

            if (origin->GetEdge(destination)) {
                OSD_ERROR("ERROR : An edge connecting two vertices was specified more than once. "
                          "It's likely that an incident face was flipped");
                valid = false;
            }
        }

        if ( valid ) 
        {
            if (scheme == HbrMeshUtil::kLoop) {
                // For Loop subdivision, triangulate from vertex indices
                int triangle[3];
                triangle[0] = vIndex[0];
                for (int fvi = 2; fvi < nFaceVerts; ++fvi) {
                    triangle[1] = vIndex[fvi-1];
                    triangle[2] = vIndex[fvi];
                    hbrMesh->NewFace(3, triangle, 0);
                }
                /* ptex not fully implemented for loop, yet */
                /* fvar not fully implemented for loop, yet */

            } else {

                // For Catmull-Clark subdivision, create a quad face from vertices
                /* bilinear subdivision not fully implemented */
                OsdHbrFace *face = hbrMesh->NewFace(nFaceVerts, &(vIndex[0]), 0);

                if (usingPtex) {
                    // Ptex textures will be used, set up ptex coordinates
                    face->SetPtexIndex(ptxIdx);
                    ptxIdx += (nFaceVerts == 4) ? 1 : nFaceVerts;
                }

                if (fvarData) {
                    // Face-varying data has been passed in, get pointer to data
                    int fvarWidth = hbrMesh->GetTotalFVarWidth();
                    const float *faceData = &(*fvarData)[ fvcOffset*fvarWidth ];

                    // For each face vertex copy fvar data into hbr mesh
                    for(int fvi=0; fvi<nFaceVerts; ++fvi)
                    {
                        OsdHbrVertex *v = hbrMesh->GetVertex( vIndex[fvi] );
                        OsdHbrFVarData& fvarData = v->GetFVarData(face);
                        if ( ! fvarData.IsInitialized() )
                        {
                            fvarData.SetAllData( fvarWidth, faceData );
                        }
                        else if (!fvarData.CompareAll(fvarWidth, faceData))
                        {
                            // If data exists for this face vertex, but is different
                            // (e.g. we're on a UV seam) create another fvar datum
                            OsdHbrFVarData& fvarData = v->NewFVarData(face);
                            fvarData.SetAllData( fvarWidth, faceData );
                        }

                        // Advance pointer to next set of face-varying data
                        faceData += fvarWidth;
                    }
                }
            }
        } else {
            OSD_ERROR("Face %d will be ignored\n", fi);
        }

        fvcOffset += nFaceVerts;
    }

    // Assign boundary interpolation methods
    hbrMesh->SetInterpolateBoundaryMethod(interpBoundary);
    if ( fvarDesc ) 
        hbrMesh->SetFVarInterpolateBoundaryMethod(fvarDesc->getInterpBoundary());

    // Set edge crease in two different indexing way
    size_t nEdgeCreases = edgeCreases1.size();
    for (size_t i = 0; i < nEdgeCreases; ++i) {
        if (edgeCreases1[i] <= 0.0)
            continue;

        OsdHbrHalfedge * e = hbrMesh->
            GetFace(edgeCrease1Indices[i*2])->
            GetEdge(edgeCrease1Indices[i*2+1]);

        if (!e) {
            OSD_ERROR("Can't find edge (face %d edge %d)\n",
                      edgeCrease1Indices[i*2], edgeCrease1Indices[i*2+1]);
            continue;
        }
        e->SetSharpness(static_cast<float>(edgeCreases1[i]));
    }
    nEdgeCreases = edgeCreases2.size();
    for (size_t i = 0; i < nEdgeCreases; ++i) {
        if (edgeCreases2[i] <= 0.0)
            continue;

        OsdHbrVertex * v0 = hbrMesh->GetVertex(edgeCrease2Indices[i*2]);
        OsdHbrVertex * v1 = hbrMesh->GetVertex(edgeCrease2Indices[i*2+1]);
        OsdHbrHalfedge * e = NULL;

        if (v0 && v1)
            if (!(e = v0->GetEdge(v1)))
                e = v1->GetEdge(v0);
        if (!e) {
            OSD_ERROR("ERROR can't find edge");
            continue;
        }
        e->SetSharpness(static_cast<float>(edgeCreases2[i]));
    }

    // Set corner
    {
        size_t nVertexCreases = vtxCreases.size();
        for (size_t i = 0; i < nVertexCreases; ++i) {
            if (vtxCreases[i] <= 0.0)
                continue;
            OsdHbrVertex * v = hbrMesh->GetVertex(vtxCreaseIndices[i]);
            if (!v) {
                OSD_ERROR("Can't find vertex %d\n", vtxCreaseIndices[i]);
                continue;
            }
            v->SetSharpness(static_cast<float>(vtxCreases[i]));
        }
    }

    // Call finish to complete build of HBR mesh
    hbrMesh->Finish();

    return hbrMesh;
}


// `dyu:  do we need to have one of these dudes for shader.glsl?`
