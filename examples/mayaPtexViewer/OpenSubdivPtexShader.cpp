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

#if not defined(__APPLE__)
    #include <GL/glew.h>
    #if defined(WIN32)
        #include <GL/wglew.h>
    #endif
#endif

#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MDrawContext.h>

#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>
#include <osd/glPtexTexture.h>

#include <Ptexture.h>
#include <PtexUtils.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "../common/maya_util.h"
#include "OpenSubdivPtexShader.h"
#include "osdPtexMeshData.h"

// Identifiers
MTypeId OpenSubdivPtexShader::id(0x88111);
MString OpenSubdivPtexShader::drawRegistrantId("OpenSubdivPtexShaderPlugin");

// Attributes
MObject OpenSubdivPtexShader::aLevel;
MObject OpenSubdivPtexShader::aTessFactor;
MObject OpenSubdivPtexShader::aScheme;
MObject OpenSubdivPtexShader::aKernel;
MObject OpenSubdivPtexShader::aInterpolateBoundary;
MObject OpenSubdivPtexShader::aAdaptive;
MObject OpenSubdivPtexShader::aWireframe;

MObject OpenSubdivPtexShader::aDiffuse;
MObject OpenSubdivPtexShader::aAmbient;
MObject OpenSubdivPtexShader::aSpecular;

// Ptex texture attributes
MObject OpenSubdivPtexShader::aDiffuseEnvironmentMapFile;
MObject OpenSubdivPtexShader::aSpecularEnvironmentMapFile;
MObject OpenSubdivPtexShader::aColorFile;
MObject OpenSubdivPtexShader::aDisplacementFile;
MObject OpenSubdivPtexShader::aOcclusionFile;

MObject OpenSubdivPtexShader::aEnableDisplacement;
MObject OpenSubdivPtexShader::aEnableColor;
MObject OpenSubdivPtexShader::aEnableOcclusion;
MObject OpenSubdivPtexShader::aEnableNormal;

MObject OpenSubdivPtexShader::aFresnelBias;
MObject OpenSubdivPtexShader::aFresnelScale;
MObject OpenSubdivPtexShader::aFresnelPower;

// Shader
MObject OpenSubdivPtexShader::aShaderSource;

static const char *defaultShaderSource =
#include "shader.inc"
;


// --------------------------------------------------------------------------------------
//          
//      Override of OpenSubdiv::OsdGLDrawRegistry 
//
// --------------------------------------------------------------------------------------
struct Effect 
{
    bool color;
    bool occlusion;
    bool displacement;
    bool normal;

    bool operator < (const Effect &e) const 
    {
        // no precedence, just need to know if the two are not equal
        return (color == false && e.color != false) || 
                  (color == e.color && (occlusion == false && e.occlusion != false) || 
                      (occlusion == e.occlusion && (displacement == false && e.displacement != false) ||
                          (displacement == e.displacement && (normal == false && e.normal != false))));

    }
};

typedef std::pair<OpenSubdiv::OsdPatchDescriptor, Effect> EffectDesc;

class EffectDrawRegistry : public OpenSubdiv::OsdGLDrawRegistry<EffectDesc> 
{

public:
    EffectDrawRegistry() : _isAdaptive(false),
                           _ptexColorValid(false),
                           _ptexDisplacementValid(false),
                           _ptexOcclusionValid(false),
                           _diffuseEnvironmentId(0),
                           _shaderSource( defaultShaderSource )
            {}

    // isAdaptive
    void setIsAdaptive( bool ad ) { resetIfChanged(ad, _isAdaptive); }
    bool getIsAdaptive() const { return _isAdaptive; }

    // ptexColorValid
    void setPtexColorValid( bool pcv ) { resetIfChanged(pcv, _ptexColorValid); }
    bool getPtexColorValid() const { return _ptexColorValid; }

    // ptexDisplacementValid
    void setPtexDisplacementValid( bool pdv ) { resetIfChanged(pdv, _ptexDisplacementValid); }
    bool getPtexDisplacementValid() const { return _ptexDisplacementValid; }

    // ptexOcclusionValid
    void setPtexOcclusionValid( bool pov ) { resetIfChanged(pov, _ptexOcclusionValid); }
    bool getPtexOcclusionValid() const { return _ptexOcclusionValid; }

    // diffuseEnvironmentId
    void setDiffuseEnvironmentId( GLuint dId ) { resetIfChanged(dId, _diffuseEnvironmentId); }
    GLuint getDiffuseEnvironmentId() const { return _diffuseEnvironmentId; }

    // specularEnvironmentId
    void setSpecularEnvironmentId( GLuint eId ) { resetIfChanged(eId, _specularEnvironmentId); }
    GLuint getSpecularEnvironmentId() const { return _specularEnvironmentId; }

    // shaderSource
    void setShaderSource( std::string const & src ) { resetIfChanged(src, _shaderSource); }
    std::string const & getShaderSource() const { return _shaderSource; }

protected:
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig);

    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc);

private:

    // convenience method
    template< typename T>
    void resetIfChanged( T newVal, T& curVal )
    {
        if ( newVal != curVal ) {
            curVal = newVal;
            Reset();
        }
    }

    // parameters
    bool        _isAdaptive;
    bool        _ptexColorValid;
    bool        _ptexDisplacementValid;
    bool        _ptexOcclusionValid;
    GLuint      _diffuseEnvironmentId;
    GLuint      _specularEnvironmentId;
    std::string _shaderSource;
};


EffectDrawRegistry::SourceConfigType *
EffectDrawRegistry::_CreateDrawSourceConfig(DescType const & desc) 
{
    Effect effect = desc.second;

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first);
    sconfig->commonShader.AddDefine("USE_PTEX_COORD");           // used by built-in OSD shaders
    sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");    // used by built-in OSD shaders
    sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");

    bool quad = true;
    if (desc.first.type != OpenSubdiv::kNonPatch) {
        quad = false;
        sconfig->tessEvalShader.source = _shaderSource +
            sconfig->tessEvalShader.source;
        sconfig->tessEvalShader.version = "#version 410\n";

        if (_ptexDisplacementValid) {
            if (effect.displacement) {
                sconfig->tessEvalShader.AddDefine("USE_PTEX_DISPLACEMENT");
            }
            if (effect.normal) {
                sconfig->fragmentShader.AddDefine("USE_PTEX_NORMAL");
            } else if (effect.displacement) {
                sconfig->geometryShader.AddDefine("FLAT_NORMALS");
            }
        }

    } else {
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.source = _shaderSource;
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");

        if (effect.displacement && _ptexDisplacementValid) {
            sconfig->geometryShader.AddDefine("USE_PTEX_DISPLACEMENT");
            sconfig->geometryShader.AddDefine("FLAT_NORMALS");
        }
    }
    assert(sconfig);

    sconfig->vertexShader.AddDefine("NUM_ELEMENTS", "3");

    sconfig->geometryShader.version = "#version 410\n";
    sconfig->geometryShader.source = _shaderSource;
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    sconfig->fragmentShader.version = "#version 410\n";
    sconfig->fragmentShader.source = _shaderSource;
    sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");

    if (quad) {
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
    } else {
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
    }

    if (effect.color && _ptexColorValid) {
        sconfig->fragmentShader.AddDefine("USE_PTEX_COLOR");
    }
    if (effect.occlusion && _ptexOcclusionValid) {
        sconfig->fragmentShader.AddDefine("USE_PTEX_OCCLUSION");
    }
    if (_diffuseEnvironmentId != 0) {
        sconfig->fragmentShader.AddDefine("USE_DIFFUSE_ENV_MAP");
    }
    if (_specularEnvironmentId != 0) {
        sconfig->fragmentShader.AddDefine("USE_SPECULAR_ENV_MAP");
    }

    return sconfig;
}


// XXX can/should these be in the registry instead of global?
GLuint g_transformUB = 0,
       g_transformBinding = 0,
       g_tessellationUB = 0,
       g_tessellationBinding = 0,
       g_lightingUB = 0,
       g_lightingBinding = 0;

EffectDrawRegistry::ConfigType *
EffectDrawRegistry::_CreateDrawConfig(
        DescType const & desc,
        SourceConfigType const * sconfig) 
{
    ConfigType * config = BaseRegistry::_CreateDrawConfig(desc.first, sconfig);
    assert(config);

    // XXXdyu can use layout(binding=) with GLSL 4.20 and beyond
    g_transformBinding = 0;
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Transform"),
        g_transformBinding);

    g_tessellationBinding = 1;
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Tessellation"),
        g_tessellationBinding);

    g_lightingBinding = 2;
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Lighting"),
        g_lightingBinding);
    CHECK_GL_ERROR("CreateDrawConfig B \n");

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

    CHECK_GL_ERROR("CreateDrawConfig leave\n");

    return config;
}

// XXX could be a singleton
EffectDrawRegistry effectRegistry;
// --------------------------------------------------------------------------------------



// --------------------------------------------------------------------------------------

OpenSubdivPtexShader::OpenSubdivPtexShader()
    : _level(3),
      _tessFactor(2),
      _adaptive(true),
      _wireframe(false),
      _scheme(OsdPtexMeshData::kCatmark),
      _kernel(OsdPtexMeshData::kCPU),
      _interpolateBoundary(OsdPtexMeshData::kInterpolateBoundaryNone),
      _enableColor(true),
      _enableDisplacement(true),
      _enableOcclusion(true),
      _enableNormal(true),
      _ptexColor(NULL), 
      _ptexDisplacement(NULL), 
      _ptexOcclusion(NULL),
      _shaderSource( defaultShaderSource ),
      _adaptiveDirty(false),
      _diffEnvMapDirty(true),
      _specEnvMapDirty(true),
      _ptexColorDirty(true),
      _ptexDisplacementDirty(true),
      _ptexOcclusionDirty(true),
      _shaderSourceDirty(false)
{
}

OpenSubdivPtexShader::~OpenSubdivPtexShader() 
{
    if (_ptexColor)         delete _ptexColor;
    if (_ptexDisplacement)  delete _ptexDisplacement;
    if (_ptexOcclusion)     delete _ptexOcclusion;
}

void *
OpenSubdivPtexShader::creator() 
{
    return new OpenSubdivPtexShader();
}

MStatus
OpenSubdivPtexShader::initialize() 
{
    MFnTypedAttribute typedAttr;
    MFnNumericAttribute numAttr;
    MFnEnumAttribute enumAttr;

    // level
    aLevel = numAttr.create("level", "lv", MFnNumericData::kLong, 3);
    numAttr.setInternal(true);
    numAttr.setMin(1);
    numAttr.setSoftMax(5);
    numAttr.setMax(10);

    // tessFactor
    aTessFactor = numAttr.create("tessFactor", "tessf", MFnNumericData::kLong, 2);
    numAttr.setInternal(true);
    numAttr.setMin(1);
    numAttr.setMax(10);

    // scheme
    aScheme = enumAttr.create("scheme", "sc", OsdPtexMeshData::kCatmark);
    enumAttr.addField("Catmull-Clark",  OsdPtexMeshData::kCatmark);
    enumAttr.addField("Loop",           OsdPtexMeshData::kLoop);
    enumAttr.addField("Bilinear",       OsdPtexMeshData::kBilinear);
    enumAttr.setInternal(true);

    // kernel
    aKernel = enumAttr.create("kernel", "kn", OsdPtexMeshData::kCPU);
    enumAttr.addField("CPU",    OsdPtexMeshData::kCPU);
#ifdef OPENSUBDIV_HAS_OPENMP
    enumAttr.addField("OpenMP", OsdPtexMeshData::kOPENMP);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    enumAttr.addField("CL",     OsdPtexMeshData::kCL);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    enumAttr.addField("CUDA",   OsdPtexMeshData::kCUDA);
#endif
    enumAttr.setInternal(true);

    // interpolateBoundary
    aInterpolateBoundary = enumAttr.create("interpolateBoundary", "ib", 
                    OsdPtexMeshData::kInterpolateBoundaryNone);
    enumAttr.addField("None",            OsdPtexMeshData::kInterpolateBoundaryNone);
    enumAttr.addField("Edge Only",       OsdPtexMeshData::kInterpolateBoundaryEdgeOnly);
    enumAttr.addField("Edge and Corner", OsdPtexMeshData::kInterpolateBoundaryEdgeAndCorner);
    enumAttr.addField("Always Sharp",    OsdPtexMeshData::kInterpolateBoundaryAlwaysSharp);
    enumAttr.setInternal(true);

    // adaptive
    aAdaptive = numAttr.create("adaptive", "adp", MFnNumericData::kBoolean, true);
    numAttr.setInternal(true);

    // wireframe
    aWireframe = numAttr.create("wireframe", "wf", MFnNumericData::kBoolean, false);

    // material attributes
    aDiffuse = numAttr.createColor("diffuse",   "d");
    numAttr.setDefault(0.6f, 0.6f, 0.7f);
    aAmbient = numAttr.createColor("ambient",   "a");
    numAttr.setDefault(0.1f, 0.1f, 0.1f);
    aSpecular = numAttr.createColor("specular", "s");
    numAttr.setDefault(0.3f, 0.3f, 0.3f);

    // Ptex Texture Attributes
    //
    // diffuseEnvironmentMapFile;
    aDiffuseEnvironmentMapFile = typedAttr.create("diffuseEnvironmentMap", "difenv", MFnData::kString);
    typedAttr.setInternal(true);
    // don't let maya hold on to string when fileNode is disconnected
    typedAttr.setDisconnectBehavior(MFnAttribute::kReset);

    // specularEnvironmentMapFile;
    aSpecularEnvironmentMapFile = typedAttr.create("specularEnvironmentMap", "specenv", MFnData::kString);
    typedAttr.setInternal(true);
    // don't let maya hold on to string when fileNode is disconnected
    typedAttr.setDisconnectBehavior(MFnAttribute::kReset);

    // colorFile;
    aColorFile = typedAttr.create("colorFile", "cf", MFnData::kString);
    typedAttr.setInternal(true);

    // displacementFile;
    aDisplacementFile = typedAttr.create("displacementFile", "df", MFnData::kString);
    typedAttr.setInternal(true);

    // occlusionFile;
    aOcclusionFile = typedAttr.create("occlusionFile", "of", MFnData::kString);
    typedAttr.setInternal(true);

    // enableDisplacement;
    aEnableDisplacement = numAttr.create("enableDisplacement", "end", MFnNumericData::kBoolean, 1);
    numAttr.setInternal(true);

    // enableColor;
    aEnableColor = numAttr.create("enableColor", "enc", MFnNumericData::kBoolean, 1);
    numAttr.setInternal(true);

    // enableOcclusion;
    aEnableOcclusion = numAttr.create("enableOcclusion", "eno", MFnNumericData::kBoolean, 1);
    numAttr.setInternal(true);

    // enableNormal;
    aEnableNormal = numAttr.create("enableNormal", "enn", MFnNumericData::kBoolean, 1);
    numAttr.setInternal(true);

    // fresnelBias;
    aFresnelBias = numAttr.create("fresnelBias", "fb", MFnNumericData::kFloat, 0.2f);
    numAttr.setMin(0);
    numAttr.setMax(1);

    // fresnelScale;
    aFresnelScale = numAttr.create("fresnelScale", "fs", MFnNumericData::kFloat, 1.0f);
    numAttr.setMin(0);
    numAttr.setSoftMax(1);

    // fresnelPower;
    aFresnelPower = numAttr.create("fresnelPower", "fp", MFnNumericData::kFloat, 5.0f);
    numAttr.setMin(0);
    numAttr.setSoftMax(10);

    // shaderSource;
    aShaderSource = typedAttr.create("shaderSource", "ssrc", MFnData::kString);
    typedAttr.setInternal(true);


    // add attributes
    addAttribute(aLevel);
    addAttribute(aTessFactor);
    addAttribute(aScheme);
    addAttribute(aKernel);
    addAttribute(aInterpolateBoundary);
    addAttribute(aAdaptive);
    addAttribute(aWireframe);

    addAttribute(aDiffuse);
    addAttribute(aAmbient);
    addAttribute(aSpecular);

    addAttribute(aShaderSource);

    addAttribute(aDiffuseEnvironmentMapFile);
    addAttribute(aSpecularEnvironmentMapFile);
    addAttribute(aColorFile);
    addAttribute(aDisplacementFile);
    addAttribute(aOcclusionFile);

    addAttribute(aEnableDisplacement);
    addAttribute(aEnableColor);
    addAttribute(aEnableOcclusion);
    addAttribute(aEnableNormal);

    addAttribute(aFresnelBias);
    addAttribute(aFresnelScale);
    addAttribute(aFresnelPower);

    return MS::kSuccess;
}

void
OpenSubdivPtexShader::postConstructor() 
{
    setMPSafe(false);
}

MStatus
OpenSubdivPtexShader::compute(const MPlug &plug, MDataBlock &data) 
{
    return MS::kSuccess;
}

bool
OpenSubdivPtexShader::getInternalValueInContext(const MPlug &plug, MDataHandle &handle, MDGContext &) 
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
 
    } else if (plug == aShaderSource) {
        handle.setString( _shaderSourceFilename );

    } else if (plug == aDiffuseEnvironmentMapFile) {
        handle.setString(_diffEnvMapFile);
    } else if (plug == aSpecularEnvironmentMapFile) {
        handle.setString(_specEnvMapFile);
    } else if (plug == aColorFile) {
        handle.setString(_colorFile);
    } else if (plug == aDisplacementFile) {
        handle.setString(_displacementFile);
    } else if (plug == aOcclusionFile) {
        handle.setString(_occlusionFile);
    } else if (plug == aEnableColor) {
        handle.setBool(_enableColor);
    } else if (plug == aEnableDisplacement) {
        handle.setBool(_enableDisplacement);
    } else if (plug == aEnableOcclusion) {
        handle.setBool(_enableOcclusion);
    } else if (plug == aEnableNormal) {
        handle.setBool(_enableNormal);
    }

    return false;
}

bool
OpenSubdivPtexShader::setInternalValueInContext(const MPlug &plug, const MDataHandle &handle, MDGContext &) 
{
    if (plug == aLevel) {
        _hbrMeshDirty = true;
        _level = handle.asLong();
    } else if (plug == aTessFactor) {
        _tessFactor = handle.asLong();
    } else if (plug == aScheme) {
        _hbrMeshDirty = true;
        _scheme = (OsdPtexMeshData::SchemeType)handle.asShort();
    } else if (plug == aKernel) {
        _hbrMeshDirty = true;
        _kernel = (OsdPtexMeshData::KernelType)handle.asShort();
    } else if (plug == aInterpolateBoundary) {
        _hbrMeshDirty = true;
        _interpolateBoundary = (OsdPtexMeshData::InterpolateBoundaryType)handle.asShort();
    } else if (plug == aAdaptive) {
        _hbrMeshDirty = true;
        _adaptiveDirty = true;
        _adaptive = handle.asBool();

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

    } else if (plug == aDiffuseEnvironmentMapFile) {
        _diffEnvMapDirty = true;
        _diffEnvMapFile = handle.asString();
    } else if (plug == aSpecularEnvironmentMapFile) {
        _specEnvMapDirty = true;
        _specEnvMapFile = handle.asString();
    } else if (plug == aColorFile) {
        _ptexColorDirty = true;
        _colorFile = handle.asString();
    } else if (plug == aDisplacementFile) {
        _ptexDisplacementDirty = true;
        _displacementFile = handle.asString();
    } else if (plug == aOcclusionFile) {
        _ptexOcclusionDirty = true;
        _occlusionFile = handle.asString();
    } else if (plug == aEnableColor) {
        _enableColor = handle.asBool();
    } else if (plug == aEnableDisplacement) {
        _enableDisplacement = handle.asBool();
    } else if (plug == aEnableOcclusion) {
        _enableOcclusion = handle.asBool();
    } else if (plug == aEnableNormal) {
        _enableNormal = handle.asBool();
    }

    return false;
}


MStatus
OpenSubdivPtexShader::renderSwatchImage(MImage & image) 
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

void
OpenSubdivPtexShader::updateAttributes() 
{
    MObject object = thisMObject();
    MFnDependencyNode depFn(object);

    // retrieve non-internal attributes
    _diffuse = getColor(object, aDiffuse);
    _ambient = getColor(object, aAmbient);
    _specular = getColor(object, aSpecular);

    getAttribute(object, aWireframe, &_wireframe);

    getAttribute(object, aFresnelBias, &_fresnelBias);
    getAttribute(object, aFresnelScale, &_fresnelScale);
    getAttribute(object, aFresnelPower, &_fresnelPower);

    // pull on any plugs that might be connected
}

void
OpenSubdivPtexShader::draw(const MHWRender::MDrawContext &mDrawContext,
                           OsdPtexMeshData *data)
{
    MStatus status;

    glPushAttrib(GL_POLYGON_BIT|GL_ENABLE_BIT);

    // in ptexviewer, cull back face even in wireframe mode (for performance)
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    if (_wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    CHECK_GL_ERROR("draw begin\n");

    // If shader source attribute has changed, update effectRegistry
    updateRegistry();

    GLuint bPosition = data->bindPositionVBO();
    GLuint bNormal = data->bindNormalVBO();
    OpenSubdiv::OsdGLDrawContext *osdDrawContext = data->getDrawContext();

    glBindBuffer(GL_ARRAY_BUFFER, bPosition);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);
    glEnableVertexAttribArray(0);

    if (not _adaptive) {
        glBindBuffer(GL_ARRAY_BUFFER, bNormal);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);
        glEnableVertexAttribArray(1);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, osdDrawContext->patchIndexBuffer);

    OpenSubdiv::OsdPatchArrayVector const & patches =
        osdDrawContext->patchArrays;

    for (size_t i = 0; i < patches.size(); ++i) {
        OpenSubdiv::OsdPatchArray const & patch = patches[i];

        GLint surfaceProgram = bindProgram(mDrawContext, osdDrawContext, patch);

        if (patch.desc.type != OpenSubdiv::kNonPatch) {
            glPatchParameteri(GL_PATCH_VERTICES, patch.desc.GetPatchSize());

            if (osdDrawContext->GetVertexTextureBuffer()) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_BUFFER,
                              osdDrawContext->GetVertexTextureBuffer());
                glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, bPosition);
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
            glActiveTexture(GL_TEXTURE0);

            glDrawElements(GL_PATCHES,
                           patch.numIndices, GL_UNSIGNED_INT,
                           reinterpret_cast<void *>(patch.firstIndex *
                                                    sizeof(unsigned int)));
        } else {
            GLint nonAdaptiveLevel = glGetUniformLocation(surfaceProgram,
                                                          "nonAdaptiveLevel");
            if (nonAdaptiveLevel != -1) {
                glProgramUniform1i(surfaceProgram, nonAdaptiveLevel, _level);
            }

            glDrawElements(_scheme == OsdPtexMeshData::kLoop ? GL_TRIANGLES : GL_LINES_ADJACENCY,
                           patch.numIndices, GL_UNSIGNED_INT,
                           reinterpret_cast<void *>(patch.firstIndex *
                                                    sizeof(unsigned int)));
        }
        CHECK_GL_ERROR("post draw\n");
    }

    glUseProgram(0);

    glDisableVertexAttribArray(0);
    if (not _adaptive)
        glDisableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glPopAttrib();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    CHECK_GL_ERROR("draw end\n");
}


// ----------------------------------------------------------------------
//      private methods
//

GLuint 
OpenSubdivPtexShader::bindTexture( const MString& filename, int textureUnit )
{
    GLuint textureId = 0;

    MHWRender::MTexture *mTexture = _theTextureManager->acquireTexture(filename);
    if (mTexture) {
        textureId = *(GLuint*)mTexture->resourceHandle();
        glActiveTexture(GL_TEXTURE0+textureUnit);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        fprintf(stderr,"Can't read texture file: \"%s\"\n", filename.asChar());
    }
    return textureId;
}


// 
//      updateRegistry
//
//      When attributes change which affect the shader compilation the
//      effectRegistry needs to be updated with the new values
//
void
OpenSubdivPtexShader::updateRegistry()
{
    // adaptive toggle
    if (_adaptiveDirty) {
        effectRegistry.setIsAdaptive(_adaptive);
        _adaptiveDirty = false;
    }

    // ptex color file
    if (_ptexColorDirty) {
        bool ptexColorValid = bindPtexTexture(_colorFile, &_ptexColor, CLR_TEXTURE_UNIT);
        effectRegistry.setPtexColorValid(ptexColorValid);
        _ptexColorDirty = false;
    }

    // ptex displacement file
    if (_ptexDisplacementDirty) {
        bool ptexDisplacementValid = bindPtexTexture(_displacementFile, &_ptexDisplacement, DISP_TEXTURE_UNIT);
        effectRegistry.setPtexDisplacementValid(ptexDisplacementValid);
        _ptexDisplacementDirty = false;
    }

    // ptex occlusion file
    if (_ptexOcclusionDirty) {
        bool ptexOcclusionValid = bindPtexTexture(_occlusionFile, &_ptexOcclusion, OCC_TEXTURE_UNIT);
        effectRegistry.setPtexOcclusionValid(ptexOcclusionValid);
        _ptexOcclusionDirty = false;
    }


    MHWRender::MRenderer *theRenderer = MHWRender::MRenderer::theRenderer();
    _theTextureManager = theRenderer->getTextureManager();

    // diffuse environment map file
    if (_diffEnvMapDirty) {
        GLuint diffEnvMapId = bindTexture( _diffEnvMapFile, DIFF_TEXTURE_UNIT );
        effectRegistry.setDiffuseEnvironmentId(diffEnvMapId);
        _diffEnvMapDirty = false;
    }

    // specular environment map file
    if (_specEnvMapDirty) {
        GLuint specEnvMapId = bindTexture( _specEnvMapFile, ENV_TEXTURE_UNIT );
        effectRegistry.setSpecularEnvironmentId(specEnvMapId);
        _specEnvMapDirty = false;
    }

    // shader source
    if (_shaderSourceDirty) {
        if ( _shaderSource.empty() ) {
            if ( effectRegistry.getShaderSource() != defaultShaderSource ) {
                effectRegistry.setShaderSource(defaultShaderSource);
            }
        } else {
            if ( effectRegistry.getShaderSource() != _shaderSource ) {
                effectRegistry.setShaderSource(_shaderSource);
            }
        }
        _shaderSourceDirty = false;
    }
}

GLuint
OpenSubdivPtexShader::bindProgram(const MHWRender::MDrawContext &     mDrawContext,
                                        OpenSubdiv::OsdGLDrawContext *osdDrawContext,
                                  const OpenSubdiv::OsdPatchArray &   patch)
{

    CHECK_GL_ERROR("bindProgram begin\n");

    // Build shader
    Effect effect;
    effect.color = _enableColor;
    effect.occlusion = _enableOcclusion;
    effect.displacement = _enableDisplacement;
    effect.normal = _enableNormal;
    EffectDesc effectDesc( patch.desc, effect );
    EffectDrawRegistry::ConfigType *
        config = effectRegistry.GetDrawConfig(effectDesc);

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
        int GregoryQuadOffsetBase;
        int PrimitiveIdBase;
    } tessellationData;

    tessellationData.TessLevel = static_cast<float>(1 << _tessFactor);
    tessellationData.GregoryQuadOffsetBase = patch.GetQuadOffsetBase;
    tessellationData.PrimitiveIdBase = patch.GetPatchIndex();;

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


#ifdef USE_NON_IMAGE_BASED_LIGHTING
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

    for (int i = 0; i < numLights && i < 1; ++i) {
        MFloatPointArray positions;
        MFloatVector direction;
        float intensity;
        MColor color;
        bool hasDirection, hasPosition;
        mDrawContext.getLightInformation(i, positions, direction, intensity,
                                    color, hasDirection, hasPosition);

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
#endif

    GLint eye = glGetUniformLocation(program, "eyePositionInWorld");
    MPoint e = MPoint(0, 0, 0) *
        mDrawContext.getMatrix(MHWRender::MDrawContext::kWorldViewInverseMtx);
    glProgramUniform3f(program, eye,
                       static_cast<float>(e.x),
                       static_cast<float>(e.y),
                       static_cast<float>(e.z));

    // update other uniforms
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
                       glGetUniformLocation(program, "fresnelBias"),
                       _fresnelBias);
    glProgramUniform1f(program,
                       glGetUniformLocation(program, "fresnelScale"),
                       _fresnelScale);
    glProgramUniform1f(program,
                       glGetUniformLocation(program, "fresnelPower"),
                       _fresnelPower);


    // Ptex bindings 
    // color ptex
    if (effectRegistry.getPtexColorValid()) {
        GLint texData = glGetUniformLocation(program, "textureImage_Data");
        glProgramUniform1i(program, texData, CLR_TEXTURE_UNIT + 0);
        GLint texPacking = glGetUniformLocation(program, "textureImage_Packing");
        glProgramUniform1i(program, texPacking, CLR_TEXTURE_UNIT + 1);
        GLint texPages = glGetUniformLocation(program, "textureImage_Pages");
        glProgramUniform1i(program, texPages, CLR_TEXTURE_UNIT + 2);
    }

    // displacement ptex
    if (effectRegistry.getPtexDisplacementValid()) {
        GLint texData = glGetUniformLocation(program, "textureDisplace_Data");
        glProgramUniform1i(program, texData, DISP_TEXTURE_UNIT + 0);
        GLint texPacking = glGetUniformLocation(program, "textureDisplace_Packing");
        glProgramUniform1i(program, texPacking, DISP_TEXTURE_UNIT + 1);
        GLint texPages = glGetUniformLocation(program, "textureDisplace_Pages");
        glProgramUniform1i(program, texPages, DISP_TEXTURE_UNIT + 2);
    }

    // occlusion ptex
    if (effectRegistry.getPtexOcclusionValid()) {
        GLint texData = glGetUniformLocation(program, "textureOcclusion_Data");
        glProgramUniform1i(program, texData, OCC_TEXTURE_UNIT + 0);
        GLint texPacking = glGetUniformLocation(program, "textureOcclusion_Packing");
        glProgramUniform1i(program, texPacking, OCC_TEXTURE_UNIT + 1);
        GLint texPages = glGetUniformLocation(program, "textureOcclusion_Pages");
        glProgramUniform1i(program, texPages, OCC_TEXTURE_UNIT + 2);
    }

    // diffuse environment map
    if (effectRegistry.getDiffuseEnvironmentId() != 0) {
        GLint difmap = glGetUniformLocation(program, "diffuseEnvironmentMap");
        glProgramUniform1i(program, difmap, DIFF_TEXTURE_UNIT);
    }

    // specular environment map
    if (effectRegistry.getSpecularEnvironmentId() != 0) {
        GLint envmap = glGetUniformLocation(program, "specularEnvironmentMap");
        glProgramUniform1i(program, envmap, ENV_TEXTURE_UNIT);
    }

    glActiveTexture(GL_TEXTURE0);

    CHECK_GL_ERROR("bindProgram leave\n");

    return program;
}

OpenSubdiv::OsdGLPtexTexture *
OpenSubdivPtexShader::loadPtex(const MString &filename) 
{
    if (filename.length()) {
        printf("Load ptex %s\n", filename.asChar());
        Ptex::String ptexError;
        PtexTexture *ptex = PtexTexture::open(filename.asChar(),
                                              ptexError, true);
        if (ptex) {
            // create osdptex
            OpenSubdiv::OsdGLPtexTexture *osdptex =
                OpenSubdiv::OsdGLPtexTexture::Create(ptex, 0,
                                                     /*gutterWidth=*/1,
                                                     /*pageMargin=*/8);
            ptex->release();
            return osdptex;
        }
        printf("Load ptex failed on file: \"%s\"\n", filename.asChar());
    }
    return NULL;
}

bool
OpenSubdivPtexShader::bindPtexTexture(const MString& ptexFilename, 
                                   OpenSubdiv::OsdGLPtexTexture **osdPtexPtr,
                                   int samplerUnit)
{
    // Reload ptex texture
    if (*osdPtexPtr) delete *osdPtexPtr;
    *osdPtexPtr = loadPtex(ptexFilename);
    if (*osdPtexPtr == NULL)
        return false;

    // Rebind
    glActiveTexture(GL_TEXTURE0 + samplerUnit + 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, (*osdPtexPtr)->GetTexelsTexture());

    glActiveTexture(GL_TEXTURE0 + samplerUnit + 1);
    glBindTexture(GL_TEXTURE_BUFFER, (*osdPtexPtr)->GetLayoutTextureBuffer());

    glActiveTexture(GL_TEXTURE0 + samplerUnit + 2);
    glBindTexture(GL_TEXTURE_BUFFER, (*osdPtexPtr)->GetPagesTextureBuffer());

    // Reset
    glActiveTexture(GL_TEXTURE0);

    return true;
}

