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

#include <GL/glew.h>

// Include this first to avoid winsock2.h problems on Windows:
#include <maya/MTypes.h>

#include <maya/MFnPlugin.h>
#include <maya/MFnPluginData.h>
#include <maya/MGlobal.h>
#include <maya/MFnMesh.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MIntArray.h>
#include <maya/MUintArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MPointArray.h>
#include <maya/MItMeshEdge.h>

#include <maya/MShaderManager.h>
#include <maya/MViewport2Renderer.h>
#include <maya/MDrawRegistry.h>
#include <maya/MPxHwShaderNode.h>
#include <maya/MPxShaderOverride.h>
#include <maya/MPxGeometryOverride.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MHWShaderSwatchGenerator.h>
#include <maya/MPxVertexBufferGenerator.h>
#include <maya/MStateManager.h>

#include <osd/mutex.h>

#include <hbr/mesh.h>
#include <hbr/bilinear.h>
#include <hbr/catmark.h>
#include <hbr/loop.h>

#include <osd/mesh.h>
#include <osd/pTexture.h>
#include <osd/cpuDispatcher.h>
#include <osd/elementArrayBuffer.h>
#include <osd/ptexCoordinatesTextureBuffer.h>
#include <Ptexture.h>
#include <PtexUtils.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <osd/cudaDispatcher.h>
extern void cudaInit();
#include "hbrUtil.h"

static const char *defaultShaderSource = 
#include "shader.inc"
;

#define CHECK_GL_ERROR(...)  \
    if(GLuint err = glGetError()) {   \
    printf("GL error %x :", err); \
    printf(__VA_ARGS__); \
    }

#ifdef __linux__
#define MAYA2013_PREVIEW
#endif

//---------------------------------------------------------------------------
template<class T> static int
FindAttribute( MFnDependencyNode &fnDN, const char *nm, T *val )
{
    MStatus s;
    MPlug p;
    T ss;
    p = fnDN.findPlug(nm, &s);
    if(s != MS::kSuccess) return -1;
    s = p.getValue(ss);
    if( s != MS::kSuccess ) return -1;
    *val = ss;
    return 0;
}
//---------------------------------------------------------------------------
template<class T> static bool
CompareArray(const T &a, const T &b)
{
    if(a.length() != b.length()) return false;
    for(unsigned int i = 0; i < a.length(); ++i){
        if(a[i] != b[i]) return false;
    }
    return true;
}
//---------------------------------------------------------------------------
class OpenSubdivPtexShader : public MPxHwShaderNode
{
public:
    OpenSubdivPtexShader();
    virtual ~OpenSubdivPtexShader();

    static void *creator();
    static MStatus initialize();

    virtual void postConstructor() 
        {
            setMPSafe(false);
        }

    virtual MStatus compute(const MPlug &plug, MDataBlock &data);
    virtual MStatus bind(const MDrawRequest &request, M3dView &view);
    virtual MStatus unbind(const MDrawRequest &request, M3dView &view);
    virtual MStatus geometry( const MDrawRequest& request,
                              M3dView& view,
                              int prim,
                              unsigned int writable,
                              int indexCount,
                              const unsigned int * indexArray,
                              int vertexCount,
                              const int * vertexIDs,
                              const float * vertexArray,
                              int normalCount,
                              const float ** normalArrays,
                              int colorCount,
                              const float ** colorArrays,
                              int texCoordCount,
                              const float ** texCoordArrays);
    virtual MStatus glBind(const MDagPath &);
    virtual MStatus glUnBind(const MDagPath &);
    virtual MStatus glGeometry(const MDagPath &path,
                               int prim,
                               unsigned int writable,
                               int indexCount,
                               const unsigned int * indexArray,
                               int vertexCount,
                               const int * vertexIDs,
                               const float * vertexArray,
                               int normalCount,
                               const float ** normalArrays,
                               int colorCount,
                               const float ** colorArrays,
                               int texCoordCount,
                               const float ** texCoordArrays);
    virtual MStatus renderSwatchImage( MImage & image )
        {
            unsigned int width, height;
            image.getSize(width, height);
            unsigned char *p = image.pixels();
            for(unsigned int i=0; i<width*height; i++){
                *p++ = 0;
                *p++ = 0;
                *p++ = 0;
                *p++ = 255;
            }
            return MS::kSuccess;
        }

    
//    virtual bool isBounded() const;
//    virtual MBoundingBox boundingBox() const;
    virtual bool setInternalValueInContext(const MPlug &plug, const MDataHandle &handle, MDGContext &);
    
    void bindTextures();
    GLuint getProgram() const {return _program; }
    void updateUniforms();
    bool isWireframe() const { return _wireframe; }
    int getLevel() const { return _level; }
    int getScheme() const { return _scheme; }

    static MTypeId id;
    static MString drawRegistrantId;

private:
    OpenSubdiv::OsdPTexture * loadPtex(MString &filename);
    void bindPTexture(OpenSubdiv::OsdPTexture *osdPTex, GLuint data, GLuint packing, GLuint pages, int samplerUnit) const;
    void linkProgram();
    GLuint compileShader(GLenum shaderType, const char *section, const char *shaderSource);

    static MObject colorFile;
    static MObject displacementFile;
    static MObject occlusionFile;
    static MObject level;
    static MObject scheme;
    static MObject diffuseMapFile;
    static MObject environmentMapFile;
    static MObject shaderSource;

    static MObject enableDisplacement;
    static MObject enableColor;
    static MObject enableOcclusion;

    static MObject aDiffuse;
    static MObject aSpecular;
    static MObject aAmbient;
    static MObject aFresnelBias;
    static MObject aFresnelScale;
    static MObject aFresnelPower;
    static MObject aLight;
    static MObject aWireframe;

    enum { kCatmark, kLoop, kBilinear };

    MString _ptexColorFile, _ptexDisplacementFile, _ptexOcclusionFile;
    OpenSubdiv::OsdPTexture * _ptexColor;
    OpenSubdiv::OsdPTexture * _ptexDisplacement;
    OpenSubdiv::OsdPTexture * _ptexOcclusion;

    MString _diffuseMap;
    MString _environmentMap;
    bool _invalidateDiffuseMap;
    bool _invalidateEnvMap;

    bool _enableColor;
    bool _enableDisplacement;
    bool _enableOcclusion;

    GLuint _program;
    bool _wireframe;
    int _level;
    int _scheme;
    std::string _shaderSource;
};

MTypeId OpenSubdivPtexShader::id(0x88111);
MString OpenSubdivPtexShader::drawRegistrantId("OpenSubdivPtexShaderPlugin");
MObject OpenSubdivPtexShader::colorFile;
MObject OpenSubdivPtexShader::displacementFile;
MObject OpenSubdivPtexShader::occlusionFile;
MObject OpenSubdivPtexShader::level;
MObject OpenSubdivPtexShader::scheme;
MObject OpenSubdivPtexShader::diffuseMapFile;
MObject OpenSubdivPtexShader::environmentMapFile;
MObject OpenSubdivPtexShader::shaderSource;
MObject OpenSubdivPtexShader::enableDisplacement;
MObject OpenSubdivPtexShader::enableColor;
MObject OpenSubdivPtexShader::enableOcclusion;
MObject OpenSubdivPtexShader::aDiffuse;
MObject OpenSubdivPtexShader::aSpecular;
MObject OpenSubdivPtexShader::aAmbient;
MObject OpenSubdivPtexShader::aFresnelBias;
MObject OpenSubdivPtexShader::aFresnelScale;
MObject OpenSubdivPtexShader::aFresnelPower;
MObject OpenSubdivPtexShader::aLight;
MObject OpenSubdivPtexShader::aWireframe;

OpenSubdivPtexShader::OpenSubdivPtexShader()
    : _ptexColor(NULL), _ptexDisplacement(NULL), _ptexOcclusion(NULL),
      _invalidateDiffuseMap(true),
      _invalidateEnvMap(true),
      _enableColor(true),
      _enableDisplacement(true),
      _enableOcclusion(true),
      _program(0),
      _wireframe(false),
      _level(1),
      _scheme(0)
{
    _shaderSource = defaultShaderSource;
    linkProgram();
}

OpenSubdivPtexShader::~OpenSubdivPtexShader()
{
    if (_ptexColor) delete _ptexColor;
    if (_ptexDisplacement) delete _ptexDisplacement;
    if (_ptexOcclusion) delete _ptexOcclusion;

    if (_program)
        glDeleteProgram(_program);
}

void *
OpenSubdivPtexShader::creator()
{
    return new OpenSubdivPtexShader();
}

MStatus
OpenSubdivPtexShader::initialize()
{
    MStatus stat;
    MFnTypedAttribute typedAttr;
    MFnNumericAttribute numAttr;
    MFnEnumAttribute enumAttr;

    colorFile = typedAttr.create("colorFile", "cf", MFnData::kString, &stat);
    typedAttr.setInternal(true);
    displacementFile = typedAttr.create("displacementFile", "df", MFnData::kString, &stat);
    typedAttr.setInternal(true);
    occlusionFile = typedAttr.create("occlusionFile", "of", MFnData::kString, &stat);
    typedAttr.setInternal(true);

    level = numAttr.create("level", "lv", MFnNumericData::kLong, 1);
    numAttr.setInternal(true);
    numAttr.setMin(1);
    numAttr.setSoftMax(5);

    scheme = enumAttr.create("scheme", "sc");
    enumAttr.addField("Catmull-Clark", kCatmark);
    enumAttr.addField("Loop", kLoop);
    enumAttr.addField("Bilinear", kBilinear);
    enumAttr.setInternal(true);

    diffuseMapFile = typedAttr.create("diffuseMap", "difmap", MFnData::kString);
    typedAttr.setInternal(true);

    environmentMapFile = typedAttr.create("environmentMap", "envmap", MFnData::kString);
    typedAttr.setInternal(true);

    shaderSource = typedAttr.create("shaderSource", "ssrc", MFnData::kString);
    typedAttr.setInternal(true);


    enableColor = numAttr.create("enableColor", "enc", MFnNumericData::kBoolean, 1);
    numAttr.setInternal(true);
    enableDisplacement = numAttr.create("enableDisplacement", "end", MFnNumericData::kBoolean, 1);
    numAttr.setInternal(true);
    enableOcclusion = numAttr.create("enableOcclusion", "eno", MFnNumericData::kBoolean, 1);
    numAttr.setInternal(true);
   
    aDiffuse = numAttr.createColor("diffuse", "d");
    aSpecular = numAttr.createColor("specular", "s");
    aAmbient = numAttr.createColor("ambient", "a");

    aLight = numAttr.create("light", "l", MFnNumericData::k3Float);

    aWireframe = numAttr.create("wireframe", "wf", MFnNumericData::kBoolean);

    aFresnelBias = numAttr.create("fresnelBias", "fb", MFnNumericData::kFloat, 0.2f);
    numAttr.setMin(0);
    numAttr.setMax(1);
    aFresnelScale = numAttr.create("fresnelScale", "fs", MFnNumericData::kFloat, 1.0f);
    numAttr.setMin(0);
    numAttr.setSoftMax(1);
    aFresnelPower = numAttr.create("fresnelPower", "fp", MFnNumericData::kFloat, 5.0f);
    numAttr.setMin(0);
    numAttr.setSoftMax(10);

    addAttribute(colorFile);
    addAttribute(displacementFile);
    addAttribute(occlusionFile);
    addAttribute(level);
    addAttribute(scheme);
    addAttribute(diffuseMapFile);
    addAttribute(environmentMapFile);
    addAttribute(shaderSource);
    addAttribute(enableColor);
    addAttribute(enableDisplacement);
    addAttribute(enableOcclusion);

    addAttribute(aDiffuse);
    addAttribute(aSpecular);
    addAttribute(aAmbient);
    addAttribute(aFresnelBias);
    addAttribute(aFresnelScale);
    addAttribute(aFresnelPower);
    addAttribute(aLight);
    addAttribute(aWireframe);


    return MS::kSuccess;
}

MStatus
OpenSubdivPtexShader::compute(const MPlug &plug, MDataBlock &data)
{
    return MS::kSuccess;
}

OpenSubdiv::OsdPTexture *
OpenSubdivPtexShader::loadPtex(MString &filename)
{
    if (filename.length() ){
        printf("Load ptex %s\n", filename.asChar());
        Ptex::String ptexError;
        PtexTexture *ptex = PtexTexture::open(filename.asChar(), ptexError, true);
        if (ptex) {
            // create osdptex
            OpenSubdiv::OsdPTexture *osdptex = OpenSubdiv::OsdPTexture::Create(ptex, 0);
            ptex->release();
            return osdptex;
        }
        printf("Fail in load ptex\n");
    }
    return NULL;
}

void
OpenSubdivPtexShader::bindPTexture(OpenSubdiv::OsdPTexture *osdPTex, GLuint data, GLuint packing, GLuint pages, int samplerUnit) const
{
    glProgramUniform1i(_program, data, samplerUnit + 0);
    glActiveTexture(GL_TEXTURE0 + samplerUnit + 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, osdPTex->GetTexelsTexture());

    glProgramUniform1i(_program, packing, samplerUnit + 1);
    glActiveTexture(GL_TEXTURE0 + samplerUnit + 1);
    glBindTexture(GL_TEXTURE_BUFFER, osdPTex->GetLayoutTextureBuffer());

    glProgramUniform1i(_program, pages, samplerUnit + 2);
    glActiveTexture(GL_TEXTURE0 + samplerUnit + 2);
    glBindTexture(GL_TEXTURE_BUFFER, osdPTex->GetPagesTextureBuffer());

    glActiveTexture(GL_TEXTURE0);
}

void
OpenSubdivPtexShader::bindTextures()
{
    if (_ptexColor)
    {
        // color ptex
        GLint texData = glGetUniformLocation(_program, "textureImage_Data");
        GLint texPacking = glGetUniformLocation(_program, "textureImage_Packing");
        GLint texPages = glGetUniformLocation(_program, "textureImage_Pages");
        bindPTexture(_ptexColor, texData, texPacking, texPages, 1);
    }

    // displacement ptex
    if (_ptexDisplacement) {
        GLint texData = glGetUniformLocation(_program, "textureDisplace_Data");
        GLint texPacking = glGetUniformLocation(_program, "textureDisplace_Packing");
        GLint texPages = glGetUniformLocation(_program, "textureDisplace_Pages");
        bindPTexture(_ptexDisplacement, texData, texPacking, texPages, 4);
    }
    
    // occlusion ptex
    if (_ptexOcclusion) {
        GLint texData = glGetUniformLocation(_program, "textureOcclusion_Data");
        GLint texPacking = glGetUniformLocation(_program, "textureOcclusion_Packing");
        GLint texPages = glGetUniformLocation(_program, "textureOcclusion_Pages");
        bindPTexture(_ptexOcclusion, texData, texPacking, texPages, 7);
    }

    MHWRender::MRenderer *theRenderer = MHWRender::MRenderer::theRenderer();
    MHWRender::MTextureManager *theTextureManager = theRenderer->getTextureManager();

    if (_invalidateDiffuseMap)
    {
        MHWRender::MTexture *texture = theTextureManager->acquireTexture(_diffuseMap);
        if (texture) {
            GLint difmap = glGetUniformLocation(_program, "diffuseMap");
            glProgramUniform1i(_program, difmap, 10);
            glActiveTexture(GL_TEXTURE0+10);
            glBindTexture(GL_TEXTURE_2D, *(GLuint*)texture->resourceHandle());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            _invalidateDiffuseMap = false;
        } else {
            printf("Can't read %s\n", _diffuseMap.asChar());
        }
    }
    if (_invalidateEnvMap)
    {
        MHWRender::MTexture *texture = theTextureManager->acquireTexture(_environmentMap);
        if (texture) {
            GLint envmap = glGetUniformLocation(_program, "environmentMap");
            glProgramUniform1i(_program, envmap, 11);
            glActiveTexture(GL_TEXTURE0+11);
            glBindTexture(GL_TEXTURE_2D, *(GLuint*)texture->resourceHandle());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            _invalidateEnvMap = false;
        } else {
            printf("Can't read %s\n", _environmentMap.asChar());
        }
    }
}

static MColor getColor(MObject object, MObject attr)
{
    MPlug plug(object, attr);
    
    MObject data;
    plug.getValue(data);
    MFnNumericData numFn(data);
    float color[3];
    numFn.getData(color[0], color[1], color[2]);
    return MColor(color[0], color[1], color[2]);
}

static MFloatVector getVector(MObject object, MObject attr)
{
    MPlug plug(object, attr);
    
    MObject data;
    plug.getValue(data);
    MFnNumericData numFn(data);
    float color[3];
    numFn.getData(color[0], color[1], color[2]);
    return MVector(color[0], color[1], color[2]);
}

void
OpenSubdivPtexShader::updateUniforms()
{
    MObject object = thisMObject();
    MColor diffuse = getColor(object, aDiffuse);
    MColor ambient = getColor(object, aAmbient);
    MColor specular = getColor(object, aSpecular);
    MFnDependencyNode depFn(object);
    float fresnelBias, fresnelScale, fresnelPower;
    FindAttribute(depFn, "fresnelBias", &fresnelBias);
    FindAttribute(depFn, "fresnelScale", &fresnelScale);
    FindAttribute(depFn, "fresnelPower", &fresnelPower);
    MFloatVector light = getVector(object, aLight);
    FindAttribute(depFn, "wireframe", &_wireframe);

    glProgramUniform3f(_program, glGetUniformLocation(_program, "diffuse"),
                       diffuse.r, diffuse.g, diffuse.b);
    glProgramUniform3f(_program, glGetUniformLocation(_program, "ambient"),
                       ambient.r, ambient.g, ambient.b);
    glProgramUniform3f(_program, glGetUniformLocation(_program, "specular"),
                       specular.r, specular.g, specular.b);
    glProgramUniform3f(_program, glGetUniformLocation(_program, "light"),
                       light.x, light.y, light.z);

    glProgramUniform1f(_program, glGetUniformLocation(_program, "fresnelBias"), fresnelBias);
    glProgramUniform1f(_program, glGetUniformLocation(_program, "fresnelScale"), fresnelScale);
    glProgramUniform1f(_program, glGetUniformLocation(_program, "fresnelPower"), fresnelPower);
}

//------------------------------------------------------------------------------
bool
OpenSubdivPtexShader::setInternalValueInContext(const MPlug &plug, const MDataHandle &handle, MDGContext &)
{
    if(plug == level)
    {
        _level = handle.asLong();
    }
    else if(plug == scheme)
    {
        _scheme = handle.asShort();
    }
    else if(plug == environmentMapFile)
    {
        _invalidateEnvMap = true;
        _environmentMap = handle.asString();
    }
    else if(plug == diffuseMapFile)
    {
        _invalidateDiffuseMap = true;
        _diffuseMap = handle.asString();
    }
    else if(plug == colorFile)
    {
        MString filename = handle.asString();
        if (_ptexColor) delete _ptexColor;
        _ptexColor = loadPtex(filename);
    }
    else if(plug == displacementFile) 
    {
        MString filename = handle.asString();
        if (_ptexDisplacement) delete _ptexDisplacement;
        _ptexDisplacement = loadPtex(filename);
    }
    else if(plug == occlusionFile)
    {
        MString filename = handle.asString();
        if (_ptexOcclusion) delete _ptexOcclusion;
        _ptexOcclusion = loadPtex(filename);
    }
    else if(plug == shaderSource)
    {
        MString filename = handle.asString();
        std::ifstream ifs;
        ifs.open(filename.asChar());
        if (ifs.fail()) {
            printf("link default shader\n");
            _shaderSource = defaultShaderSource;
        } else {
            printf("link %s shader\n", filename.asChar());
            std::stringstream buffer;
            buffer << ifs.rdbuf();
            _shaderSource = buffer.str();
        }
        ifs.close();
        linkProgram();
    }
    else if(plug == enableColor)
    {
        _enableColor = handle.asBool();
        linkProgram();
    }
    else if(plug == enableDisplacement)
    {
        _enableDisplacement = handle.asBool();
        linkProgram();
    }
    else if(plug == enableOcclusion)
    {
        _enableOcclusion = handle.asBool();
        linkProgram();
    }
    return false;
}

MStatus
OpenSubdivPtexShader::bind(const MDrawRequest& request, M3dView &view)
{
    return MS::kSuccess;
}

MStatus
OpenSubdivPtexShader::unbind(const MDrawRequest& request, M3dView &view)
{
    return MS::kSuccess;
}

MStatus
OpenSubdivPtexShader::geometry( const MDrawRequest& request,
                                M3dView& view,
                                int prim,
                                unsigned int writable,
                                int indexCount,
                                const unsigned int * indexArray,
                                int vertexCount,
                                const int * vertexIDs,
                                const float * vertexArray,
                                int normalCount,
                                const float ** normalArrays,
                                int colorCount,
                                const float ** colorArrays,
                                int texCoordCount,
                                const float ** texCoordArrays)
{
    return MS::kSuccess;
}

MStatus OpenSubdivPtexShader::glBind(const MDagPath &)
{
    return MS::kSuccess;
}
MStatus OpenSubdivPtexShader::glUnBind(const MDagPath &)
{
    return MS::kSuccess;
}

MStatus OpenSubdivPtexShader::glGeometry(const MDagPath &path,
                               int prim,
                               unsigned int writable,
                               int indexCount,
                               const unsigned int * indexArray,
                               int vertexCount,
                               const int * vertexIDs,
                               const float * vertexArray,
                               int normalCount,
                               const float ** normalArrays,
                               int colorCount,
                               const float ** colorArrays,
                               int texCoordCount,
                               const float ** texCoordArrays)
{
    return MS::kSuccess;
}


GLuint
OpenSubdivPtexShader::compileShader(GLenum shaderType, const char *section, const char *shaderSource)
{
    int color = _enableColor ? 1 : 0;
    int displacement = _enableDisplacement ? 1 : 0;
    int occlusion = _enableOcclusion ? 1 : 0;

    const char *sources[2];
    char define[1024];
    sprintf(define,
            "#define %s\n"
            "#define USE_PTEX_COLOR %d\n"
            "#define USE_PTEX_OCCLUSION %d\n"
            "#define USE_PTEX_DISPLACEMENT %d\n",
            section, color, occlusion, displacement);

    sources[0] = define;
    sources[1] = shaderSource;

    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 2, sources, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if( status == GL_FALSE ) {
        GLchar emsg[1024];
        glGetShaderInfoLog(shader, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error compiling GLSL shader (%s): %s\n", section, emsg ); 
        return 0;
    }

    return shader;
}


void
OpenSubdivPtexShader::linkProgram()
{
    printf( "Link Program\n");
    CHECK_GL_ERROR("link program 1\n");
    if (_program)
        glDeleteProgram(_program);

    _invalidateEnvMap = true;
    _invalidateDiffuseMap = true;

    _program = glCreateProgram();

    CHECK_GL_ERROR("link program 2\n");

    const char *src = _shaderSource.c_str();

    GLuint vertexShader         = compileShader(GL_VERTEX_SHADER,
                                                "VERTEX_SHADER", src);
    GLuint geometryShader       = compileShader(GL_GEOMETRY_SHADER,
                                                "GEOMETRY_SHADER", src);
    GLuint fragmentShader       = compileShader(GL_FRAGMENT_SHADER,
                                                "FRAGMENT_SHADER", src);

    glAttachShader(_program, vertexShader);
    glAttachShader(_program, geometryShader);
    glAttachShader(_program, fragmentShader);

    glLinkProgram(_program);

    glDeleteShader(vertexShader);
    glDeleteShader(geometryShader);
    glDeleteShader(fragmentShader);

    CHECK_GL_ERROR("link program done\n");
    GLint status;  
    glGetProgramiv(_program, GL_LINK_STATUS, &status );      
    if( status == GL_FALSE ) {
        GLchar emsg[1024];
        glGetProgramInfoLog(_program, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error linking GLSL program : %s\n", emsg ); 
        _program = 0;
        return;
    }
}

//---------------------------------------------------------------------------
// Viewport 2.0 override implementation
//---------------------------------------------------------------------------
class OsdPtexMeshData : public MUserData
{
public:
    OsdPtexMeshData(MObject mesh);
    virtual ~OsdPtexMeshData();

    void populateIfNeeded(int level, int scheme);
    void updateGeometry(const MHWRender::MVertexBuffer *point, const MHWRender::MVertexBuffer *normal);
    void prepare();
    void draw(GLuint program) const;
    void update();

private:
    void initializeOsd();
    void initializeIndexBuffer();
    void bindPtexCoordinates(GLuint program) const;

    MObject _mesh;

    OpenSubdiv::OsdHbrMesh *_hbrmesh;
    OpenSubdiv::OsdMesh *_osdmesh;
//    OpenSubdiv::OsdVertexBuffer *_vertexBuffer;
    OpenSubdiv::OsdVertexBuffer *_positionBuffer, *_normalBuffer;
    OpenSubdiv::OsdElementArrayBuffer *_elementArrayBuffer;
    OpenSubdiv::OsdPtexCoordinatesTextureBuffer *_ptexCoordinatesTextureBuffer;

    MFloatPointArray _pointArray;
    MFloatVectorArray  _normalArray;

    // topology cache
    MIntArray _vertexList;
    MUintArray _edgeIds, _vtxIds;
    MDoubleArray _edgeCreaseData, _vtxCreaseData;
    int _level;
    int _maxlevel;
    int _interpBoundary;
    int _scheme;

    bool _needsUpdate;
    bool _needsInitializeOsd;
    bool _needsInitializeIndexBuffer;
};

OsdPtexMeshData::OsdPtexMeshData(MObject mesh) :
    MUserData(false), _mesh(mesh),
    _hbrmesh(NULL), _osdmesh(NULL),
//    _vertexBuffer(NULL),
    _positionBuffer(NULL), _normalBuffer(NULL),
    _elementArrayBuffer(NULL),
    _ptexCoordinatesTextureBuffer(NULL),
    _level(0),
    _interpBoundary(0),
    _scheme(0),
    _needsUpdate(false),
    _needsInitializeOsd(false),
    _needsInitializeIndexBuffer(false)
{
}

OsdPtexMeshData::~OsdPtexMeshData() 
{
    if (_hbrmesh) delete _hbrmesh;
    if (_osdmesh) delete _osdmesh;
//    if (_vertexBuffer) delete _vertexBuffer;
    if (_positionBuffer) delete _positionBuffer;
    if (_normalBuffer) delete _normalBuffer;
    if (_elementArrayBuffer) delete _elementArrayBuffer;
    if (_ptexCoordinatesTextureBuffer) delete _ptexCoordinatesTextureBuffer;
}

void
OsdPtexMeshData::populateIfNeeded(int lv, int scheme)
{
    MStatus s;

    if (scheme == 1) scheme = 0; // avoid loop for now

    MFnMesh meshFn(_mesh, &s);
    if (s != MS::kSuccess) return;

    if (lv < 1) lv =1;

    MIntArray vertexCount, vertexList;
    meshFn.getVertices(vertexCount, vertexList);
    MUintArray edgeIds;
    MDoubleArray edgeCreaseData;
    meshFn.getCreaseEdges(edgeIds, edgeCreaseData);
    MUintArray vtxIds;
    MDoubleArray vtxCreaseData;
    meshFn.getCreaseVertices(vtxIds, vtxCreaseData );

    if (vertexCount.length() == 0) return;

    short interpBoundary = 0;
    FindAttribute(meshFn, "boundaryRule", &interpBoundary);

    if(CompareArray(_vertexList, vertexList) &&
       CompareArray(_edgeIds, edgeIds) &&
       CompareArray(_edgeCreaseData, edgeCreaseData) &&
       CompareArray(_vtxIds, vtxIds) &&
       CompareArray(_vtxCreaseData, vtxCreaseData) &&
       _interpBoundary == interpBoundary &&
       _scheme == scheme &&
       _maxlevel >= lv)
    {
        if(_level != lv) {
            _level = lv;
            _needsInitializeIndexBuffer = true;
        }
        return;
    }

    printf("Populate %s, level = %d < %d\n", meshFn.name().asChar(), lv, _level);

    // update topology
    _vertexList = vertexList;
    _edgeIds = edgeIds;
    _edgeCreaseData = edgeCreaseData;
    _vtxIds = vtxIds;
    _vtxCreaseData = vtxCreaseData;
    _maxlevel = std::max(lv, _maxlevel);
    _level = lv;
    _interpBoundary = interpBoundary;
    _scheme = scheme;

    std::vector<int> numIndices, faceIndices, edgeCreaseIndices, vtxCreaseIndices;
    std::vector<float> edgeCreases, vtxCreases;
    numIndices.resize(vertexCount.length());
    faceIndices.resize(vertexList.length());
    for(int i = 0; i < (int)vertexCount.length(); ++i) numIndices[i] = vertexCount[i];
    for(int i = 0; i < (int)vertexList.length(); ++i) faceIndices[i] = vertexList[i];
    vtxCreaseIndices.resize(vtxIds.length());
    for(int i = 0; i < (int)vtxIds.length(); ++i) vtxCreaseIndices[i] = vtxIds[i];
    vtxCreases.resize(vtxCreaseData.length());
    for(int i = 0; i < (int)vtxCreaseData.length(); ++i)
        vtxCreases[i] = (float)vtxCreaseData[i];
    edgeCreases.resize(edgeCreaseData.length());
    for(int i = 0; i < (int)edgeCreaseData.length(); ++i)
        edgeCreases[i] = (float)edgeCreaseData[i];

    // edge crease index is stored as pair of <face id> <edge localid> ...
    int nEdgeIds = edgeIds.length();
    edgeCreaseIndices.resize(nEdgeIds*2);
    for(int i = 0; i < nEdgeIds; ++i){
        int2 vertices;
        if (meshFn.getEdgeVertices(edgeIds[i], vertices) != MS::kSuccess) {
            s.perror("ERROR can't get creased edge vertices");
            continue;
        }
        edgeCreaseIndices[i*2] = vertices[0];
        edgeCreaseIndices[i*2+1] = vertices[1];
    }

    _hbrmesh = ConvertToHBR(meshFn.numVertices(), numIndices, faceIndices,
                            vtxCreaseIndices, vtxCreases,
                            std::vector<int>(), std::vector<float>(),
                            edgeCreaseIndices, edgeCreases,
                            interpBoundary, scheme);

    // GL function can't be used in prepareForDraw API.
    _needsInitializeOsd = true;
}

void
OsdPtexMeshData::initializeOsd()
{
    if (!_hbrmesh)
        return;

    // create Osd mesh
    int kernel = OpenSubdiv::OsdKernelDispatcher::kCUDA;
//    int kernel = OpenSubdiv::OsdKernelDispatcher::kCL;
//    int kernel = OpenSubdiv::OsdKernelDispatcher::kCPU;
//    if (OpenSubdiv::OsdKernelDispatcher::HasKernelType(OpenSubdiv::OsdKernelDispatcher::kOPENMP)) {
//        kernel = OpenSubdiv::OsdKernelDispatcher::kOPENMP;
//    }
    if (_osdmesh)
        delete _osdmesh;

    _osdmesh = new OpenSubdiv::OsdMesh();
    _osdmesh->Create(_hbrmesh, _level, kernel);
    delete _hbrmesh;
    _hbrmesh = NULL;

    // create vertex buffer
//    if (_vertexBuffer) delete _vertexBuffer;
//    _vertexBuffer = _osdmesh->InitializeVertexBuffer(6 /* position + normal */);
    if (_positionBuffer) delete _positionBuffer;
    if (_normalBuffer) delete _normalBuffer;
    _positionBuffer = _osdmesh->InitializeVertexBuffer(3);
    _normalBuffer = _osdmesh->InitializeVertexBuffer(3);

    _needsInitializeOsd = false;
    _needsInitializeIndexBuffer = true;

    // get geometry from maya mesh
    MFnMesh meshFn(_mesh);
    meshFn.getPoints(_pointArray);
    meshFn.getVertexNormals(true, _normalArray);

    _needsUpdate = true;
}

void
OsdPtexMeshData::updateGeometry(const MHWRender::MVertexBuffer *points, const MHWRender::MVertexBuffer *normals)
{
    // Update vertex
//    if (!_vertexBuffer) return;
    if (!_positionBuffer) return;
    if (!_normalBuffer) return;

    int nCoarsePoints = _pointArray.length();

    glBindBuffer(GL_COPY_READ_BUFFER, *(GLuint*)points->resourceHandle());
    glBindBuffer(GL_COPY_WRITE_BUFFER, _positionBuffer->GetGpuBuffer());
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, nCoarsePoints*3*sizeof(float));

    glBindBuffer(GL_COPY_READ_BUFFER, *(GLuint*)normals->resourceHandle());
    glBindBuffer(GL_COPY_WRITE_BUFFER, _normalBuffer->GetGpuBuffer());
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, nCoarsePoints*3*sizeof(float));
    
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    _osdmesh->Subdivide(_positionBuffer, NULL);
    _osdmesh->Subdivide(_normalBuffer, NULL);

#if 0
    float *pbuffer = new float[nCoarsePoints*3];
    float *nbuffer = new float[nCoarsePoints*3];
    glBindBuffer(GL_ARRAY_BUFFER, *(GLuint*)points->resourceHandle());
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, nCoarsePoints*3*sizeof(float), pbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, *(GLuint*)normals->resourceHandle());
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, nCoarsePoints*3*sizeof(float), nbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    const float *p = pbuffer, *n = nbuffer;;
    for(int i = 0; i<nCoarsePoints; ++i) {
        _pointArray[i] = MFloatPoint(p[0], p[1], p[2]);
        _normalArray[i] = MFloatVector(n[0], n[1], n[2]);
        p += 3;
        n += 3;
    }
    delete[] pbuffer;
    delete[] nbuffer;

    int nPoints = _pointArray.length();
    std::vector<float> vertex;
    vertex.resize(nPoints*6);

    for(int i = 0; i < nPoints; ++i){
        vertex[i*6+0] = _pointArray[i].x;
        vertex[i*6+1] = _pointArray[i].y;
        vertex[i*6+2] = _pointArray[i].z;
        vertex[i*6+3] = _normalArray[i].x;
        vertex[i*6+4] = _normalArray[i].y;
        vertex[i*6+5] = _normalArray[i].z;
    }

    _vertexBuffer->UpdateData(&vertex.at(0), nPoints);
    _osdmesh->Subdivide(_vertexBuffer, NULL);
#endif

    _needsUpdate = false;
}

void
OsdPtexMeshData::initializeIndexBuffer()
{
    // create element array buffer
    if (_elementArrayBuffer) delete _elementArrayBuffer;
    _elementArrayBuffer = _osdmesh->CreateElementArrayBuffer(_level);

    // create ptex coordinates texture buffer
    if (_ptexCoordinatesTextureBuffer) delete _ptexCoordinatesTextureBuffer;
    _ptexCoordinatesTextureBuffer = _osdmesh->CreatePtexCoordinatesTextureBuffer(_level);

    _needsInitializeIndexBuffer = false;
}

void
OsdPtexMeshData::bindPtexCoordinates(GLuint program) const
{
    if (!_osdmesh) return;

    // bind ptexture
    GLint texIndices = glGetUniformLocation(program, "ptexIndices");
    GLint ptexLevel = glGetUniformLocation(program, "ptexLevel");
        
    glProgramUniform1i(program, ptexLevel, 1<<_level);
    glProgramUniform1i(program, texIndices, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, _ptexCoordinatesTextureBuffer->GetGlTexture());
}

void
OsdPtexMeshData::prepare()
{
    if (_needsInitializeOsd) {
        const_cast<OsdPtexMeshData*>(this)->initializeOsd();
    }
    if (_needsInitializeIndexBuffer) {
        const_cast<OsdPtexMeshData*>(this)->initializeIndexBuffer();
    }
}

void
OsdPtexMeshData::draw(GLuint program) const
{
    MStatus status;

    bindPtexCoordinates(program);

    CHECK_GL_ERROR("draw begin\n");

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
#if 0
    GLuint bVertex = _vertexBuffer->GetGpuBuffer();
    glBindBuffer(GL_ARRAY_BUFFER, bVertex);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (float*)12);
#else
    GLuint bPosition = _positionBuffer->GetGpuBuffer();
    GLuint bNormal = _normalBuffer->GetGpuBuffer();
    glBindBuffer(GL_ARRAY_BUFFER, bPosition);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);

    glBindBuffer(GL_ARRAY_BUFFER, bNormal);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
#endif

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _elementArrayBuffer->GetGlBuffer());

    CHECK_GL_ERROR("before draw elements\n");
    glDrawElements(GL_LINES_ADJACENCY, _elementArrayBuffer->GetNumIndices(), GL_UNSIGNED_INT, 0);
    
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    CHECK_GL_ERROR("draw done\n");
}

// ------------------------------------------------------------------------------
class OpenSubdivPtexShaderOverride : public MHWRender::MPxShaderOverride
{
public:
    static MHWRender::MPxShaderOverride* creator(const MObject &obj)
        {
            return new OpenSubdivPtexShaderOverride(obj);
        }
    virtual ~OpenSubdivPtexShaderOverride();
    
    virtual MString initialize(const MInitContext &initContext, MInitFeedback &initFeedback)
        {
            MString empty;

            {
                MHWRender::MVertexBufferDescriptor positionDesc(
                    empty,
                    MHWRender::MGeometry::kPosition,
                    MHWRender::MGeometry::kFloat,
                    3);
                addGeometryRequirement(positionDesc);
            }

            MHWRender::MVertexBufferDescriptor positionDesc(
                "osdPosition",
                MHWRender::MGeometry::kTangent,
                MHWRender::MGeometry::kFloat,
                3);
            positionDesc.setSemanticName("osdPosition");
            addGeometryRequirement(positionDesc);

            MHWRender::MVertexBufferDescriptor normalDesc(
                "osdNormal",
                MHWRender::MGeometry::kBitangent,
                MHWRender::MGeometry::kFloat,
                3);
            normalDesc.setSemanticName("osdNormal");
            addGeometryRequirement(normalDesc);

            if (initFeedback.customData == NULL) {
                OsdPtexMeshData *data = new OsdPtexMeshData(initContext.dagPath.node());
                initFeedback.customData = data;
            }

            return MString("OpenSubdivPtexShaderOverride");
        }

    virtual void updateDG(MObject object)
        {
            if (object == MObject::kNullObj) 
                return;

            _shader = (OpenSubdivPtexShader*)MPxHwShaderNode::getHwShaderNodePtr(object);
            if (_shader) {
//                MFnDependencyNode depFn(object);
//                FindAttribute(depFn, "level", &_level);

                _shader->updateUniforms();
            }
        }
    virtual void updateDevice()
        {
            // only place to access GPU device safely
        }
    virtual void endUpdate()
        {
        }

/*
    virtual void activateKey(MHWRender::MDrawContext &context, const MString &key)
        {
        }

    virtual void terminateKey(MHWRender::MDrawContext &context, const MString &key)
        {
        }
*/

    virtual bool draw(MHWRender::MDrawContext &context, const MHWRender::MRenderItemList &renderItemList) const;

    virtual bool rebuildAlways()
        {
            return false;
        }

    virtual MHWRender::DrawAPI supportedDrawAPIs() const 
        {
            return MHWRender::kOpenGL;
        }
    virtual bool isTransparent()
        {
            return true;
        }
    

private:
    OpenSubdivPtexShaderOverride(const MObject &obj);

    OpenSubdivPtexShader *_shader;
    int _level;

};


OpenSubdivPtexShaderOverride::OpenSubdivPtexShaderOverride(const MObject &obj)
    : MHWRender::MPxShaderOverride(obj),
      _shader(NULL)
{
}

OpenSubdivPtexShaderOverride::~OpenSubdivPtexShaderOverride()
{
}


bool
OpenSubdivPtexShaderOverride::draw(MHWRender::MDrawContext &context, const MHWRender::MRenderItemList &renderItemList) const
{
	using namespace MHWRender;
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

            for(int i = 0; i < (desc.independentBlendEnable ? MHWRender::MBlendState::kMaxTargets : 1); ++i) 
            {    
                desc.targetBlends[i].blendEnable = false;
            }
            blendState = stateMgr->acquireBlendState(desc);
        }

        stateMgr->setDepthStencilState(depthState);
        stateMgr->setBlendState(blendState);
    }

    GLuint program = _shader->getProgram();

    for(int i=0; i< renderItemList.length(); i++){
        const MHWRender::MRenderItem *renderItem = renderItemList.itemAt(i);
//        printf("draw: %d %s\n", i, renderItem->sourceDagPath().fullPathName().asChar());
//        printf("# of indexBuffer = %d\n", renderItem->geometry()->indexBufferCount());

//        const OsdPtexMeshData *data = (dynamic_cast<const OsdPtexMeshData*>(renderItem->customData());
        OsdPtexMeshData *data = (OsdPtexMeshData*)(renderItem->customData());
        if (data == NULL) {
            return false;
        }
        
        data->populateIfNeeded(_shader->getLevel(), _shader->getScheme());

        const MHWRender::MVertexBuffer *position = NULL, *normal = NULL;
        {
            const MHWRender::MGeometry *geometry = renderItem->geometry();
//            printf("Vertex buffer count = %d\n", geometry->vertexBufferCount());
            for(int i = 0; i < geometry->vertexBufferCount(); i++){
                const MHWRender::MVertexBuffer *vb = geometry->vertexBuffer(i);
                const MHWRender::MVertexBufferDescriptor &vdesc = vb->descriptor();
                if (vdesc.name() == "osdPosition")
                    position = vb;
                if (vdesc.name() == "osdNormal")
                    normal = vb;
//                printf("%d: %s, offset=%d, stride=%d\n", i, vdesc.name().asChar(), vdesc.offset(), vdesc.stride());
            }
        }

        glUseProgram(program);
        {
            // shader uniform setting
            GLint position = glGetUniformLocation(program, "lightSource[0].position");
            GLint ambient = glGetUniformLocation(program, "lightSource[0].ambient");
            GLint diffuse = glGetUniformLocation(program, "lightSource[0].diffuse");
            GLint specular = glGetUniformLocation(program, "lightSource[0].specular");

            glProgramUniform4f(program, position, 0, 0.2f, 1, 0);
            glProgramUniform4f(program, ambient, 0.4f, 0.4f, 0.4f, 1.0f);
            glProgramUniform4f(program, diffuse, 0.3f, 0.3f, 0.3f, 1.0f);
            glProgramUniform4f(program, specular, 0.2f, 0.2f, 0.2f, 1.0f);
        
            GLint otcMatrix = glGetUniformLocation(program, "objectToClipMatrix");
            GLint oteMatrix = glGetUniformLocation(program, "objectToEyeMatrix");
            GLint etoMatrix = glGetUniformLocation(program, "eyeToObjectMatrix");

            float modelview[4][4], mvp[4][4], modelviewI[4][4];
            context.getMatrix(MHWRender::MDrawContext::kWorldViewMtx).get(modelview);
            context.getMatrix(MHWRender::MDrawContext::kWorldViewProjMtx).get(mvp);
            context.getMatrix(MHWRender::MDrawContext::kWorldViewInverseMtx).get(modelviewI);

            glProgramUniformMatrix4fv(program, otcMatrix, 1, false, (float*)mvp);
            glProgramUniformMatrix4fv(program, oteMatrix, 1, false, (float*)modelview);
            glProgramUniformMatrix4fv(program, etoMatrix, 1, false, (float*)modelviewI);

            GLint eye = glGetUniformLocation(program, "eyePositionInWorld");
            MPoint e = MPoint(0,0,0) * context.getMatrix(MHWRender::MDrawContext::kWorldViewInverseMtx);
            glProgramUniform3f(program, eye, (float)e.x, (float)e.y, (float)e.z);
        }

        if (_shader->isWireframe()) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        _shader->bindTextures();

//        CHECK_GL_ERROR("pre draw");
        // draw meshdata
        data->prepare();

        data->updateGeometry(position, normal);

        data->draw(program);

//        CHECK_GL_ERROR("post draw");

        glUseProgram(0);
    }
    return true;
}
// ------------------------------------------------------------------------------

using namespace MHWRender;

class OsdBufferGenerator : public MHWRender::MPxVertexBufferGenerator
{
public:
    OsdBufferGenerator(bool normal) : _normal(normal) {}
    virtual ~OsdBufferGenerator() {}

    virtual bool getSourceIndexing(const MDagPath &dagPath, MHWRender::MComponentDataIndexing &sourceIndexing) const
        {
            MStatus status;
            MFnMesh mesh(dagPath.node());
            if (!status) return false;

            MIntArray vertexCount, vertexList;
            mesh.getVertices(vertexCount, vertexList);

            MUintArray &vertices = sourceIndexing.indices();
            for(unsigned int i = 0; i < vertexList.length(); ++i)
                vertices.append((unsigned int)vertexList[i]);

            sourceIndexing.setComponentType(MComponentDataIndexing::kFaceVertex); // .... although not facevertex

            return true;
        }
    virtual bool getSourceStreams(const MDagPath &dagPath, MStringArray &) const
        {
            return false;
        }
#ifdef MAYA2013_PREVIEW
    virtual void createVertexStream(const MDagPath &dagPath, MVertexBuffer &vertexBuffer,
                                    const MComponentDataIndexing &targetIndexing, const MComponentDataIndexing &,
                                    const MVertexBufferArray &) const 
#else
    virtual void createVertexStream(const MDagPath &dagPath, MVertexBuffer &vertexBuffer,
                                    const MComponentDataIndexing &targetIndexing) const 
#endif
        {
            const MVertexBufferDescriptor &desc = vertexBuffer.descriptor();

            MFnMesh meshFn(dagPath);
            int nVertices = meshFn.numVertices();
            if (!_normal) {
                MFloatPointArray points;
                meshFn.getPoints(points);
                
#ifdef MAYA2013_PREVIEW
                float *buffer = (float*)vertexBuffer.acquire(nVertices, true);
#else
                float *buffer = (float*)vertexBuffer.acquire(nVertices);
#endif
                float *dst = buffer;
                for(int i=0; i < nVertices; ++i){
                    *dst++ = points[i].x;
                    *dst++ = points[i].y;
                    *dst++ = points[i].z;
                }
                vertexBuffer.commit(buffer);
            } else {
                MFloatVectorArray normals;
                meshFn.getVertexNormals(true, normals);

#ifdef MAYA2013_PREVIEW
                float *buffer = (float*)vertexBuffer.acquire(nVertices, true);
#else
                float *buffer = (float*)vertexBuffer.acquire(nVertices);
#endif
                float *dst = buffer;
                for(int i=0; i < nVertices; ++i){
                    *dst++ = normals[i].x;
                    *dst++ = normals[i].y;
                    *dst++ = normals[i].z;
                }
                vertexBuffer.commit(buffer);
            }
        }

    static MPxVertexBufferGenerator *positionCreator()
        {
            return new OsdBufferGenerator(false);
        }
    static MPxVertexBufferGenerator *normalCreator()
        {
            return new OsdBufferGenerator(true);
        }
private:
    bool _normal;
};

//---------------------------------------------------------------------------
// Plugin Registration
//---------------------------------------------------------------------------
MStatus
initializePlugin( MObject obj )
{
    MStatus status;
    MFnPlugin plugin( obj, "Pixar", "3.0", "Any" );// vender,version,apiversion
    
    MString swatchName = MHWShaderSwatchGenerator::initialize();
    MString userClassify("shader/surface/utility/:drawdb/shader/surface/OpenSubdivPtexShader:swatch/"+swatchName);

    // shader node
    status = plugin.registerNode( "openSubdivPtexShader",
                                  OpenSubdivPtexShader::id, 
                                  &OpenSubdivPtexShader::creator, 
                                  &OpenSubdivPtexShader::initialize,
                                  MPxNode::kHwShaderNode,
                                  &userClassify);

    MHWRender::MDrawRegistry::registerVertexBufferGenerator("osdPosition",
                                                            OsdBufferGenerator::positionCreator);
    MHWRender::MDrawRegistry::registerVertexBufferGenerator("osdNormal",
                                                            OsdBufferGenerator::normalCreator);

    // shaderoverride
    status = MHWRender::MDrawRegistry::registerShaderOverrideCreator(
        "drawdb/shader/surface/OpenSubdivPtexShader",
        OpenSubdivPtexShader::drawRegistrantId,
        OpenSubdivPtexShaderOverride::creator);

    glewInit();
    OpenSubdiv::OsdCpuKernelDispatcher::Register();
    OpenSubdiv::OsdCudaKernelDispatcher::Register();
    cudaInit();

    OpenSubdiv::OsdPTexture::SetGutterWidth(1);
    OpenSubdiv::OsdPTexture::SetPageMargin(8);
    OpenSubdiv::OsdPTexture::SetGutterDebug(0);

    return status;
}

MStatus
uninitializePlugin( MObject obj )
{
    MFnPlugin plugin( obj );
  
    MStatus status;
    status = plugin.deregisterNode(OpenSubdivPtexShader::id);

    MHWRender::MDrawRegistry::deregisterVertexBufferGenerator("osdPosition");
    MHWRender::MDrawRegistry::deregisterVertexBufferGenerator("osdNormal");

    status = MHWRender::MDrawRegistry::deregisterShaderOverrideCreator(
        "drawdb/shader/surface/OpenSubdivPtexShader",
        OpenSubdivPtexShader::drawRegistrantId);

    return status;
}
