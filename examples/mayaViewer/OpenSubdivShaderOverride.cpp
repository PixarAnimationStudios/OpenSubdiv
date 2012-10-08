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
#include <osd/cpuDispatcher.h>
#include <osd/clDispatcher.h>
#include <osd/cudaDispatcher.h>
#include <osd/elementArrayBuffer.h>

extern void cudaInit();
#include "hbrUtil.h"

#define CHECK_GL_ERROR(...)  \
    if(GLuint err = glGetError()) {   \
    printf("GL error %x :", err); \
    printf(__VA_ARGS__); \
    }

#define MAYA2013_PREVIEW

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
class OpenSubdivShader : public MPxHwShaderNode
{
public:
    OpenSubdivShader();
    virtual ~OpenSubdivShader();

    static void *creator();
    static MStatus initialize();

    virtual void postConstructor() 
        {
            setMPSafe(false);
        }

    virtual MStatus compute(const MPlug &plug, MDataBlock &data);
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
    
    virtual bool setInternalValueInContext(const MPlug &plug, const MDataHandle &handle, MDGContext &);
    
    void updateAttributes();
    bool isWireframe() const { return _wireframe; }
    int getLevel() const { return _level; }
    int getScheme() const { return _scheme; }
    int getKernel() const { return _kernel; }

    static MTypeId id;
    static MString drawRegistrantId;

    enum { kCatmark, kLoop, kBilinear };

private:
    static MObject aLevel;
    static MObject aScheme;
    static MObject aKernel;
    static MObject aDiffuse;
    static MObject aSpecular;
    static MObject aAmbient;
    static MObject aShininess;
    static MObject aLight;
    static MObject aWireframe;

    bool _wireframe;
    int _level;
    int _scheme;
    int _kernel;
};

MTypeId OpenSubdivShader::id(0x88110);
MString OpenSubdivShader::drawRegistrantId("OpenSubdivShaderPlugin");
MObject OpenSubdivShader::aLevel;
MObject OpenSubdivShader::aScheme;
MObject OpenSubdivShader::aKernel;
MObject OpenSubdivShader::aDiffuse;
MObject OpenSubdivShader::aSpecular;
MObject OpenSubdivShader::aAmbient;
MObject OpenSubdivShader::aShininess;
MObject OpenSubdivShader::aLight;
MObject OpenSubdivShader::aWireframe;

OpenSubdivShader::OpenSubdivShader()
    : _wireframe(false),
      _level(1),
      _scheme(kCatmark),
      _kernel(OpenSubdiv::OsdKernelDispatcher::kCPU)
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
    MStatus stat;
    MFnTypedAttribute typedAttr;
    MFnNumericAttribute numAttr;
    MFnEnumAttribute enumAttr;

    aLevel = numAttr.create("level", "lv", MFnNumericData::kLong, 1);
    numAttr.setInternal(true);
    numAttr.setMin(1);
    numAttr.setSoftMax(5);

    aScheme = enumAttr.create("scheme", "sc");
    enumAttr.addField("Catmull-Clark", kCatmark);
    enumAttr.addField("Loop", kLoop);
    enumAttr.addField("Bilinear", kBilinear);
    enumAttr.setInternal(true);

    aKernel = enumAttr.create("kernel", "kn");
    enumAttr.addField("CPU", OpenSubdiv::OsdKernelDispatcher::kCPU);
    if (OpenSubdiv::OsdKernelDispatcher::HasKernelType(OpenSubdiv::OsdKernelDispatcher::kOPENMP))
        enumAttr.addField("OpenMP", OpenSubdiv::OsdKernelDispatcher::kOPENMP);
    if (OpenSubdiv::OsdKernelDispatcher::HasKernelType(OpenSubdiv::OsdKernelDispatcher::kCL))
        enumAttr.addField("CL", OpenSubdiv::OsdKernelDispatcher::kCL);
    if (OpenSubdiv::OsdKernelDispatcher::HasKernelType(OpenSubdiv::OsdKernelDispatcher::kCUDA))
        enumAttr.addField("CUDA", OpenSubdiv::OsdKernelDispatcher::kCUDA);
    enumAttr.setInternal(true);

    aDiffuse = numAttr.createColor("diffuse", "d");
    numAttr.setDefault(1.0f, 1.0f, 1.0f);
    aSpecular = numAttr.createColor("specular", "s");
    numAttr.setDefault(1.0f, 1.0f, 1.0f);
    aAmbient = numAttr.createColor("ambient", "a");
    numAttr.setDefault(1.0f, 1.0f, 1.0f);
    aShininess = numAttr.create("shininess", "shin", MFnNumericData::kFloat, 16.0f);
    numAttr.setMin(0);
    numAttr.setMax(128.0f);

    aWireframe = numAttr.create("wireframe", "wf", MFnNumericData::kBoolean);

    addAttribute(aLevel);
    addAttribute(aScheme);
    addAttribute(aKernel);
    addAttribute(aDiffuse);
    addAttribute(aSpecular);
    addAttribute(aAmbient);
    addAttribute(aShininess);
    addAttribute(aWireframe);

    return MS::kSuccess;
}

MStatus
OpenSubdivShader::compute(const MPlug &plug, MDataBlock &data)
{
    return MS::kSuccess;
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

void
OpenSubdivShader::updateAttributes()
{
    MObject object = thisMObject();
    MColor diffuse = getColor(object, aDiffuse);
    MColor ambient = getColor(object, aAmbient);
    MColor specular = getColor(object, aSpecular);

    float shininess = 0.0f;
    MFnDependencyNode depFn(object);
    FindAttribute(depFn, "wireframe", &_wireframe);
    FindAttribute(depFn, "shininess", &shininess);
    
    float color[4] = { 0, 0, 0, 1 };
    diffuse.get(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    ambient.get(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
    specular.get(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
}

//------------------------------------------------------------------------------
bool
OpenSubdivShader::setInternalValueInContext(const MPlug &plug, const MDataHandle &handle, MDGContext &)
{
    if(plug == aLevel)
    {
        _level = handle.asLong();
    }
    else if(plug == aScheme)
    {
        _scheme = handle.asShort();
    }
    else if(plug == aKernel)
    {
        _kernel = handle.asShort();
    }
    return false;
}

//---------------------------------------------------------------------------
// Viewport 2.0 override implementation
//---------------------------------------------------------------------------
class OsdMeshData : public MUserData
{
public:
    OsdMeshData(MObject mesh);
    virtual ~OsdMeshData();

    void populateIfNeeded(int level, int scheme, int kernel);
    void updateGeometry(const MHWRender::MVertexBuffer *point, const MHWRender::MVertexBuffer *normal);
    void prepare();
    void draw() const;
    void update();

private:
    void initializeOsd();
    void initializeIndexBuffer();

    MObject _mesh;

    OpenSubdiv::OsdHbrMesh *_hbrmesh;
    OpenSubdiv::OsdMesh *_osdmesh;
    OpenSubdiv::OsdVertexBuffer *_positionBuffer, *_normalBuffer;
    OpenSubdiv::OsdElementArrayBuffer *_elementArrayBuffer;

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
    int _kernel;

    bool _needsUpdate;
    bool _needsInitializeOsd;
    bool _needsInitializeIndexBuffer;
};

OsdMeshData::OsdMeshData(MObject mesh) :
    MUserData(false), _mesh(mesh),
    _hbrmesh(NULL), _osdmesh(NULL),
    _positionBuffer(NULL), _normalBuffer(NULL),
    _elementArrayBuffer(NULL),
    _level(0),
    _interpBoundary(0),
    _scheme(OpenSubdivShader::kCatmark),
    _kernel(OpenSubdiv::OsdKernelDispatcher::kCPU),
    _needsUpdate(false),
    _needsInitializeOsd(false),
    _needsInitializeIndexBuffer(false)
{
}

OsdMeshData::~OsdMeshData() 
{
    if (_hbrmesh) delete _hbrmesh;
    if (_osdmesh) delete _osdmesh;
    if (_positionBuffer) delete _positionBuffer;
    if (_normalBuffer) delete _normalBuffer;
    if (_elementArrayBuffer) delete _elementArrayBuffer;
}

void
OsdMeshData::populateIfNeeded(int lv, int scheme, int kernel)
{
    MStatus s;

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
       _kernel == kernel &&
       _maxlevel >= lv)
    {
        if(_level != lv) {
            _level = lv;
            _needsInitializeIndexBuffer = true;
        }
        return;
    }

    // printf("Populate %s, level = %d < %d\n", meshFn.name().asChar(), lv, _level);

    // update topology
    _vertexList = vertexList;
    _edgeIds = edgeIds;
    _edgeCreaseData = edgeCreaseData;
    _vtxIds = vtxIds;
    _vtxCreaseData = vtxCreaseData;
    _maxlevel = lv;
    _level = lv;
    _interpBoundary = interpBoundary;
    _scheme = scheme;
    _kernel = kernel;

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
OsdMeshData::initializeOsd()
{
    if (!_hbrmesh)
        return;

    // create Osd mesh
    if (_osdmesh)
        delete _osdmesh;

    _osdmesh = new OpenSubdiv::OsdMesh();
    _osdmesh->Create(_hbrmesh, _level, _kernel);
    delete _hbrmesh;
    _hbrmesh = NULL;

    // create vertex buffer
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
OsdMeshData::updateGeometry(const MHWRender::MVertexBuffer *points, const MHWRender::MVertexBuffer *normals)
{
    // Update coarse vertex
    if (!_positionBuffer) return;
    if (!_normalBuffer) return;

    int nCoarsePoints = _pointArray.length();

    OpenSubdiv::OsdCpuVertexBuffer *cpuPos = dynamic_cast<OpenSubdiv::OsdCpuVertexBuffer*>(_positionBuffer);
    OpenSubdiv::OsdCpuVertexBuffer *cpuNormal = dynamic_cast<OpenSubdiv::OsdCpuVertexBuffer*>(_normalBuffer);

    if (cpuPos) {
        // I know, this is very inefficient...
        float *d_pos = cpuPos->GetCpuBuffer();
        float *d_normal = cpuNormal->GetCpuBuffer();

        glBindBuffer(GL_ARRAY_BUFFER, *(GLuint*)points->resourceHandle());
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, nCoarsePoints*3*sizeof(float), d_pos);
        glBindBuffer(GL_ARRAY_BUFFER, *(GLuint*)normals->resourceHandle());
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, nCoarsePoints*3*sizeof(float), d_normal);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    } else {
        glBindBuffer(GL_COPY_READ_BUFFER, *(GLuint*)points->resourceHandle());
        glBindBuffer(GL_COPY_WRITE_BUFFER, _positionBuffer->GetGpuBuffer());
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, nCoarsePoints*3*sizeof(float));
        
        glBindBuffer(GL_COPY_READ_BUFFER, *(GLuint*)normals->resourceHandle());
        glBindBuffer(GL_COPY_WRITE_BUFFER, _normalBuffer->GetGpuBuffer());
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, nCoarsePoints*3*sizeof(float));
        
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }

    _osdmesh->Subdivide(_positionBuffer, NULL);
    _osdmesh->Subdivide(_normalBuffer, NULL);

    _needsUpdate = false;
}

void
OsdMeshData::initializeIndexBuffer()
{
    // create element array buffer
    if (_elementArrayBuffer) delete _elementArrayBuffer;
    _elementArrayBuffer = _osdmesh->CreateElementArrayBuffer(_level);

    _needsInitializeIndexBuffer = false;
}

void
OsdMeshData::prepare()
{
    if (_needsInitializeOsd) {
        const_cast<OsdMeshData*>(this)->initializeOsd();
    }
    if (_needsInitializeIndexBuffer) {
        const_cast<OsdMeshData*>(this)->initializeIndexBuffer();
    }
}

void
OsdMeshData::draw() const
{
    MStatus status;

    CHECK_GL_ERROR("draw begin\n");

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    GLuint bPosition = _positionBuffer->GetGpuBuffer();
    GLuint bNormal = _normalBuffer->GetGpuBuffer();
    glBindBuffer(GL_ARRAY_BUFFER, bPosition);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, bNormal);
    glNormalPointer(GL_FLOAT, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _elementArrayBuffer->GetGlBuffer());

    glDrawElements(_scheme == OpenSubdivShader::kLoop ? GL_TRIANGLES : GL_QUADS,
                   _elementArrayBuffer->GetNumIndices(), GL_UNSIGNED_INT, 0);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    CHECK_GL_ERROR("draw end\n");
}

// ------------------------------------------------------------------------------
class OpenSubdivShaderOverride : public MHWRender::MPxShaderOverride
{
public:
    static MHWRender::MPxShaderOverride* creator(const MObject &obj)
        {
            return new OpenSubdivShaderOverride(obj);
        }
    virtual ~OpenSubdivShaderOverride();
    
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
                OsdMeshData *data = new OsdMeshData(initContext.dagPath.node());
                initFeedback.customData = data;
            }

            return MString("OpenSubdivShaderOverride");
        }

    virtual void updateDG(MObject object)
        {
            if (object == MObject::kNullObj) 
                return;

            _shader = (OpenSubdivShader*)MPxHwShaderNode::getHwShaderNodePtr(object);
            if (_shader) {
                _shader->updateAttributes();
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
    OpenSubdivShaderOverride(const MObject &obj);

    OpenSubdivShader *_shader;
    int _level;

};


OpenSubdivShaderOverride::OpenSubdivShaderOverride(const MObject &obj)
    : MHWRender::MPxShaderOverride(obj),
      _shader(NULL)
{
}

OpenSubdivShaderOverride::~OpenSubdivShaderOverride()
{
}


bool
OpenSubdivShaderOverride::draw(MHWRender::MDrawContext &context, const MHWRender::MRenderItemList &renderItemList) const
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

    for(int i=0; i< renderItemList.length(); i++){
        const MHWRender::MRenderItem *renderItem = renderItemList.itemAt(i);
        OsdMeshData *data = (OsdMeshData*)(renderItem->customData());
        if (data == NULL) {
            return false;
        }
        
        data->populateIfNeeded(_shader->getLevel(), _shader->getScheme(), _shader->getKernel());

        const MHWRender::MVertexBuffer *position = NULL, *normal = NULL;
        {
            const MHWRender::MGeometry *geometry = renderItem->geometry();
            for(int i = 0; i < geometry->vertexBufferCount(); i++){
                const MHWRender::MVertexBuffer *vb = geometry->vertexBuffer(i);
                const MHWRender::MVertexBufferDescriptor &vdesc = vb->descriptor();
                if (vdesc.name() == "osdPosition")
                    position = vb;
                if (vdesc.name() == "osdNormal")
                    normal = vb;
            }
        }

        float diffuse[4] = {1, 1, 1, 1};
        float ambient[4] = {0.1f, 0.1f, 0.1f, 0.1f};
        float specular[4] = {1, 1, 1, 1};
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular);

        glPushAttrib(GL_POLYGON_BIT);
        glPushAttrib(GL_ENABLE_BIT);

        if (_shader->isWireframe()) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDisable(GL_LIGHTING);
            glDisable(GL_LIGHT0);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glEnable(GL_LIGHTING);
            glEnable(GL_LIGHT0);
        }

        // draw meshdata
        data->prepare();

        data->updateGeometry(position, normal);

        data->draw();

        glPopAttrib();
        glPopAttrib();
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

            sourceIndexing.setComponentType(MComponentDataIndexing::kFaceVertex);

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
    MString userClassify("shader/surface/utility/:drawdb/shader/surface/OpenSubdivShader:swatch/"+swatchName);

    glewInit();
    OpenSubdiv::OsdCpuKernelDispatcher::Register();
    OpenSubdiv::OsdCudaKernelDispatcher::Register();
    OpenSubdiv::OsdClKernelDispatcher::Register();
    cudaInit();

    // shader node
    status = plugin.registerNode( "openSubdivShader",
                                  OpenSubdivShader::id, 
                                  &OpenSubdivShader::creator, 
                                  &OpenSubdivShader::initialize,
                                  MPxNode::kHwShaderNode,
                                  &userClassify);

    MHWRender::MDrawRegistry::registerVertexBufferGenerator("osdPosition",
                                                            OsdBufferGenerator::positionCreator);
    MHWRender::MDrawRegistry::registerVertexBufferGenerator("osdNormal",
                                                            OsdBufferGenerator::normalCreator);

    // shaderoverride
    status = MHWRender::MDrawRegistry::registerShaderOverrideCreator(
        "drawdb/shader/surface/OpenSubdivShader",
        OpenSubdivShader::drawRegistrantId,
        OpenSubdivShaderOverride::creator);

    return status;
}

MStatus
uninitializePlugin( MObject obj )
{
    MFnPlugin plugin( obj );
  
    MStatus status;
    status = plugin.deregisterNode(OpenSubdivShader::id);

    MHWRender::MDrawRegistry::deregisterVertexBufferGenerator("osdPosition");
    MHWRender::MDrawRegistry::deregisterVertexBufferGenerator("osdNormal");

    status = MHWRender::MDrawRegistry::deregisterShaderOverrideCreator(
        "drawdb/shader/surface/OpenSubdivShader",
        OpenSubdivShader::drawRegistrantId);

    return status;
}
