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

#include <osd/cpuDispatcher.h>
#include <osd/cudaDispatcher.h>
#include <osd/mesh.h>
#include <osd/vertexBuffer.h>

#include "hbrUtil.h"

// Maya API includes
#include <maya/MDagPath.h>
#include <maya/MFnPlugin.h>
#include <maya/MFnMesh.h>
#include <maya/MFloatPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MUintArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MPxCommand.h>
#include <maya/MSyntax.h>
#include <maya/MArgDatabase.h>

// Viewport 2.0 includes
#include <maya/MDrawRegistry.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MGlobal.h>
#include <maya/MSelectionList.h>

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

class SubdivUserData : public MUserData
{
public:
    SubdivUserData(bool loop);
    virtual ~SubdivUserData();

    void Populate(MObject mesh);
    void UpdatePoints(MObject mesh);

    int GetElementBuffer() const { return _index; }
    int GetNumIndices() const { return _numIndices; }
    int GetVertexBuffer() const { return _vertexBuffer->GetGpuBuffer(); }
    int GetVaryingBuffer() const { return _varyingBuffer->GetGpuBuffer(); }
    int GetVertexStride() const { return _vertexBuffer->GetNumElements() * sizeof(float); }
    int GetVaryingStride() const { return _varyingBuffer->GetNumElements() * sizeof(float); }
    int GetPrimType() const { return _loop ? GL_TRIANGLES : GL_QUADS; }

    // XXX
    bool fIsSelected;

private:
    // topology cache
    MIntArray _vertexList;
    MUintArray _edgeIds, _vtxIds;
    MDoubleArray _edgeCreaseData, _vtxCreaseData;

    int _level;
    int _interpBoundary;
    bool _loop;

    OpenSubdiv::OsdMesh *_osdmesh;
    OpenSubdiv::OsdVertexBuffer *_vertexBuffer, *_varyingBuffer;
    GLuint _index;

    int _numIndices;
    float _cachedTotal;
};

//---------------------------------------------------------------------------

class OpenSubdivDrawOverride : public MHWRender::MPxDrawOverride
{
public:
    static MHWRender::MPxDrawOverride* Creator(const MObject& obj) {
            return new OpenSubdivDrawOverride(obj);
        }

    virtual ~OpenSubdivDrawOverride();

    virtual MBoundingBox boundingBox(
        const MDagPath& objPath,
        const MDagPath& cameraPath) const;

    virtual MUserData* prepareForDraw(
        const MDagPath& objPath,
        const MDagPath& cameraPath,
        MUserData* oldData);

    static void draw(const MHWRender::MDrawContext& context, const MUserData* data);

    static void setLoopSubdivision(bool loop) { _loop = loop; }

private:
    OpenSubdivDrawOverride(const MObject& obj);

    bool getSelectionStatus(const MDagPath& objPath) const;
    static bool _loop;
};

bool OpenSubdivDrawOverride::_loop = false;

//---------------------------------------------------------------------------

SubdivUserData::SubdivUserData(bool loop) :
    MUserData(false /*don't delete after draw */),
    _loop(loop)
{
    _osdmesh = new OpenSubdiv::OsdMesh();
    glGenBuffers(1, &_index);
}

SubdivUserData::~SubdivUserData()
{
    delete _osdmesh;
    glDeleteBuffers(1, &_index);
}

void
SubdivUserData::Populate(MObject mesh)
{
    MStatus s;
    MFnMesh meshFn(mesh);
    MIntArray vertexCount, vertexList;
    meshFn.getVertices(vertexCount, vertexList);
    MUintArray edgeIds;
    MDoubleArray edgeCreaseData;
    meshFn.getCreaseEdges(edgeIds, edgeCreaseData);
    MUintArray vtxIds;
    MDoubleArray vtxCreaseData;
    meshFn.getCreaseVertices(vtxIds, vtxCreaseData );

    short level = 1;
    FindAttribute(meshFn, "smoothLevel", &level);
    if(level < 1) level = 1;

    short interpBoundary = 0;
    FindAttribute(meshFn, "boundaryRule", &interpBoundary);

    if(CompareArray(_vertexList, vertexList) &&
       CompareArray(_edgeIds, edgeIds) &&
       CompareArray(_edgeCreaseData, edgeCreaseData) &&
       CompareArray(_vtxIds, vtxIds) &&
       CompareArray(_vtxCreaseData, vtxCreaseData) &&
        _level == level &&
       _interpBoundary == interpBoundary)
    {
        return;
    }

    // update topology
    _vertexList = vertexList;
    _edgeIds = edgeIds;
    _edgeCreaseData = edgeCreaseData;
    _vtxIds = vtxIds;
    _vtxCreaseData = vtxCreaseData;
    _level = level;
    _interpBoundary = interpBoundary;

    if(_loop){
        MIntArray triangleCounts;
        meshFn.getTriangles(triangleCounts, vertexList);
        int numTriangles = vertexList.length()/3;
        vertexCount.clear();
        for(int i = 0; i < numTriangles; ++i){
            vertexCount.append(3);
        }
    }

    // XXX redundant copy... replace _vertexList with numIndices, etc

    // create Osd mesh
    std::vector<int> numIndices, faceIndices, edgeCreaseIndices, vtxCreaseIndices;
    std::vector<float> edgeCreases, vtxCreases;
    numIndices.resize(vertexCount.length());
    faceIndices.resize(vertexList.length());
    for(int i = 0; i < (int)vertexCount.length(); ++i) numIndices[i] = vertexCount[i];
    for(int i = 0; i < (int)vertexList.length(); ++i) faceIndices[i] = vertexList[i];
    vtxCreaseIndices.resize(vtxIds.length());
    for(int i = 0; i < (int)vtxIds.length(); ++i) vtxCreaseIndices[i] = vtxIds[i];
    vtxCreases.resize(vtxCreaseData.length());
    for(int i = 0; i < (int)vtxCreaseData.length(); ++i) vtxCreases[i] = (float)vtxCreaseData[i];
    edgeCreases.resize(edgeCreaseData.length());
    for(int i = 0; i < (int)edgeCreaseData.length(); ++i) edgeCreases[i] = (float)edgeCreaseData[i];

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

    OpenSubdiv::OsdHbrMesh *hbrMesh = ConvertToHBR(meshFn.numVertices(), numIndices, faceIndices,
                                                   vtxCreaseIndices, vtxCreases,
                                                   std::vector<int>(), std::vector<float>(),
                                                   edgeCreaseIndices, edgeCreases,
                                                   interpBoundary, _loop);

    if (_vertexBuffer) delete _vertexBuffer;

    int kernel = OpenSubdiv::OsdKernelDispatcher::kCPU;
    if (OpenSubdiv::OsdKernelDispatcher::HasKernelType(OpenSubdiv::OsdKernelDispatcher::kOPENMP)) {
        kernel = OpenSubdiv::OsdKernelDispatcher::kOPENMP;
    }
    _osdmesh->Create(hbrMesh, level, kernel);

    // create vertex buffer
    _vertexBuffer = _osdmesh->InitializeVertexBuffer(6 /* position + normal */);

    delete hbrMesh;

    // update element array buffer
    const std::vector<int> indices = _osdmesh->GetFarMesh()->GetFaceVertices(level);
    _numIndices = (int)indices.size();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _index);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*_numIndices,
                 &(indices[0]), GL_STATIC_DRAW);

    _cachedTotal = -1;
    UpdatePoints(mesh);
}

void
SubdivUserData::UpdatePoints(MObject mesh)
{
    // update coarse vertex array
    MFnMesh meshFn(mesh);
    MStatus status;

    int nPoints = meshFn.numVertices();
    const float *points = meshFn.getRawPoints(&status);

    // XXX: looking for other good way to detect change
    float total = 0;
    for(int i = 0; i < 3*nPoints; ++i) total += points[i];
    if(_cachedTotal == total) return;
    _cachedTotal = total;

    MFloatVectorArray normals;
    meshFn.getVertexNormals(true, normals);
    if (nPoints != normals.length()) return; // XXX: error


    // Update vertex
    std::vector<float> vertex;
    vertex.resize(nPoints*6);

    for(int i = 0; i < nPoints; ++i){
        vertex[i*6+0] = points[i*3+0];
        vertex[i*6+1] = points[i*3+1];
        vertex[i*6+2] = points[i*3+2];
        vertex[i*6+3] = normals[i].x;
        vertex[i*6+4] = normals[i].y;
        vertex[i*6+5] = normals[i].z;
    }

    _vertexBuffer->UpdateData(&vertex.at(0), nPoints);

/* XXX
    float *varying = new float[_osdmesh.GetNumVaryingElements()];
    _osdmesh.BeginUpdateCoarseVertexVarying();
    for(int i = 0; i < nPoints; ++i){
        varying[0] = normals[i].x;
        varying[1] = normals[i].y;
        varying[2] = normals[i].z;
        _osdmesh.UpdateCoarseVertexVarying(i, varying);
    }
    _osdmesh.EndUpdateCoarseVertexVarying();
    delete[] varying;
*/

    // subdivide
    _osdmesh->Subdivide(_vertexBuffer, NULL);
}

//---------------------------------------------------------------------------

OpenSubdivDrawOverride::OpenSubdivDrawOverride(const MObject& obj)
  : MHWRender::MPxDrawOverride(obj, OpenSubdivDrawOverride::draw)
{
}

OpenSubdivDrawOverride::~OpenSubdivDrawOverride()
{
}

bool OpenSubdivDrawOverride::getSelectionStatus(const MDagPath& objPath) const
{
    // retrieve the selection status of the node
    MStatus status;
    MSelectionList selectedList;
    status = MGlobal::getActiveSelectionList(selectedList);
    if(!status)
        return false;

    MDagPath pathCopy = objPath;
    do {
        if(selectedList.hasItem(pathCopy)) return true;
        status = pathCopy.pop();
    } while(status);

    return false;
}

MBoundingBox OpenSubdivDrawOverride::boundingBox(const MDagPath& objPath, const MDagPath& cameraPath) const
{
    MPoint corner1( -1.0, -1.0, -1.0 );
    MPoint corner2( 1.0, 1.0, 1.0);

    return MBoundingBox(corner1, corner2);
}

MUserData* OpenSubdivDrawOverride::prepareForDraw(
    const MDagPath& objPath,
    const MDagPath& cameraPath,
    MUserData* oldData)
{
    SubdivUserData* data = dynamic_cast<SubdivUserData*>(oldData);
    if (!data)
    {
        // data did not exist or was incorrect type, create new
        // XXX by the way, who release this object?
        bool loop = _loop;
        data = new SubdivUserData(loop);
    }
    data->fIsSelected = getSelectionStatus(objPath);
    data->Populate(objPath.node());
    data->UpdatePoints(objPath.node());

    return data;
}

void OpenSubdivDrawOverride::draw(const MHWRender::MDrawContext& context, const MUserData* data)
{
    // get cached data
    bool isSelected = false;
    SubdivUserData* mesh = const_cast<SubdivUserData*>(dynamic_cast<const SubdivUserData*>(data));
    if (mesh)
    {
        isSelected = mesh->fIsSelected;
    }

    // set colour
    static const float colorData[] = {1.0f, 0.0f, 0.0f};
    static const float selectedColorData[] = {0.0f, 1.0f, 0.0f};
    if(isSelected)
        glColor3fv(selectedColorData);
    else
        glColor3fv(colorData);
    MStatus status;

    // set world matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    MMatrix transform =
        context.getMatrix(MHWRender::MDrawContext::kWorldViewMtx, &status);
    if (status)
    {
        glLoadMatrixd(transform.matrix[0]);
    }

    // set projection matrix
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    MMatrix projection =
        context.getMatrix(MHWRender::MDrawContext::kProjectionMtx, &status);
    if (status)
    {
        glLoadMatrixd(projection.matrix[0]);
    }


    const int displayStyle = context.getDisplayStyle();
    glPushAttrib( GL_CURRENT_BIT );
    glPushAttrib( GL_ENABLE_BIT);

    if(displayStyle & MHWRender::MDrawContext::kGouraudShaded) {
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }else if(displayStyle & MHWRender::MDrawContext::kWireFrame){
        glDisable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }

    {
        int vertexStride = mesh->GetVertexStride();
//        int varyingStride = mesh->GetVaryingStride();
        //printf("Draw. stride = %d\n", stride);
        glBindBuffer(GL_ARRAY_BUFFER, mesh->GetVertexBuffer());
        glVertexPointer(3, GL_FLOAT, vertexStride, ((char*)(0)));
        glEnableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, mesh->GetVertexBuffer());
        glNormalPointer(GL_FLOAT, vertexStride, ((char*)(12)));
//        glBindBuffer(GL_ARRAY_BUFFER, mesh->GetVaryingBuffer());
//        glNormalPointer(GL_FLOAT, varyingStride, ((char*)(0)));
        glEnableClientState(GL_NORMAL_ARRAY);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->GetElementBuffer());
        glDrawElements(mesh->GetPrimType(), mesh->GetNumIndices(), GL_UNSIGNED_INT, NULL);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    glPopAttrib();
    glPopAttrib();

    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glColor3f(1, 1, 1);
}
//---------------------------------------------------------------------------
// Control command

class OpenSubdivCommand : public MPxCommand
{
public:
    virtual MStatus doIt(const MArgList &args);
    static void *Creator();
};

void*
OpenSubdivCommand::Creator()
{
    return new OpenSubdivCommand();
}

MStatus
OpenSubdivCommand::doIt(const MArgList &args)
{
    MSyntax syntax;
    syntax.addFlag("m", "method", MSyntax::kString);
    MArgDatabase argDB(syntax, args);

    if(argDB.isFlagSet("m")){
        MString method;
        argDB.getFlagArgument("m", 0, method);
        if(method == "loop"){
            OpenSubdivDrawOverride::setLoopSubdivision(true);
        }else{
            OpenSubdivDrawOverride::setLoopSubdivision(false);
        }
    }

    return MS::kSuccess;
}

//---------------------------------------------------------------------------
// Plugin Registration
//---------------------------------------------------------------------------

MString drawDbClassification("drawdb/geometry/mesh");
MString drawRegistrantId("OpenSubdivDrawOverridePlugin");

MStatus initializePlugin( MObject obj )
{
#if defined(_WIN32) && defined(_DEBUG)
    // Disable buffering for stdout and stderr when using debug versions
    // of the C run-time library.
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
#endif

    MStatus   status;
    MFnPlugin plugin( obj, PLUGIN_COMPANY, "3.0", "Any");

    status = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
        drawDbClassification,
        drawRegistrantId,
        OpenSubdivDrawOverride::Creator);
    if (!status) {
        status.perror("registerDrawOverrideCreator");
        return status;
    }

    status = plugin.registerCommand("openSubdivControl", OpenSubdivCommand::Creator);
    if (!status) {
        status.perror("registerCommand");
        return status;
    }

    glewInit();

    //XXX:cleanup  Need to register other kernel dispatchers.
    OpenSubdiv::OsdCpuKernelDispatcher::Register();
#if OPENSUBDIV_HAS_CUDA
    OpenSubdiv::OsdCudaKernelDispatcher::Register();
#endif

    return status;
}

MStatus uninitializePlugin( MObject obj)
{
    MStatus   status;
    MFnPlugin plugin( obj );

    status = MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
        drawDbClassification,
        drawRegistrantId);
    if (!status) {
        status.perror("deregisterDrawOverrideCreator");
        return status;
    }

    status = plugin.deregisterCommand("openSubdivControl");

    return status;
}
