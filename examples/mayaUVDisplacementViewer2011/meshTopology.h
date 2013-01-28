#include <maya/MFnMesh.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MFloatArray.h>
#include <maya/MFloatVectorArray.h>

#include <GL/glew.h>
#include <maya/MFnPlugin.h>
#include <maya\MPxNode.h>

#include <maya/MDataBlock.h>
#include <maya/MPlug.h>
#include <maya/MFnMesh.h>
#include <maya/MFnMeshData.h>

#include <maya/MMeshSmoothOptions.h>

#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>

#include <maya/MPointArray.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MIntArray.h>

#include <maya/MItMeshEdge.h>
#include <maya/MItMeshVertex.h>

#include <maya/MMatrix.h>

#include <maya/MGlobal.h>
#include <maya/MString.h>

#include <maya/MPxSurfaceShape.h>
#include <maya/MPxSurfaceShapeUI.h>

#include <maya/MDrawRequest.h>
#include <maya/M3dView.h>
#include <maya/MDrawData.h>

#include <maya/MFnEnumAttribute.h>

#include <GL/GL.H>

#include <maya/MMaterial.h>
#include <maya/MDagPath.h>

#include <maya/MFloatArray.h>

#include <maya/MRenderUtil.h>
#include <maya/MPlugArray.h>
#include <maya/MFloatVectorArray.h>
#include <maya/MFloatVector.h>
#include <maya/MFloatMatrix.h>

#include <maya/MGlobal.h>

#include <maya/MItMeshPolygon.h>
#include <maya/MItMeshFaceVertex.h>

#include <osd/mutex.h>
#include <osd/mesh.h>
#include <osd/cpuDispatcher.h>
#include <osd/clDispatcher.h>
#include <osd/cudaDispatcher.h>
#include <osd/elementArrayBuffer.h>
#include <osd/ptexCoordinatesTextureBuffer.h>

#include <maya/MUintArray.h>

#include <maya/MSelectInfo.h>
#include <maya/MSelectionList.h>
#include <maya/MSelectionMask.h>
#include <maya/MFloatPointArray.h>
#include <maya/MFloatPoint.h>

#include <maya/MMatrix.h>

#include <osd/glslDispatcher.h>

#include <far/table.h>

#include <exception>

#include "hbrUtil.h"
extern void cudaInit();

#define LEAD_COLOR				18	// green
#define ACTIVE_COLOR			15	// white
#define ACTIVE_AFFECTED_COLOR	8	// purple
#define DORMANT_COLOR			4	// blue
#define HILITE_COLOR			17	// pale blue
#define DORMANT_VERTEX_COLOR	8	// purple
#define ACTIVE_VERTEX_COLOR		16	// yellow

const int NOTHINGCHANGE=0;
const int TOPOLOGYCHANGE=1;
const int UVCHANGE=2;
const int VERTEXCHANGE=3;

template<class T> static bool
	CompareArray(const T &a, const T &b)
{
	if(a.length() != b.length()) return false;
	for(unsigned int i = 0; i < a.length(); ++i){
		if(a[i] != b[i]) return false;
	}
	return true;
}

template<class T> static bool
	CompareVectorArray(const T &a, const T &b)
{
	if(a.size() != b.size()) return false;
	for(unsigned int i = 0; i < a.size(); ++i){
		if(a[i] != b[i]) return false;
	}
	return true;
}

struct meshTopology
{
	

	int _level;

	int changeStatus;

	std::vector<int> vertexCount, vertexList, edgeCreaseIndices, vtxCreaseIndices;
	std::vector<int> edgeCreaseVtxIndices;
	std::vector<float> edgeCreases, vtxCreases;
	std::vector<std::vector<float>> uvWeight;

	int faceCount;
	int vertexNum;
	std::vector<int> grpIdx;

	MPointArray positions;
	MFloatVectorArray normals;
	MFloatArray uvDatas;

	meshTopology(){_level=3;changeStatus=NOTHINGCHANGE;faceCount=0;}
	meshTopology(const MObject &meshObj)
	{
		MIntArray _vertexCount;
		//std::vector<int> vertexCount;
		MIntArray _vertexList;
		//std::vector<int> vertexList;
		MUintArray _edgeIds, _vtxIds;
		//std::vector<int> edgeCreaseIndices, vtxCreaseIndices;
		//std::vector<int> edgeCreaseVtxIndices;
		MDoubleArray _edgeCreaseData, _vtxCreaseData;
		//std::vector<float> edgeCreases, vtxCreases;
		
		MFnMesh fnMesh(meshObj);
		fnMesh.getVertices(_vertexCount,_vertexList);
		fnMesh.getCreaseEdges(_edgeIds,_edgeCreaseData);
		fnMesh.getCreaseVertices(_vtxIds,_vtxCreaseData);

		vertexCount.resize(_vertexCount.length());
		for(int i = 0; i < (int)_vertexCount.length(); ++i) 
			vertexCount[i] = _vertexCount[i];
		vertexList.resize(_vertexList.length());
		for(int i = 0; i < (int)_vertexList.length(); ++i) 
			vertexList[i] = _vertexList[i];
		vtxCreaseIndices.resize(_vtxIds.length());
		for(int i = 0; i < (int)_vtxIds.length(); ++i) 
			vtxCreaseIndices[i] = _vtxIds[i];
		vtxCreases.resize(_vtxCreaseData.length());
		for(int i = 0; i < (int)_vtxCreaseData.length(); ++i) 
			vtxCreases[i] = (float)_vtxCreaseData[i];
		edgeCreases.resize(_edgeCreaseData.length());
		for(int i = 0; i < (int)_edgeCreaseData.length(); ++i) 
			edgeCreases[i] = (float)_edgeCreaseData[i];

		int nEdgeIds = _edgeIds.length();
		edgeCreaseVtxIndices.resize(nEdgeIds*2);
		edgeCreaseIndices.resize(nEdgeIds);
		for(int i = 0; i < nEdgeIds; ++i)
		{
			int2 vertices;
			fnMesh.getEdgeVertices(_edgeIds[i], vertices) ;
			edgeCreaseVtxIndices[i*2] = vertices[0];
			edgeCreaseVtxIndices[i*2+1] = vertices[1];
			edgeCreaseIndices[i]=_edgeIds[i];
		}

		uvWeight.resize(_vertexCount.length());
		MItMeshPolygon itFace(meshObj);
		for(itFace.reset();!itFace.isDone();itFace.next())
		{
			uvWeight[itFace.index()].resize(itFace.polygonVertexCount()*4);
			for(int i=0;i<itFace.polygonVertexCount();i++)
			{
				float2 uvpoint;
				itFace.getUV(i,uvpoint);
				uvDatas.append(uvpoint[0]);
				uvDatas.append(uvpoint[1]);

				uvWeight[itFace.index()][i*4]=0;
				uvWeight[itFace.index()][i*4+1]=0;
				uvWeight[itFace.index()][i*4+2]=0;
				uvWeight[itFace.index()][i*4+3]=0;
				uvWeight[itFace.index()][i*4+i]=1;
			}
			if(itFace.polygonVertexCount()==3)
			{
				uvDatas.append(0);
				uvDatas.append(0);
			}
		}
		fnMesh.getPoints(positions,MSpace::kObject);
		fnMesh.getVertexNormals(false,normals);

		faceCount=fnMesh.numPolygons();
		cout<<"face"<<faceCount<<" "<<vertexCount.size()<<endl;
		vertexNum=fnMesh.numVertices();
		grpIdx.resize(faceCount,0);
	}

	void update(meshTopology& newMeshTopo)
	{
		vertexCount.resize(newMeshTopo.vertexCount.size());
		for(int i=0;i<newMeshTopo.vertexCount.size();i++)
			vertexCount[i]=newMeshTopo.vertexCount[i];

		vertexList.resize(newMeshTopo.vertexList.size());
		for(int i=0;i<newMeshTopo.vertexList.size();i++)
			vertexList[i]=newMeshTopo.vertexList[i];

		edgeCreaseIndices.resize(newMeshTopo.edgeCreaseIndices.size());
		for(int i=0;i<newMeshTopo.edgeCreaseIndices.size();i++)
			edgeCreaseIndices[i]=newMeshTopo.edgeCreaseIndices[i];

		edgeCreaseVtxIndices.resize(newMeshTopo.edgeCreaseVtxIndices.size());
		for(int i=0;i<newMeshTopo.edgeCreaseVtxIndices.size();i++)
			edgeCreaseVtxIndices[i]=newMeshTopo.edgeCreaseVtxIndices[i];

		edgeCreases.resize(newMeshTopo.edgeCreases.size());
		for(int i=0;i<newMeshTopo.edgeCreases.size();i++)
			edgeCreases[i]=newMeshTopo.edgeCreases[i];

		vtxCreaseIndices.resize(newMeshTopo.vtxCreaseIndices.size());
		for(int i=0;i<newMeshTopo.vtxCreaseIndices.size();i++)
			vtxCreaseIndices[i]=newMeshTopo.vtxCreaseIndices[i];

		vtxCreases.resize(newMeshTopo.vtxCreases.size());
		for(int i=0;i<newMeshTopo.vtxCreases.size();i++)
			vtxCreases[i]=newMeshTopo.vtxCreases[i];

		uvWeight.resize(newMeshTopo.uvWeight.size());
		for(int i=0;i<newMeshTopo.uvWeight.size();i++)
		{
			uvWeight[i].resize(newMeshTopo.uvWeight[i].size());
			for(int j=0;j<newMeshTopo.uvWeight[i].size();j++)
				uvWeight[i][j]=newMeshTopo.uvWeight[i][j];
		}
	
/*
// 		vertexList.clear();
// 		vertexList.swap(newMeshTopo.vertexList);
// 		edgeCreaseIndices.clear();
// 		edgeCreaseIndices.swap(newMeshTopo.edgeCreaseIndices);
// 		edgeCreaseVtxIndices.clear();
// 		edgeCreaseVtxIndices.swap(newMeshTopo.edgeCreaseVtxIndices);
// 		edgeCreases.clear();
// 		edgeCreases.swap(newMeshTopo.edgeCreases);
// 		vtxCreaseIndices.clear();
// 		vtxCreaseIndices.swap(newMeshTopo.vtxCreaseIndices);
// 		vtxCreases.clear();
// 		vtxCreases.swap(newMeshTopo.vtxCreases);
// 		uvWeight.clear();
// 		uvWeight.swap(newMeshTopo.uvWeight);
*/
		uvDatas=newMeshTopo.uvDatas;
		positions=newMeshTopo.positions;
		normals=newMeshTopo.normals;
		faceCount=newMeshTopo.faceCount;
		vertexNum=newMeshTopo.vertexNum;
		grpIdx.resize(faceCount,0);
	}

	void getUpdateInfo(meshTopology& newMeshTopo)
	{
		if(		!(CompareVectorArray(vertexCount,newMeshTopo.vertexCount)&&
				CompareVectorArray(vertexList,newMeshTopo.vertexList)&&
				CompareVectorArray(edgeCreaseIndices,newMeshTopo.edgeCreaseIndices)&&
				CompareVectorArray(vtxCreaseIndices,newMeshTopo.vtxCreaseIndices)&&
				CompareVectorArray(edgeCreases,newMeshTopo.edgeCreases)&&
				CompareVectorArray(vtxCreases,newMeshTopo.vtxCreases)))
		{
			update(newMeshTopo);
			changeStatus=TOPOLOGYCHANGE;
		}
		else if(!CompareArray(uvDatas,newMeshTopo.uvDatas))
		{
			update(newMeshTopo);
			changeStatus=UVCHANGE;
		}
		else if(!CompareArray(positions,newMeshTopo.positions))
		{
			update(newMeshTopo);
			changeStatus=VERTEXCHANGE;
		}
		else
			changeStatus=NOTHINGCHANGE;

	}
	void getUpdateInfo(int subLevel)
	{
		if(_level!=subLevel)
		{
			_level=subLevel;
			changeStatus=TOPOLOGYCHANGE;
		}
	}
};

struct drawData
{
	OpenSubdiv::OsdGpuVertexBuffer *_gpuVertexBuffer,*_gpuNormalBuffer;
	OpenSubdiv::OsdElementArrayBuffer *_elementArrayBuffer;
	int length;
	bool draw;
	drawData()
	{
		cout<<"drawData"<<endl;
		length=0;
		cout<<"drawData"<<endl;
		_gpuNormalBuffer=NULL;
		_gpuVertexBuffer=NULL;
		_elementArrayBuffer=NULL;
/*
// 		if(_gpuNormalBuffer)
// 			delete _gpuNormalBuffer;
// 		cout<<"drawData"<<endl;
// 		if(_gpuVertexBuffer)
// 			delete _gpuVertexBuffer;
// 		cout<<"drawData"<<endl;
// 		if(_elementArrayBuffer)
// 			delete _elementArrayBuffer;
// 		cout<<"drawData"<<endl;
*/
	}

 	~drawData()
 	{
 		if(_gpuNormalBuffer)
 			delete _gpuNormalBuffer;
 		if(_gpuVertexBuffer)
 			delete _gpuVertexBuffer;
 		if(_elementArrayBuffer)
 			delete _elementArrayBuffer;
		length=0;
 	}

};

struct dataCompute
{
	vector<float> vertex,normal,oriVertex,oriNormal;
	int length;
	MFloatPointArray oriVertexFor3D;
	MFloatVectorArray color;
	//vector<float> color;
	bool dirty;
	dataCompute(){}
	void setLength(int length)
	{
		this->length=length;
		vertex.resize(length);
		normal.resize(length);
		oriVertex.resize(length);
		oriNormal.resize(length);
		oriVertexFor3D.setLength(length/3);
		color.setLength(length/3);
	}
	void build3DVertex()
	{
		for(int i=0;i<length/3;i++)
		{
			oriVertexFor3D.set(MFloatPoint(oriVertex[3*i+0],
															oriVertex[3*i+1],
															oriVertex[3*i+2]),
											i);
		}
	}
	
	void sample(int & dirtyCount,MString texName)
	{
		if(dirty)
		{
			dirtyCount++;
			//sample
			if (texName!="")
			{
				MFloatMatrix cam;
				MFloatVectorArray resultTrans;
				MRenderUtil::sampleShadingNetwork(texName,length/3,false,false,
					cam,&oriVertexFor3D,NULL,NULL,NULL,&oriVertexFor3D,NULL,NULL,NULL,color,resultTrans);
			}
			else
			{
				cout<<"clear3d"<<endl;
				color.clear();
				color=MFloatVectorArray(length/3);
			}
			dirty=false;
		}
	}
};

struct uvInfo
{
	float u;
	float v;
	int faceid;
	float w0,w1,w2,w3;

	uvInfo()
	{
		faceid=-1;
	}
};

struct uv2DColor
{
	vector<uvInfo> uv2dInfo;
	vector<float> color;//finalColor
	int length;
	bool dirty;
	//MFloatVectorArray color;
	uv2DColor(){}
	void setLength(int length)
	{
		this->length=length;
		uv2dInfo.resize(this->length);
		color.resize(this->length);
	}

	void updateUV(MFloatArray uvDatas)
	{
		for(int i=0;i<length;i++)
		{
#ifdef _OPENMP
#pragma omp parallel for
#endif
			if(uv2dInfo[i].faceid==-1)
				continue;
			uv2dInfo[i].u=uvDatas[8*uv2dInfo[i].faceid]*uv2dInfo[i].w0
				+uvDatas[8*uv2dInfo[i].faceid+2]*uv2dInfo[i].w1
				+uvDatas[8*uv2dInfo[i].faceid+4]*uv2dInfo[i].w2
				+uvDatas[8*uv2dInfo[i].faceid+6]*uv2dInfo[i].w3+10;
			uv2dInfo[i].u-=(int)uv2dInfo[i].u;
			uv2dInfo[i].v=uvDatas[8*uv2dInfo[i].faceid+1]*uv2dInfo[i].w0
				+uvDatas[8*uv2dInfo[i].faceid+3]*uv2dInfo[i].w1
				+uvDatas[8*uv2dInfo[i].faceid+5]*uv2dInfo[i].w2
				+uvDatas[8*uv2dInfo[i].faceid+7]*uv2dInfo[i].w3+10;
			uv2dInfo[i].v-=(int)uv2dInfo[i].v;
		}
	}
};

struct picture2D
{
	int size;
	MFloatVectorArray color;
	MString texture;
	bool dirty;
	picture2D()
	{
		dirty=true;
	}
	picture2D(MString texture,int size)
	{
		this->texture=texture;
		this->size=size;
		dirty=true;
		color.setLength(size*size);
	}
	void update(const picture2D& newPicture)
	{
		texture=newPicture.texture;
		size=newPicture.size;
		dirty=true;
		color.setLength(size*size);
	}
	void setSize(int size)
	{
		if(this->size!=size)
		{
			dirty=true;
			color.setLength(size*size);
		}
		this->size=size;
	}
	void sample(int & dirtyCount)
	{
		cout<<"sampe2d"<<" "<<dirty<<endl;
		if(dirty)
		{
			dirtyCount++;
			//sample
			MFloatArray uarray(size*size);
			MFloatArray varray(size*size);

			//set uv value

			for(int i=0;i<size;i++)
			{
				for(int j=0;j<size;j++)
				{
#ifdef _OPENMP
#pragma omp parallel for
#endif
					uarray[i*size+j]=(1.0*j)/size;
					varray[i*size+j]=(1.0*i)/size;
				}
			}

			MFloatMatrix cam;
			MFloatVectorArray resultTrans;
			MRenderUtil::sampleShadingNetwork(texture,size*size,false,false,
				cam,NULL,&uarray,&varray,NULL,NULL,NULL,NULL,NULL,color,resultTrans);
		}
		dirty=false;
	}
};

struct meshSubdivide
{
	//vector<int> grouptID;
	meshTopology meshInfo;
	OpenSubdiv::OsdMesh *_osdmesh;
	OpenSubdiv::OsdHbrMesh *_hbrMesh;
	vector<int> remap;
	drawData dataForUI;
	dataCompute dataForCompute;
	uv2DColor uv2DColorInfo;

	int varCount;
	int fvarindice[4];
	int fvarwidths[4];
	int totalfarwidth;

	meshSubdivide()
	{
		cout<<"meshSubdivide"<<endl;
		_osdmesh=NULL;
		cout<<"meshSubdivide"<<endl;
		_osdmesh = new OpenSubdiv::OsdMesh();
		cout<<"meshSubdivide"<<endl;
/*
// 		if(_hbrMesh)
// 			delete _hbrMesh;
*/
		cout<<"meshSubdivide"<<endl;
		_hbrMesh=NULL;
		cout<<"meshSubdivide"<<endl;
		varCount=4;
		fvarindice[0]=0;
		fvarindice[1]=1;
		fvarindice[2]=2;
		fvarindice[3]=3;
		fvarwidths[0]=1;
		fvarwidths[1]=1;
		fvarwidths[2]=1;
		fvarwidths[3]=1;
		totalfarwidth=4;
		cout<<"meshSubdivide"<<endl;
	}
	~meshSubdivide()
	{
		if(_osdmesh)
			delete _osdmesh;
		if(_hbrMesh)
			delete _hbrMesh;
	}
	
	

	void setMeshClear()
	{
		meshInfo.changeStatus=NOTHINGCHANGE;
		dataForCompute.dirty=false;
	}

	void update(int level)
	{
		meshInfo.getUpdateInfo(level);
		//todo:dealing with stateChange
		switch (meshInfo.changeStatus)
		{
		case TOPOLOGYCHANGE:
			topologyChange();
			break;
		case UVCHANGE:
			uvChange();
			break;
		case VERTEXCHANGE:
			vertexChange();
			break;
		default:
			break;
		}
	}

	void update(meshTopology& newMeshTopo)
	{
		cout<<"ddddd"<<endl;
		meshInfo.getUpdateInfo(newMeshTopo);
		cout<<meshInfo.changeStatus<<endl;
		//todo:dealing with stateChange
		switch (meshInfo.changeStatus)
		{
		case TOPOLOGYCHANGE:
			cout<<"topo"<<endl;
			topologyChange();
			break;
		case UVCHANGE:
			cout<<"uv"<<endl;
			uvChange();
			break;
		case VERTEXCHANGE:
			cout<<"vertex"<<endl;
			vertexChange();
			break;
		default:
			break;
		}
	}

	void topologyChange()
	{
		cout<<"1"<<endl;
		buildOsdHbr();
		cout<<"2"<<endl;
		buildDrawData();
		cout<<"3"<<endl;
		buildUVInfo();
		cout<<"4"<<endl;
		uvChange();
		cout<<"5"<<endl;
		vertexChange();
		cout<<"6"<<endl;
	}

	void buildOsdHbr()
	{
		cout<<"aaa"<<endl;
		

		if(_hbrMesh)
			delete _hbrMesh;
		cout<<meshInfo.vertexCount.size()<<" "
			<<meshInfo.vertexList.size()<<" "
			<<meshInfo.uvWeight.size()<<endl;
		_hbrMesh = ConvertToHBR(meshInfo.vertexNum, meshInfo.vertexCount, meshInfo.vertexList,
			meshInfo.vtxCreaseIndices, meshInfo.vtxCreases,
			std::vector<int>(), std::vector<float>(),
			meshInfo.edgeCreaseVtxIndices, meshInfo.edgeCreases,
			meshInfo.uvWeight,
			2, false,
			varCount,fvarindice,fvarwidths,totalfarwidth);
		_hbrMesh->PrintStats(cout);
		cout<<"aaa"<<endl;
		int kernel = OpenSubdiv::OsdKernelDispatcher::kCPU;
		if (OpenSubdiv::OsdKernelDispatcher::HasKernelType(OpenSubdiv::OsdKernelDispatcher::kOPENMP)) {
			kernel = OpenSubdiv::OsdKernelDispatcher::kOPENMP;
		}
		if (OpenSubdiv::OsdKernelDispatcher::HasKernelType(OpenSubdiv::OsdKernelDispatcher::kCUDA)) {
			kernel = OpenSubdiv::OsdKernelDispatcher::kCUDA;
			cout<<"cuda"<<endl;
		}
		cout<<"aaa"<<endl;
		remap.clear();
		cout<<meshInfo._level<<endl;
		//,&remap
		bool test=_osdmesh->Create(_hbrMesh, meshInfo._level, kernel,&remap);
		cout<<"aaa"<<test<<endl;
	}

	void buildDrawData()
	{
		cout<<_osdmesh->GetTotalVertices()<<"ttt"<<endl;
		if(dataForUI._gpuNormalBuffer)
			delete dataForUI._gpuNormalBuffer;
		cout<<_osdmesh->GetTotalVertices()<<"ttt"<<endl;
		if(dataForUI._gpuVertexBuffer)
			delete dataForUI._gpuVertexBuffer;
		cout<<_osdmesh->GetTotalVertices()<<"ttt"<<endl;
		dataForUI._gpuVertexBuffer=dynamic_cast<OpenSubdiv::OsdGpuVertexBuffer *>(_osdmesh->InitializeVertexBuffer(3));
		//dataForUI._gpuVertexBuffer=_osdmesh->InitializeVertexBuffer(3);
		cout<<_osdmesh->GetTotalVertices()<<"ttt"<<endl;
		dataForUI._gpuNormalBuffer=dynamic_cast<OpenSubdiv::OsdGpuVertexBuffer *>(_osdmesh->InitializeVertexBuffer(3));
		//dataForUI._gpuNormalBuffer=_osdmesh->InitializeVertexBuffer(3);
		cout<<_osdmesh->GetTotalVertices()<<"ttt"<<endl;
		if(dataForUI._elementArrayBuffer)
			delete dataForUI._elementArrayBuffer;
		cout<<meshInfo._level<<"ttt"<<endl;
		dataForUI._elementArrayBuffer=_osdmesh->CreateElementArrayBuffer(meshInfo._level);
		cout<<_osdmesh->GetTotalVertices()<<"ttt"<<endl;
		dataForCompute.setLength(3*_osdmesh->GetTotalVertices());
		cout<<_osdmesh->GetTotalVertices()<<"ttt"<<endl;
	}

	void buildUVInfo()
	{
		uv2DColorInfo.setLength(_osdmesh->GetTotalVertices());
		for(int i=0;i<_osdmesh->GetTotalVertices();i++)
		{
			int farVID=remap[i];
			OpenSubdiv::OsdHbrVertex* vertex=_hbrMesh->GetVertex(i);
			OpenSubdiv::OsdHbrFace * face=vertex->GetFace();
			float *f=vertex->GetFVarData(face).GetData(0);
			uv2DColorInfo.uv2dInfo[farVID].w0=f[0];
			uv2DColorInfo.uv2dInfo[farVID].w1=f[1];
			uv2DColorInfo.uv2dInfo[farVID].w2=f[2];
			uv2DColorInfo.uv2dInfo[farVID].w3=f[3];
			uv2DColorInfo.uv2dInfo[farVID].faceid=face->GetPath().topface;
		}
	}

	void uvChange()
	{
		uv2DColorInfo.updateUV(meshInfo.uvDatas);
		uv2DColorInfo.dirty=true;
	}

	void vertexChange()
	{
		std::vector<float> vertexOnly;
		vertexOnly.resize(meshInfo.vertexNum*3);

		std::vector<float> normalOnly;
		normalOnly.resize(meshInfo.vertexNum*3);

		for(int i = 0; i < meshInfo.vertexNum; ++i)
		{
			vertexOnly[i*3+0]=meshInfo.positions[i].x;
			vertexOnly[i*3+1]=meshInfo.positions[i].y;
			vertexOnly[i*3+2]=meshInfo.positions[i].z;

			normalOnly[i*3+0]=meshInfo.normals[i].x;
			normalOnly[i*3+1]=meshInfo.normals[i].y;
			normalOnly[i*3+2]=meshInfo.normals[i].z;
		}

		dataForUI._gpuVertexBuffer->UpdateData(&vertexOnly.at(0),meshInfo.vertexNum);
		dataForUI._gpuNormalBuffer->UpdateData(&normalOnly.at(0),meshInfo.vertexNum);

		_osdmesh->Subdivide(dataForUI._gpuVertexBuffer,NULL);
		_osdmesh->Subdivide(dataForUI._gpuNormalBuffer,NULL);

		prepareComputeData();
	}

	void prepareComputeData()
	{
		dataForUI._gpuVertexBuffer->GetBufferData(&dataForCompute.oriVertex[0],0,_osdmesh->GetTotalVertices());
		dataForUI._gpuNormalBuffer->GetBufferData(&dataForCompute.oriNormal[0],0,_osdmesh->GetTotalVertices());
		//normalize(&dataForCompute.oriNormal[0]);
		dataForCompute.build3DVertex();
		dataForCompute.dirty=true;
	}

	void updateColor(map<int,picture2D>& picture2dInfo,vector<int>& grpIdx)
	{
		map<int,picture2D>::iterator pictureIterTest=picture2dInfo.find(0);
		if(pictureIterTest!=picture2dInfo.end())
		{
			cout<<"exist"<<endl;
		}
		else
		{
			cout<<"uexist"<<endl;
		}
		for(int i=0;i<uv2DColorInfo.length;i++)
		{
			//cout<<i<<endl;
 			int faceid=uv2DColorInfo.uv2dInfo[i].faceid;
 			if(faceid==-1)
 				continue;
			//cout<<faceid<<" "<<grpIdx.size()<<endl;
			int gid=0;
			if(faceid<grpIdx.size())
 				gid=grpIdx.at(faceid);
			//cout<<gid<<endl;
 			map<int,picture2D>::iterator pictureIter=picture2dInfo.find(gid);
 			if(pictureIter==picture2dInfo.end())
 			{
				//cout<<"3d"<<endl;
 				uv2DColorInfo.color[i]=dataForCompute.color[i].x;
 				continue;
 			}
 			int w=min(max(0,(int)(uv2DColorInfo.uv2dInfo[i].u*pictureIter->second.size)),pictureIter->second.size-1);
 			int h=min(max(0,(int)(uv2DColorInfo.uv2dInfo[i].v*pictureIter->second.size)),pictureIter->second.size-1);
			//cout<<i<<endl;
			uv2DColorInfo.color[i]=pictureIter->second.color[h*pictureIter->second.size+w].x+dataForCompute.color[i].x;
			//uv2DColorInfo.color[i]=0;
		}
		updatePositionNormal();
	}

	void updatePositionNormal()
	{
		cout<<"ffff"<<uv2DColorInfo.length<<" "<<dataForCompute.length<<" "<<dataForUI.length<<endl;
		if(uv2DColorInfo.length<=0)
			return;
		for(int i=0;i<uv2DColorInfo.length;i++)
		{
#ifdef _OPENMP
#pragma omp parallel for
#endif
			dataForCompute.vertex[i*3+0]=dataForCompute.oriVertex[i*3+0]+dataForCompute.oriNormal[i*3+0]*uv2DColorInfo.color[i];
			dataForCompute.vertex[i*3+1]=dataForCompute.oriVertex[i*3+1]+dataForCompute.oriNormal[i*3+1]*uv2DColorInfo.color[i];
			dataForCompute.vertex[i*3+2]=dataForCompute.oriVertex[i*3+2]+dataForCompute.oriNormal[i*3+2]*uv2DColorInfo.color[i];
		}
		calcNormals();
		dataForUI._gpuVertexBuffer->UpdateData(&dataForCompute.vertex[0],uv2DColorInfo.length);
		dataForUI._gpuNormalBuffer->UpdateData(&dataForCompute.normal[0],uv2DColorInfo.length);
		dataForUI.length=uv2DColorInfo.length;
	}

	void calcNormals()
	{
		//calc normal vectors
		int nverts = (int)uv2DColorInfo.length/3;

		int nfaces = _hbrMesh->GetNumFaces();

		for (int i = 0; i < nfaces; ++i) 
		{
#ifdef _OPENMP
#pragma omp parallel for
#endif

			OpenSubdiv::OsdHbrFace * f = _hbrMesh->GetFace(i);

			float const * p0 = &dataForCompute.vertex[remap[f->GetVertex(0)->GetID()]*3],
				* p1 = &dataForCompute.vertex[remap[f->GetVertex(1)->GetID()]*3],
				* p2 = &dataForCompute.vertex[remap[f->GetVertex(2)->GetID()]*3];

			float n[3];
			cross( n, p0, p1, p2 );

			for (int j = 0; j < f->GetNumVertices(); j++) {
				int idx = remap[f->GetVertex(j)->GetID()] * 3;
				dataForCompute.normal[idx  ] += n[0];
				dataForCompute.normal[idx+1] += n[1];
				dataForCompute.normal[idx+2] += n[2];
			}
		}
		for (int i = 0; i < nverts; ++i)
		{
#ifdef _OPENMP
#pragma omp parallel for
#endif
			normalize( &dataForCompute.normal[i*3] );
		}
	}

	void cross(float *n, const float *p0, const float *p1, const float *p2) 
	{

		float a[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
		float b[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
		n[0] = a[1]*b[2]-a[2]*b[1];
		n[1] = a[2]*b[0]-a[0]*b[2];
		n[2] = a[0]*b[1]-a[1]*b[0];

		float rn = 1.0f/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
		n[0] *= rn;
		n[1] *= rn;
		n[2] *= rn;
	}

	void normalize(float * p) {

		float dist = sqrtf( p[0]*p[0] + p[1]*p[1]  + p[2]*p[2] );
		p[0]/=dist;
		p[1]/=dist;
		p[2]/=dist;
	}
	
};