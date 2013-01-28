#pragma once



//#include "typedef.h"

//#include "displacementMap.h"
#include "meshTopology.h"
//#include "drawData.h"



class vixo_smooth : public MPxSurfaceShape
{
private:
	//drawData drawDataUI;
	meshSubdivide meshSubInfo;
	vector<int> groupInfo;
	void getGroupInfo(int faceNum);
	map<int,picture2D> picture2dInfo;
	void updatePicture2d(int idx,int& dirtyCount);
	void clean()
	{
		isMeshDirty=false;
		isSubLevelDirty=false;
		dirty2D.clear();
		isMapPixelDirty=false;
		is3dDirty=false;
		isGroupIdDirty=false;
		meshSubInfo.setMeshClear();
	}
public:
	drawData* getDrawData();

public:
	vixo_smooth(void);
	virtual~vixo_smooth(void);

	virtual MStatus	compute( const MPlug&, MDataBlock& );
	virtual MStatus setDependentsDirty(const MPlug& plug, MPlugArray& plugArray);

	static  void *		creator();
	static  MStatus		initialize();

	virtual void postConstructor()
	{
		setRenderable(true);
	}

	virtual MBoundingBox boundingBox() const
	{
		MBoundingBox box;
		MPlug plugtest(thisMObject(),inOriMesh);
		if(plugtest.isConnected()==false)
			return box;

		MPlugArray plugArr;
		plugtest.connectedTo(plugArr,true,false);
		if(plugArr[0].node().hasFn(MFn::kMesh)==false)
			return box;

		MFnDagNode fnDag(plugArr[0].node());
		MBoundingBox oriBox=fnDag.boundingBox();

		return oriBox;
	}
	virtual bool isBounded() const
	{
		return true;
	}

	static MTypeId id;
	static MObject inOriMesh;
	static MObject subdivideLevel;

	static MObject in2DDisplacementValue;
	static MObject in3DDisplacementValue;

	static MObject mapPixel;
	static MObject inGroupId;

private:
	bool isMeshDirty;
	bool isSubLevelDirty;
	MIntArray dirty2D;
	bool isMapPixelDirty;
	bool is3dDirty;
	bool isGroupIdDirty;

};


class vixo_smoothUI : public MPxSurfaceShapeUI
{
public:
	vixo_smoothUI();
	virtual ~vixo_smoothUI(); 

	virtual void	getDrawRequests( const MDrawInfo & info,
		bool objectAndActiveOnly,
		MDrawRequestQueue & requests );
	virtual void	draw( const MDrawRequest & request,
		M3dView & view ) const;

	static  void *  creator();

	bool select(MSelectInfo &selectInfo, MSelectionList &selectionList, MPointArray &worldSpaceSelectPts) const
	{

		bool selected = false;
		bool componentSelected = false;
		bool hilited = false;


		if ( !selected ) {

			vixo_smooth* meshNode = (vixo_smooth*)surfaceShape();

			// NOTE: If the geometry has an intersect routine it should
			// be called here with the selection ray to determine if the
			// the object was selected.

			selected = true;
			MSelectionMask priorityMask( MSelectionMask::kSelectMeshes );
			MSelectionList item;
			item.add( selectInfo.selectPath() );
			MPoint xformedPt;
			if ( selectInfo.singleSelection() ) {
				MPoint center = meshNode->boundingBox().center();
				xformedPt = center;
				xformedPt *= selectInfo.selectPath().inclusiveMatrix();
			}

			selectInfo.addSelection( item, xformedPt, selectionList,
				worldSpaceSelectPts, priorityMask, false );
		}

		return selected;
	}

	void drawWireFrame(const MDrawRequest & request,M3dView & view) const ;
	void drawShaded(const MDrawRequest & request,M3dView & view) const ;
};

MStatus initializePlugin( MObject obj )
{ 
	glewInit();
	OpenSubdiv::OsdCpuKernelDispatcher::Register();
	OpenSubdiv::OsdCudaKernelDispatcher::Register();
	cudaInit();
	MFnPlugin plugin( obj, PLUGIN_COMPANY, "3.0", "Any");
	return plugin.registerShape( "vixo_smooth", vixo_smooth::id,
		&vixo_smooth::creator,
		&vixo_smooth::initialize,
		&vixo_smoothUI::creator  );
}

MStatus uninitializePlugin( MObject obj)
{
	MFnPlugin plugin( obj );
	return plugin.deregisterNode( vixo_smooth::id );
}


