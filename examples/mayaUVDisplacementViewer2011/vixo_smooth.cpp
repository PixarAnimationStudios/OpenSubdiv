#include "vixo_smooth.h"


MTypeId	vixo_smooth::id( 0x23655 );
MObject vixo_smooth::inOriMesh;
MObject vixo_smooth::in2DDisplacementValue;
MObject vixo_smooth::in3DDisplacementValue;
MObject vixo_smooth::mapPixel;
MObject vixo_smooth::inGroupId;
MObject vixo_smooth::subdivideLevel;

vixo_smooth::vixo_smooth(void)
{

}


vixo_smooth::~vixo_smooth(void)
{

}

void vixo_smooth::getGroupInfo(int faceTotal)
{

	MPlug plugGroupID(thisMObject(),inGroupId);

	MIntArray plugGroupIdExist;
	plugGroupID.getExistingArrayAttributeIndices(plugGroupIdExist);

	MIntArray plugGroupIdConn;
	plugGroupIdConn.clear();
	for(int i=0;i<plugGroupIdExist.length();i++)
	{
		MPlugArray srcPlugsG;
		plugGroupID.elementByLogicalIndex(plugGroupIdExist[i]).connectedTo(srcPlugsG,true,false);
		if(srcPlugsG.length()==0)
			continue;
		plugGroupIdConn.append(plugGroupIdExist[i]);
	}

	groupInfo.resize(faceTotal,-1);

	if(plugGroupIdConn.length()==0)
	{
		groupInfo.clear();
		groupInfo.resize(faceTotal,0);
	}
	else
	{
		for(int i=0;i<plugGroupIdConn.length();i++)
		{
			int gid=plugGroupIdConn[i];
			MPlugArray srcPlugsG;
			plugGroupID.elementByLogicalIndex(gid).connectedTo(srcPlugsG,true,false);
			//listConnections -p 1 -t "mesh" -s 0 -d 1 groupId241.groupId;
			//body_Shape1.instObjGroups.objectGroups[8].objectGroupId
			//objectGrpCompList
			MStringArray res;
			MGlobal::executeCommand("listConnections -p 1 -t \"mesh\" -s 0 -d 1 "+srcPlugsG[0].info(),res);
			if(res.length()==0)
				continue;

			MStringArray resSplit;
			res[0].split('.',resSplit);
			MString nodeName=resSplit[0];
			MString idx=resSplit[2];

			res.clear();
			MGlobal::executeCommand("getAttr "+nodeName+".instObjGroups."+idx+".objectGrpCompList",res);
			MString flat="ls -fl";
			for(int mm=0;mm<res.length();mm++)
			{
				flat=flat+" "+nodeName+"."+res[mm];
			}
			res.clear();
			MGlobal::executeCommand(flat,res);

			if(res.length()==0)
				continue;
			MIntArray faceidxele(res.length());
			set<int> faceidxset;
			for(int mm=0;mm<res.length();mm++)
			{
				MStringArray splitres;
				res[mm].split('[',splitres);
				MString temp=splitres[1];
				splitres.clear();
				temp.split(']',splitres);
				faceidxele[mm]=splitres[0].asInt();
				faceidxset.insert(faceidxele[mm]);
				groupInfo[faceidxele[mm]]=gid;
			}

		}
	}

}

drawData* vixo_smooth::getDrawData()
{
	MPlug inMeshPlug(thisMObject(),inOriMesh);
	if(inMeshPlug.isConnected()==false)
	{
		meshSubInfo.dataForUI.draw=false;
		//if(meshSubInfo.dataForUI._gpuVertexBuffer)
		//	delete meshSubInfo.dataForUI._gpuVertexBuffer;
		//if(meshSubInfo.dataForUI._gpuNormalBuffer)
		//	delete meshSubInfo.dataForUI._gpuVertexBuffer;
		//if(meshSubInfo.dataForUI._elementArrayBuffer)
		//	delete meshSubInfo.dataForUI._elementArrayBuffer;
		meshSubInfo.dataForUI.length=0;
		meshSubInfo.meshInfo.vertexCount.clear();
		return &meshSubInfo.dataForUI;
	}
	MObject inMeshObj=inMeshPlug.asMObject();
	if(meshSubInfo.meshInfo.faceCount<=0)
	{
		isMeshDirty=true;
		is3dDirty=true;
		isSubLevelDirty=true;
		isMapPixelDirty=true;
		isGroupIdDirty=true;
		MPlug displace2d(thisMObject(),in2DDisplacementValue);
		displace2d.getExistingArrayAttributeIndices(dirty2D);
	}
	if(isMeshDirty)
	{
		MFnMesh fnMesh(inMeshObj);
		//cout<<fnMesh.numVertices()<<endl;
		meshTopology newMeshTopo(inMeshObj);
		//cout<<"apple"<<endl;
		meshSubInfo.update(newMeshTopo);
		//cout<<meshSubInfo.dataForUI.length<<endl;
	}
	if(is3dDirty)
	{
		meshSubInfo.dataForCompute.dirty=true;
	}
	if(isSubLevelDirty)
	{
		int level=MPlug(thisMObject(),subdivideLevel).asInt();
		meshSubInfo.update(level);
	}
	if(isGroupIdDirty)
	{
		//update group id info
		getGroupInfo(meshSubInfo.meshInfo.faceCount);
	}
	if(isMapPixelDirty)
	{
		int mapSize=MPlug(thisMObject(),mapPixel).asInt();
		map<int,picture2D>::iterator pictureIter;
		for(pictureIter=picture2dInfo.begin();pictureIter!=picture2dInfo.end();pictureIter++)
		{
			pictureIter->second.setSize(mapSize);
		}
	}
	//dirty2D
	int dirtyCount=0;
	map<int,picture2D>::iterator pictureIter;
	//cout<<"dddddmmmm"<<dirty2D.length()<<endl;
	for(int i=0;i<dirty2D.length();i++)
	{
		//cout<<"here"<<endl;
		//unconnect 
			//delete
		//connect
			//setdirty or add
		updatePicture2d(dirty2D[i],dirtyCount);
	}

	
	//clean 2d picture
	//cout<<"test"<<picture2dInfo.size()<<endl;
	for(pictureIter=picture2dInfo.begin();pictureIter!=picture2dInfo.end();pictureIter++)
	{
		//resample
		cout<<"resample"<<endl;
		pictureIter->second.sample(dirtyCount);
		cout<<dirtyCount<<endl;
	}
	//clean 3d picture
	MPlug map3dPlug(thisMObject(),in3DDisplacementValue);
	MString tex3d="";
	if(map3dPlug.isConnected())
	{
		MPlugArray plugArr;
		map3dPlug.connectedTo(plugArr,true,false);
		tex3d=plugArr[0].info();
	}
	meshSubInfo.dataForCompute.sample(dirtyCount,tex3d);
	cout<<dirtyCount<<endl;
	if(dirtyCount>0)
	{
		cout<<"yyyy"<<endl;
		//compute final color,position,normal
		meshSubInfo.updateColor(picture2dInfo,groupInfo);
		cout<<"yyyy"<<endl;
	}
	//cout<<"aa:"<<meshSubInfo.dataForUI.length<<endl;
	clean();
	return &meshSubInfo.dataForUI;
}

void vixo_smooth::updatePicture2d(int idx,int& dirtyCount)
{
	MPlug plug(thisMObject(),in2DDisplacementValue);
	plug=plug.elementByLogicalIndex(idx);
	if(!plug.isConnected())
	{
		plug=plug.child(0);
		if(!plug.isConnected())
		{
			picture2dInfo.erase(idx);
			dirtyCount++;
			return;
		}
	}
/*
// 	{
// 		cout<<"unconn"<<plug.info().asChar()<<endl;
// 		picture2dInfo.erase(idx);
// 		dirtyCount++;
// 	}
*/
	//else
	//{
		cout<<"conn"<<plug.info().asChar()<<endl;
		MPlugArray texPlugs;
		plug.connectedTo(texPlugs,true,false);
		map<int,picture2D>::iterator pictureIter=picture2dInfo.find(idx);
		picture2D pictureEle(texPlugs[0].info(),MPlug(thisMObject(),mapPixel).asInt());
		if(pictureIter==picture2dInfo.end())
		{
			picture2dInfo.insert(make_pair(idx,pictureEle));
		}
		else
		{
			pictureIter->second.update(pictureEle);
		}
	//}
}

MStatus vixo_smooth::setDependentsDirty(const MPlug& plug, MPlugArray& plugArray)
{
	cout<<plug.info().asChar()<<endl;
	if(plug==MPlug(thisMObject(),inOriMesh))
		isMeshDirty=true;
	else if(plug==MPlug(thisMObject(),subdivideLevel))
		isSubLevelDirty=true;
	else if(plug==MPlug(thisMObject(),in3DDisplacementValue))
		is3dDirty=true;
	else if(plug.isElement()&&plug.array()==MPlug(thisMObject(),inGroupId))
		isGroupIdDirty=true;
	else if(plug.array()==MPlug(thisMObject(),in2DDisplacementValue))
	{
		dirty2D.append(plug.logicalIndex());
		cout<<"dirty:"<<plug.logicalIndex()<<endl;
	}
	else if(plug.parent().array()==MPlug(thisMObject(),in2DDisplacementValue))
	{
		dirty2D.append(plug.parent().logicalIndex());
		cout<<"dirtyrgb:"<<plug.parent().logicalIndex()<<endl;
	}
	else if(plug==MPlug(thisMObject(),mapPixel))
		isMapPixelDirty=true;

	return MS::kSuccess;
}


MStatus	vixo_smooth::compute( const MPlug& plug, MDataBlock& data )
{
	return MS::kSuccess;
}

void*	vixo_smooth::creator()
{
	return new vixo_smooth;
}

MStatus	vixo_smooth::initialize()
{
	MFnTypedAttribute tAttr;
	MFnNumericAttribute nAttr;

	in2DDisplacementValue=nAttr.createColor("DisplacementValue2D","ddv");
	nAttr.setCached(false);
	nAttr.setArray(true);
	nAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	in3DDisplacementValue=nAttr.createColor("DisplacementValue3D","tdv");
	nAttr.setCached(false);
	inGroupId=nAttr.create("inGroupId","igi",MFnNumericData::kLong);
	nAttr.setCached(false);
	nAttr.setArray(true);
	nAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	subdivideLevel=nAttr.create("subdivideLevel","subl",MFnNumericData::kInt,3);
	nAttr.setMin(1);
	nAttr.setMax(5);
	nAttr.setCached(false);
	inOriMesh=tAttr.create("inOriMesh","iom",MFnData::kMesh);
	tAttr.setCached(false);
	mapPixel=nAttr.create("mapPixel","mps",MFnNumericData::kInt,128);
	nAttr.setCached(false);

	addAttribute(in2DDisplacementValue);
	addAttribute(in3DDisplacementValue);
	addAttribute(inGroupId);
	addAttribute(subdivideLevel);
	addAttribute(mapPixel);
	addAttribute(inOriMesh);

	return MS::kSuccess;
}


vixo_smoothUI::vixo_smoothUI() {}
vixo_smoothUI::~vixo_smoothUI() {}

void* vixo_smoothUI::creator()
{
	return new vixo_smoothUI();
}


void vixo_smoothUI::getDrawRequests( const MDrawInfo & info, bool objectAndActiveOnly, MDrawRequestQueue & queue )
{
	cout<<"request"<<endl;
	MDrawData data;
	MDrawRequest request = info.getPrototype( *this );
	vixo_smooth* shapeNode = (vixo_smooth*)surfaceShape();
	drawData* drawDatas=shapeNode->getDrawData();
	cout<<"end"<<endl;

	if(drawDatas->_gpuVertexBuffer==NULL)
		cout<<"null"<<endl;
	if(drawDatas->_gpuNormalBuffer==NULL)
		cout<<"null"<<endl;
	if(drawDatas->_elementArrayBuffer==NULL)
		cout<<"null"<<endl;

	getDrawData( drawDatas, data );
	cout<<"end"<<endl;
	request.setDrawData( data );
	cout<<"end"<<endl;

	M3dView::DisplayStyle  appearance    = info.displayStyle();
	M3dView::DisplayStatus displayStatus = info.displayStatus();

	if ( ! info.objectDisplayStatus( M3dView::kDisplayMeshes ) )
		return;

	switch ( appearance )
	{
	case M3dView::kWireFrame :
		{
			request.setToken(M3dView::kWireFrame);
			M3dView::ColorTable activeColorTable = M3dView::kActiveColors;
			M3dView::ColorTable dormantColorTable = M3dView::kDormantColors;
			switch ( displayStatus )
			{
			case M3dView::kLead :
				request.setColor( LEAD_COLOR, activeColorTable );
				break;
			case M3dView::kActive :
				request.setColor( ACTIVE_COLOR, activeColorTable );
				break;
			case M3dView::kActiveAffected :
				request.setColor( ACTIVE_AFFECTED_COLOR, activeColorTable );
				break;
			case M3dView::kDormant :
				request.setColor( DORMANT_COLOR, dormantColorTable );
				break;
			case M3dView::kHilite :
				request.setColor( HILITE_COLOR, activeColorTable );
				break;
			default:	
				break;
			}
			cout<<"end"<<endl;
			queue.add( request );
			cout<<"end"<<endl;
			break;
		}
	default:
		{
			request.setToken(M3dView::kGouraudShaded);
			queue.add( request );
			if(displayStatus==M3dView::kActive||displayStatus==M3dView::kLead||displayStatus==M3dView::kHilite)
			{
				MDrawRequest wireRequest=info.getPrototype(*this);
				wireRequest.setDrawData(data);
				wireRequest.setToken(M3dView::kFlatShaded);
				wireRequest.setDisplayStyle(M3dView::kWireFrame);

				M3dView::ColorTable activeColorTable = M3dView::kActiveColors;
				switch ( displayStatus )
				{
				case M3dView::kLead :
					wireRequest.setColor( LEAD_COLOR, activeColorTable );
					break;
				case M3dView::kActive :
					wireRequest.setColor( ACTIVE_COLOR, activeColorTable );
					break;
				case M3dView::kHilite :
					wireRequest.setColor( HILITE_COLOR, activeColorTable );
					break;
				default:	
					break;
				}

				queue.add(wireRequest);

			}
			break;
		}
		cout<<"ttttt"<<endl;
	}
	cout<<"end"<<endl;
}

void vixo_smoothUI::draw( const MDrawRequest & request, M3dView & view ) const
{
	cout<<"draw"<<endl;
	MDrawData data = request.drawData();
	drawData* drawDatas=(drawData*)data.geometry();
	if(drawDatas->length<=0)
		return;
	if(drawDatas->_gpuVertexBuffer==NULL)
		return;
	if(drawDatas->_gpuNormalBuffer==NULL)
		return;
	if(drawDatas->_elementArrayBuffer==NULL)
		return;
	int token=request.token();
	switch(token)
	{
		case M3dView::kWireFrame:
		{
			drawWireFrame(request,view);
			break;
		}
		case M3dView::kGouraudShaded:
		{
			drawShaded(request,view);
			break;
		}
		case M3dView::kFlatShaded:
		{
			drawWireFrame(request,view);
			break;
		}
	default:
		break;
	}
}

void vixo_smoothUI::drawWireFrame(const MDrawRequest & request, M3dView & view) const 
{
	cout<<"wire"<<endl;
	MDrawData data = request.drawData();
	drawData* drawDatas=(drawData*)data.geometry();
	int token=request.token();
	
	view.beginGL();

	glPushAttrib( GL_CURRENT_BIT );
	glPushAttrib( GL_ENABLE_BIT);


 	bool lightingWasOn = glIsEnabled( GL_LIGHTING ) ? true : false;
 	if ( lightingWasOn ) {
 		glDisable( GL_LIGHTING );
 	}


	if(token==M3dView::kFlatShaded)
		glDepthMask(false);

	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	{
		int vertexStride = 3*sizeof(float);
		//        int varyingStride = mesh->GetVaryingStride();
		//printf("Draw. stride = %d\n", stride);
		//glBindBuffer(GL_ARRAY_BUFFER, drawDatas->_vertexBuffer->GetGpuBuffer());
		cout<<"drawwwww"<<endl;
		glBindBuffer(GL_ARRAY_BUFFER,drawDatas->_gpuVertexBuffer->GetGpuBuffer());
		glVertexPointer(3, GL_FLOAT, vertexStride, ((char*)(0)));
		glEnableClientState(GL_VERTEX_ARRAY);
		cout<<"drawwwww"<<endl;

		//glBindBuffer(GL_ARRAY_BUFFER, drawDatas->_gpuNormalBuffer->GetGpuBuffer());
		//glNormalPointer(GL_FLOAT, vertexStride, ((char*)(0)));
		//        glBindBuffer(GL_ARRAY_BUFFER, mesh->GetVaryingBuffer());
		//        glNormalPointer(GL_FLOAT, varyingStride, ((char*)(0)));
		//glEnableClientState(GL_NORMAL_ARRAY);
		//cout<<"drawwwww"<<endl;

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawDatas->_elementArrayBuffer->GetGlBuffer());
		glDrawElements(GL_QUADS, drawDatas->_elementArrayBuffer->GetNumIndices(), GL_UNSIGNED_INT, NULL);
		cout<<"drawwwww"<<endl;

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


	if ( lightingWasOn ) {
		glEnable( GL_LIGHTING );
	}

	if(token==M3dView::kFlatShaded)
		glDepthMask(true);

	glPopAttrib();
	glPopAttrib();

	view.endGL(); 
}

void vixo_smoothUI::drawShaded(const MDrawRequest & request, M3dView & view)const 
{
	MDrawData data = request.drawData();

	drawData* drawDatas=(drawData*)data.geometry();

	view.beginGL();
	glPushAttrib( GL_CURRENT_BIT );
	glPushAttrib( GL_ENABLE_BIT);

	//glShadeModel(GL_SMOOTH);
	glMaterialf(GL_FRONT,GL_SPECULAR,0.5f);
	glMaterialf(GL_FRONT,GL_SHININESS,50.0f);

	glLightf(GL_LIGHT0,GL_POSITION,1.0f);
	glLightf(GL_LIGHT0,GL_DIFFUSE,1.0f);
	glLightf(GL_LIGHT0,GL_SPECULAR,1.0f);
	glLightModelf(GL_LIGHT_MODEL_AMBIENT,0.2f);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	int vertexStride = 3*sizeof(float);

	glBindBuffer(GL_ARRAY_BUFFER,drawDatas->_gpuVertexBuffer->GetGpuBuffer());
	glVertexPointer(3, GL_FLOAT, vertexStride, ((char*)(0)));
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, drawDatas->_gpuNormalBuffer->GetGpuBuffer());
	glNormalPointer(GL_FLOAT, vertexStride, ((char*)(0)));
	
	glEnableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawDatas->_elementArrayBuffer->GetGlBuffer());
	glDrawElements(GL_QUADS, drawDatas->_elementArrayBuffer->GetNumIndices(), GL_UNSIGNED_INT, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHTING);

	glPopAttrib();
	glPopAttrib();

	view.endGL();
}
