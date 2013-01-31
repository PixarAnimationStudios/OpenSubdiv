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
#ifndef _EXAMPLE_MAYA_UTIL_H_
#define _EXAMPLE_MAYA_UTIL_H_

#include <maya/MColor.h>
#include <maya/MFloatVector.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnNumericData.h>
#include <maya/MMatrix.h>
#include <maya/MPlug.h>
#include <maya/MString.h>
#include <maya/MVector.h>


#define CHECK_GL_ERROR(...)  \
    if (GLuint err = glGetError()) {                \
        fprintf(stderr, "GL error %x :", err);      \
        fprintf(stderr, "%s", __VA_ARGS__);         \
    }


#define MERROR(status, msg)                                                                         \
    {                                                                                               \
        MGlobal::displayError(MString(msg));                                                        \
        fprintf(stderr, "%s [ %s:%d ]\n", msg, __FILE__, __LINE__);  \
    }                               

#define MCHECK_PRINT(status, msg)                                                                   \
    if (status.error())                                                                             \
    {                                                                                               \
        MGlobal::displayError(MString(msg));                                                        \
        fprintf(stderr, "%s [ %s:%d ]\n", msg, __FILE__, __LINE__);  \
    }                               

#define MCHECK_RETURN(status, msg)                                                                  \
    if (status.error())                                                                             \
    {                                                                                               \
        MGlobal::displayError(MString(msg));                                                        \
        fprintf(stderr, "%s [ %s:%d ]\n", msg, __FILE__, __LINE__);  \
        return status;                                                                              \
    }                               


// 
//      Templated funtions able to retrieve any attribute that
//      can be retrieved via the overloaded MPlug::getValue()
//      methods.  Any attributes that need MPlug::getData()
//      Need to use non-templated functions below
//
template<class T> static int
findAttribute( MFnDependencyNode &depFn, const char *attr, T *val )
{
    MStatus stat;
    MPlug plug;
    T tmp;

    // careful version - returns -1 if attribute missing
    plug = depFn.findPlug(attr, &stat);
    if (stat != MS::kSuccess) return -1;

    stat = plug.getValue(tmp);
    if ( stat != MS::kSuccess ) return -1;

    *val = tmp;
    return 0;
}


template<class T> static void
getAttribute( MObject& object, MObject& attr, T *val )
{
    // fast version - crash & burn if attribute missing
    MPlug plug(object, attr);
    plug.getValue(*val);
}


static MColor
getColor(MObject object, MObject attr) 
{
    MPlug plug(object, attr);
    MObject data;
    plug.getValue(data);
    MFnNumericData numFn(data);
    float color[3];
    numFn.getData(color[0], color[1], color[2]);
    return MColor(color[0], color[1], color[2]);
}

static MFloatVector
getVector(MObject object, MObject attr) 
{
    MPlug plug(object, attr);
    MObject data;
    plug.getValue(data);
    MFnNumericData numFn(data);
    float color[3];
    numFn.getData(color[0], color[1], color[2]);
    return MVector(color[0], color[1], color[2]);
}


static void
// reverse to dst,src ?
setMatrix(const MMatrix &mat, float *dst) 
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            dst[i*4+j] = float(mat(i, j));
}


#endif // _EXAMPLE_MAYA_UTIL_H_

