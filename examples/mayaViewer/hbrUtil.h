#ifndef OSD_HBR_UTIL_H
#define OSD_HBR_UTIL_H

#include <vector>
#include <osd/mesh.h>

extern "C" OpenSubdiv::OsdHbrMesh * ConvertToHBR(int nVertices,
                                                 std::vector<int>   const & numIndices,
                                                 std::vector<int>   const & faceIndices,
                                                 std::vector<int>   const & vtxCreaseIndices,
                                                 std::vector<float> const & vtxCreases,
                                                 std::vector<int>   const & edgeCrease1Indices,
                                                 std::vector<float> const & edgeCreases1,
                                                 std::vector<int>   const & edgeCrease2Indices,
                                                 std::vector<float> const & edgeCreases2,
                                                 int interpBoundary,
                                                 bool loop);
#endif
