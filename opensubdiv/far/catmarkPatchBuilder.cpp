//
//   Copyright 2018 DreamWorks Animation LLC.
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#include "../far/catmarkPatchBuilder.h"
#include "../vtr/stackBuffer.h"

#include <cassert>
#include <cmath>
#include <cstdio>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

using Vtr::Array;
using Vtr::ConstArray;
using Vtr::internal::StackBuffer;

namespace Far {

//
//  Core functions for computing Catmark limit properties that are used 
//  in the conversion to multiple patch types.
//
//  This struct/class is just a means of grouping common functions.
//
//  There is a long and unclear history to the details of the computations
//  involved in the patch conversion here...
//
//  The formulae for computing the Gregory patch points do not follow the
//  more widely accepted work of Loop, Shaefer et al or Myles et al.  The
//  formulae for the limit points and tangents also ultimately need to be
//  retrieved from Sdc::Scheme to ensure they conform, so future factoring
//  of the formulae is still necessary.
//
//  XXXX (barfowl) - /internal computations are not implemented in terms
//  of <REAL>, only the interface supports <REAL>.  Need to eventually
//  remove remaining computations using "float" and define a mechanism
//  to invoke the appropriate precision math functions, e.g. calling
//  something like Math<float>::Cos(), etc.
//
template <typename REAL>
struct CatmarkLimits {
public:
    static void ComputeInteriorPointWeights(int valence, int faceInRing,
                REAL* pWeights, REAL* epWeights, REAL* emWeights);

    static void ComputeBoundaryPointWeights(int valence, int faceInRing,
                REAL* pWeights, REAL* epWeights, REAL* emWeights);

private:
    //
    //  Lookup table and formula for the scale factor applied to limit
    //  tangents that arises from eigen values of the subdivision matrix
    //
    static inline float computeCoefficient(int valence) {
        // precomputed coefficient table up to valence 29
        static float efTable[] = {
            0, 0, 0,
            0.812816f, 0.500000f, 0.363644f, 0.287514f,
            0.238688f, 0.204544f, 0.179229f, 0.159657f,
            0.144042f, 0.131276f, 0.120632f, 0.111614f,
            0.103872f, 0.09715f, 0.0912559f, 0.0860444f,
            0.0814022f, 0.0772401f, 0.0734867f, 0.0700842f,
            0.0669851f, 0.0641504f, 0.0615475f, 0.0591488f,
            0.0569311f, 0.0548745f, 0.0529621f
        };
        assert(valence > 0);
        if (valence < 30) return efTable[valence];

        float t = 2.0f * float(M_PI) / float(valence);
        return 1.0f / (valence * (cosf(t) + 5.0f +
                                  sqrtf((cosf(t) + 9) * (cosf(t) + 1)))/16.0f);
    }
};

template <typename REAL>
void
CatmarkLimits<REAL>::ComputeInteriorPointWeights(int valence, int faceInRing,
                REAL* pWeights, REAL* epWeights, REAL* emWeights) {

    //
    //  For the limit tangents of an interior vertex, the second tangent is a
    //  rotation of the first, i.e. the coefficients for the ring around the
    //  vertex can be simply shifted by two.  So there is really no need to
    //  compute it explicitly here.  The single tangent can similarly be
    //  oriented along the corresponding edges for Ep and Em and scaled and
    //  offset by P accordingly.
    //
    //  The formula used for tangents here differs from Sdc::Scheme for
    //  Catmark -- the direction is the same but the length varies due to the
    //  different terms used to scale the results (both based on eigenvalues).
    //  The main difference in the computation here though is that each edge-
    //  point is a function of three cos() terms:
    //      cos(i*theta), cos((i-1)*theta), cos((i+1)theta)
    //  while the Sdc::Scheme weight depends only on cos(i*theta), and so they
    //  are accumulated here rather than assigned directly.
    //
    //  Ultimately the Sdc::Scheme formulae are a little more efficient but we
    //  don't want to impact positions of Ep and Em slightly by switching to
    //  them until such a change can be given more justification and visibility
    //  (e.g. major version).
    //
    bool computeEdgePoints = epWeights && emWeights;

    float fValence        = float(valence);
    float oneOverValence  = 1.0f / fValence;
    float oneOverValPlus5 = 1.0f / (fValence + 5.0f);

    float pCoeff   = oneOverValence * oneOverValPlus5;
    float tanCoeff = computeCoefficient(valence) * 0.5f * oneOverValPlus5;

    float faceAngle = 2.0f * float(M_PI) * oneOverValence;

    //
    //  Assign position weights directly while accumulating an intermediate set
    //  of weights for the limit tangent.  And skip over the first weight for
    //  the corner vertex once assigned (zero for tangents) so that we don't
    //  have to deal with the off-by-one offset within the loop:
    //
    int weightWidth = 1 + 2 * valence;
    StackBuffer<REAL, 64, true> tanWeights(weightWidth);
    std::memset(&tanWeights[0], 0, weightWidth * sizeof(REAL));

    pWeights[0] = fValence * oneOverValPlus5;

    REAL *pW = &pWeights[1];
    REAL *tW = &tanWeights[1];
    for (int i = 0; i < valence; ++i) {
        pW[2*i]     = pCoeff * 4.0f;
        pW[2*i + 1] = pCoeff;

        if (computeEdgePoints) {
            int iPrev = (i + valence - 1) % valence;
            int iNext = (i + 1) % valence;

            float cosICoeff = tanCoeff * cosf(float(i) * faceAngle);

            tW[2*iPrev]     += cosICoeff * 2.0f;
            tW[2*iPrev + 1] += cosICoeff;
            tW[2*i]         += cosICoeff * 4.0f;
            tW[2*i + 1]     += cosICoeff;
            tW[2*iNext]     += cosICoeff * 2.0f;
        }
    }

    //
    //  Rotate/permute the scaled tangent weights along edges and add to P:
    //
    if (computeEdgePoints) {
        int epOffset = 2 * ((valence - faceInRing) % valence);
        int emOffset = 2 * ((valence - faceInRing - 1 + valence) % valence);

        epWeights[0] = pWeights[0];
        emWeights[0] = pWeights[0];
        for (int i = 1; i < weightWidth; ++i) {
            int ip = i + epOffset;
            if (ip >= weightWidth) ip -= weightWidth - 1;

            int im = i + emOffset;
            if (im >= weightWidth) im -= weightWidth - 1;

            epWeights[i] = pWeights[i] + tanWeights[ip];
            emWeights[i] = pWeights[i] + tanWeights[im];
        }
    }
}

template <typename REAL>
void
CatmarkLimits<REAL>::ComputeBoundaryPointWeights(int valence, int faceInRing,
                REAL* pWeights, REAL* epWeights, REAL* emWeights) {

    int   numFaces  = valence - 1;
    float faceAngle = float(M_PI) / float(numFaces);

    int weightWidth = 2 * valence;

    int N = weightWidth - 1;

    //
    //  Position weights are trivial:
    //
    std::memset(&pWeights[0],  0, weightWidth * sizeof(float));

    pWeights[0] = 4.0f / 6.0f;
    pWeights[1] = 1.0f / 6.0f;
    pWeights[N] = 1.0f / 6.0f;

    if ((epWeights == 0) && (emWeights == 0)) return;

    //
    //  Ep and Em weights are computed by combining weights for the boundary
    //  and interior tangents.  The boundary tangent is trivially represented
    //  by two non-zero weights, so allocate and compute weights for the
    //  interior tangent:
    //
    float tBoundaryWeight_1 =  1.0f / 6.0f;
    float tBoundaryWeight_N = -1.0f / 6.0f;

    StackBuffer<REAL, 64, true> tanWeights(weightWidth);
    {
        float k = float(numFaces);
        float theta = faceAngle;
        float c = cosf(theta);
        float s = sinf(theta);
        float div3 = 1.0f / 3.0f;
        float div3kc = 1.0f / (3.0f*k+c);
        float gamma = -4.0f * s * div3kc;
        float alpha_0k = -((1.0f+2.0f*c) * sqrtf(1.0f+c)) * div3kc / sqrtf(1.0f-c);
        float beta_0 = s * div3kc;

        tanWeights[0] = gamma * div3;
        tanWeights[1] = alpha_0k * div3;
        tanWeights[2] = beta_0 * div3;
        tanWeights[N] = alpha_0k * div3;

        for (int i = 1; i < valence - 1; ++i) {
            float alpha = 4.0f * sinf(float(i)*theta) * div3kc;
            float beta = (sinf(float(i)*theta) + sinf(float(i+1)*theta)) * div3kc;

            tanWeights[1 + 2*i]     = alpha * div3;
            tanWeights[1 + 2*i + 1] = beta * div3;
        }
    }

    //
    //  Compute Ep weights -- trivial case if on the leading face and edge:
    //
    if (faceInRing == 0) {
        //  Ep is on boundary edge and has only two weights:  w[1] and w[N]
        std::memset(&epWeights[0], 0, weightWidth * sizeof(float));

        epWeights[0] = 2.0f / 3.0f;
        epWeights[1] = 1.0f / 3.0f;
    } else {
        //  Ep is on interior edge and has all weights
        int iEdgeNext = faceInRing;
        float faceAngleNext = faceAngle * float(iEdgeNext);
        float cosAngleNext  = cosf(faceAngleNext);
        float sinAngleNext  = sinf(faceAngleNext);

        for (int i = 0; i < weightWidth; ++i) {
            epWeights[i] = tanWeights[i] * sinAngleNext;
        }
        epWeights[0] += pWeights[0];
        epWeights[1] += pWeights[1] + tBoundaryWeight_1 * cosAngleNext;
        epWeights[N] += pWeights[N] + tBoundaryWeight_N * cosAngleNext;
    }

    //
    //  Compute Em weights -- trivial case if on the trailing face and edge:
    //
    if (faceInRing == (numFaces - 1)) {
        //  Em is on boundary edge and has only two weights:  w[1] and w[N]
        std::memset(&emWeights[0], 0, weightWidth * sizeof(float));

        emWeights[0] = 2.0f / 3.0f;
        emWeights[N] = 1.0f / 3.0f;
    } else {
        //  Em is on interior edge and has all weights
        int iEdgePrev = (faceInRing + 1) % valence;
        float faceAnglePrev = faceAngle * float(iEdgePrev);
        float cosAnglePrev  = cosf(faceAnglePrev);
        float sinAnglePrev  = sinf(faceAnglePrev);

        for (int i = 0; i < weightWidth; ++i) {
            emWeights[i] = tanWeights[i] * sinAnglePrev;
        }
        emWeights[0] += pWeights[0];
        emWeights[1] += pWeights[1] + tBoundaryWeight_1 * cosAnglePrev;
        emWeights[N] += pWeights[N] + tBoundaryWeight_N * cosAnglePrev;
    }
}

//
//  SparseMatrixPoint
//
//  This is a utility class representing a row of a SparseMatrix -- which
//  in turn corresponds to a point of a resulting patch.
//
//  This interface was originally transitional (supporting a migration away
//  from the former GregoryBasis::Point class) and its unclear if it will
//  persist.  Its needs have been simplified and given the usual pre-sizing
//  of a sparse row, a simple Assign() method may be all that is necessary.
//
//  Use of the AddOrAppend() method is highly discouraged as it requires
//  iteration and testing of potentially all entries, before appending if
//  the specified entry does not exist.  It is currently only used to
//  simplify the awkward case of valence-2 interior vertices -- where the
//  1-ring of neighboring vertices overlaps and has duplicate entries.
//  Alternative to simplify this without resorting to AddOrAppend() are
//  under consideration.
//
template <typename REAL>
class SparseMatrixPoint {
public:
    typedef Index index_type;
    typedef REAL  weight_type;

    typedef SparseMatrix<weight_type> matrix_type;
public:
    SparseMatrixPoint(matrix_type & matrix, int row, int size = -1);

    int GetSize() const { return _size; }
    int GetCapacity() const { return _indices.size(); }

    void Append(              index_type index, weight_type weight);
    void AddOrAppend(         index_type index, weight_type weight);
    void Assign(int rowEntry, index_type index, weight_type weight);

    void Copy(SparseMatrixPoint<weight_type> const & other);
public:
    int _size;
    Array<index_type> _indices;
    Array<weight_type> _weights;
};

template <typename REAL>
inline
SparseMatrixPoint<REAL>::SparseMatrixPoint(matrix_type & matrix, int row, int size) {
    _indices = matrix.SetRowColumns(row);
    _weights = matrix.SetRowElements(row);
    _size    = (size < 0) ? _weights.size() : size;
}
template <typename REAL>
inline void
SparseMatrixPoint<REAL>::Assign(int rowEntry, index_type index, weight_type weight) {
    _indices[rowEntry] = index;
    _weights[rowEntry] = weight;
}
template <typename REAL>
inline void
SparseMatrixPoint<REAL>::Append(index_type index, weight_type weight) {
    assert(GetSize() < GetCapacity());
    _indices[_size] = index;
    _weights[_size] = weight;
    _size ++;
}
template <typename REAL>
inline void
SparseMatrixPoint<REAL>::AddOrAppend(index_type index, weight_type weight) {
    for (int i = 0; i < GetSize(); ++i) {
        if (_indices[i] == index) {
            _weights[i] += weight;
            return;
        }
    }
    Append(index, weight);
}
template <typename REAL>
inline void
SparseMatrixPoint<REAL>::Copy(SparseMatrixPoint const & other) {
    assert(GetCapacity() == other.GetCapacity());
    _size = other._size;
    std::memcpy(&_indices[0], &other._indices[0], _size * sizeof(index_type));
    std::memcpy(&_weights[0], &other._weights[0], _size * sizeof(weight_type));
}


//
//  Simple utility functions for dealing with SparseMatrix:
//
namespace {
    template <typename REAL>
    void
    _printMatrix(SparseMatrix<REAL> const & matrix, bool printIndices = true,
                                                    bool printWeights = true) {

        printf("Matrix %d x %d, %d elements:\n",
            matrix.GetNumRows(), matrix.GetNumColumns(), matrix.GetNumElements());

        for (int i = 0; i < matrix.GetNumRows(); ++i) {
            int rowSize = matrix.GetRowSize(i);
            printf("  Row %d (size = %d):\n", i, rowSize);

            if (printIndices) {
                ConstArray<int> indices = matrix.GetRowColumns(i);
                printf("    Indices:  ");
                for (int j = 0; j < rowSize; ++j) {
                    printf("%6d ", indices[j]);
                }
                printf("\n");
            }
            if (printWeights) {
                ConstArray<REAL> weights = matrix.GetRowElements(i);
                printf("    Weights:  ");
                for (int j = 0; j < rowSize; ++j) {
                    printf("%6.3f ", (REAL)weights[j]);
                }
                printf("\n");
            }
        }
    }

    template <typename REAL>
    void
    _initializeFullMatrix(SparseMatrix<REAL> & M, int nRows, int nColumns) {

        M.Resize(nRows, nColumns, nRows * nColumns);

        //  Fill row 0 with index for every column:
        M.SetRowSize(0, nColumns);
        Array<int> row0Columns = M.SetRowColumns(0);
        for (int i = 0; i < nColumns; ++i) {
            row0Columns[i] = i;
        }

        //  Copy row 0's indices into all other rows:
        for (int row = 1; row < nRows; ++row) {
            M.SetRowSize(row, nColumns);
            Array<int> dstRowColumns = M.SetRowColumns(row);
            std::memcpy(&dstRowColumns[0], &row0Columns[0], nColumns * sizeof(int));
        }
    }

    template <typename REAL>
    void
    _resizeMatrix(SparseMatrix<REAL> & matrix,
                  int numRows, int numColumns, int numElements,
                  int const rowSizes[]) {

        matrix.Resize(numRows, numColumns, numElements);
        for (int i = 0; i < numRows; ++i) {
            matrix.SetRowSize(i, rowSizes[i]);
        }
        assert(matrix.GetNumElements() == numElements);
    }

    template <typename REAL>
    void
    _addSparsePointToFullRow(REAL * fullRow,
                             SparseMatrixPoint<REAL> const & p,
                             REAL s, int * indexMask) {

        for (int i = 0; i < p.GetSize(); ++i) {
            int index = p._indices[i];

            fullRow[index] += s * p._weights[i];

            indexMask[index] = 1 + index;
        }
    }

    template <typename REAL>
    void
    _addSparseRowToFull(REAL * fullRow,
                        SparseMatrix<REAL> const & M, int sparseRow, REAL s) {

        ConstArray<int>  indices = M.GetRowColumns(sparseRow);
        ConstArray<REAL> weights = M.GetRowElements(sparseRow);

        for (int i = 0; i < indices.size(); ++i) {
            fullRow[indices[i]] += s * weights[i];
        }
    }

    template <typename REAL>
    void
    _combineSparseMatrixRowsInFull(SparseMatrix<REAL> & dstMatrix, int dstRowIndex,
           SparseMatrix<REAL> const & srcMatrix, int numSrcRows,
           int const srcRowIndices[], REAL const * srcRowWeights) {

        REAL * dstRow = &dstMatrix.SetRowElements(dstRowIndex)[0];

        std::memset(dstRow, 0, dstMatrix.GetNumColumns() * sizeof(REAL));

        for (int i = 0; i < numSrcRows; ++i) {
            _addSparseRowToFull(dstRow, srcMatrix, srcRowIndices[i], srcRowWeights[i]);
        }
    }

    template <typename REAL>
    void
    _matrixPrintDensity(const char* prefix, SparseMatrix<REAL> const & M) {
    
        int fullSize = M.GetNumRows() * M.GetNumColumns();
        int sparseSize = M.GetNumElements();

        int nonZeroSize = 0;
        for (int i = 0; i < M.GetNumRows(); ++i) {
            ConstArray<REAL> elements = M.GetRowElements(i);
            for (int j = 0; j < elements.size(); ++j) {
                nonZeroSize += (elements[j] != 0);
            }
        }
        printf("%s(%dx%d = %d):  elements = %d, non-zero = %d, density = %.1f\n",
            prefix, M.GetNumRows(), M.GetNumColumns(), fullSize,
            sparseSize, nonZeroSize, (REAL)nonZeroSize * 100.0f / (REAL)fullSize);
    }
} // end namespace for SparseMatrix utilities


//
//  GregoryConverter
//
//  The GregoryConverter class essentially provides a change-of-basis matrix
//  from source vertices in a Catmull-Clark mesh to the 20 control points of a
//  Gregory patch.
//
//  Historically the source topology was specified as a Vtr::Level and face index,
//  from which contributions of all 1-ring vertices that support the 20 points of
//  the patch are determined.  The source topology is now specified via a simple
//  SourcePatch, so a matrix can be determined for a particular configuration and
//  re-used for any similar instance.
//
//  Control points are labeled using the convention from:  "Approximating
//  Subdivision Surfaces with Gregory Patches for Hardware Tessellation" Loop,
//  Schaefer, Ni, Castano (ACM ToG Siggraph Asia 2009)
//
//     P3         e3-      e2+         P2
//        x--------x--------x--------x
//        |        |        |        |
//        |        |        |        |
//        |        | f3-    | f2+    |
//        |        x        x        |
//    e3+ x------x            x------x e2-
//        |     f3+          f2-     |
//        |                          |
//        |                          |
//        |     f0-          f1+     |
//    e0- x------x            x------x e1+
//        |        x        x        |
//        |        | f0+    | f1-    |
//        |        |        |        |
//        |        |        |        |
//        x--------x--------x--------x
//     P0         e0+      e1-         P1
//
template <typename REAL>
class GregoryConverter {
public:
    typedef REAL                      Weight;
    typedef SparseMatrix<Weight>      Matrix;
    typedef SparseMatrixPoint<Weight> Point;
public:
    GregoryConverter() : _numSourcePoints(0) { }
    GregoryConverter(SourcePatch const & sourcePatch);
    GregoryConverter(SourcePatch const & sourcePatch, Matrix & sparseMatrix);

    void Initialize(SourcePatch const & sourcePatch);

    bool IsIsolatedInteriorPatch() const { return _isIsolatedInteriorPatch; }
    int  GetIsolatedInteriorCorner() const { return _isolatedCorner; }
    int  GetIsolatedInteriorValence() const { return _isolatedValence; }

    void Convert(Matrix & sparseMatrix) const;

private:
    //
    //  Local nested class for GregoryConverter to cache information for the
    //  corners of the source patch.  It copies some information from the
    //  SourcePatch so that we don't have to keep it around, but it contains
    //  additional information relevant to the determination of the Gregory
    //  points -- most notably classifications of the face-points and the
    //  sines/cosines of angles for the face corners that are used repeatedly.
    //
    struct CornerTopology {
        //  Basic flags copied from the SourcePatch
        unsigned int isBoundary  : 1;
        unsigned int isSharp     : 1;
        unsigned int isDart      : 1;
        unsigned int isRegular   : 1;
        unsigned int isVal2Int   : 1;

        //  Flags for edge- and face-points relating to adjacent corners:
        unsigned int fpIsRegular : 1;
        unsigned int fmIsRegular : 1;
        unsigned int fpIsCopied  : 1;
        unsigned int fmIsCopied  : 1;

        unsigned int val2InRing  : 1;
        unsigned int epAdjToVal2 : 1;
        unsigned int emAdjToVal2 : 1;

        //  Other values stored for repeated use:
        int  valence;
        int  numFaces;
        int  faceInRing;

        REAL faceAngle;
        REAL cosFaceAngle;
        REAL sinFaceAngle;

        //  Its useful to have the ring for each corner immediately available:
        StackBuffer<int, 40, true> ringPoints;
    };

    //
    //  Methods to resize the matrix before populating it:
    //
    void resizeMatrixUnisolated(Matrix & matrix) const;

    void resizeMatrixIsolatedIrregular(Matrix & matrix, int irregCornerIndex,
                                                        int irregValence) const;

    //
    //  Methods to compute the various rows of points in the matrix:
    //
    void assignRegularEdgePoints(int cornerIndex, Matrix & matrix) const;
    void computeIrregularEdgePoints(int cornerIndex, Matrix & matrix,
                                    Weight *weightBuffer) const;

    void assignRegularFacePoints(int cornerIndex, Matrix & matrix) const;
    void computeIrregularFacePoints(int cornerIndex, Matrix & matrix,
                                    Weight *weightBuffer, int *indexBuffer) const;

    void computeIrregularInteriorEdgePoints(int cornerIndex,
                                            Point & P, Point & Ep, Point & Em,
                                            Weight *weightBuffer) const;
    void computeIrregularBoundaryEdgePoints(int cornerIndex,
                                            Point & P, Point & Ep, Point & Em,
                                            Weight *weightBuffer) const;

    int  getIrregularFacePointSize(int cornerNear, int cornerFar) const;
    void computeIrregularFacePoint(
                int cornerNear, int edgeInNearRing, int cornerFar,
                Point const & p, Point const & eNear, Point const & eFar,
                Point & fNear, REAL signForSideOfEdge /* -1.0 or 1.0 */,
                Weight *rowWeights, int *columnMask) const;

private:
    int _numSourcePoints;
    int _maxValence;

    bool _isIsolatedInteriorPatch;
    int  _isolatedCorner;
    int  _isolatedValence;

    CornerTopology _corners[4];
};


//
//  GregoryConverter
//
//  Construction and initialization/computation of the change-of-basis
//  matrix to a Gregory patch.
//
template <typename REAL>
GregoryConverter<REAL>::GregoryConverter(
        SourcePatch const & sourcePatch) {

    Initialize(sourcePatch);
}

template <typename REAL>
GregoryConverter<REAL>::GregoryConverter(
        SourcePatch const & sourcePatch, Matrix & sparseMatrix) {

    Initialize(sourcePatch);
    Convert(sparseMatrix);
}

template <typename REAL>
void
GregoryConverter<REAL>::Initialize(SourcePatch const & sourcePatch) {

    //
    //  Allocate and gather the 1-rings for the corner vertices and other
    //  topological information for more immediate access:
    //
    int width = sourcePatch.GetNumSourcePoints();
    _numSourcePoints = width;
    _maxValence      = sourcePatch.GetMaxValence();

    int boundaryCount = 0;
    int irregularCount = 0;
    int irregularCorner = -1;
    int irregularValence = -1;
    int sharpCount = 0;

    for (int cIndex = 0; cIndex < 4; ++cIndex) {
        SourcePatch::Corner srcCorner = sourcePatch._corners[cIndex];

        CornerTopology& corner = _corners[cIndex];

        corner.isBoundary = srcCorner._boundary;
        corner.isSharp    = srcCorner._sharp;
        corner.isDart     = srcCorner._dart;
        corner.numFaces   = srcCorner._numFaces;
        corner.faceInRing = srcCorner._patchFace;
        corner.isVal2Int  = srcCorner._val2Interior;
        corner.val2InRing = srcCorner._val2Adjacent;
        corner.valence    = corner.numFaces + corner.isBoundary;

        corner.isRegular = ((corner.numFaces << corner.isBoundary) == 4)
                         && !corner.isSharp;
        if (corner.isRegular) {
            corner.faceAngle    = REAL(M_PI_2);
            corner.cosFaceAngle = 0.0f;
            corner.sinFaceAngle = 1.0f;
        } else {
            // XXXX (barfowl) - use of sine/cosine here needs to respect <REAL>
            corner.faceAngle =
                (corner.isBoundary ? REAL(M_PI) : (2.0f * REAL(M_PI)))
                    / REAL(corner.numFaces);
            corner.cosFaceAngle = cosf(corner.faceAngle);
            corner.sinFaceAngle = sinf(corner.faceAngle);
        }

        corner.ringPoints.SetSize(sourcePatch.GetCornerRingSize(cIndex));
        sourcePatch.GetCornerRingPoints(cIndex, corner.ringPoints);

        //  Accumulate topology information to categorize the patch as a whole:
        boundaryCount += corner.isBoundary;
        if (!corner.isRegular) {
            irregularCount ++;
            irregularCorner = cIndex;
            irregularValence = corner.valence;
        }
        sharpCount += corner.isSharp;
    }

    //  Make a second pass to assign tags dependent on adjacent corners
    for (int cIndex = 0; cIndex < 4; ++cIndex) {
        CornerTopology& corner = _corners[cIndex];

        int cNext = (cIndex + 1) & 0x3;
        int cPrev = (cIndex + 3) & 0x3;

        corner.epAdjToVal2 = _corners[cNext].isVal2Int;
        corner.emAdjToVal2 = _corners[cPrev].isVal2Int;

        //
        //  Identify if the face points are regular or shared/copied from
        //  one of the pair:
        //
        corner.fpIsRegular = corner.isRegular && _corners[cNext].isRegular;
        corner.fmIsRegular = corner.isRegular && _corners[cPrev].isRegular;

        corner.fpIsCopied = false;
        corner.fmIsCopied = false;

        if (corner.isBoundary) {
            //  Both face points are same when one of the two corners' edges
            //  is discontinuous -- one is then copied from the other (unless
            //  regular)
            if (corner.numFaces > 1) {
                if (_corners[cNext].isBoundary || _corners[cNext].isDart) {
                    corner.fpIsRegular = corner.fmIsRegular;
                    corner.fpIsCopied  = !corner.fpIsRegular;
                }
                if (_corners[cPrev].isBoundary || _corners[cPrev].isDart) {
                    corner.fmIsRegular = corner.fpIsRegular;
                    corner.fmIsCopied  = !corner.fmIsRegular;
                }
            } else {
                //  The case of a corner patch is always regular
                corner.fpIsRegular = true;
                corner.fmIsRegular = true;
            }
        }
    }
    _isIsolatedInteriorPatch = (irregularCount == 1) && (boundaryCount == 0) &&
                               (irregularValence > 2) && (sharpCount == 0);
    if (_isIsolatedInteriorPatch) {
        _isolatedCorner  = irregularCorner;
        _isolatedValence = irregularValence;
    }
}

template <typename REAL>
void
GregoryConverter<REAL>::Convert(Matrix & matrix) const {

    //
    //  Initialize the sparse matrix to accomodate the coefficients for each
    //  row/point -- identify common topological cases to treat more easily
    //  (and note that specializing the popoluation of the matrix may also be
    //  worthwhile in such cases)
    //
    if (_isIsolatedInteriorPatch) {
        resizeMatrixIsolatedIrregular(matrix, _isolatedCorner, _isolatedValence);
    } else {
        resizeMatrixUnisolated(matrix);
    }

    //
    //  Compute the corner and edge points P, Ep and Em first.  Since face
    //  points Fp and Fm involve edge points for two adjacent corners, their
    //  computation must follow:
    //
    int maxRingSize      = 1 + 2 * _maxValence;
    int weightBufferSize = std::max(3 * maxRingSize, 2 * _numSourcePoints);

    StackBuffer<Weight, 128, true> weightBuffer(weightBufferSize);
    StackBuffer<int, 128, true>    indexBuffer(weightBufferSize);

    for (int cIndex = 0; cIndex < 4; ++cIndex) {
        if (_corners[cIndex].isRegular) {
            assignRegularEdgePoints(cIndex, matrix);
        } else {
            computeIrregularEdgePoints(cIndex, matrix, weightBuffer);
        }
    }

    for (int cIndex = 0; cIndex < 4; ++cIndex) {
        if (_corners[cIndex].fpIsRegular || _corners[cIndex].fmIsRegular) {
            assignRegularFacePoints(cIndex, matrix);
        }
        if (!_corners[cIndex].fpIsRegular || !_corners[cIndex].fmIsRegular) {
            computeIrregularFacePoints(cIndex, matrix, weightBuffer, indexBuffer);
        }
    }
}

template <typename REAL>
void
GregoryConverter<REAL>::resizeMatrixIsolatedIrregular(
        Matrix & matrix, int cornerIndex, int cornerValence) const {

    int irregRingSize = 1 + 2 * cornerValence;

    int irregCorner   =  cornerIndex;
    int irregPlus     = (cornerIndex + 1) & 0x3;
    int irregOpposite = (cornerIndex + 2) & 0x3;
    int irregMinus    = (cornerIndex + 3) & 0x3;

    int   rowSizes[20];
    int * rowSizePtr = 0;

    rowSizePtr = rowSizes + irregCorner * 5;
    *rowSizePtr++ = irregRingSize;
    *rowSizePtr++ = irregRingSize;
    *rowSizePtr++ = irregRingSize;
    *rowSizePtr++ = irregRingSize;
    *rowSizePtr++ = irregRingSize;

    rowSizePtr = rowSizes + irregPlus * 5;
    *rowSizePtr++ = 9;
    *rowSizePtr++ = 6;
    *rowSizePtr++ = 6;
    *rowSizePtr++ = 4;
    *rowSizePtr++ = 3 + irregRingSize;

    rowSizePtr = rowSizes + irregOpposite * 5;
    *rowSizePtr++ = 9;
    *rowSizePtr++ = 6;
    *rowSizePtr++ = 6;
    *rowSizePtr++ = 4;
    *rowSizePtr++ = 4;

    rowSizePtr = rowSizes + irregMinus * 5;
    *rowSizePtr++ = 9;
    *rowSizePtr++ = 6;
    *rowSizePtr++ = 6;
    *rowSizePtr++ = 3 + irregRingSize;
    *rowSizePtr++ = 4;

    int numElements = 7*irregRingSize + 85;

    _resizeMatrix(matrix, 20, _numSourcePoints, numElements, rowSizes);
}

template <typename REAL>
void
GregoryConverter<REAL>::resizeMatrixUnisolated(Matrix & matrix) const {

    int rowSizes[20];

    int numElements = 0;

    for (int cIndex = 0; cIndex < 4; ++cIndex) {
        int * rowSize = rowSizes + cIndex*5;

        CornerTopology const & corner = _corners[cIndex];

        //  First, the corner and pair of edge points:
        if (corner.isRegular) {
            if (! corner.isBoundary) {
                rowSize[0] = 9 - corner.val2InRing;
                rowSize[1] = 6 - corner.epAdjToVal2;
                rowSize[2] = 6 - corner.emAdjToVal2;
            } else {
                rowSize[0] = 3;
                rowSize[1] = corner.faceInRing ? 6 : 2;
                rowSize[2] = corner.faceInRing ? 2 : 6;
            }
        } else {
            if (corner.isSharp) {
                rowSize[0] = 1;
                rowSize[1] = 2;
                rowSize[2] = 2;
            } else if (! corner.isBoundary) {
                int ringSize = 1 + 2 * corner.valence - corner.val2InRing;
                rowSize[0] = ringSize;
                rowSize[1] = ringSize;
                rowSize[2] = ringSize;
            } else if (corner.numFaces > 1) {
                int ringSize = 1 + corner.valence + corner.numFaces - corner.val2InRing;
                rowSize[0] = 3;
                rowSize[1] = (corner.faceInRing > 0) ? ringSize : 2;
                rowSize[2] = (corner.faceInRing < (corner.numFaces - 1)) ? ringSize : 2;
            } else {
                rowSize[0] = 3;
                rowSize[1] = 2;
                rowSize[2] = 2;
            }
        }
        numElements += rowSize[0] + rowSize[1] + rowSize[2];

        //  Second, the pair of face points:
        rowSize[3] = 4;
        rowSize[4] = 4;
        if (!corner.fpIsRegular || !corner.fmIsRegular) {
            int cNext = (cIndex + 1) & 0x3;
            int cPrev = (cIndex + 3) & 0x3;
            if (!corner.fpIsRegular) {
                rowSize[3] = getIrregularFacePointSize(cIndex,
                                corner.fpIsCopied ? cPrev : cNext);
            }
            if (!corner.fmIsRegular) {
                rowSize[4] = getIrregularFacePointSize(cIndex,
                                corner.fmIsCopied ? cNext : cPrev);
            }
        }
        numElements += rowSize[3] + rowSize[4];
    }

    _resizeMatrix(matrix, 20, _numSourcePoints, numElements, rowSizes);
}

template <typename REAL>
void
GregoryConverter<REAL>::assignRegularEdgePoints(int cIndex, Matrix & matrix) const {

    //  Declare with 0 size for use with Append()
    Point p (matrix, 5*cIndex + 0, 0);
    Point ep(matrix, 5*cIndex + 1, 0);
    Point em(matrix, 5*cIndex + 2, 0);

    CornerTopology const & corner = _corners[cIndex];

    int const * cRing = corner.ringPoints;

    if (! corner.isBoundary) {
        p.Append(cIndex,   4.0f / 9.0f);
        p.Append(cRing[0], 1.0f / 9.0f);
        p.Append(cRing[2], 1.0f / 9.0f);
        p.Append(cRing[4], 1.0f / 9.0f);
        p.Append(cRing[6], 1.0f / 9.0f);
        p.Append(cRing[1], 1.0f / 36.0f);
        if (!corner.val2InRing) {
            p.Append(cRing[3], 1.0f / 36.0f);
            p.Append(cRing[5], 1.0f / 36.0f);
            p.Append(cRing[7], 1.0f / 36.0f);
        } else {
            p.AddOrAppend(cRing[3], 1.0f / 36.0f);
            p.AddOrAppend(cRing[5], 1.0f / 36.0f);
            p.AddOrAppend(cRing[7], 1.0f / 36.0f);
        }

        //  Identify the edges along Ep and Em and those opposite them:
        int iEdgeEp = 2 *   corner.faceInRing;
        int iEdgeEm = 2 * ((corner.faceInRing + 1) & 0x3);
        int iEdgeOp = 2 * ((corner.faceInRing + 2) & 0x3);
        int iEdgeOm = 2 * ((corner.faceInRing + 3) & 0x3);

        ep.Append(cIndex,             4.0f / 9.0f);
        ep.Append(cRing[iEdgeEp],     2.0f / 9.0f);
        ep.Append(cRing[iEdgeEm],     1.0f / 9.0f);
        ep.Append(cRing[iEdgeOm],     1.0f / 9.0f);
        ep.Append(cRing[iEdgeEp + 1], 1.0f / 18.0f);
        if (corner.epAdjToVal2) {
            ep.AddOrAppend(cRing[iEdgeOm + 1], 1.0f / 18.0f);
        } else {
            ep.Append(cRing[iEdgeOm + 1], 1.0f / 18.0f);
        }

        em.Append(cIndex,             4.0f / 9.0f);
        em.Append(cRing[iEdgeEm],     2.0f / 9.0f);
        em.Append(cRing[iEdgeEp],     1.0f / 9.0f);
        em.Append(cRing[iEdgeOp],     1.0f / 9.0f);
        em.Append(cRing[iEdgeEp + 1], 1.0f / 18.0f);
        if (corner.emAdjToVal2) {
            em.AddOrAppend(cRing[iEdgeEm + 1], 1.0f / 18.0f);
        } else {
            em.Append(cRing[iEdgeEm + 1], 1.0f / 18.0f);
        }
    } else {
        //  Decide which point corresponds to interior vs exterior tangent:
        Point & eInterior = corner.faceInRing ? ep : em;
        Point & eBoundary = corner.faceInRing ? em : ep;
        int     iBoundary = corner.faceInRing ? 4 : 0;

        p.Append(cIndex,   2.0f / 3.0f);
        p.Append(cRing[0], 1.0f / 6.0f);
        p.Append(cRing[4], 1.0f / 6.0f);

        eBoundary.Append(cIndex,           2.0f / 3.0f);
        eBoundary.Append(cRing[iBoundary], 1.0f / 3.0f);

        eInterior.Append(cIndex,   4.0f / 9.0f);
        eInterior.Append(cRing[2], 2.0f / 9.0f);
        eInterior.Append(cRing[0], 1.0f / 9.0f);
        eInterior.Append(cRing[4], 1.0f / 9.0f);
        eInterior.Append(cRing[1], 1.0f / 18.0f);
        eInterior.Append(cRing[3], 1.0f / 18.0f);
    }
    assert(matrix.GetRowSize(5*cIndex + 0) == p.GetSize());
    assert(matrix.GetRowSize(5*cIndex + 1) == ep.GetSize());
    assert(matrix.GetRowSize(5*cIndex + 2) == em.GetSize());
}

template <typename REAL>
void
GregoryConverter<REAL>::computeIrregularEdgePoints(int cIndex,
        Matrix & matrix, Weight *weightBuffer) const {

    //  Declare with 0 size for use with Append()
    Point p (matrix, 5*cIndex + 0, 0);
    Point ep(matrix, 5*cIndex + 1, 0);
    Point em(matrix, 5*cIndex + 2, 0);

    //
    //  The corner and edge points P, Ep and Em  are completely determined
    //  by the 1-ring of vertices around (and including) the corner vertex.
    //  We combine full sets of coefficients for the vertex and its 1-ring.
    //
    CornerTopology const & corner = _corners[cIndex];

    if (corner.isSharp) {
        //
        //  The sharp case -- both interior and boundary...
        //
        p.Append(cIndex, 1.0f);

        // Approximating these for now, pending future investigation...
        ep.Append(cIndex,           2.0f / 3.0f);
        ep.Append((cIndex+1) & 0x3, 1.0f / 3.0f);

        em.Append(cIndex,           2.0f / 3.0f);
        em.Append((cIndex+3) & 0x3, 1.0f / 3.0f);
    } else if (! corner.isBoundary) {
        //
        //  The irregular interior case:
        //
        computeIrregularInteriorEdgePoints(cIndex, p, ep, em, weightBuffer);
    } else if (corner.numFaces > 1) {
        //
        //  The irregular boundary case:
        //
        computeIrregularBoundaryEdgePoints(cIndex, p, ep, em, weightBuffer);
    } else {
        //
        //  The irregular/smooth corner case:
        //
        p.Append(cIndex,           4.0f / 6.0f);
        p.Append((cIndex+1) & 0x3, 1.0f / 6.0f);
        p.Append((cIndex+3) & 0x3, 1.0f / 6.0f);

        ep.Append(cIndex,           2.0f / 3.0f);
        ep.Append((cIndex+1) & 0x3, 1.0f / 3.0f);

        em.Append(cIndex,           2.0f / 3.0f);
        em.Append((cIndex+3) & 0x3, 1.0f / 3.0f);
    }

    assert(matrix.GetRowSize(5*cIndex + 0) == p.GetSize());
    assert(matrix.GetRowSize(5*cIndex + 1) == ep.GetSize());
    assert(matrix.GetRowSize(5*cIndex + 2) == em.GetSize());
}


template <typename REAL>
void
GregoryConverter<REAL>::computeIrregularInteriorEdgePoints(
        int cIndex,
        Point& p, Point& ep, Point& em,
        Weight *ringWeights) const {

    CornerTopology const & corner = _corners[cIndex];

    int valence = corner.valence;
    int weightWidth = 1 + 2 * valence;

    Weight* pWeights  = &ringWeights[0];
    Weight* epWeights = pWeights + weightWidth;
    Weight* emWeights = pWeights + weightWidth * 2;

    //
    //  The interior (smooth) case -- invoke the public static method that
    //  computes pre-allocated ring weights for P, Ep and Em:
    //
    CatmarkLimits<REAL>::ComputeInteriorPointWeights(valence, corner.faceInRing,
            pWeights, epWeights, emWeights);

    //
    //  Transer the weights for the ring into the stencil form of the required
    //  Point type.  The limit mask for position involves all ring weights, and
    //  since Ep and Em depend on it, there should be no need to filter weights
    //  with value 0:
    //
    //  The presence of overlapping contributions in the ring when a neighboring
    //  vertex is valence-2 interior is a nuisance and highly unlikely but still
    //  possible if we have a triangle near a valence-2 interior vertex -- they
    //  will still not be isolated at level 1.
    //
    //  For now just use the much-less-efficient AddOrAppend() in this case
    //  until we can identify the overlap more specifically and have a more
    //  general mechanism availble to deal with it.
    //
    p.Append( cIndex, pWeights[0]);
    ep.Append(cIndex, epWeights[0]);
    em.Append(cIndex, emWeights[0]);

    if (!corner.val2InRing) {
        for (int i = 1; i < weightWidth; ++i) {
            int pRingPoint = corner.ringPoints[i-1];

            p.Append( pRingPoint, pWeights[i]);
            ep.Append(pRingPoint, epWeights[i]);
            em.Append(pRingPoint, emWeights[i]);
        }
    } else {
        for (int i = 1; i < weightWidth; ++i) {
            int pRingPoint = corner.ringPoints[i-1];

            p.AddOrAppend( pRingPoint, pWeights[i]);
            ep.AddOrAppend(pRingPoint, epWeights[i]);
            em.AddOrAppend(pRingPoint, emWeights[i]);
        }
    }
}


template <typename REAL>
void
GregoryConverter<REAL>::computeIrregularBoundaryEdgePoints(
        int cIndex,
        Point& p, Point& ep, Point& em,
        Weight *ringWeights) const {

    CornerTopology const & corner = _corners[cIndex];

    int valence = corner.valence;
    int weightWidth = 1 + corner.valence + corner.numFaces;

    Weight* pWeights  = &ringWeights[0];
    Weight* epWeights = pWeights + weightWidth;
    Weight* emWeights = pWeights + weightWidth * 2;

    //
    //  The boundary (smooth) case -- invoke the public static method that
    //  computes pre-allocated ring weights for P, Ep and Em:
    //
    CatmarkLimits<REAL>::ComputeBoundaryPointWeights(valence, corner.faceInRing,
                pWeights, epWeights, emWeights);

    //
    //  Transfer ring weights into points -- exploiting cases where they
    //  are known to be non-zero only along the two boundary edges:
    //
    int N = weightWidth - 1;

    int p0 = cIndex;
    int p1 = corner.ringPoints[0];
    int pN = corner.ringPoints[2*(valence-1)];

    p.Append(p0, pWeights[0]);
    p.Append(p1, pWeights[1]);
    p.Append(pN, pWeights[N]);

    //  If Ep is on the boundary edge, it has only two non-zero weights along
    //  that edge:
    ep.Append(p0, epWeights[0]);
    ep.Append(p1, epWeights[1]);
    if (corner.faceInRing > 0) {
        for (int i = 2; i < weightWidth; ++i) {
            ep.Append(corner.ringPoints[i-1], epWeights[i]);
        }
    }

    //  If Em is on the boundary edge, it has only two non-zero weights along
    //  that edge:
    em.Append(p0, emWeights[0]);
    if (corner.faceInRing < (corner.numFaces - 1)) {
        for (int i = 1; i < N; ++i) {
            em.Append(corner.ringPoints[i-1], emWeights[i]);
        }
    }
    em.Append(pN, emWeights[N]);
}


template <typename REAL>
int
GregoryConverter<REAL>::getIrregularFacePointSize(
        int cIndexNear, int cIndexFar) const {

    CornerTopology const & corner    = _corners[cIndexNear];
    CornerTopology const & adjCorner = _corners[cIndexFar];

    int thisSize = corner.isSharp
                 ? (6 - (corner.valence == 2))
                 : (1 + corner.ringPoints.GetSize() - corner.val2InRing);

    int adjSize = (adjCorner.isRegular || adjCorner.isSharp || (adjCorner.valence == 2))
                ? 0
                : (1 + adjCorner.ringPoints.GetSize() - 6);

    return thisSize + adjSize;
}

template <typename REAL>
void
GregoryConverter<REAL>::computeIrregularFacePoint(
        int cIndexNear, int edgeInNearCornerRing, int cIndexFar,
        Point const & p, Point const & eNear, Point const & eFar, Point & fNear,
        REAL signForSideOfEdge, Weight *rowWeights, int *columnMask) const {

    CornerTopology const & cornerNear = _corners[cIndexNear];
    CornerTopology const & cornerFar  = _corners[cIndexFar];

    int valence = cornerNear.valence;

    Weight cosNear = cornerNear.cosFaceAngle;
    Weight cosFar  = cornerFar.cosFaceAngle;

    Weight pCoeff     =                          cosFar  / 3.0f;
    Weight eNearCoeff = (3.0f - 2.0f * cosNear - cosFar) / 3.0f;
    Weight eFarCoeff  =         2.0f * cosNear           / 3.0f;

    int fullRowSize = _numSourcePoints;
    std::memset(&columnMask[0], 0, fullRowSize * sizeof(int));
    std::memset(&rowWeights[0], 0, fullRowSize * sizeof(Weight));

    _addSparsePointToFullRow(rowWeights, p,     pCoeff,     columnMask);
    _addSparsePointToFullRow(rowWeights, eNear, eNearCoeff, columnMask);
    _addSparsePointToFullRow(rowWeights, eFar,  eFarCoeff,  columnMask);

    //  Remember that R is to be computed about an interior edge and is
    //  comprised of the two pairs of points opposite the interior edge
    //
    //  Remember also that val-2-overlap may cause two of these to be the
    //  same -- doesn't matter if we accumulate here will if we assign:
    //
    int iEdgeInterior = edgeInNearCornerRing;
    int iEdgePrev     = (iEdgeInterior + valence - 1) % valence;
    int iEdgeNext     = (iEdgeInterior + 1) % valence;

    rowWeights[cornerNear.ringPoints[2*iEdgePrev]]         += -signForSideOfEdge /  9.0f;
    rowWeights[cornerNear.ringPoints[2*iEdgePrev     + 1]] += -signForSideOfEdge / 18.0f;
    rowWeights[cornerNear.ringPoints[2*iEdgeInterior + 1]] +=  signForSideOfEdge / 18.0f;
    rowWeights[cornerNear.ringPoints[2*iEdgeNext]]         +=  signForSideOfEdge /  9.0f;

    for (int i = 0; i < fullRowSize; ++i) {
        if (columnMask[i]) {
            fNear.Append(columnMask[i] - 1, rowWeights[i]);
        }
    }
}

template <typename REAL>
void
GregoryConverter<REAL>::assignRegularFacePoints(int cIndex, Matrix & matrix) const {

    //  Declare with 0 size for use with Append()
    Point fp(matrix, 5*cIndex + 3, 0);
    Point fm(matrix, 5*cIndex + 4, 0);

    CornerTopology const & corner = _corners[cIndex];

    int cNext = (cIndex+1) & 0x3;
    int cOpp  = (cIndex+2) & 0x3;
    int cPrev = (cIndex+3) & 0x3;

    //  Assign regular Fp and/or Fm:
    if (corner.fpIsRegular) {
        fp.Append(cIndex, 4.0f / 9.0f);
        fp.Append(cPrev,  2.0f / 9.0f);
        fp.Append(cNext,  2.0f / 9.0f);
        fp.Append(cOpp,   1.0f / 9.0f);
        assert(matrix.GetRowSize(5*cIndex + 3) == fp.GetSize());
    }
    if (corner.fmIsRegular) {
        fm.Append(cIndex, 4.0f / 9.0f);
        fm.Append(cPrev,  2.0f / 9.0f);
        fm.Append(cNext,  2.0f / 9.0f);
        fm.Append(cOpp,   1.0f / 9.0f);
        assert(matrix.GetRowSize(5*cIndex + 4) == fm.GetSize());
    }
}

template <typename REAL>
void
GregoryConverter<REAL>::computeIrregularFacePoints(int cIndex,
        Matrix & matrix, Weight *rowWeights, int *columnMask) const {

    //  Identify neighboring corners:
    CornerTopology const & corner = _corners[cIndex];

    int cNext = (cIndex+1) & 0x3;
    int cPrev = (cIndex+3) & 0x3;

    Point epPrev(matrix, 5*cPrev  + 1);
    Point em    (matrix, 5*cIndex + 2);
    Point p     (matrix, 5*cIndex + 0);
    Point ep    (matrix, 5*cIndex + 1);
    Point emNext(matrix, 5*cNext  + 2);

    //  Declare with 0 size for use with Append()
    Point fp(matrix, 5*cIndex + 3, 0);
    Point fm(matrix, 5*cIndex + 4, 0);

    //
    //  Compute the face points Fp and Fm in terms of the corner (P) and edge
    //  points (Ep and Em) previously computed.  The caller provides a buffer
    //  of the appropriate size (twice the width of the matrix) to use for
    //  combining weights, along with an integer buffer used to identify
    //  non-zero weights and preserve the sparsity of the combinations (note
    //  they use index + 1 to detect index 0 when cleared with 0 entries).
    //
    if (!corner.fpIsRegular && !corner.fpIsCopied) {
        int iEdgeP = corner.faceInRing;
        computeIrregularFacePoint(cIndex, iEdgeP, cNext,
                p, ep, emNext, fp, 1.0, rowWeights, columnMask);
    }
    if (!corner.fmIsRegular && !corner.fmIsCopied) {
        int iEdgeM = (corner.faceInRing + 1) % corner.valence;
        computeIrregularFacePoint(cIndex, iEdgeM, cPrev,
                p, em, epPrev, fm, -1.0, rowWeights, columnMask);
    }

    //  Copy Fp or Fm now that any shared values were computed above:
    if (corner.fpIsCopied) {
        fp.Copy(fm);
    }
    if (corner.fmIsCopied) {
        fm.Copy(fp);
    }

    if (!corner.fpIsRegular) assert(matrix.GetRowSize(5*cIndex + 3) == fp.GetSize());
    if (!corner.fmIsRegular) assert(matrix.GetRowSize(5*cIndex + 4) == fm.GetSize());
}

//
//  BSplineConverter
//
//  The BSplineConverter is far less complicated than GregoryConverter -- and
//  actually makes use of GregroyConverter in some cases.  It provides a direct
//  mapping from the original Catmull-Clark points to a set of BSpline points
//  fit to the limit position and tangent plane of a single/isolated irregular
//  interior corner. In the case of all other irregularities, the set of Gregory
//  points are first determined (using the GregoryConverter) and then converted
//  to BSpline.
//
//  In this latter case, none of the BSpline points derived correspond to the
//  original source points.
//
template <typename REAL>
class BSplineConverter {
public:
    typedef REAL                   Weight;
    typedef SparseMatrix<Weight>   Matrix;
public:
    BSplineConverter() : _sourcePatch(0) { }
    BSplineConverter(SourcePatch const & sourcePatch);
    BSplineConverter(SourcePatch const & sourcePatch, Matrix & sparseMatrix);

    void Initialize(SourcePatch const & sourcePatch);
    void Convert(Matrix & matrix) const;

private:
    void convertIrregularCorner(int irregularCornerIndex, Matrix & matrix) const;
    void buildIrregularCornerMatrix(int irregularCornerValence,
                                    int numSourcePoints,
                                    int const rowsForXPoints[7],
                                    Matrix & matrix) const;

    void convertFromGregory(Matrix const & gregoryMatrix, Matrix & matrix) const;

private:
    SourcePatch const * _sourcePatch;
    GregoryConverter<REAL> _gregoryConverter;
};

template <typename REAL>
BSplineConverter<REAL>::BSplineConverter(SourcePatch const & sourcePatch) {

    Initialize(sourcePatch);
}
template <typename REAL>
BSplineConverter<REAL>::BSplineConverter(SourcePatch const & sourcePatch,
                                         Matrix & matrix) {

    Initialize(sourcePatch);
    Convert(matrix);
}

template <typename REAL>
void
BSplineConverter<REAL>::Initialize(SourcePatch const & sourcePatch) {

    _sourcePatch = &sourcePatch;
    _gregoryConverter.Initialize(sourcePatch);
}

template <typename REAL>
void
BSplineConverter<REAL>::Convert(Matrix & matrix) const {

    if (_gregoryConverter.IsIsolatedInteriorPatch()) {
        convertIrregularCorner(_gregoryConverter.GetIsolatedInteriorCorner(),
                               matrix);
    } else {
        Matrix gregoryMatrix;
        _gregoryConverter.Convert(gregoryMatrix);

        convertFromGregory(gregoryMatrix, matrix);
    }
}

template <typename REAL>
void
BSplineConverter<REAL>::convertFromGregory(Matrix const & G, Matrix & B) const {

    //
    //  The change of basis matrix from Gregory/Bezier to BSpline contains three
    //  unique sets of weights corresponding to corner, boundary and interior
    //  points:
    //
    static REAL const wCorner[9]   = { 49.f,-42.f,-42.f, 36.f,-14.f,-14.f, 12.f, 12.f, 4.f };
    static REAL const wBoundary[6] = {-14.f, 12.f,  7.f, -6.f,  4.f, -2.f };
    static REAL const wInterior[4] = {  4.f, -2.f, -2.f,  1.f };

    //
    //  The points of the BSpline and Gregory matrices are oriented and correlated
    //  as follows:
    //
    //      B = { 12, 13, 14, 15 }     G = { 15, 17, 11, 10 }
    //          {  8,  9, 10, 11 }         { 16, 18, 13, 12 }
    //          {  4,  5,  6,  7 }         {  2,  3,  8,  6 }
    //          {  0,  1,  2,  3 }         {  0,  1,  7,  5 }
    //
    //  With four symmetric quadrants the dependencies of the BSpline points on the
    //  Gregory/Bezier points are as follows -- using the "p", "ep", "em" and "f"
    //  naming from the Gregory points:
    //
    static int const pIndices[4][9]  = { {  3,  1,  2,  0,  8, 18,  7, 16, 13 },
                                         {  8,  6,  7,  5,  3, 13, 12,  1, 18 },
                                         { 13, 11, 12, 10, 18,  8, 17,  6,  3 },
                                         { 18, 16, 17, 15, 13,  3,  2, 11,  8 } };

    static int const epIndices[4][6] = { {  3,  1,  8,  7, 18, 13 },
                                         {  8,  6, 13, 12,  3, 18 },
                                         { 13, 11, 18, 17,  8,  3 },
                                         { 18, 16,  3,  2, 13,  8 } };
    static int const emIndices[4][6] = { {  3,  2, 18, 16,  8, 13 },
                                         {  8,  7,  3,  1, 13, 18 },
                                         { 13, 12,  8,  6, 18,  3 },
                                         { 18, 17, 13, 11,  3,  8 } };

    static int const fIndices[4][4]  = { {  3,  8, 18, 13 },
                                         {  8, 13,  3, 18 },
                                         { 13, 18,  8,  3 },
                                         { 18,  3, 13,  8 } };

    //
    //  The matrix is not very sparse -- build it full for now for simplicity and
    //  consider pruning later.
    //
    //  Remember that to use variable/sparse row sizes requires processing rows in
    //  order unless we can pre-assign the row sizes (difficult here).
    //
    _initializeFullMatrix(B, 16, G.GetNumColumns());

    _combineSparseMatrixRowsInFull(B,  0, G, 9, pIndices[0],  wCorner);
    _combineSparseMatrixRowsInFull(B,  1, G, 6, epIndices[0], wBoundary);
    _combineSparseMatrixRowsInFull(B,  2, G, 6, emIndices[1], wBoundary);
    _combineSparseMatrixRowsInFull(B,  3, G, 9, pIndices[1],  wCorner);

    _combineSparseMatrixRowsInFull(B,  4, G, 6, emIndices[0], wBoundary);
    _combineSparseMatrixRowsInFull(B,  5, G, 4, fIndices[0],  wInterior);
    _combineSparseMatrixRowsInFull(B,  6, G, 4, fIndices[1],  wInterior);
    _combineSparseMatrixRowsInFull(B,  7, G, 6, epIndices[1], wBoundary);

    _combineSparseMatrixRowsInFull(B,  8, G, 6, epIndices[3], wBoundary);
    _combineSparseMatrixRowsInFull(B,  9, G, 4, fIndices[3],  wInterior);
    _combineSparseMatrixRowsInFull(B, 10, G, 4, fIndices[2],  wInterior);
    _combineSparseMatrixRowsInFull(B, 11, G, 6, emIndices[2], wBoundary);

    _combineSparseMatrixRowsInFull(B, 12, G, 9, pIndices[3],  wCorner);
    _combineSparseMatrixRowsInFull(B, 13, G, 6, emIndices[3], wBoundary);
    _combineSparseMatrixRowsInFull(B, 14, G, 6, epIndices[2], wBoundary);
    _combineSparseMatrixRowsInFull(B, 15, G, 9, pIndices[2],  wCorner);
}

template <typename REAL>
void
BSplineConverter<REAL>::buildIrregularCornerMatrix(
        int irregularCornerValence,
        int numSourcePoints, int const rowsForXPoints[7],
        Matrix & matrix) const {

    int ringSizePlusCorner = 1 + 2 * irregularCornerValence;

    int numElements = 7 * ringSizePlusCorner + 11;

    int rowSizes[16];
    for (int i = 0; i < 16; ++i) {
        rowSizes[i] = 1;
    }
    rowSizes[rowsForXPoints[0]] = ringSizePlusCorner;
    rowSizes[rowsForXPoints[1]] = ringSizePlusCorner;
    rowSizes[rowsForXPoints[2]] = ringSizePlusCorner;
    rowSizes[rowsForXPoints[3]] = ringSizePlusCorner;
    rowSizes[rowsForXPoints[4]] = ringSizePlusCorner;
    rowSizes[rowsForXPoints[5]] = ringSizePlusCorner + 1;
    rowSizes[rowsForXPoints[6]] = ringSizePlusCorner + 1;

    matrix.Resize(16, numSourcePoints, numElements);
    for (int i = 0; i < 16; ++i) {
        matrix.SetRowSize(i, rowSizes[i]);

        REAL * firstElement = &matrix.SetRowElements(i)[0];
        if (rowSizes[i] == 1) {
            *firstElement = REAL(1.0);
        } else {
            std::memset(firstElement, 0, rowSizes[i] * sizeof(REAL));
        }
    }
}

template <typename REAL>
void
BSplineConverter<REAL>::convertIrregularCorner(int irregularCorner,
                                               Matrix & matrix) const {
    //
    //  Labeling/ordering of source points P[] and derived points X[] for the
    //  final patch, where P0* denotes the extra-ordinary vertex and P5 "does
    //  not exist", i.e. it serves as a place-holder for the remainder of the
    //  exterior ring of arbitrary size around P0:
    //
    //        ...
    //    (P5)   P4----P15---P14          X0----X2----X4----X6
    //   .        |     |     |            |     |     |     |
    //   .        |     |     |            |     |     |     |
    //     P6----P0*---P3----P13          X1----P0*---P3----P13 
    //      |     |P' Em|     |    --->    |     |     |     |
    //      |     |Ep   |     |            |     |     |     |
    //     P7----P1----P2----P12          X3----P1----P2----P12
    //      |     |     |     |            |     |     |     |
    //      |     |     |     |            |     |     |     |
    //     P8----P9----P10---P11          X5----P9----P10---P11
    //
    //  The formulae deriving X[] on the right are in terms of the P[] on the
    //  left along with the limit position and edge points (P', Ep and Em) and
    //  other X[].  Given dependencies between the Xi formulae, the order of
    //  evaluation is important.
    //
    //  Listed in terms of symmetric pairs, we compute X0 last:
    //
    //      X1 = 1/3 * ( 36Ep - 16P0 - 8P1 - 2P2 - 4P3 - P6 - 2P7)
    //      X2 = 1/3 * ( 36Em - 16P0 - 4P1 - 2P2 - 8P3 - P4 - 2P15)
    //
    //      X3 = 1/3 * (-18Ep + 8P0 + 4P1 + P2 + 2P3 + 4P7  + 2P6)
    //      X4 = 1/3 * (-18Em + 8P0 + 2P1 + P2 + 4P3 + 4P15 + 2P4)
    //
    //      X5 = X1 + (P8  - P6)
    //      X6 = X2 + (P14 - P4)
    //
    //      X0 = 36P' - 16P0 - 4(P1 + P3 + X2 + X1) - (P2 + X3 + X4)
    //
    //  Since the limit points (P', Ep and Em) are all defined in terms of the
    //  1-ring around P0, and with terms generally involving source points P[]
    //  also part of that ring, almost all Xi are fully determined by points in
    //  the ring.  Only X5 and X6 involve additional points, and then only one
    //  additional point each, so its simple to amend these cases separately.
    //
    //  So we compute the Xi by combining sets of coefficients for the 1-ring
    //  around P0 (with that ring including PO as the first entry).
    //

    //
    //  Compute limit points P, Ep and Em in terms of weights of the 1-ring for the
    //  corner and identify the indices of relevant points within the ring:
    //
    int valence    = _sourcePatch->_corners[irregularCorner]._numFaces;
    int faceInRing = _sourcePatch->_corners[irregularCorner]._patchFace;

    int ringSizePlusCorner = 1 + 2 * valence;

    StackBuffer<REAL, 120, true> limitPointWeights(3 * ringSizePlusCorner);

    Weight * wP  = &limitPointWeights[0];
    Weight * wEp = wP + ringSizePlusCorner;
    Weight * wEm = wEp + ringSizePlusCorner;

    assert(valence > 2);
    CatmarkLimits<REAL>::ComputeInteriorPointWeights(valence, faceInRing, wP, wEp, wEm);

    //
    //  Resize the sparse matrix (and all of its rows) to hold coefficients for
    //  X and identify arrays for each X where we will compute the weights:
    //
    static int const xRowsAll[4][7] = { {  0,  1,  4,  2,  8,  3, 12 },
                                        {  3,  7,  2, 11,  1, 15,  0 },
                                        { 15, 14, 11, 13,  7, 12,  3 },
                                        { 12,  8, 13,  4, 14,  0, 15 } };

    int const * xRows = xRowsAll[irregularCorner];

    int numSourcePoints = _sourcePatch->GetNumSourcePoints();

    buildIrregularCornerMatrix(valence, numSourcePoints, xRows, matrix);

    Weight * wX0 = &matrix.SetRowElements(xRows[0])[0];
    Weight * wX1 = &matrix.SetRowElements(xRows[1])[0];
    Weight * wX2 = &matrix.SetRowElements(xRows[2])[0];
    Weight * wX3 = &matrix.SetRowElements(xRows[3])[0];
    Weight * wX4 = &matrix.SetRowElements(xRows[4])[0];
    Weight * wX5 = &matrix.SetRowElements(xRows[5])[0];
    Weight * wX6 = &matrix.SetRowElements(xRows[6])[0];

    //
    //  We use the ordering of points in the retrieved 1-ring for which weights
    //  of the Catmark limit points are computed.  So rather than re-order the
    //  ring to accomodate contributing source points, identify the locations
    //  of the source points in the 1-ring so we can set coefficients
    //  appropriately:
    //
    int faceInRingPlus1  = (faceInRing + 1) % valence;
    int faceInRingPlus2  = (faceInRing + 2) % valence;
    int faceInRingMinus1 = (faceInRing + valence - 1) % valence;

    int p0inRing  = 0;
    int p1inRing  = 1 + 2*faceInRing;
    int p2inRing  = 1 + 2*faceInRing + 1;
    int p3inRing  = 1 + 2*faceInRingPlus1;
    int p15inRing = 1 + 2*faceInRingPlus1 + 1;
    int p4inRing  = 1 + 2*faceInRingPlus2;
    int p6inRing  = 1 + 2*faceInRingMinus1;
    int p7inRing  = 1 + 2*faceInRingMinus1 + 1;
    int p8inRing  = ringSizePlusCorner;
    int p14inRing = ringSizePlusCorner;

    //
    //  Assign the weights for the X[] in symmetric pairs -- first initializing
    //  entries for contributions of source points P[], then combining the
    //  contributions of P[] with those for the limit points and dependent X[]:
    //
    //  X1 = 1/3 * (36Ep - (16P0 + 8P1 + 2P2 + 4P3 + P6 + 2P7))
    //  X2 = 1/3 * (36pm - (16P0 + 8P3 + 2P2 + 4P1 + P4 + 2P15))
    wX1[p0inRing] = wX2[p0inRing]  = 16.0f;
    wX1[p1inRing] = wX2[p3inRing]  =  8.0f;
    wX1[p2inRing] = wX2[p2inRing]  =  2.0f;
    wX1[p3inRing] = wX2[p1inRing]  =  4.0f;
    wX1[p6inRing] = wX2[p4inRing]  =  1.0f;
    wX1[p7inRing] = wX2[p15inRing] =  2.0f;

    //  X3 = 1/3 * (-18Ep + (8P0 + 4P1 + P2 + 2P3 + 2P6 + 4P7))
    //  X4 = 1/3 * (-18Em + (8P0 + 4P3 + P2 + 2P1 + 2P4 + 4P15))
    wX3[p0inRing] = wX4[p0inRing]  =  8.0f;
    wX3[p1inRing] = wX4[p3inRing]  =  4.0f;
    wX3[p2inRing] = wX4[p2inRing]  =  1.0f;
    wX3[p3inRing] = wX4[p1inRing]  =  2.0f;
    wX3[p6inRing] = wX4[p4inRing]  =  2.0f;
    wX3[p7inRing] = wX4[p15inRing] =  4.0f;

    //  X5 = X1 + (P8  - P6)
    //  X6 = X2 + (P14 - P4)
    wX5[p6inRing] = wX6[p4inRing]  = -1.0f;
    wX5[p8inRing] = wX6[p14inRing] =  1.0f;

    //  X0 = 36P' - 16P0 - 4(P1 + P3 + X2 + X1) - (P2 + X3 + X4)
    //     = 36P' - (16P0 + 4P1 + P2 + 4P3) - 4(X2 + X1) - (X3 + X4)
    wX0[p0inRing] = 16.0f;
    wX0[p1inRing] =  4.0f;
    wX0[p2inRing] =  1.0f;
    wX0[p3inRing] =  4.0f;

    //  Combine weights for all X[] in one iteration through the ring:
    const REAL oneThird = 1.0f / 3.0f;
    for (int i = 0; i < ringSizePlusCorner; ++i) {
        wX1[i] = (36.0f * wEp[i] - wX1[i]) * oneThird;
        wX2[i] = (36.0f * wEm[i] - wX2[i]) * oneThird;

        wX3[i] = - wEp[i] * 6.0f + wX3[i] * oneThird;
        wX4[i] = - wEm[i] * 6.0f + wX4[i] * oneThird;

        wX5[i] += wX1[i];
        wX6[i] += wX2[i];

        wX0[i] =  wP[i] * 36.0f - wX0[i] - (wX2[i] + wX1[i]) * 4.0f - (wX3[i] + wX4[i]);
    }

    //
    //  The weights for the rows for X[] are now computed, and with identity
    //  rows of the remaining source points already assigned a weight of 1.0,
    //  all weights in the conversion matrix are now assigned.
    //
    //  We now need to assign the indices.  Indices for the 1-ring around the
    //  corner are trivially retrieved and complete rows for all X[] except
    //  the last entries for X5 and X6.  So identify the source points needed
    //  for these two trailing entries and those for other source points that
    //  are referenced by the matrix.
    //
    //  We've already identified those involved in the equations above -- the
    //  rest can be determined from the orientation of points in SourcePatch:
    //  all exterior points follow in a counter-clockwise order after the four
    //  interior points, and we only care about the exterior points P8 through
    //  P14.
    //
    StackBuffer<int, 40, true> ringPoints(ringSizePlusCorner);

    ringPoints[0] = irregularCorner;
    _sourcePatch->GetCornerRingPoints(irregularCorner, &ringPoints[1]);

    //  Identify P8 through P14 (no need to identify all 16):
    int pPoints[16];
    int pNext = ringPoints[p7inRing] + 1;
    for (int i = 8; i < 16; ++i, ++pNext) {
        pPoints[i] = (pNext < numSourcePoints) ? pNext
                   : (pNext - numSourcePoints + 4);
    }

    //  Assign the ring of indices for the rows of X[] -- amending X5 and X6:
    int * xIndices[7];
    for (int i = 0; i < 7; ++i) {
        xIndices[i] = &matrix.SetRowColumns(xRows[i])[0];
        std::memcpy(xIndices[i], ringPoints, ringSizePlusCorner * sizeof(int));
    }
    xIndices[5][ringSizePlusCorner] = pPoints[8];
    xIndices[6][ringSizePlusCorner] = pPoints[14];

    //  Assign the index for the rows of the four interior points -- these are
    //  fixed given the interior points precede the exterior:
    matrix.SetRowColumns( 5)[0] = 0;
    matrix.SetRowColumns( 6)[0] = 1;
    matrix.SetRowColumns( 9)[0] = 3;
    matrix.SetRowColumns(10)[0] = 2;

    //  Assign the index for the rows of remaining exterior source points
    //  (P9 through P13) -- identify the rows from a lookup table based on
    //  the irregular corner:
    static int const extPointRowsAll[4][5] = { {  7, 11, 15, 14, 13 },
                                               { 14, 13, 12,  8,  4 },
                                               {  8,  4,  0,  1,  2 },
                                               {  1,  2,  3,  7, 11 } };
    int const * extPointRows = extPointRowsAll[irregularCorner];

    matrix.SetRowColumns(extPointRows[0])[0] = pPoints[ 9];
    matrix.SetRowColumns(extPointRows[1])[0] = pPoints[10];
    matrix.SetRowColumns(extPointRows[2])[0] = pPoints[11];
    matrix.SetRowColumns(extPointRows[3])[0] = pPoints[12];
    matrix.SetRowColumns(extPointRows[4])[0] = pPoints[13];
}

//
//  LinearConverter
//
//  The LinearConverter is far less complicated than any of the others.  There's
//  not much more to it than a single conversion method -- it follows the pattern
//  for consistency.
//
template <typename REAL>
class LinearConverter {
public:
    typedef REAL                      Weight;
    typedef SparseMatrix<Weight>      Matrix;
    typedef SparseMatrixPoint<Weight> MatrixPoint;
public:
    LinearConverter() : _sourcePatch(0) { }
    LinearConverter(SourcePatch const & sourcePatch);
    LinearConverter(SourcePatch const & sourcePatch, Matrix & sparseMatrix);

    void Initialize(SourcePatch const & sourcePatch);
    void Convert(Matrix & matrix) const;

private:
    SourcePatch const * _sourcePatch;
};

template <typename REAL>
LinearConverter<REAL>::LinearConverter(SourcePatch const & sourcePatch) {

    Initialize(sourcePatch);
}
template <typename REAL>
LinearConverter<REAL>::LinearConverter(SourcePatch const & sourcePatch, Matrix & matrix) {

    Initialize(sourcePatch);
    Convert(matrix);
}

template <typename REAL>
void
LinearConverter<REAL>::Initialize(SourcePatch const & sourcePatch) {

    _sourcePatch = &sourcePatch;
}

template <typename REAL>
void
LinearConverter<REAL>::Convert(Matrix & matrix) const {

    StackBuffer<Index,64,true>  indexBuffer(1 + _sourcePatch->GetMaxRingSize());
    StackBuffer<Weight,64,true> weightBuffer(1 + _sourcePatch->GetMaxRingSize());

    int numElements = 4 * (1 + _sourcePatch->GetMaxRingSize());

    matrix.Resize(4, _sourcePatch->GetNumSourcePoints(), numElements);

    for (int cIndex = 0; cIndex < 4; ++cIndex) {
        //  Deal with the trivial sharp case first:
        if (_sourcePatch->_corners[cIndex]._sharp) {
            matrix.SetRowSize(cIndex, 1);
            matrix.SetRowColumns(cIndex)[0] = cIndex;
            matrix.SetRowElements(cIndex)[0] = 1.0f;
            continue;
        }

        SourcePatch::Corner const & sourceCorner = _sourcePatch->_corners[cIndex];

        int ringSize = _sourcePatch->GetCornerRingSize(cIndex);
        if (sourceCorner._boundary) {
            matrix.SetRowSize(cIndex, 3);
        } else {
            matrix.SetRowSize(cIndex, 1 + ringSize - sourceCorner._val2Adjacent);
        }

        Array<Index>  rowIndices = matrix.SetRowColumns(cIndex);
        Array<Weight> rowWeights = matrix.SetRowElements(cIndex);

        indexBuffer[0] = cIndex;
        _sourcePatch->GetCornerRingPoints(cIndex, &indexBuffer[1]);

        if (sourceCorner._boundary) {
            CatmarkLimits<REAL>::ComputeBoundaryPointWeights(
                1 + sourceCorner._numFaces, sourceCorner._patchFace,
                &weightBuffer[0], 0, 0);

            rowIndices[0] = indexBuffer[0];
            rowIndices[1] = indexBuffer[1];
            rowIndices[2] = indexBuffer[ringSize];

            rowWeights[0] = weightBuffer[0];
            rowWeights[1] = weightBuffer[1];
            rowWeights[2] = weightBuffer[ringSize];
        } else {
            CatmarkLimits<REAL>::ComputeInteriorPointWeights(
                sourceCorner._numFaces, sourceCorner._patchFace,
                &weightBuffer[0], 0, 0);

            if (sourceCorner._val2Adjacent) {
                MatrixPoint row(matrix, cIndex, 0);
                for (int i = 0; i <= ringSize; ++i) {
                    row.AddOrAppend(indexBuffer[i], weightBuffer[i]);
                }
            } else {
                std::memcpy(
                    &rowIndices[0], indexBuffer, (1 + ringSize) * sizeof(Index));
                std::memcpy(
                    &rowWeights[0], weightBuffer, (1 + ringSize) * sizeof(Weight));
            }
        }
    }
}


//
//  Internal utilities more relevant to the CatmarkPatchBuilder:
//
namespace {
    //
    //  The patch type associated with each basis for Catmark -- quickly
    //  indexed from an array.  The patch type here is essentially the
    //  quad form of each basis.
    //
    PatchDescriptor::Type const patchTypeFromBasisArray[] = {
            PatchDescriptor::NON_PATCH,      // undefined
            PatchDescriptor::REGULAR,        // regular
            PatchDescriptor::GREGORY_BASIS,  // Gregory
            PatchDescriptor::QUADS,          // linear
            PatchDescriptor::NON_PATCH       // Bezier -- for future use
    };
}

int
CatmarkPatchBuilder::convertToPatchType(SourcePatch const &   sourcePatch,
                                        PatchDescriptor::Type patchType,
                                        SparseMatrix<float> & matrix) const {

    assert(_schemeType == Sdc::SCHEME_CATMARK);

    //
    //  XXXX (barfowl) - consider a CatmarkPatch class to wrap SourcePatch
    //  with the additional corner information that it initializes.  That
    //  can then be used for conversion to all destination patch types...
    //
    
    if (patchType == PatchDescriptor::GREGORY_BASIS) {
        GregoryConverter<float>(sourcePatch, matrix);
    } else if (patchType == PatchDescriptor::REGULAR) {
        BSplineConverter<float>(sourcePatch, matrix);
    } else if (patchType == PatchDescriptor::QUADS) {
        LinearConverter<float>(sourcePatch, matrix);
    } else {
        assert("Unknown or unsupported patch type" == 0);
    }
    return matrix.GetNumRows();
}

CatmarkPatchBuilder::CatmarkPatchBuilder(
    TopologyRefiner const& refiner, Options const& options) :
        PatchBuilder(refiner, options) {

    _regPatchType   = patchTypeFromBasisArray[_options.regBasisType];
    _irregPatchType = (_options.irregBasisType == BASIS_UNSPECIFIED)
                    ? _regPatchType
                    : patchTypeFromBasisArray[_options.irregBasisType];

    _nativePatchType = patchTypeFromBasisArray[BASIS_REGULAR];
    _linearPatchType = patchTypeFromBasisArray[BASIS_LINEAR];
}

CatmarkPatchBuilder::~CatmarkPatchBuilder() {
}

PatchDescriptor::Type
CatmarkPatchBuilder::patchTypeFromBasis(BasisType basis) const {

    return patchTypeFromBasisArray[(int)basis];
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
