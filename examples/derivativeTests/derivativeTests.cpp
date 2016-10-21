//------------------------------------------------------------------------------
// Exampledescription:
//
// Demonstrate the double-precision LimitStencilTable evaluation feature.
// I copied some code from far_tutorial_0.cpp to quickly grab a mesh and a corresponding TopologyRefiner.
//

#include <far/stencilTableFactory.h>
#include <far/topologyDescriptor.h>
#include <far/primvarRefiner.h>

#include <iostream>

#include <cstdio>

#include <stdlib.h>
#include <time.h>

//typedef float FD;
typedef double FD;

// Cube geometry from catmark_cube.h

static FD g_verts[8][3] = {{ (FD)-0.5, (FD)-0.5, (FD) 0.5 },
                           { (FD) 0.5, (FD)-0.5, (FD) 0.5 },
                           { (FD)-0.5, (FD) 0.5, (FD) 0.5 },
                           { (FD) 0.5, (FD) 0.5, (FD) 0.5 },
                           { (FD)-0.5, (FD) 0.5, (FD)-0.5 },
                           { (FD) 0.5, (FD) 0.5, (FD)-0.5 },
                           { (FD)-0.5, (FD)-0.5, (FD)-0.5 },
                           { (FD) 0.5, (FD)-0.5, (FD)-0.5 }};

static int g_nverts = 8,
g_nfaces = 6;

static int g_vertsperface[6] = { 4, 4, 4, 4, 4, 4 };

static int g_vertIndices[24] = { 0, 1, 3, 2,
                                 2, 3, 5, 4,
                                 4, 5, 7, 6,
                                 6, 7, 1, 0,
                                 1, 7, 5, 3,
                                 6, 0, 2, 4  };

using namespace OpenSubdiv;
using namespace OPENSUBDIV_VERSION;
using namespace Far;

struct vect {
    void Clear() { x = 0; y = 0; z = 0; }
    void AddWithWeight(const vect& other, FD weight)
    {
        x += weight * other.x;
        y += weight * other.y;
        z += weight * other.z;
    }

    FD norm()
    {
        return sqrt( x*x + y*y + z*z );
    }

    FD x;
    FD y;
    FD z;
};

std::ostream& operator<<(std::ostream& out, const vect& v)
{
    out << v.x << ' ' << v.y << ' ' << v.z;
    return out;
}

vect operator+(const vect& a, const vect& b)
{
    vect out = { a.x + b.x , a.y + b.y , a.z + b.z };
    return out;
}

vect operator-(const vect& a, const vect& b)
{
    vect out = { a.x - b.x , a.y - b.y , a.z - b.z };
    return out;
}

vect operator*(FD c, const vect& a)
{
    vect out = { c * a.x , c * a.y , c * a.z };
    return out;
}

vect operator/(const vect& a, FD c)
{
    vect out = { a.x / c , a.y / c , a.z / c };
    return out;
}

struct pointData {
    vect pos;
    vect ds;
    vect dt;
    vect dss;
    vect dst;
    vect dtt;
};

std::ostream& operator<<(std::ostream& out, const pointData& v)
{
    out << v.pos << '\n';
    out << v.ds << '\n';
    out << v.dt << '\n';
    out << v.dss << '\n';
    out << v.dst << '\n';
    out << v.dtt << '\n';
    return out;
}

// Compute limit surface position + derivative data at a specified point.
pointData compute(TopologyRefiner* refiner, int face, FD s, FD t)
{
    std::vector< LimitStencilTableFactoryG<FD>::LocationArray > myLoc( 1 );
    myLoc[ 0 ].numLocations = 1;
    myLoc[ 0 ].ptexIdx = face;
    myLoc[ 0 ].s = &s;
    myLoc[ 0 ].t = &t;

    LimitStencilTableFactoryG<FD>::Options lstOptions;
    lstOptions.generate2ndDerivatives = 1;

    const LimitStencilTableG<FD>*
        myLST = LimitStencilTableFactoryG<FD>::Create( *refiner , myLoc , 0 , 0 , lstOptions );


    pointData out;

    myLST->UpdateValues   ( (vect*)g_verts , &out.pos );
    myLST->UpdateDerivs   ( (vect*)g_verts , &out.ds  , &out.dt  );
    myLST->Update2ndDerivs( (vect*)g_verts , &out.dss , &out.dst , &out.dtt );

    delete myLST;

    return out;
}

void dtest(TopologyRefiner* refiner, int face, FD s, FD t, FD ep)
{
    std::cout << "\n\n\n\nSTART DERIVATIVE TESTING: face = " << face << " s = " << s << " t = " << t
              << " initial ep = " << ep << '\n';

    pointData center = compute( refiner , face , s , t );

    FD epDs = ep;
    std::cerr << "\n\n\n\n------------------ TESTING DS ------------------" << std::endl;
    FD minRelOrAbsErrorDs = 1e9;
    std::cerr << "ds EXACT: " << center.ds << std::endl;
    for ( int i = 0 ; i < 20 ; ++i ) {
        pointData datPlus  = compute( refiner , face , s+epDs , t );
        pointData datMinus = compute( refiner , face , s-epDs , t );

        vect dApprox = (datPlus.pos - datMinus.pos) / (2.0*epDs);

        FD error = (center.ds - dApprox).norm();
        FD relError = error / std::max( center.ds.norm() , dApprox.norm() );

        std::cerr << "ep = "<<epDs<<" approx = "<<dApprox<<" error = "<< error << " relerror = "<< relError << std::endl;

        minRelOrAbsErrorDs = std::min( minRelOrAbsErrorDs , relError );
        minRelOrAbsErrorDs = std::min( minRelOrAbsErrorDs , error );

        epDs /= 2.0;
    }

    FD epDt = ep;
    std::cerr << "\n\n\n\n------------------ TESTING DT ------------------" << std::endl;
    FD minRelOrAbsErrorDt = 1e9;
    std::cerr << "dt EXACT: " << center.dt << std::endl;
    for ( int i = 0 ; i < 20 ; ++i ) {
        pointData datPlus  = compute( refiner , face , s , t+epDt );
        pointData datMinus = compute( refiner , face , s , t-epDt );

        vect dApprox = (datPlus.pos - datMinus.pos) / (2.0*epDt);

        FD error = (center.dt - dApprox).norm();
        FD relError = error / std::max( center.ds.norm() , dApprox.norm() );

        std::cerr << "ep = "<<epDt<<" approx = "<<dApprox<<" error = "<< error << " relerror = "<< relError << std::endl;

        minRelOrAbsErrorDt = std::min( minRelOrAbsErrorDt , relError );
        minRelOrAbsErrorDt = std::min( minRelOrAbsErrorDt , error );

        epDt /= 2.0;
    }

    FD epDss = ep;
    std::cerr << "\n\n\n\n------------------ TESTING DSS -----------------" << std::endl;
    FD minRelOrAbsErrorDss = 1e9;
    std::cerr << "dss EXACT: " << center.dss << std::endl;
    for ( int i = 0 ; i < 20 ; ++i ) {
        pointData datPlus  = compute( refiner , face , s+epDss , t );
        pointData datMinus = compute( refiner , face , s-epDss , t );

        vect dApprox = (datPlus.ds - datMinus.ds) / (2.0*epDss);

        FD error = (center.dss - dApprox).norm();
        FD relError = error / std::max( center.dss.norm() , dApprox.norm() );

        std::cerr << "ep = "<<epDss<<" approx = "<<dApprox<<" error = "<< error << " relerror = "<< relError << std::endl;

        minRelOrAbsErrorDss = std::min( minRelOrAbsErrorDss , relError );
        minRelOrAbsErrorDss = std::min( minRelOrAbsErrorDss , error );

        epDss /= 2.0;
    }

    FD epDst = ep;
    std::cerr << "\n\n\n\n------------------ TESTING DST -----------------" << std::endl;
    FD minRelOrAbsErrorDst = 1e9;
    std::cerr << "dst EXACT: " << center.dst << std::endl;
    for ( int i = 0 ; i < 20 ; ++i ) {
        pointData datPlus  = compute( refiner , face , s+epDst , t );
        pointData datMinus = compute( refiner , face , s-epDst , t );

        vect dApprox = (datPlus.dt - datMinus.dt) / (2.0*epDst);

        FD error = (center.dst - dApprox).norm();
        FD relError = error / std::max( center.dst.norm() , dApprox.norm() );

        std::cerr << "ep = "<<epDst<<" approx = "<<dApprox<<" error = "<< error << " relerror = "<< relError << std::endl;

        minRelOrAbsErrorDst = std::min( minRelOrAbsErrorDst , relError );
        minRelOrAbsErrorDst = std::min( minRelOrAbsErrorDst , error );

        epDst /= 2.0;
    }

    FD epDtt = ep;
    std::cerr << "\n\n\n\n------------------ TESTING DTT -----------------" << std::endl;
    FD minRelOrAbsErrorDtt = 1e9;
    std::cerr << "dtt EXACT: " << center.dtt << std::endl;
    for ( int i = 0 ; i < 20 ; ++i ) {
        pointData datPlus  = compute( refiner , face , s , t+epDtt );
        pointData datMinus = compute( refiner , face , s , t-epDtt );

        vect dApprox = (datPlus.dt - datMinus.dt) / (2.0*epDtt);

        FD error = (center.dtt - dApprox).norm();
        FD relError = error / std::max( center.dtt.norm() , dApprox.norm() );

        std::cerr << "ep = "<<epDtt<<" approx = "<<dApprox<<" error = "<< error << " relerror = "<< relError << std::endl;

        minRelOrAbsErrorDtt = std::min( minRelOrAbsErrorDtt , relError );
        minRelOrAbsErrorDtt = std::min( minRelOrAbsErrorDtt , error );

        epDtt /= 2.0;
    }
    std::cerr << "\n\n\n\n";

    std::cout << "Best rel-or-abs errors are:" << std::endl;
    std::cout << "ds: " << minRelOrAbsErrorDs << std::endl;
    std::cout << "dt: " << minRelOrAbsErrorDt << std::endl;
    std::cout << "dss: " << minRelOrAbsErrorDss << std::endl;
    std::cout << "dst: " << minRelOrAbsErrorDst << std::endl;
    std::cout << "dtt: " << minRelOrAbsErrorDtt << std::endl;
}

void dtestrand(TopologyRefiner* refiner)
{
    // pick a random point on a random face. Then pick a random direction and run derivative testing.
    FD s = rand() / (FD)RAND_MAX;
    FD t = rand() / (FD)RAND_MAX;

    FD mindist = 1e-6;

    int face = rand() % g_nfaces;

    FD ep = 0.1;
    ep = std::min( ep , s - mindist );
    ep = std::min( ep , t - mindist );
    ep = std::min( ep , 1 - s - mindist );
    ep = std::min( ep , 1 - t - mindist );

    dtest( refiner , face , s , t , ep );
}

TopologyRefiner* buildRefiner()
{
    // Populate a topology descriptor with our mesh's data

    Sdc::SchemeType type = Sdc::SCHEME_CATMARK;

    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_NONE);

    TopologyDescriptor desc;
    desc.numVertices  = g_nverts;
    desc.numFaces     = g_nfaces;
    desc.numVertsPerFace = g_vertsperface;
    desc.vertIndicesPerFace  = g_vertIndices;


    // Instantiate a FarTopologyRefiner from the descriptor
    TopologyRefiner * refiner = TopologyRefinerFactory<TopologyDescriptor>::Create(desc,
                                            TopologyRefinerFactory<TopologyDescriptor>::Options(type, options));


    int subdLevels = 2;
    TopologyRefiner::AdaptiveOptions trOptions( subdLevels );
    refiner->RefineAdaptive( trOptions );

    return refiner;
}

//------------------------------------------------------------------------------
int main(int, char **) {

    srand (time(NULL));

    std::cout.precision( 15 );
    std::cerr.precision( 15 );

    TopologyRefiner * refiner = buildRefiner();


    // Now run derivative tests centered around a few "randomly" chosen points.

    dtest(refiner,
        3, // face
        0.5, // s
        0.3, // t
        0.1 // ep
    );

    dtest(refiner,
        5, // face
        0.5, // s
        0.5, // t
        0.1 // ep
    );

    // This one is specifically chosen to lie in a Gregory patch to make sure those derivatives get tested too.
    dtest(refiner,
        2, // face
        0.01, // s
        0.99, // t
        0.005 // ep
    );


    // If we don't trust the result yet, we could always run some more with randomly chosen start points...
    for ( int i = 0 ; i < 3 ; ++i )
        dtestrand( refiner );


    delete refiner;

    return 0;
}
