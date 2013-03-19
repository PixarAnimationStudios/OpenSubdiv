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
static const std::string loop_icosahedron =
"# This file uses centimeters as units for non-parametric coordinates.\n"
"\n"
"v 0.850651 0.000000 -0.525731\n"
"v 0.850651 -0.000000 0.525731\n"
"v -0.850651 -0.000000 0.525731\n"
"v -0.850651 0.000000 -0.525731\n"
"v 0.000000 -0.525731 0.850651\n"
"v 0.000000 0.525731 0.850651\n"
"v 0.000000 0.525731 -0.850651\n"
"v 0.000000 -0.525731 -0.850651\n"
"v -0.525731 -0.850651 -0.000000\n"
"v 0.525731 -0.850651 -0.000000\n"
"v 0.525731 0.850651 0.000000\n"
"v -0.525731 0.850651 0.000000\n"
"vt 0.181818 0.250000\n"
"vt 0.363636 0.250000\n"
"vt 0.545455 0.250000\n"
"vt 0.727273 0.250000\n"
"vt 0.909091 0.250000\n"
"vt 0.090909 0.416667\n"
"vt 0.272727 0.416667\n"
"vt 0.454545 0.416667\n"
"vt 0.636364 0.416667\n"
"vt 0.818182 0.416667\n"
"vt 1.000000 0.416667\n"
"vt 0.000000 0.583333\n"
"vt 0.181818 0.583333\n"
"vt 0.363636 0.583333\n"
"vt 0.545455 0.583333\n"
"vt 0.727273 0.583333\n"
"vt 0.909091 0.583333\n"
"vt 0.090909 0.750000\n"
"vt 0.272727 0.750000\n"
"vt 0.454545 0.750000\n"
"vt 0.636364 0.750000\n"
"vt 0.818182 0.750000\n"
"vn 0.850651 -0.000000 0.525731\n"
"vn 0.525731 -0.850651 -0.000000\n"
"vn 0.850651 0.000000 -0.525731\n"
"vn 0.525731 0.850651 0.000000\n"
"vn -0.000000 -0.525731 -0.850651\n"
"vn 0.000000 0.525731 -0.850651\n"
"vn -0.000000 -0.525731 0.850651\n"
"vn -0.000000 0.525731 0.850651\n"
"vn -0.850651 0.000000 -0.525731\n"
"vn -0.525731 -0.850651 0.000000\n"
"vn -0.850651 0.000000 0.525731\n"
"vn -0.525731 0.850651 0.000000\n"
"f 2/17/1 10/22/2 1/16/3\n"
"f 1/16/3 11/10/4 2/17/1\n"
"f 1/16/3 8/15/5 7/9/6\n"
"f 1/16/3 7/9/6 11/10/4\n"
"f 1/16/3 10/21/2 8/15/5\n"
"f 5/13/7 2/12/1 6/6/8\n"
"f 10/18/2 2/12/1 5/13/7\n"
"f 2/17/1 11/10/4 6/11/8\n"
"f 4/8/9 9/14/10 3/7/11\n"
"f 3/7/11 12/2/12 4/8/9\n"
"f 5/13/7 6/6/8 3/7/11\n"
"f 3/7/11 9/14/10 5/13/7\n"
"f 6/6/8 12/1/12 3/7/11\n"
"f 7/9/6 8/15/5 4/8/9\n"
"f 4/8/9 12/3/12 7/9/6\n"
"f 4/8/9 8/15/5 9/14/10\n"
"f 5/13/7 9/14/10 10/19/2\n"
"f 6/11/8 11/10/4 12/5/12\n"
"f 7/9/6 12/4/12 11/10/4\n"
"f 8/15/5 10/20/2 9/14/10\n"
"t interpolateboundary 1/0/0 2\n"
"\n"
;
