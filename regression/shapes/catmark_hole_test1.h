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
static char const * catmark_hole_test1 =
"# This file uses centimeters as units for non-parametric coordinates.\n"
"\n"
"v 0.000000 -1.414214 1.000000\n"
"v 1.414214 0.000000 1.000000\n"
"v -1.414214 0.000000 1.000000\n"
"v 0.000000 1.414214 1.000000\n"
"v -1.414214 0.000000 -1.000000\n"
"v 0.000000 1.414214 -1.000000\n"
"v 0.000000 -1.414214 -1.000000\n"
"v 1.414214 0.000000 -1.000000\n"
"vt 0.375000 0.000000\n"
"vt 0.625000 0.000000\n"
"vt 0.375000 0.250000\n"
"vt 0.625000 0.250000\n"
"vt 0.375000 0.500000\n"
"vt 0.625000 0.500000\n"
"vt 0.375000 0.750000\n"
"vt 0.625000 0.750000\n"
"vt 0.375000 1.000000\n"
"vt 0.625000 1.000000\n"
"vt 0.875000 0.000000\n"
"vt 0.875000 0.250000\n"
"vt 0.125000 0.000000\n"
"vt 0.125000 0.250000\n"
"vn 0.000000 0.000000 1.000000\n"
"vn 0.000000 0.000000 1.000000\n"
"vn 0.000000 0.000000 1.000000\n"
"vn 0.000000 0.000000 1.000000\n"
"vn -0.707107 0.707107 0.000000\n"
"vn -0.707107 0.707107 0.000000\n"
"vn -0.707107 0.707107 0.000000\n"
"vn -0.707107 0.707107 0.000000\n"
"vn 0.000000 0.000000 -1.000000\n"
"vn 0.000000 0.000000 -1.000000\n"
"vn 0.000000 0.000000 -1.000000\n"
"vn 0.000000 0.000000 -1.000000\n"
"vn 0.707107 -0.707107 0.000000\n"
"vn 0.707107 -0.707107 0.000000\n"
"vn 0.707107 -0.707107 0.000000\n"
"vn 0.707107 -0.707107 0.000000\n"
"vn 0.707107 0.707107 0.000000\n"
"vn 0.707107 0.707107 0.000000\n"
"vn 0.707107 0.707107 0.000000\n"
"vn 0.707107 0.707107 0.000000\n"
"vn -0.707107 -0.707107 0.000000\n"
"vn -0.707107 -0.707107 0.000000\n"
"vn -0.707107 -0.707107 0.000000\n"
"vn -0.707107 -0.707107 0.000000\n"
"s off\n"
"f 1/1/1 2/2/2 4/4/3 3/3/4\n"
"f 3/3/5 4/4/6 6/6/7 5/5/8\n"
"f 5/5/9 6/6/10 8/8/11 7/7/12\n"
"f 7/7/13 8/8/14 2/10/15 1/9/16\n"
"f 2/2/17 8/11/18 6/12/19 4/4/20\n"
"f 7/13/21 1/1/22 3/3/23 5/14/24\n"
"t hole 1/0/0 1\n"
;
