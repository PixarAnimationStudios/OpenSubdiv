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
static const std::string catmark_pyramid_creases0 =
"# This file uses centimeters as units for non-parametric coordinates.\n"
"\n"
"v 0.0 0.0 2.0\n"
"v 0.0 -2.0 0.0\n"
"v 2.0 0.0 0.0\n"
"v 0.0 2.0 0.0\n"
"v -2.0 0.0 0.0\n"
"vt 0.0 0.0\n"
"vt 0.0 -2.0\n"
"vt 2.0 0.0\n"
"vt 0.0 2.0\n"
"vt -2.0 0.0\n"
"s off\n"
"f 1/1/1 2/2/2 3/3/3\n"
"f 1/1/1 3/3/3 4/4/4\n"
"f 1/1/1 4/4/4 5/5/5\n"
"f 1/1/1 5/5/5 2/2/2\n"
"f 5/5/5 4/4/4 3/3/3 2/2/2\n"
"t crease 2/1/0 4 3 3.0\n"
"t crease 2/1/0 3 2 3.0\n"
"t crease 2/1/0 2 1 3.0\n"
"t crease 2/1/0 1 4 3.0\n"
"t interpolateboundary 1/0/0 2\n"
"\n"
;
