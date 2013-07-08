..  
       Copyright (C) Pixar. All rights reserved.
  
       This license governs use of the accompanying software. If you
       use the software, you accept this license. If you do not accept
       the license, do not use the software.
  
       1. Definitions
       The terms "reproduce," "reproduction," "derivative works," and
       "distribution" have the same meaning here as under U.S.
       copyright law.  A "contribution" is the original software, or
       any additions or changes to the software.
       A "contributor" is any person or entity that distributes its
       contribution under this license.
       "Licensed patents" are a contributor's patent claims that read
       directly on its contribution.
  
       2. Grant of Rights
       (A) Copyright Grant- Subject to the terms of this license,
       including the license conditions and limitations in section 3,
       each contributor grants you a non-exclusive, worldwide,
       royalty-free copyright license to reproduce its contribution,
       prepare derivative works of its contribution, and distribute
       its contribution or any derivative works that you create.
       (B) Patent Grant- Subject to the terms of this license,
       including the license conditions and limitations in section 3,
       each contributor grants you a non-exclusive, worldwide,
       royalty-free license under its licensed patents to make, have
       made, use, sell, offer for sale, import, and/or otherwise
       dispose of its contribution in the software or derivative works
       of the contribution in the software.
  
       3. Conditions and Limitations
       (A) No Trademark License- This license does not grant you
       rights to use any contributor's name, logo, or trademarks.
       (B) If you bring a patent claim against any contributor over
       patents that you claim are infringed by the software, your
       patent license from such contributor to the software ends
       automatically.
       (C) If you distribute any portion of the software, you must
       retain all copyright, patent, trademark, and attribution
       notices that are present in the software.
       (D) If you distribute any portion of the software in source
       code form, you may do so only under this license by including a
       complete copy of this license with your distribution. If you
       distribute any portion of the software in compiled or object
       code form, you may only do so under a license that complies
       with this license.
       (E) The software is licensed "as-is." You bear the risk of
       using it. The contributors give no express warranties,
       guarantees or conditions. You may have additional consumer
       rights under your local laws which this license cannot change.
       To the extent permitted under your local laws, the contributors
       exclude the implied warranties of merchantability, fitness for
       a particular purpose and non-infringement.
  

Getting Started
---------------

.. contents::
   :local:
   :backlinks: none


Getting started with Git and accessing the source code. 


Downloading the code
====================

The code is hosted on a Github public repository. Download and setup information 
for Git tools can be found `here <https://help.github.com/articles/set-up-git>`__.

You can access the OpenSubdiv Git repository at https://github.com/PixarAnimationStudios/OpenSubdiv

From there, there are several ways of downloading the OpenSubdiv source code.

    - Zip archive : downloaded from `here <https://github.com/PixarAnimationStudios/OpenSubdiv/archive/dev.zip>`__
      
    - Using a GUI client : you can find a list `here <http://git-scm.com/downloads/guis>`__
      Please refer to the documentation of your preferred application.

    - From the GitShell, Cygwin or the CLI : assuming that you have the Git tools 
      installed, you can clone the OpenSubdiv repository directly with the 
      following command:
      
      .. code:: c++
      
          git clone https://github.com/PixarAnimationStudios/OpenSubdiv.git
      
      

These methods only pull static archives, which is are not under the version 
control system and therefore cannot pull updates or push changes back. If you
intend on contributing features or fixes to the main trunk of the code, you will
need to create a free Github account and clone a fork of the OpenSubdiv repository.

Submissions to the main code trunk can be sent using Git's pull-request mechanisms.
Please note that we are using the git flow tools so all changes should be made to
our 'dev' branch. Before we can accept submissions however, we will need a signed 
`Contributor's License Agreement <intro.html#contributing>`__.

----

Branches & Git Flow
===================

Since version 1.1.0, OpenSubdiv has adopted the `Git Flow 
<http://nvie.com/posts/a-successful-git-branching-model/>`__ branching model .

Our active development branch is named 'dev' : all new features and buf fixes should
be submitted to this branch. The changes submitted to the dev branch are periodically
patched to the 'master' branch as new versions are released.

Checking out branches
_____________________

The Git Flow `tools <https://github.com/nvie/gitflow>`__ are not a requisite for 
working with the OpenSubdiv code base, but new work should always be performed in
the 'dev' branch, or dedicated feature-branches. By default, a cloned repository
will be pointing to the 'master' branch. You can switch to the 'dev' branch using
the following command:

.. code:: c++

    git branch dev

You can check that the branch has now been switched simply with:

.. code:: c++

    git branch

Which should return:

.. code:: c++

    * dev
      master


API Versions
____________

OpenSubdiv maintains an internal API versioning system. The version number can be
read from the file `./opensubdiv/version.h <https://github.com/PixarAnimationStudios/OpenSubdiv/blob/master/opensubdiv/version.h>`__.
Following the Git-Flow pattern, our releases are indexed using Git's tagging
system.

List of the existing tags:

.. code:: c++

    git tag --list

Checking out version 1.2.0:

.. code:: c++

    git checkout v1_2_0

Making Changes
______________

Direct push access to the OpenSubdiv master repository is currently limited to a 
small internal development team, so external submissions should be made by sending 
`pull-requests <https://help.github.com/articles/using-pull-requests>`__ from 
forks of our 'dev' branch. 

----

Code Overview
=============

The OpenSubdiv code base contains the following main areas:

**./opensubdiv/**

  The main subdivision APIs : Hbr, Far and Osd. 


**./regression/**

  Standalone regression tests and baseline data to help maintain the integrity of
  our APIs. If GPU SDKs are detected, some tests will attempt to run computations
  on those GPUs.

**./examples/**

  A small collection of standalone applications that illustrate how to deploy the
  various features and optimizations of the OpenSubdiv APIs. The GL-based examples
  rely on the cross-platform GLFW API for interactive window management, while the
  DirectX ones are OS-native.

**./python/**

  Python-SWIG bindings for a minimal uniform refinement wrapper 

**./documentation/**

  The reStructuredText source files along with python scripts that generate the HTML
  documentation site.

----

Next : `Building OpenSubdiv <cmake_build.html>`__
