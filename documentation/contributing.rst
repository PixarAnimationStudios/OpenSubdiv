..
     Copyright 2013 Pixar

     Licensed under the Apache License, Version 2.0 (the "Apache License")
     with the following modification; you may not use this file except in
     compliance with the Apache License and the following modification to it:
     Section 6. Trademarks. is deleted and replaced with:

     6. Trademarks. This License does not grant permission to use the trade
        names, trademarks, service marks, or product names of the Licensor
        and its affiliates, except as required to comply with Section 4(c) of
        the License and to reproduce the content of the NOTICE file.

     You may obtain a copy of the Apache License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the Apache License with the above modification is
     distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
     KIND, either express or implied. See the Apache License for the specific
     language governing permissions and limitations under the Apache License.

Contributing to OpenSubdiv
--------------------------

.. contents::
   :local:
   :backlinks: none


----

Contributor License Agreement
=============================

In order for us to accept code submissions (merge git pull-requests), contributors
need to sign the Contributor License Agreement (CLA). There are two CLAs, one for
individuals and one for corporations. As for the end-user license, both are based
on Apache. They are found in the code repository (`individual form
<https://github.com/PixarAnimationStudios/OpenSubdiv/blob/release/OpenSubdivCLA_individual.pdf>`__,
`corporate form <https://github.com/PixarAnimationStudios/OpenSubdiv/blob/release/OpenSubdivCLA_corporate.pdf>`__).
Please email the signed CLA to opensubdiv-cla@pixar.com.

Understand Git
==============

First, you should familiarize yourself with the Git data model and commands.

For small changes you may not need to understand Git deeply, but for larger
changes or working with the codebase over a long period of time, it becomes
critical to understand more of what's going on under the hood.

There are many free resources on the internet, one which we've found useful is
the following e-book:

`<https://github.com/pluralsight/git-internals-pdf/releases>`_

Recommended Git Workflow
========================

Once you have a local development tree cloned and working, you can start making
changes. You will need to integrate changes from the source tree as you work;
the following outlines the workflow used by core OpenSubdiv engineers at Pixar
and DreamWorks:

#. Fork the repository into your own local copy. This can be done via the
   GitHub website using the "fork" button.

#. Clone your fork locally:

     | git clone <your_fork_url> OpenSubdiv.<your_name>
     |
     | e.g.:
     | git clone https://github.com/yourusername/OpenSubdiv.git OpenSubdiv.yourusername

#. Setup two remotes, **origin** and **upstream**. Origin will be setup as a
   result of cloning your remote repository, but upstream must be setup manually:

     | git remote add **upstream** https://github.com/PixarAnimationStudios/OpenSubdiv.git

   Verify your remotes are setup correctly:

     | git remote -v

   Which should look something like:

     | origin https://github.com/yourusername/OpenSubdiv.git (fetch)
     | origin https://github.com/yourusername/OpenSubdiv.git (push)
     | upstream https://github.com/PixarAnimationStudios/OpenSubdiv.git (fetch)
     | upstream https://github.com/PixarAnimationStudios/OpenSubdiv.git (push)

   Finally, fetch the upstream content (this is required for the next step):

     | git fetch upstream

#. Setup a new branch for each change. Working with branches in Git is its
   greatest pleasure, we strongly suggest setting up a new branch for each
   change which you plan to pull-request.

   All work is done in the "dev" branch, so be sure to keep your change in sync
   with this upstream branch. To begin, start your new branch from the dev
   branch:

     | git checkout -b dev-feature upstream/dev

#. As you are working on your feature, new changes will be merged into the
   upstream repository, to sync these changes down and preserve your local
   edits, you can continually rebase your local work:

     | git pull --rebase upstream dev

   Notice the "--rebase" option here. It updates the current branch to the
   upstream/dev branch and rebases all edits so they are at the head of your
   local feature branch.

   Alternatively, you can rebase all your work at once when your feature is
   complete.

Sending a Pull Request
======================

First, rebase and squash your changes appropriately to produce a clean set of
changes at the head of your tree. We require changes to be grouped locally to
ensure that rolling back changes can be done easily.

If you've followed the steps above, your pending change should already be queued
up as required. If you have not, you may need to rebase and squash changes at
this point.

Once the change is clean, push your changes to "origin" and go to the GitHub
website to submit your pull request.

Be sure to submit your request against the "dev" branch.
