Setup:
======
 * clone docco
        git clone https://github.com/jashkenas/docco.git
 * install node.js
        http://nodejs.org/
 * npm install commander
 * sudo easy_install Pygments

Generate documentation:
=======================
$ ~/src/docco/bin/docco simpleCpuSubdivision.cpp



Trina's Setup
(setup above didn't work for me, here's what I did, YMMV)

To Generate docco html file:

* sudo easy_install Pygments

* install node from http://nodejs.org/
* put source into /usr/local/src

    % cd /usr/local/src/node-VERSION
    % ./configure
    % make
    % make install
    % npm install -g docco

* docco should now be in /usr/local/bin
* rehash
* cd /your/source/code/directory
* docco yoursource.cpp
    voila!
* docs go into /your/source/code/directory/docs

