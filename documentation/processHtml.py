#!/usr/bin/env python

#
#     Copyright (C) Pixar. All rights reserved.
#
#     This license governs use of the accompanying software. If you
#     use the software, you accept this license. If you do not accept
#     the license, do not use the software.
#
#     1. Definitions
#     The terms "reproduce," "reproduction," "derivative works," and
#     "distribution" have the same meaning here as under U.S.
#     copyright law.  A "contribution" is the original software, or
#     any additions or changes to the software.
#     A "contributor" is any person or entity that distributes its
#     contribution under this license.
#     "Licensed patents" are a contributor's patent claims that read
#     directly on its contribution.
#
#     2. Grant of Rights
#     (A) Copyright Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free copyright license to reproduce its contribution,
#     prepare derivative works of its contribution, and distribute
#     its contribution or any derivative works that you create.
#     (B) Patent Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free license under its licensed patents to make, have
#     made, use, sell, offer for sale, import, and/or otherwise
#     dispose of its contribution in the software or derivative works
#     of the contribution in the software.
#
#     3. Conditions and Limitations
#     (A) No Trademark License- This license does not grant you
#     rights to use any contributor's name, logo, or trademarks.
#     (B) If you bring a patent claim against any contributor over
#     patents that you claim are infringed by the software, your
#     patent license from such contributor to the software ends
#     automatically.
#     (C) If you distribute any portion of the software, you must
#     retain all copyright, patent, trademark, and attribution
#     notices that are present in the software.
#     (D) If you distribute any portion of the software in source
#     code form, you may do so only under this license by including a
#     complete copy of this license with your distribution. If you
#     distribute any portion of the software in compiled or object
#     code form, you may only do so under a license that complies
#     with this license.
#     (E) The software is licensed "as-is." You bear the risk of
#     using it. The contributors give no express warranties,
#     guarantees or conditions. You may have additional consumer
#     rights under your local laws which this license cannot change.
#     To the extent permitted under your local laws, the contributors
#     exclude the implied warranties of merchantability, fitness for
#     a particular purpose and non-infringement.
#

import os
import sys
import string
import re
import HTMLParser

class HtmlToTextParser(HTMLParser.HTMLParser):
    def __init__(self):
        HTMLParser.HTMLParser.__init__(self)
        self.m_text = []
        self.m_inTitle = False
        self.m_inScript = False
        self.m_inStyle = False
        self.m_title = ""
        self.m_navigation = False

    def handle_data(self, data):
        if self.m_inScript or self.m_inStyle:
            return
        text = data.strip()
        if len(text) > 0:
            text = re.sub('[\s]+', ' ', text)
            text = re.sub('[^\.,\- a-zA-Z0-9_]+', '', text)
            self.m_text.append(text + ' ')
        if self.m_inTitle:
            self.m_title = str(text)
            
    def handle_endtag(self, tag):
        if tag.lower() == "title": self.m_inTitle = False
        if tag.lower() == "script": self.m_inScript = False
        if tag.lower() == "style": self.m_inStyle = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title": self.m_inTitle = True
        if tag.lower() == "script": self.m_inScript = True
        if tag.lower() == "style": self.m_inStyle = True        
        if tag.lower() == "div":
            for attr in attrs:
                if (len(attr)>=2 and \
                    attr[0].lower()=="class" and \
                    attr[1].lower()=="navigation"):
                    self.m_navigation = True

    def HasNavigationSection(self):
        return self.m_navigation

    def GetText(self):
        return ''.join(self.m_text).strip()
        
    def GetTitle(self):
        return self.m_title

#-------------------------------------------------------------------------------
def StripHTMLComments(data):
    regex = re.compile('\<![ \r\n\t]*(--([^\-]|[\r\n]|-[^\-])*--[ \r\n\t]*)\>')
    return regex.sub('',data)

#-------------------------------------------------------------------------------
def ReadNavigationTemplate( filePath ):

    navHtml = ""

    try:
        navFile = open( filePath, "r")
    except IOError:
        print "Could not open file \'"+filePath+"\'"
    
    with navFile:
        print "Navigation template: \'"+filePath+"\'"
        navHtml = navFile.read()
        navHtml = StripHTMLComments(navHtml)
        navFile.close()
        navHtml = StripHTMLComments(navHtml)
        
    return navHtml
    
#-------------------------------------------------------------------------------
def WriteIndexFile( outputFile, content ):
    outputPath = os.path.dirname( outputFile )
    
    try:
        os.makedirs( outputPath );
    except:
        pass

    print "Creating Search-Index File : \""+outputFile+"\""

    f = open(outputFile, "w")
    f.write(content)
    f.close()

#-------------------------------------------------------------------------------
def Usage():
    print str(sys.argv[0])+" <input directory> <output directory> <html template>"
    exit(1);


#-------------------------------------------------------------------------------
# Main
if (len(sys.argv)<3):
    Usage()

rootDir = str(sys.argv[1])

navTemplate = str(sys.argv[2])
    
navHtml = ReadNavigationTemplate( navTemplate )

print "Scanning : \'"+rootDir+"\'"

searchIndex = 'var tipuesearch = { "pages": [ '

# recursively scan sub-directories for HTML files
for root, dirs, files in os.walk(rootDir):

    # skip doxygen generated HTML
    if 'doxy_html' in dirs:
        dirs.remove('doxy_html')

    for f in files:

        inputFile = os.path.join(root, f)
        if inputFile.endswith(".html") or inputFile.endswith(".htm") :

            f = open(inputFile, "r+")
            html = f.read()

            # parse the ReST generated HTML
            parser = HtmlToTextParser()
            try:
                parser.feed(html)
                title = parser.GetTitle()
                text = parser.GetText()
            except HTMLParser.HTMLParseError:
                continue

            msg = "    \""+inputFile+"\" - "

            # index the contents of the page for search
            if (not inputFile.lower().endswith("search.html")):
                if title == "":
                    title = "untitled"
                loc = os.path.relpath(inputFile, rootDir)
                searchIndex += '{"title":"'+title+'", "text":"'+text+'", "tags": "", "loc":"'+loc+'"}, \n'
                msg += "indexed - "

            # if necessary, insert navigation html
            if (not parser.HasNavigationSection()):
                loc = string.find(html,"<body>")
                html = html[:loc+6] + navHtml + html[loc+6:]

                msg += "added navigation"

                f.seek(0)
                f.write(html)
                f.close()

            print msg

searchIndex = searchIndex + "]};"

WriteIndexFile( os.path.join(rootDir, "tipuesearch", "tipuesearch_content.js"), searchIndex )

