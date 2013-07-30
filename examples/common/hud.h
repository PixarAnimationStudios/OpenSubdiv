//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//
#ifndef HUD_H
#define HUD_H

#include <vector>
#include <string>
#include <cstdarg>

#include "hud.h"

class Hud
{
public:
    typedef void (*RadioButtonCallback)(int c);
    typedef void (*CheckBoxCallback)(bool checked, int data);

    struct RadioButton 
    {
        int x, y, w, h;
        int group;
        int localIndex;
        std::string label;
        bool checked;
        RadioButtonCallback callback;
        int callbackData;
        int shortcut;
        bool sharedShortcut;
    };

    struct CheckBox
    {
        int x, y, w, h;
        std::string label;
        bool checked;
        CheckBoxCallback callback;
        int callbackData;
        int shortcut;
    };

    Hud();
    virtual ~Hud();
    
    virtual void Init(int width, int height);

    virtual void Rebuild(int width, int height);

    virtual bool Flush();

    bool IsVisible() const;

    void SetVisible(bool visible);

    void DrawString(int x, int y, const char *fmt, ...);

    void DrawString(int x, int y, float r, float g, float b, const char *fmt, ...);

    void Clear();

    void AddRadioButton(int group, const char *label, bool checked, int x, int y,
                        RadioButtonCallback callback=0, int data=0, int shortcut=0);

    void AddCheckBox(const char *label, bool checked, int x, int y,
                     CheckBoxCallback callback=0, int data=0, int shortcut=0);

    bool KeyDown(int key);

    bool MouseClick(int x, int y);

    int GetWidth() const;
    
    int GetHeight() const;

protected:
    int drawString(std::vector<float> &vboSource, int x, int y, float r, float g, float b, const char *c) const;
    int drawChar(std::vector<float> &vboSource, int x, int y, float r, float g, float b, char ch) const;
    const std::vector<RadioButton> & getRadioButtons() const;
    const std::vector<CheckBox> & getCheckBoxes() const;
    std::vector<float> & getVboSource();

private:
    bool _visible;
    std::vector<float> _vboSource;
    int _windowWidth, _windowHeight;
    bool _requiresRebuildStatic;
    std::vector<RadioButton> _radioButtons;
    std::vector<CheckBox> _checkBoxes;
};

#endif // HUD_H
