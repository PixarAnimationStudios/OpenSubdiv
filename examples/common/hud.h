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
