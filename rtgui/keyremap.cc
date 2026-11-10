/*
 *  This file is part of RawTherapee.
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RawTherapee is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RawTherapee.  If not, see <https://www.gnu.org/licenses/>.
 */
#include "keyremap.h"

#include <sstream>
#include <map>
#include <string>
#include <utility>

#include <glibmm/miscutils.h>
#include <gtk/gtk.h>

#include "options.h"

namespace
{

const guint MOD_MASK = GDK_CONTROL_MASK | GDK_SHIFT_MASK | GDK_MOD1_MASK;

struct Combo {
    guint keyval;   // always stored lower-case
    guint mods;
    bool operator<(const Combo& o) const
    {
        return std::make_pair(keyval, mods) < std::make_pair(o.keyval, o.mods);
    }
};

std::map<Combo, Combo> keymap;
bool loaded = false;

// Written on first start. Every mapping is commented out, so behaviour is
// unchanged until the user enables lines.
const char* const TEMPLATE =
    "# RawTherapee keymap\n"
    "#\n"
    "# Each line:   <key you press> = <built-in shortcut it triggers>\n"
    "# Modifiers:   <Ctrl> <Shift> <Alt>     Example:  <Ctrl><Shift>d = <Ctrl>b\n"
    "# Key names:   letters, digits, F1..F12, comma, period, less, greater,\n"
    "#              bracketleft, bracketright, plus, minus, Left, Right, ...\n"
    "# Use 'none' on the right side to disable a key.\n"
    "# Lines starting with # are ignored. Restart RawTherapee after editing.\n"
    "# Remapping only applies to shortcuts, not to typing in text fields.\n"
    "#\n"
    "# ---- Capture One style (remove the leading # to enable) ----\n"
    "# r = s                    # Straighten tool        (built-in: S)\n"
    "# y = <Shift>b             # Before / After         (built-in: Shift+B)\n"
    "# period = z               # Zoom to 100%           (built-in: Z)\n"
    "# comma = f                # Zoom to fit            (built-in: F)\n"
    "# <Ctrl>0 = f              # Zoom to fit\n"
    "# <Ctrl><Alt>0 = z         # Zoom to 100%\n"
    "# <Ctrl>plus = plus        # Zoom in                (built-in: +)\n"
    "# <Ctrl>equal = plus\n"
    "# <Ctrl>minus = minus      # Zoom out               (built-in: -)\n"
    "# <Ctrl>d = <Ctrl>s        # Process / export       (built-in: Ctrl+S)\n"
    "# <Ctrl><Shift>d = <Ctrl>b # Add to batch queue     (built-in: Ctrl+B)\n"
    "# <Ctrl>e = greater        # Highlight warning      (built-in: >)\n"
    "# <Ctrl><Shift>e = less    # Shadow warning         (built-in: <)\n"
    "# <Ctrl><Alt>e = <Ctrl>e   # Edit in external app   (built-in: Ctrl+E)\n"
    "# <Ctrl>Right = <Shift>F4  # Next image             (built-in: Shift+F4)\n"
    "# <Ctrl>Left = <Shift>F3   # Previous image         (built-in: Shift+F3)\n"
    "# <Ctrl>t = <Alt>l         # Show / hide tools      (built-in: Alt+L)\n"
    "# <Alt>r = r               # Red channel preview    (built-in: R)\n";

std::string trim(const std::string& s)
{
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return {};
    }
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool parseCombo(const std::string& text, Combo& out)
{
    if (text == "none") {
        out = {GDK_KEY_VoidSymbol, 0};
        return true;
    }

    guint key = 0;
    GdkModifierType mods = GdkModifierType(0);
    gtk_accelerator_parse(text.c_str(), &key, &mods);

    if (!key) {
        return false;
    }

    out = {gdk_keyval_to_lower(key), guint(mods) & MOD_MASK};
    return true;
}

void load()
{
    loaded = true;

    // GLib file functions handle non-ASCII paths on Windows.
    const std::string path = Glib::build_filename(Options::rtdir, "keymap");
    gchar* contents = nullptr;

    if (!g_file_get_contents(path.c_str(), &contents, nullptr, nullptr)) {
        g_file_set_contents(path.c_str(), TEMPLATE, -1, nullptr);
        return;
    }

    std::istringstream in(contents);
    g_free(contents);

    std::string line;

    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line.erase(hash);
        }

        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        Combo from, to;
        if (parseCombo(trim(line.substr(0, eq)), from) && parseCombo(trim(line.substr(eq + 1)), to)
                && from.keyval != GDK_KEY_VoidSymbol) {
            keymap[from] = to;
        } else {
            g_warning("keymap: cannot parse line \"%s\"", line.c_str());
        }
    }
}

}

void KeyRemap::apply(GdkEventKey* event)
{
    if (!loaded) {
        load();
    }

    if (keymap.empty()) {
        return;
    }

    const guint lower = gdk_keyval_to_lower(event->keyval);
    const guint mods = event->state & MOD_MASK;

    auto it = keymap.find({lower, mods});

    // Symbols typed with Shift (e.g. "<", "+") can be written without <Shift>.
    if (it == keymap.end() && (mods & GDK_SHIFT_MASK) && lower == gdk_keyval_to_upper(lower)) {
        it = keymap.find({lower, mods & ~guint(GDK_SHIFT_MASK)});
    }

    if (it == keymap.end()) {
        return;
    }

    const Combo& to = it->second;

    event->keyval = (to.mods & GDK_SHIFT_MASK) ? gdk_keyval_to_upper(to.keyval) : to.keyval;
    event->state = (event->state & ~MOD_MASK) | to.mods;
    event->hardware_keycode = 0;

    // Some handlers switch on the hardware keycode instead of the keyval.
    GdkKeymapKey* keys = nullptr;
    gint n = 0;

    if (to.keyval != GDK_KEY_VoidSymbol
            && gdk_keymap_get_entries_for_keyval(gdk_keymap_get_for_display(gdk_display_get_default()), to.keyval, &keys, &n)) {
        if (n > 0) {
            event->hardware_keycode = keys[0].keycode;
        }
        g_free(keys);
    }
}
