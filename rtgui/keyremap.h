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
#pragma once

#include <gdk/gdk.h>

/**
 * User-configurable keyboard shortcut remapping.
 *
 * Shortcuts are hard-coded throughout the GUI. Instead of touching every
 * site, key events are translated once at the window level according to
 * the "keymap" file in the user's config directory: each line maps the
 * key the user presses to the built-in shortcut it should trigger.
 */
namespace KeyRemap
{

/// Rewrites the event in place if the pressed key is remapped.
void apply(GdkEventKey* event);

}
