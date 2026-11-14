/*
 *  AI denoise / AI upscale for several photos at once (from the file browser).
 *  Part of the personal Capture-One-like fork; GPL-3.0-or-later like RawTherapee.
 */
#pragma once

#include <vector>
#include <glibmm/ustring.h>
#include <gtkmm/window.h>

#include "rtengine/procparams.h"

namespace AIBatch
{

struct Item {
    Glib::ustring file;
    rtengine::procparams::ProcParams params;   // the photo's current edits
    bool hasParams = false;
};

enum class Mode { DENOISE, UPSCALE };

// Asks for the settings, then works through the photos while showing the progress.
void run(Gtk::Window& parent, const std::vector<Item>& items, Mode mode);

}
