/*
 *  AI denoise: runs a neural-network denoiser (ONNX Runtime) on a developed image.
 *  Part of the personal Capture-One-like fork; GPL-3.0-or-later like the rest of RawTherapee.
 */
#pragma once

#include <functional>
#include <glibmm/ustring.h>

namespace rtengine
{
class IImagefloat;
}

namespace AIDenoise
{

// True when this build contains ONNX Runtime support.
bool compiledIn();

// Full path of the model file that is expected next to the program.
Glib::ustring modelPath();

// Denoises the image in place (values 0..65535 as produced by the RawTherapee pipeline).
// Returns an empty string on success, otherwise a human readable error message.
// progress is called from the worker thread with a value in 0..1.
Glib::ustring run(rtengine::IImagefloat* img, const std::function<void(double)>& progress);

}
