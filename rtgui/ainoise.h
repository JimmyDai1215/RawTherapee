/*
 *  AI denoise: denoises a Bayer raw file with a neural network (ONNX Runtime) and
 *  writes the result as a new DNG raw file.
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

// Full paths of the model and of the runtime library expected in the program folder.
Glib::ustring modelPath();
Glib::ustring runtimePath();

// True when the model and the runtime are both installed.
bool available();

// Whether the graphics card may be used, and what the last run actually used ("GPU"/"CPU").
void setGpuAllowed(bool allowed);
Glib::ustring lastDevice();

// Loads rawFile, denoises its mosaic (strength 0..1) and writes dngFile.
// Returns an empty string on success, otherwise a human readable error message.
// progress is called from the calling thread with a value in 0..1.
Glib::ustring rawToDng(const Glib::ustring& rawFile, const Glib::ustring& dngFile, float strength, const std::function<void(double)>& progress);

// --- 2x upscaling of a developed image

Glib::ustring upscaleModelPath();
bool upscaleAvailable();

// Upscales the developed image (values 0..65535) to twice its size and writes a 16 bit TIFF.
// Returns an empty string on success, otherwise a human readable error message.
Glib::ustring upscaleToTiff(rtengine::IImagefloat* img, const Glib::ustring& tifFile, const std::function<void(double)>& progress);

}
