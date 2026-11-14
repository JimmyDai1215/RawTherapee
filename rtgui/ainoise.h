/*
 *  AI denoise: denoises a Bayer raw file with a neural network (ONNX Runtime) and
 *  writes the result as a new DNG raw file.
 *  Part of the personal Capture-One-like fork; GPL-3.0-or-later like the rest of RawTherapee.
 */
#pragma once

#include <functional>
#include <vector>
#include <glibmm/ustring.h>

#include "airawcore.h"

namespace rtengine
{
class IImagefloat;
class InitialImage;
namespace procparams
{
class ProcParams;
}
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
double lastGpuDeviation();   // percent, negative when not measured

// Loads rawFile, denoises its mosaic (strength 0..1) and writes dngFile.
// Returns an empty string on success, otherwise a human readable error message.
// progress is called from the calling thread with a value in 0..1.
Glib::ustring rawToDng(const Glib::ustring& rawFile, const Glib::ustring& dngFile, float strength, const std::function<void(double)>& progress);

// --- 2x upscaling of a developed image

// model: 0 = RealPLKSR (faithful, keeps fine texture), 1 = BSRGAN (clean and crisp)
Glib::ustring upscaleModelPath(int model);
bool upscaleAvailable(int model);

// Upscales the developed image (values 0..65535) to twice its size and writes a 16 bit TIFF.
// Returns an empty string on success, otherwise a human readable error message.
// metadataFrom: the photo whose EXIF/IPTC/XMP data the TIFF gets (may be empty).
Glib::ustring upscaleToTiff(rtengine::IImagefloat* img, const Glib::ustring& tifFile, int model, const std::function<void(double)>& progress,
                            const Glib::ustring& metadataFrom = Glib::ustring());

// --- AI mask (object selection) for local adjustments

struct MaskPoint {
    double x, y;   // 0..1 of the full (uncropped) image
    bool add;      // true: belongs to the selection; false: does not
};

struct MaskRequest {
    double centerX = 0.5;   // a point on the object, 0..1 of the full (uncropped) image
    double centerY = 0.5;
    bool useBox = false;    // optional box around the object, 0..1
    double boxX0 = 0.0, boxY0 = 0.0, boxX1 = 1.0, boxY1 = 1.0;
    std::vector<MaskPoint> more;   // further points
};

struct MaskResult {
    double x0 = 0.0, y0 = 0.0, x1 = 1.0, y1 = 1.0;   // bounding box of the selection, 0..1
    double area = 0.0;                               // selected fraction of the image
    float quality = 0.f;                             // the network's own estimate, 0..1
    bool refined = false;                            // a second, zoomed-in pass was made
    std::vector<unsigned char> preview;              // RGB preview with the selection in red
    int previewW = 0;
    int previewH = 0;
};

// Develops the full frame (w <= 0) or a region of it (in full-image pixels), scaled down so
// that its longest side is at most maxSide. nativeW/H receive the size before scaling.
typedef std::function<bool(int x, int y, int w, int h, int maxSide, std::vector<unsigned char>& rgb, int& outW, int& outH, int& nativeW, int& nativeH)> MaskRender;

// A renderer that develops the given image with the given edits (crop, resize, sharpening and
// local adjustments are switched off).
MaskRender makeRenderer(rtengine::InitialImage* image, const rtengine::procparams::ProcParams& params);

bool maskAvailable();

// Selects the object at the requested point and writes the mask file.
// Returns an empty string on success, otherwise a human readable error message.
Glib::ustring makeMask(const MaskRender& render, const MaskRequest& request, const Glib::ustring& maskFile, MaskResult& result, const std::function<void(double)>& progress);

// The same in steps, for the mask editor. maskBegin develops the photo and loads it into the
// network; maskView then draws any state of the mask over the photo quickly; maskFinish
// writes the final mask file (and what it was made from next to it); maskEnd frees
// everything.
Glib::ustring maskBegin(const MaskRender& render, const std::function<void(double)>& progress);
Glib::ustring maskView(const AIRawCore::MaskEdit& edit, double cx, double cy, double zoom, int outW, int outH,
                       std::vector<unsigned char>& rgb, double& scaleX, double& scaleY, double& offX, double& offY);
Glib::ustring maskFinish(const AIRawCore::MaskEdit& edit, const Glib::ustring& maskFile, MaskResult& result, const std::function<void(double)>& progress);
void maskEnd();

// What a mask was made from. Returns false if that is not known.
bool loadMaskEdit(const Glib::ustring& maskFile, AIRawCore::MaskEdit& edit);

// Shows or hides the mask of the selected spot in the previews.
void setMaskOverlay(bool show);
bool maskOverlay();

}
