/*
 *  RAW-domain AI denoise core: Bayer mosaic in, denoised Bayer mosaic out, plus a
 *  minimal DNG writer. Deliberately free of GTK/GLib so it can be tested on its own.
 *  The tile scheme follows darktable's restore_raw_bayer.c (GPL-3.0-or-later);
 *  the model is the RawNIND UtNet2 Bayer denoiser by Benoit Brummer (GPL-3.0).
 *  Part of the personal Capture-One-like fork; GPL-3.0-or-later like RawTherapee.
 */
#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace AIRawCore
{

struct BayerInfo {
    int width = 0;
    int height = 0;
    int cfa[2][2] = {{0, 1}, {1, 2}};   // colour at (row & 1, col & 1): 0 = R, 1 = G, 2 = B
    float black[2][2] = {{0, 0}, {0, 0}};
    float white = 65535.f;
};

struct DngMeta {
    std::string make;
    std::string model;
    std::string lens;
    std::string dateTime;        // "YYYY:MM:DD HH:MM:SS" or empty
    int orientation = 1;         // TIFF orientation
    int iso = 0;
    double shutter = 0.0;        // seconds
    double fnumber = 0.0;
    double focal = 0.0;          // mm
    double colorMatrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // XYZ (D65) -> camera
    double asShotNeutral[3] = {1, 1, 1};
};

// Loads the ONNX Runtime library from the given full path and initialises the API.
// Must succeed once before denoiseBayer is used. Returns "" or an error text.
#ifdef _WIN32
std::string initRuntime(const wchar_t* libraryPath);
#else
std::string initRuntime(const char* libraryPath);
#endif

// "GPU" or "CPU": what the most recent denoise or upscale run actually used. The graphics
// card (DirectML) is used when it works and its results match the CPU; setting the
// environment variable RT_AI_DEVICE=cpu forces the CPU.
std::string lastDevice();

// Relative difference (in percent) between the graphics card and the CPU measured when the
// network of the last run was checked, or a negative value when no check has been made.
double lastGpuDeviation();

// Allows or forbids the graphics card for the following runs (default: allowed).
void setGpuAllowed(bool allowed);

// Denoises the mosaic in place. rows[r] points to `width` samples in raw sensor units.
// strength is 0..1 (blend between the original and the denoised data).
// Returns an empty string on success, an error text otherwise.
#ifdef _WIN32
std::string denoiseBayer(const wchar_t* modelPath, float** rows, const BayerInfo& info, float strength, const std::function<void(double)>& progress);
#else
std::string denoiseBayer(const char* modelPath, float** rows, const BayerInfo& info, float strength, const std::function<void(double)>& progress);
#endif

// Upscales an RGB image to twice its size and streams the result to `out` as an
// uncompressed 16 bit RGB TIFF. r/g/b[row] point to `width` samples in 0..65535, already
// encoded with the output profile's tone curve. icc may be empty. modelKey names the
// network (used to remember whether the graphics card works for it).
// Returns an empty string on success, an error text otherwise.
#ifdef _WIN32
std::string upscale2x(const wchar_t* modelPath, const float* const* r, const float* const* g, const float* const* b,
                      int width, int height, FILE* out, const std::string& icc, const std::string& modelKey, const std::function<void(double)>& progress);
#else
std::string upscale2x(const char* modelPath, const float* const* r, const float* const* g, const float* const* b,
                      int width, int height, FILE* out, const std::string& icc, const std::string& modelKey, const std::function<void(double)>& progress);
#endif

// Writes an uncompressed 16 bit CFA DNG. Returns false on a write error.
bool writeDng(FILE* f, float** rows, const BayerInfo& info, const DngMeta& meta);

// --- object selection (Segment Anything 2.1) -------------------------------------------

struct SelectPoint {
    double x, y;   // 0..1 of the full image
    bool add;      // true: this belongs to the selection; false: this does not
};

struct SelectRequest {
    double cx = 0.5, cy = 0.5;     // a point on the object, 0..1 of the full image
    bool useBox = false;           // optional box around the object, 0..1
    double bx0 = 0.0, by0 = 0.0, bx1 = 1.0, by1 = 1.0;
    std::vector<SelectPoint> more; // further points
};

struct SelectResult {
    int w = 0, h = 0;                      // mask size (covers the full image)
    std::vector<unsigned char> mask;       // 0..255
    double x0 = 0.0, y0 = 0.0, x1 = 1.0, y1 = 1.0;   // bounding box of the selection, 0..1
    double area = 0.0;                     // selected fraction of the image
    float quality = 0.f;                   // the network's own estimate, 0..1
    bool zoomed = false;                   // a zoomed-in pass on a small object was made
    int tiles = 0;                         // number of tiles refined at higher resolution
    std::vector<unsigned char> preview;    // RGB preview with the selection in red
    int previewW = 0, previewH = 0;
};

// Develops the full frame (w <= 0) or a region of it (in full-image pixels), scaled down so
// that its longest side is at most maxSide. nativeW/H receive the size before scaling.
typedef std::function<bool(int x, int y, int w, int h, int maxSide, std::vector<unsigned char>& rgb, int& outW, int& outH, int& nativeW, int& nativeH)> RenderFn;

// Selects the object at the requested point: a first pass over the whole frame, then either
// a zoomed-in pass (small objects) or a tile-by-tile pass at higher resolution (large
// regions), edge snapping and removal of specks and pinholes.
// Returns "" or an error text ("nothing" when no object was found at the point).
#ifdef _WIN32
std::string selectObject(const wchar_t* encoderPath, const wchar_t* decoderPath, const RenderFn& render,
                         const SelectRequest& request, SelectResult& result, const std::function<void(double)>& progress);
#else
std::string selectObject(const char* encoderPath, const char* decoderPath, const RenderFn& render,
                         const SelectRequest& request, SelectResult& result, const std::function<void(double)>& progress);
#endif

// The same in steps, for interactive use: selectBegin develops the frame and loads it into
// the network (slow); selectPreview then answers within a fraction of a second for any set
// of points and returns the frame with the rough selection in red; selectFinish makes the
// final, refined mask; selectEnd frees everything. One session at a time.
#ifdef _WIN32
std::string selectBegin(const wchar_t* encoderPath, const wchar_t* decoderPath, const RenderFn& render, const std::function<void(double)>& progress);
#else
std::string selectBegin(const char* encoderPath, const char* decoderPath, const RenderFn& render, const std::function<void(double)>& progress);
#endif
std::string selectPreview(const std::vector<SelectPoint>& points, int maxW, int maxH, std::vector<unsigned char>& rgb, int& w, int& h, double& area);
std::string selectFinish(const std::vector<SelectPoint>& points, const SelectRequest* box, SelectResult& result, const std::function<void(double)>& progress);
void selectEnd();

}
