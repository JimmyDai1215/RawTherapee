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

// Denoises the mosaic in place. rows[r] points to `width` samples in raw sensor units.
// strength is 0..1 (blend between the original and the denoised data).
// Returns an empty string on success, an error text otherwise.
#ifdef _WIN32
std::string denoiseBayer(const wchar_t* modelPath, float** rows, const BayerInfo& info, float strength, const std::function<void(double)>& progress);
#else
std::string denoiseBayer(const char* modelPath, float** rows, const BayerInfo& info, float strength, const std::function<void(double)>& progress);
#endif

// Writes an uncompressed 16 bit CFA DNG. Returns false on a write error.
bool writeDng(FILE* f, float** rows, const BayerInfo& info, const DngMeta& meta);

}
