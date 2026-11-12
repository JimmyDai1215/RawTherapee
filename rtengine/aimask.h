/*
 *  AI mask for local adjustments: an arbitrary-shape, soft raster mask stored in a small
 *  file next to the photo and multiplied into the shape of a Locallab spot.
 *  Part of the personal Capture-One-like fork; GPL-3.0-or-later like RawTherapee.
 */
#pragma once

#include <memory>
#include <vector>
#include <glibmm/ustring.h>

namespace rtengine
{

struct AiMask {
    int w = 0;
    int h = 0;
    std::vector<unsigned char> data;   // 0..255, row by row, covers the whole image

    // u, v in 0..1 over the full (uncropped) image; returns 0..1
    float sample(float u, float v) const
    {
        float x = u * w - 0.5f;
        float y = v * h - 0.5f;
        if (x < 0.f) {
            x = 0.f;
        } else if (x > w - 1.001f) {
            x = w - 1.001f;
        }
        if (y < 0.f) {
            y = 0.f;
        } else if (y > h - 1.001f) {
            y = h - 1.001f;
        }
        const int x0 = static_cast<int>(x);
        const int y0 = static_cast<int>(y);
        const float fx = x - x0;
        const float fy = y - y0;
        const unsigned char* p = data.data() + static_cast<size_t>(y0) * w + x0;
        const float top = p[0] + (p[1] - p[0]) * fx;
        const float bottom = p[w] + (p[w + 1] - p[w]) * fx;
        return (top + (bottom - top) * fy) * (1.f / 255.f);
    }
};

// Loads a mask file; results are cached. Returns an empty pointer if the file cannot be read.
std::shared_ptr<const AiMask> loadAiMask(const Glib::ustring& path);

// Writes a mask file (w and h must be at least 2). Returns false on error.
bool saveAiMask(const Glib::ustring& path, int w, int h, const unsigned char* data);

}
