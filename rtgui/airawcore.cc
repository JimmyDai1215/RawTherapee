#include "airawcore.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#if defined(HAVE_ONNXRT) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef HAVE_ONNXRT
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#endif

namespace AIRawCore
{

namespace
{

constexpr int T = 512;          // model input size in packed (half resolution) pixels
constexpr int O = 32;           // overlap on each side, packed pixels
constexpr int STEP = T - 2 * O; // distance between tiles, packed pixels
constexpr int TS = 2 * T;       // tile size in sensor pixels

// Reflects i into [lo, hi) without repeating the border sample; keeps the CFA phase.
inline int mirrorIn(int i, int lo, int hi)
{
    const int n = hi - lo;
    if (n <= 1) {
        return lo;
    }
    const int period = 2 * (n - 1);
    int d = (i - lo) % period;
    if (d < 0) {
        d += period;
    }
    return lo + (d < n ? d : period - d);
}

inline float tileRamp(int i)
{
    const float d = std::min(i + 0.5f, TS - i - 0.5f);
    return std::min(1.f, d / (4.f * O));
}

}

#ifdef HAVE_ONNXRT

#ifdef _WIN32
// onnxruntime.dll is loaded by its full path: Windows has an older copy of its own in
// System32 that must not be picked up, and RawTherapee must start without the library.
std::string initRuntime(const wchar_t* libraryPath)
{
    static bool loaded = false;
    if (loaded) {
        return {};
    }
    HMODULE h = LoadLibraryExW(libraryPath, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {
        return "cannot load the ONNX Runtime library (Windows error " + std::to_string(static_cast<unsigned long>(GetLastError())) + ")";
    }
    typedef const OrtApiBase* (ORT_API_CALL * GetBaseFn)(void);
    GetBaseFn getBase = reinterpret_cast<GetBaseFn>(reinterpret_cast<void*>(GetProcAddress(h, "OrtGetApiBase")));
    if (!getBase) {
        return "the ONNX Runtime library is not valid";
    }
    const OrtApi* api = getBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        return "the ONNX Runtime library is too old";
    }
    Ort::InitApi(api);
    loaded = true;
    return {};
}
#else
std::string initRuntime(const char*)
{
    // not Windows: the program is linked against the library
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    return {};
}
#endif

#ifdef _WIN32
std::string denoiseBayer(const wchar_t* modelPath, float** rows, const BayerInfo& info, float strength, const std::function<void(double)>& progress)
#else
std::string denoiseBayer(const char* modelPath, float** rows, const BayerInfo& info, float strength, const std::function<void(double)>& progress)
#endif
{
    const int W = info.width;
    const int H = info.height;

    // position of the red sample inside the 2x2 cell; tiles are aligned to it so that
    // the network always sees an RGGB layout
    int y0 = -1, x0 = -1;
    for (int y = 0; y < 2 && y0 < 0; ++y) {
        for (int x = 0; x < 2; ++x) {
            if (info.cfa[y][x] == 0) {
                y0 = y;
                x0 = x;
                break;
            }
        }
    }
    if (y0 < 0 || info.cfa[y0 ^ 1][x0 ^ 1] != 2 || info.cfa[y0][x0 ^ 1] != 1 || info.cfa[y0 ^ 1][x0] != 1) {
        return "unsupported sensor layout (not a Bayer pattern)";
    }

    const int Hh = (H - y0) / 2;
    const int Wh = (W - x0) / 2;
    if (Hh < 8 || Wh < 8) {
        return "image too small";
    }
    const int ylo = y0, yhi = y0 + 2 * Hh;
    const int xlo = x0, xhi = x0 + 2 * Wh;

    float range[2][2];
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            range[y][x] = std::max(1.f, info.white - info.black[y][x]);
        }
    }

    strength = std::min(1.f, std::max(0.f, strength));

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "rawtherapee");
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, modelPath, so);
        Ort::AllocatorWithDefaultOptions alloc;
        auto inName = session.GetInputNameAllocated(0, alloc);
        auto outName = session.GetOutputNameAllocated(0, alloc);
        const char* inNames[] = { inName.get() };
        const char* outNames[] = { outName.get() };
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int64_t shape[4] = { 1, 4, T, T };
        const size_t inPlane = static_cast<size_t>(T) * T;
        const size_t outPlane = static_cast<size_t>(TS) * TS;

        std::vector<float> in(inPlane * 4);
        std::vector<float> acc(static_cast<size_t>(W) * H, 0.f);   // weighted sum of the network output
        std::vector<float> wy(H, 0.f), wx(W, 0.f);                // the weights are separable
        std::vector<float> ramp(TS);
        for (int i = 0; i < TS; ++i) {
            ramp[i] = tileRamp(i);
        }

        const int ny = (Hh + STEP - 1) / STEP;
        const int nx = (Wh + STEP - 1) / STEP;

        for (int ty = 0; ty < ny; ++ty) {
            const int sy = y0 + 2 * (ty * STEP - O);
            for (int i = 0; i < TS; ++i) {
                const int r = sy + i;
                if (r >= ylo && r < yhi) {
                    wy[r] += ramp[i];
                }
            }
        }
        for (int tx = 0; tx < nx; ++tx) {
            const int sx = x0 + 2 * (tx * STEP - O);
            for (int i = 0; i < TS; ++i) {
                const int c = sx + i;
                if (c >= xlo && c < xhi) {
                    wx[c] += ramp[i];
                }
            }
        }

        int done = 0;
        for (int ty = 0; ty < ny; ++ty) {
            for (int tx = 0; tx < nx; ++tx) {
                const int sy = y0 + 2 * (ty * STEP - O);
                const int sx = x0 + 2 * (tx * STEP - O);

                for (int i = 0; i < TS; ++i) {
                    const int r = mirrorIn(sy + i, ylo, yhi);
                    const float* src = rows[r];
                    for (int j = 0; j < TS; ++j) {
                        const int c = mirrorIn(sx + j, xlo, xhi);
                        const float v = (src[c] - info.black[r & 1][c & 1]) / range[r & 1][c & 1];
                        // plane order: (0,0) R, (0,1) G, (1,0) G, (1,1) B
                        in[static_cast<size_t>(((i & 1) << 1) | (j & 1)) * inPlane + static_cast<size_t>(i >> 1) * T + (j >> 1)] = v;
                    }
                }

                Ort::Value input = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape, 4);
                auto outs = session.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 1);
                const float* out = outs[0].GetTensorData<float>();

                for (int i = 0; i < TS; ++i) {
                    const int r = sy + i;
                    if (r < ylo || r >= yhi) {
                        continue;
                    }
                    float* dst = acc.data() + static_cast<size_t>(r) * W;
                    for (int j = 0; j < TS; ++j) {
                        const int c = sx + j;
                        if (c < xlo || c >= xhi) {
                            continue;
                        }
                        // the output is RGB at sensor resolution; take the colour this site has
                        const int ch = (i & 1) + (j & 1);   // 0 R, 1 G, 2 B for an RGGB tile
                        dst[c] += out[static_cast<size_t>(ch) * outPlane + static_cast<size_t>(i) * TS + j] * ramp[i] * ramp[j];
                    }
                }

                ++done;
                if (progress) {
                    progress(static_cast<double>(done) / (ny * nx));
                }
            }
        }

        // The network output has an arbitrary constant scale. Match it to the input over
        // all samples that are safely below clipping.
        double sumIn = 0.0, sumOut = 0.0;
        for (int r = ylo; r < yhi; ++r) {
            const float* src = rows[r];
            const float* a = acc.data() + static_cast<size_t>(r) * W;
            double rowIn = 0.0, rowOut = 0.0;
            for (int c = xlo; c < xhi; ++c) {
                const float n = (src[c] - info.black[r & 1][c & 1]) / range[r & 1][c & 1];
                if (n < 0.85f) {
                    rowIn += n;
                    rowOut += a[c] / (wy[r] * wx[c]);
                }
            }
            sumIn += rowIn;
            sumOut += rowOut;
        }
        if (!(std::fabs(sumOut) > 1e-12) || !std::isfinite(sumOut) || !std::isfinite(sumIn)) {
            return "the network returned no usable data";
        }
        const float gain = static_cast<float>(sumIn / sumOut);
        if (!(gain > 0.f) || !std::isfinite(gain)) {
            return "the network output could not be scaled";
        }

        for (int r = ylo; r < yhi; ++r) {
            float* src = rows[r];
            const float* a = acc.data() + static_cast<size_t>(r) * W;
            for (int c = xlo; c < xhi; ++c) {
                const float bl = info.black[r & 1][c & 1];
                const float rg = range[r & 1][c & 1];
                const float n = (src[c] - bl) / rg;
                float d = a[c] / (wy[r] * wx[c]) * gain;
                if (!std::isfinite(d)) {
                    d = n;
                }
                // keep clipped and nearly clipped samples as they are, so that highlight
                // reconstruction still sees the original clipping
                const float keep = std::min(1.f, std::max(0.f, (n - 0.85f) / 0.10f));
                d += (n - d) * keep;
                d = n + (d - n) * strength;
                src[c] = std::min(65535.f, std::max(0.f, d * rg + bl));
            }
        }
    } catch (const Ort::Exception& e) {
        return std::string("ONNX Runtime: ") + e.what();
    } catch (const std::bad_alloc&) {
        return "out of memory";
    } catch (const std::exception& e) {
        return e.what();
    }

    return {};
}

#else

#ifdef _WIN32
std::string initRuntime(const wchar_t*)
#else
std::string initRuntime(const char*)
#endif
{
    return "built without ONNX Runtime";
}

#ifdef _WIN32
std::string denoiseBayer(const wchar_t*, float**, const BayerInfo&, float, const std::function<void(double)>&)
#else
std::string denoiseBayer(const char*, float**, const BayerInfo&, float, const std::function<void(double)>&)
#endif
{
    return "built without ONNX Runtime";
}

#endif

// ---------------------------------------------------------------------------------------
// Minimal little-endian TIFF/DNG writer

namespace
{

struct Entry {
    uint16_t tag;
    uint16_t type;      // 1 BYTE, 2 ASCII, 3 SHORT, 4 LONG, 5 RATIONAL, 10 SRATIONAL
    uint32_t count;
    std::vector<uint8_t> data;
};

void put16(std::vector<uint8_t>& v, uint16_t x)
{
    v.push_back(x & 0xff);
    v.push_back(x >> 8);
}

void put32(std::vector<uint8_t>& v, uint32_t x)
{
    for (int i = 0; i < 4; ++i) {
        v.push_back((x >> (8 * i)) & 0xff);
    }
}

Entry eBytes(uint16_t tag, std::initializer_list<uint8_t> b)
{
    return { tag, 1, static_cast<uint32_t>(b.size()), std::vector<uint8_t>(b) };
}

Entry eAscii(uint16_t tag, const std::string& s)
{
    Entry e{ tag, 2, static_cast<uint32_t>(s.size() + 1), {} };
    e.data.assign(s.begin(), s.end());
    e.data.push_back(0);
    return e;
}

Entry eShorts(uint16_t tag, std::initializer_list<uint16_t> s)
{
    Entry e{ tag, 3, static_cast<uint32_t>(s.size()), {} };
    for (uint16_t x : s) {
        put16(e.data, x);
    }
    return e;
}

Entry eLongs(uint16_t tag, std::initializer_list<uint32_t> s)
{
    Entry e{ tag, 4, static_cast<uint32_t>(s.size()), {} };
    for (uint32_t x : s) {
        put32(e.data, x);
    }
    return e;
}

Entry eRationals(uint16_t tag, const double* v, int n, bool isSigned, uint32_t den)
{
    Entry e{ tag, static_cast<uint16_t>(isSigned ? 10 : 5), static_cast<uint32_t>(n), {} };
    for (int i = 0; i < n; ++i) {
        double x = v[i];
        if (!std::isfinite(x)) {
            x = 0.0;
        }
        if (isSigned) {
            put32(e.data, static_cast<uint32_t>(static_cast<int32_t>(std::lround(x * den))));
        } else {
            put32(e.data, static_cast<uint32_t>(std::llround(std::max(0.0, x) * den)));
        }
        put32(e.data, den);
    }
    return e;
}

// Serialises an IFD that will be placed at `offset`. Values longer than 4 bytes follow the
// directory. Returns the bytes; entries must be sorted by tag.
std::vector<uint8_t> buildIfd(const std::vector<Entry>& entries, uint32_t offset)
{
    std::vector<uint8_t> dir, extra;
    const uint32_t dirSize = 2 + 12 * static_cast<uint32_t>(entries.size()) + 4;
    put16(dir, static_cast<uint16_t>(entries.size()));
    for (const Entry& e : entries) {
        put16(dir, e.tag);
        put16(dir, e.type);
        put32(dir, e.count);
        if (e.data.size() <= 4) {
            for (size_t i = 0; i < 4; ++i) {
                dir.push_back(i < e.data.size() ? e.data[i] : 0);
            }
        } else {
            put32(dir, offset + dirSize + static_cast<uint32_t>(extra.size()));
            extra.insert(extra.end(), e.data.begin(), e.data.end());
            if (extra.size() & 1) {
                extra.push_back(0);
            }
        }
    }
    put32(dir, 0);   // no next IFD
    dir.insert(dir.end(), extra.begin(), extra.end());
    return dir;
}

}

bool writeDng(FILE* f, float** rows, const BayerInfo& info, const DngMeta& meta)
{
    if (!f || info.width <= 0 || info.height <= 0) {
        return false;
    }
    const uint32_t W = info.width, H = info.height;
    const uint64_t imageBytes64 = static_cast<uint64_t>(W) * H * 2;
    if (imageBytes64 > 0xF0000000ull) {
        return false;
    }
    const uint32_t imageBytes = static_cast<uint32_t>(imageBytes64);
    const uint32_t imageOffset = 8;
    uint32_t exifOffset = imageOffset + imageBytes;
    if (exifOffset & 1) {
        ++exifOffset;
    }

    // EXIF directory
    std::vector<Entry> exif;
    if (meta.shutter > 0.0) {
        const double v = meta.shutter;
        exif.push_back(eRationals(33434, &v, 1, false, 1000000));
    }
    if (meta.fnumber > 0.0) {
        const double v = meta.fnumber;
        exif.push_back(eRationals(33437, &v, 1, false, 100));
    }
    if (meta.iso > 0) {
        exif.push_back(eShorts(34855, { static_cast<uint16_t>(std::min(meta.iso, 65535)) }));
        if (meta.iso > 65535) {
            exif.push_back(eShorts(34864, { 3 }));   // SensitivityType: ISO speed
            exif.push_back(eLongs(34867, { static_cast<uint32_t>(meta.iso) }));
        }
    }
    exif.push_back(Entry{ 36864, 7, 4, { '0', '2', '3', '0' } });   // ExifVersion
    if (!meta.dateTime.empty()) {
        exif.push_back(eAscii(36867, meta.dateTime));
    }
    if (meta.focal > 0.0) {
        const double v = meta.focal;
        exif.push_back(eRationals(37386, &v, 1, false, 100));
    }
    if (!meta.lens.empty()) {
        exif.push_back(eAscii(42036, meta.lens));
    }
    std::sort(exif.begin(), exif.end(), [](const Entry& a, const Entry& b) { return a.tag < b.tag; });
    const std::vector<uint8_t> exifBytes = buildIfd(exif, exifOffset);

    uint32_t ifdOffset = exifOffset + static_cast<uint32_t>(exifBytes.size());
    if (ifdOffset & 1) {
        ++ifdOffset;
    }

    std::vector<Entry> ifd;
    ifd.push_back(eLongs(254, { 0 }));
    ifd.push_back(eLongs(256, { W }));
    ifd.push_back(eLongs(257, { H }));
    ifd.push_back(eShorts(258, { 16 }));
    ifd.push_back(eShorts(259, { 1 }));
    ifd.push_back(eShorts(262, { 32803 }));
    ifd.push_back(eAscii(271, meta.make));
    ifd.push_back(eAscii(272, meta.model));
    ifd.push_back(eLongs(273, { imageOffset }));
    ifd.push_back(eShorts(274, { static_cast<uint16_t>(meta.orientation) }));
    ifd.push_back(eShorts(277, { 1 }));
    ifd.push_back(eLongs(278, { H }));
    ifd.push_back(eLongs(279, { imageBytes }));
    ifd.push_back(eShorts(284, { 1 }));
    ifd.push_back(eAscii(305, "RawTherapee AI denoise"));
    if (!meta.dateTime.empty()) {
        ifd.push_back(eAscii(306, meta.dateTime));
    }
    ifd.push_back(eShorts(33421, { 2, 2 }));
    ifd.push_back(eBytes(33422, { static_cast<uint8_t>(info.cfa[0][0]), static_cast<uint8_t>(info.cfa[0][1]),
                                  static_cast<uint8_t>(info.cfa[1][0]), static_cast<uint8_t>(info.cfa[1][1]) }));
    ifd.push_back(eLongs(34665, { exifOffset }));
    ifd.push_back(eBytes(50706, { 1, 4, 0, 0 }));
    ifd.push_back(eBytes(50707, { 1, 1, 0, 0 }));
    ifd.push_back(eAscii(50708, meta.make.empty() ? meta.model : meta.make + " " + meta.model));
    ifd.push_back(eBytes(50710, { 0, 1, 2 }));
    ifd.push_back(eShorts(50711, { 1 }));
    ifd.push_back(eShorts(50713, { 2, 2 }));
    {
        const double b[4] = { info.black[0][0], info.black[0][1], info.black[1][0], info.black[1][1] };
        ifd.push_back(eRationals(50714, b, 4, false, 256));
    }
    ifd.push_back(eLongs(50717, { static_cast<uint32_t>(std::lround(std::min(65535.f, std::max(1.f, info.white)))) }));
    ifd.push_back(eRationals(50721, meta.colorMatrix, 9, true, 10000));
    ifd.push_back(eRationals(50728, meta.asShotNeutral, 3, false, 1000000));
    ifd.push_back(eShorts(50778, { 21 }));   // D65
    const std::vector<uint8_t> ifdBytes = buildIfd(ifd, ifdOffset);

    // header
    std::vector<uint8_t> head = { 'I', 'I', 42, 0 };
    put32(head, ifdOffset);
    if (fwrite(head.data(), 1, head.size(), f) != head.size()) {
        return false;
    }

    std::vector<uint8_t> line(static_cast<size_t>(W) * 2);
    for (uint32_t r = 0; r < H; ++r) {
        const float* src = rows[r];
        for (uint32_t c = 0; c < W; ++c) {
            const float v = src[c];
            const uint16_t u = static_cast<uint16_t>(std::lround(std::min(65535.f, std::max(0.f, std::isfinite(v) ? v : 0.f))));
            line[2 * c] = u & 0xff;
            line[2 * c + 1] = u >> 8;
        }
        if (fwrite(line.data(), 1, line.size(), f) != line.size()) {
            return false;
        }
    }

    auto padTo = [f](uint32_t current, uint32_t target) -> bool {
        for (; current < target; ++current) {
            if (fputc(0, f) == EOF) {
                return false;
            }
        }
        return true;
    };
    if (!padTo(imageOffset + imageBytes, exifOffset)) {
        return false;
    }
    if (fwrite(exifBytes.data(), 1, exifBytes.size(), f) != exifBytes.size()) {
        return false;
    }
    if (!padTo(exifOffset + static_cast<uint32_t>(exifBytes.size()), ifdOffset)) {
        return false;
    }
    if (fwrite(ifdBytes.data(), 1, ifdBytes.size(), f) != ifdBytes.size()) {
        return false;
    }
    return fflush(f) == 0;
}

}
