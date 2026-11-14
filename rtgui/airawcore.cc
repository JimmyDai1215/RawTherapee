#include "airawcore.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <map>
#include <stdexcept>
#include <memory>
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

namespace
{

std::string g_lastDevice = "CPU";
bool g_gpuAllowed = true;
double g_lastDeviation = -1.0;

// Mirror of OrtDmlApi from dml_provider_factory.h (only the layout matters here).
struct DmlDeviceOptions {
    int preference;   // 1 = high performance
    int filter;       // 1 = GPU
};
struct DmlApi {
    void* appendDml;
    void* appendDml1;
    void* createGpuAllocation;
    void* freeGpuAllocation;
    void* getD3D12Resource;
    OrtStatus* (ORT_API_CALL* appendDml2)(OrtSessionOptions* options, DmlDeviceOptions* deviceOptions);
};

// 0 = not tested yet, 1 = the GPU gives correct results, 2 = do not use the GPU
std::map<std::string, int>& gpuState()
{
    static std::map<std::string, int> state;
    return state;
}

std::map<std::string, double>& gpuDeviation()
{
    static std::map<std::string, double> dev;
    return dev;
}

// One network with a fixed NCHW float input. Uses the graphics card through DirectML when
// that works and gives the same numbers as the CPU, otherwise the CPU.
class Net
{
public:
#ifdef _WIN32
    typedef std::wstring PathString;
#else
    typedef std::string PathString;
#endif

    Net(const PathString& path, int channels, int size, const std::string& key) :
        env_(ORT_LOGGING_LEVEL_ERROR, "rawtherapee"),
        path_(path),
        key_(key),
        mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        shape_[0] = 1;
        shape_[1] = channels;
        shape_[2] = size;
        shape_[3] = size;

        const char* dev = std::getenv("RT_AI_DEVICE");
        const bool cpuOnly = dev && std::string(dev) == "cpu";
        if (!cpuOnly && g_gpuAllowed && gpuState()[key_] != 2) {
            try {
                session_ = create(true);
                gpu_ = true;
            } catch (...) {
                session_.reset();
            }
        }
        if (!session_) {
            session_ = create(false);
            gpu_ = false;
        }
        readNames();
        g_lastDevice = gpu_ ? "GPU" : "CPU";
        g_lastDeviation = gpuDeviation().count(key_) ? gpuDeviation()[key_] : -1.0;
    }

    // Outputs of the last run (valid until the next call).
    Ort::Value& output(size_t i)
    {
        return outs_[i];
    }

    Ort::Env& env()
    {
        return env_;
    }

    // Runs the network. The returned data stays valid until the next call.
    const float* run(std::vector<float>& in)
    {
        if (gpu_) {
            bool ok = true;
            try {
                outs_ = runOn(*session_, in);
            } catch (const Ort::Exception&) {
                ok = false;
            }
            if (ok && gpuState()[key_] == 0) {
                // first use of this network on the GPU: compare with the CPU once
                std::unique_ptr<Ort::Session> cpu = create(false);
                std::vector<Ort::Value> ref = runOn(*cpu, in);
                const double dev = deviation(outs_[0], ref[0]);
                gpuDeviation()[key_] = dev;
                g_lastDeviation = dev;
                if (dev >= 0.0 && dev <= 1.0) {
                    gpuState()[key_] = 1;
                } else {
                    ok = false;
                    session_ = std::move(cpu);
                    outs_ = std::move(ref);
                    gpuState()[key_] = 2;
                    gpu_ = false;
                    g_lastDevice = "CPU";
                    return outs_[0].GetTensorData<float>();
                }
            }
            if (!ok) {
                gpuState()[key_] = 2;
                gpu_ = false;
                g_lastDevice = "CPU";
                session_ = create(false);
                outs_ = runOn(*session_, in);
            }
        } else {
            outs_ = runOn(*session_, in);
        }
        return outs_[0].GetTensorData<float>();
    }

private:
    std::unique_ptr<Ort::Session> create(bool gpu)
    {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (gpu) {
            so.DisableMemPattern();
            so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            const void* api = nullptr;
            if (OrtStatus* st = Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, &api)) {
                Ort::GetApi().ReleaseStatus(st);
                throw std::runtime_error("DirectML is not available");
            }
            const DmlApi* dml = static_cast<const DmlApi*>(api);
            if (!dml || !dml->appendDml2) {
                throw std::runtime_error("DirectML is not available");
            }
            DmlDeviceOptions opts = { 1, 1 };
            if (OrtStatus* st = dml->appendDml2(so, &opts)) {
                Ort::GetApi().ReleaseStatus(st);
                throw std::runtime_error("no usable graphics card");
            }
        } else {
            so.SetIntraOpNumThreads(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
        }
        return std::unique_ptr<Ort::Session>(new Ort::Session(env_, path_.c_str(), so));
    }

    void readNames()
    {
        Ort::AllocatorWithDefaultOptions alloc;
        inName_ = session_->GetInputNameAllocated(0, alloc).get();
        outNames_.clear();
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            outNames_.push_back(session_->GetOutputNameAllocated(i, alloc).get());
        }
    }

    std::vector<Ort::Value> runOn(Ort::Session& session, std::vector<float>& in)
    {
        const char* inNames[] = { inName_.c_str() };
        std::vector<const char*> outNames;
        for (const std::string& n : outNames_) {
            outNames.push_back(n.c_str());
        }
        Ort::Value input = Ort::Value::CreateTensor<float>(mem_, in.data(), in.size(), shape_, 4);
        return session.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames.data(), outNames.size());
    }

    // Mean absolute difference relative to the mean magnitude, in percent; negative if unusable.
    static double deviation(Ort::Value& a, Ort::Value& b)
    {
        const size_t n = a.GetTensorTypeAndShapeInfo().GetElementCount();
        if (n == 0 || n != b.GetTensorTypeAndShapeInfo().GetElementCount()) {
            return -1.0;
        }
        const float* x = a.GetTensorData<float>();
        const float* y = b.GetTensorData<float>();
        double diff = 0.0, ref = 0.0;
        for (size_t i = 0; i < n; ++i) {
            diff += std::fabs(static_cast<double>(x[i]) - y[i]);
            ref += std::fabs(static_cast<double>(y[i]));
        }
        if (!std::isfinite(diff) || !std::isfinite(ref) || !(ref > 0.0)) {
            return -1.0;
        }
        return 100.0 * diff / ref;
    }

    Ort::Env env_;
    PathString path_;
    std::string key_;
    Ort::MemoryInfo mem_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<Ort::Value> outs_;
    std::string inName_;
    std::vector<std::string> outNames_;
    int64_t shape_[4];
    bool gpu_ = false;
};

}

std::string lastDevice()
{
    return g_lastDevice;
}

void setGpuAllowed(bool allowed)
{
    g_gpuAllowed = allowed;
}

double lastGpuDeviation()
{
    return g_lastDeviation;
}

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
        Net net(modelPath, 4, T, "rawdenoise-bayer");
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

                const float* out = net.run(in);

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

std::string lastDevice()
{
    return "CPU";
}

void setGpuAllowed(bool)
{
}

double lastGpuDeviation()
{
    return -1.0;
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

// ---------------------------------------------------------------------------------------
// 2x upscaling of a developed RGB image

#ifdef HAVE_ONNXRT

namespace
{
constexpr int UT = 512;            // model input size
constexpr int UPAD = 32;           // context around the part of a tile that is used
constexpr int UFADE = 8;           // half width of the cross-fade between tiles (input pixels)
constexpr int USTEP = UT - 2 * UPAD;
constexpr int UTO = 2 * UT;        // tile output size

inline int mirror0(int i, int n)
{
    if (n <= 1) {
        return 0;
    }
    const int period = 2 * (n - 1);
    i %= period;
    if (i < 0) {
        i += period;
    }
    return i < n ? i : period - i;
}

// weight of output position q of a tile along one axis
inline float upWeight(int q, bool first, bool last)
{
    const float q0 = 2.f * (UPAD - UFADE);
    const float span = 4.f * UFADE;
    float w = 1.f;
    if (!first) {
        w = std::min(w, std::max(0.f, (q + 0.5f - q0) / span));
    }
    if (!last) {
        w = std::min(w, std::max(0.f, (UTO - q - 0.5f - q0) / span));
    }
    return w;
}
}

#ifdef _WIN32
std::string upscale2x(const wchar_t* modelPath, const float* const* r, const float* const* g, const float* const* b,
                      int width, int height, FILE* out, const std::string& icc, const std::string& modelKey, const std::function<void(double)>& progress)
#else
std::string upscale2x(const char* modelPath, const float* const* r, const float* const* g, const float* const* b,
                      int width, int height, FILE* out, const std::string& icc, const std::string& modelKey, const std::function<void(double)>& progress)
#endif
{
    const int W = width, H = height;
    if (!out || W < 16 || H < 16) {
        return "image too small";
    }
    const uint64_t OW = 2ull * W, OH = 2ull * H;
    const uint64_t imageBytes64 = OW * OH * 6ull;
    if (imageBytes64 > 0xE0000000ull) {
        return "the upscaled image would be too large for a TIFF file";
    }

    try {
        // ---- TIFF header and directory, image data follows
        std::vector<Entry> ifd;
        ifd.push_back(eLongs(256, { static_cast<uint32_t>(OW) }));
        ifd.push_back(eLongs(257, { static_cast<uint32_t>(OH) }));
        ifd.push_back(eShorts(258, { 16, 16, 16 }));
        ifd.push_back(eShorts(259, { 1 }));
        ifd.push_back(eShorts(262, { 2 }));
        ifd.push_back(eLongs(273, { 0 }));   // patched below
        ifd.push_back(eShorts(277, { 3 }));
        ifd.push_back(eLongs(278, { static_cast<uint32_t>(OH) }));
        ifd.push_back(eLongs(279, { static_cast<uint32_t>(imageBytes64) }));
        ifd.push_back(eShorts(284, { 1 }));
        ifd.push_back(eAscii(305, "RawTherapee AI upscale"));
        if (!icc.empty()) {
            Entry e{ 34675, 7, static_cast<uint32_t>(icc.size()), {} };
            e.data.assign(icc.begin(), icc.end());
            ifd.push_back(e);
        }
        std::vector<uint8_t> ifdBytes = buildIfd(ifd, 8);
        uint32_t imageOffset = 8 + static_cast<uint32_t>(ifdBytes.size());
        if (imageOffset & 1) {
            ++imageOffset;
        }
        for (Entry& e : ifd) {
            if (e.tag == 273) {
                e = eLongs(273, { imageOffset });
            }
        }
        ifdBytes = buildIfd(ifd, 8);
        std::vector<uint8_t> head = { 'I', 'I', 42, 0 };
        put32(head, 8);
        head.insert(head.end(), ifdBytes.begin(), ifdBytes.end());
        head.resize(imageOffset, 0);
        if (fwrite(head.data(), 1, head.size(), out) != head.size()) {
            return "cannot write the file";
        }

        // ---- network
        Net net(modelPath, 3, UT, modelKey);
        const size_t inPlane = static_cast<size_t>(UT) * UT;
        const size_t outPlane = static_cast<size_t>(UTO) * UTO;
        std::vector<float> in(inPlane * 3);

        const int ny = (H + USTEP - 1) / USTEP;
        const int nx = (W + USTEP - 1) / USTEP;

        // rows of the result that are still being blended; a band is at most UTO rows high
        const size_t rowLen = static_cast<size_t>(OW) * 3;
        std::vector<float> band(rowLen * UTO, 0.f);
        std::vector<uint8_t> line(rowLen * 2);
        int64_t bandStart = 0;   // output row stored in band row 0

        std::vector<float> wxs(UTO), wys(UTO);
        int done = 0;

        for (int ty = 0; ty < ny; ++ty) {
            const int iy0 = ty * USTEP - UPAD;
            for (int q = 0; q < UTO; ++q) {
                wys[q] = upWeight(q, ty == 0, ty == ny - 1);
            }
            for (int tx = 0; tx < nx; ++tx) {
                const int ix0 = tx * USTEP - UPAD;
                for (int q = 0; q < UTO; ++q) {
                    wxs[q] = upWeight(q, tx == 0, tx == nx - 1);
                }

                for (int i = 0; i < UT; ++i) {
                    const int sy = mirror0(iy0 + i, H);
                    const float* rr = r[sy];
                    const float* gg = g[sy];
                    const float* bb = b[sy];
                    for (int j = 0; j < UT; ++j) {
                        const int sx = mirror0(ix0 + j, W);
                        const size_t o = static_cast<size_t>(i) * UT + j;
                        in[o] = std::min(1.f, std::max(0.f, rr[sx] / 65535.f));
                        in[inPlane + o] = std::min(1.f, std::max(0.f, gg[sx] / 65535.f));
                        in[2 * inPlane + o] = std::min(1.f, std::max(0.f, bb[sx] / 65535.f));
                    }
                }

                const float* o = net.run(in);

                for (int qy = 0; qy < UTO; ++qy) {
                    const float wy = wys[qy];
                    const int64_t oy = 2ll * iy0 + qy;
                    if (wy <= 0.f || oy < 0 || oy >= static_cast<int64_t>(OH)) {
                        continue;
                    }
                    const int64_t br = oy - bandStart;
                    if (br < 0 || br >= UTO) {
                        return "internal error: band overflow";
                    }
                    float* dst = band.data() + static_cast<size_t>(br) * rowLen;
                    const size_t orow = static_cast<size_t>(qy) * UTO;
                    for (int qx = 0; qx < UTO; ++qx) {
                        const float w = wy * wxs[qx];
                        const int64_t ox = 2ll * ix0 + qx;
                        if (w <= 0.f || ox < 0 || ox >= static_cast<int64_t>(OW)) {
                            continue;
                        }
                        float* d = dst + static_cast<size_t>(ox) * 3;
                        d[0] += o[orow + qx] * w;
                        d[1] += o[outPlane + orow + qx] * w;
                        d[2] += o[2 * outPlane + orow + qx] * w;
                    }
                }

                ++done;
                if (progress) {
                    progress(static_cast<double>(done) / (static_cast<double>(ny) * nx));
                }
            }

            // rows that the next row of tiles no longer touches are final
            const int64_t flushEnd = ty == ny - 1
                ? static_cast<int64_t>(OH)
                : std::min<int64_t>(OH, 2ll * ((ty + 1) * USTEP - UPAD) + 2 * (UPAD - UFADE));
            const int64_t nrows = flushEnd - bandStart;
            if (nrows < 0 || nrows > UTO) {
                return "internal error: band range";
            }
            for (int64_t rIdx = 0; rIdx < nrows; ++rIdx) {
                const float* src = band.data() + static_cast<size_t>(rIdx) * rowLen;
                for (size_t k = 0; k < rowLen; ++k) {
                    const float v = src[k];
                    const uint16_t u = static_cast<uint16_t>(std::lround(std::min(1.f, std::max(0.f, std::isfinite(v) ? v : 0.f)) * 65535.f));
                    line[2 * k] = u & 0xff;
                    line[2 * k + 1] = u >> 8;
                }
                if (fwrite(line.data(), 1, line.size(), out) != line.size()) {
                    return "cannot write the file (disk full?)";
                }
            }
            // keep the rows that are still being blended
            const size_t keep = static_cast<size_t>(UTO - nrows) * rowLen;
            if (nrows > 0) {
                std::copy(band.begin() + static_cast<size_t>(nrows) * rowLen, band.end(), band.begin());
                std::fill(band.begin() + keep, band.end(), 0.f);
            }
            bandStart = flushEnd;
        }

        if (fflush(out) != 0) {
            return "cannot write the file (disk full?)";
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
std::string upscale2x(const wchar_t*, const float* const*, const float* const*, const float* const*, int, int, FILE*, const std::string&, const std::string&, const std::function<void(double)>&)
#else
std::string upscale2x(const char*, const float* const*, const float* const*, const float* const*, int, int, FILE*, const std::string&, const std::string&, const std::function<void(double)>&)
#endif
{
    return "built without ONNX Runtime";
}

#endif


// ---------------------------------------------------------------------------------------
// Object selection with Segment Anything 2.1

namespace
{

void boxBlur(std::vector<float>& a, int w, int h, int r)
{
    if (r < 1) {
        return;
    }
    std::vector<float> tmp(a.size());
    // horizontal
    for (int y = 0; y < h; ++y) {
        const float* src = a.data() + static_cast<size_t>(y) * w;
        float* dst = tmp.data() + static_cast<size_t>(y) * w;
        double sum = 0.0;
        int count = 0;
        for (int x = 0; x <= std::min(r, w - 1); ++x) {
            sum += src[x];
            ++count;
        }
        for (int x = 0; x < w; ++x) {
            dst[x] = static_cast<float>(sum / count);
            const int add = x + r + 1;
            const int rem = x - r;
            if (add < w) {
                sum += src[add];
                ++count;
            }
            if (rem >= 0) {
                sum -= src[rem];
                --count;
            }
        }
    }
    // vertical
    std::vector<double> sum(w);
    std::vector<int> count(w);
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(count.begin(), count.end(), 0);
    for (int y = 0; y <= std::min(r, h - 1); ++y) {
        const float* src = tmp.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            sum[x] += src[x];
            ++count[x];
        }
    }
    for (int y = 0; y < h; ++y) {
        float* dst = a.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            dst[x] = static_cast<float>(sum[x] / count[x]);
        }
        const int add = y + r + 1;
        const int rem = y - r;
        if (add < h) {
            const float* src = tmp.data() + static_cast<size_t>(add) * w;
            for (int x = 0; x < w; ++x) {
                sum[x] += src[x];
                ++count[x];
            }
        }
        if (rem >= 0) {
            const float* src = tmp.data() + static_cast<size_t>(rem) * w;
            for (int x = 0; x < w; ++x) {
                sum[x] -= src[x];
                --count[x];
            }
        }
    }
}

}

// Edge-preserving smoothing of `values` guided by the image (guided filter).
static void guidedFilter(const unsigned char* rgb, int w, int h, std::vector<float>& values, int radius, float eps, bool clamp01)
{
    const size_t n = static_cast<size_t>(w) * h;
    if (values.size() != n || radius < 1) {
        return;
    }
    std::vector<float> I(n), mI, mp, mIp(n), mII(n);
    for (size_t i = 0; i < n; ++i) {
        I[i] = (0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2]) * (1.f / 255.f);
        mIp[i] = I[i] * values[i];
        mII[i] = I[i] * I[i];
    }
    mI = I;
    mp = values;
    boxBlur(mI, w, h, radius);
    boxBlur(mp, w, h, radius);
    boxBlur(mIp, w, h, radius);
    boxBlur(mII, w, h, radius);
    for (size_t i = 0; i < n; ++i) {
        const float cov = mIp[i] - mI[i] * mp[i];
        const float var = mII[i] - mI[i] * mI[i];
        const float a = cov / (var + eps);
        mIp[i] = a;
        mII[i] = mp[i] - a * mI[i];
    }
    boxBlur(mIp, w, h, radius);
    boxBlur(mII, w, h, radius);
    for (size_t i = 0; i < n; ++i) {
        const float v = mIp[i] * I[i] + mII[i];
        values[i] = clamp01 ? std::min(1.f, std::max(0.f, v)) : v;
    }
}

namespace
{

// Labels the 4-connected components of the pixels equal to `value`. Returns the sizes
// (index 0 unused); labels receives the component number of each such pixel, 0 elsewhere.
std::vector<size_t> labelComponents(const std::vector<unsigned char>& bin, int w, int h, unsigned char value, std::vector<int>& labels)
{
    labels.assign(bin.size(), 0);
    std::vector<size_t> sizes(1, 0);
    std::vector<int> stack;
    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            const size_t start = static_cast<size_t>(sy) * w + sx;
            if (bin[start] != value || labels[start]) {
                continue;
            }
            const int id = static_cast<int>(sizes.size());
            size_t count = 0;
            stack.clear();
            stack.push_back(static_cast<int>(start));
            labels[start] = id;
            while (!stack.empty()) {
                const int p = stack.back();
                stack.pop_back();
                ++count;
                const int x = p % w;
                const int y = p / w;
                if (x > 0 && bin[p - 1] == value && !labels[p - 1]) {
                    labels[p - 1] = id;
                    stack.push_back(p - 1);
                }
                if (x < w - 1 && bin[p + 1] == value && !labels[p + 1]) {
                    labels[p + 1] = id;
                    stack.push_back(p + 1);
                }
                if (y > 0 && bin[p - w] == value && !labels[p - w]) {
                    labels[p - w] = id;
                    stack.push_back(p - w);
                }
                if (y < h - 1 && bin[p + w] == value && !labels[p + w]) {
                    labels[p + w] = id;
                    stack.push_back(p + w);
                }
            }
            sizes.push_back(count);
        }
    }
    return sizes;
}

struct PointPx {
    int x, y;
    bool add;
};

// Removes specks (small separate pieces) and fills pinholes of a binary selection.
// Pieces under an "add" point are always kept; holes under a "subtract" point stay open.
void cleanupSelection(std::vector<unsigned char>& bin, int w, int h, const std::vector<PointPx>& points)
{
    std::vector<int> labels;
    std::vector<size_t> sizes = labelComponents(bin, w, h, 1, labels);
    if (sizes.size() <= 1) {
        return;
    }
    size_t biggest = 0;
    for (size_t i = 1; i < sizes.size(); ++i) {
        biggest = std::max(biggest, sizes[i]);
    }
    std::vector<char> protect(sizes.size(), 0);
    for (const PointPx& p : points) {
        if (p.add) {
            const int x = std::min(w - 1, std::max(0, p.x));
            const int y = std::min(h - 1, std::max(0, p.y));
            protect[labels[static_cast<size_t>(y) * w + x]] = 1;
        }
    }
    size_t total = 0;
    for (size_t i = 0; i < bin.size(); ++i) {
        const int id = labels[i];
        if (id && !protect[id] && sizes[id] < biggest / 50) {
            bin[i] = 0;
        }
        total += bin[i];
    }
    sizes = labelComponents(bin, w, h, 0, labels);
    protect.assign(sizes.size(), 0);
    for (const PointPx& p : points) {
        if (!p.add) {
            const int x = std::min(w - 1, std::max(0, p.x));
            const int y = std::min(h - 1, std::max(0, p.y));
            protect[labels[static_cast<size_t>(y) * w + x]] = 1;
        }
    }
    for (size_t i = 0; i < bin.size(); ++i) {
        const int id = labels[i];
        if (id && !protect[id] && sizes[id] < total / 500) {
            bin[i] = 1;
        }
    }
}

// The pixel of `value` that is farthest from any other pixel (city-block distance).
// Returns false if there is no such pixel.
bool farthestInside(const unsigned char* bin, int stride, int w, int h, unsigned char value, int& bx, int& by)
{
    std::vector<int> d(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            if (bin[static_cast<size_t>(y) * stride + x] != value) {
                d[i] = 0;
            } else {
                // the tile border counts as "outside" so that points stay away from it
                int v = std::min(std::min(x, y), std::min(w - 1 - x, h - 1 - y)) + 1;
                if (x > 0) {
                    v = std::min(v, d[i - 1] + 1);
                }
                if (y > 0) {
                    v = std::min(v, d[i - w] + 1);
                }
                d[i] = v;
            }
        }
    }
    int best = 0;
    bx = by = -1;
    for (int y = h - 1; y >= 0; --y) {
        for (int x = w - 1; x >= 0; --x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            if (d[i]) {
                if (x < w - 1) {
                    d[i] = std::min(d[i], d[i + 1] + 1);
                }
                if (y < h - 1) {
                    d[i] = std::min(d[i], d[i + w] + 1);
                }
                if (d[i] > best) {
                    best = d[i];
                    bx = x;
                    by = y;
                }
            }
        }
    }
    return best > 0;
}

inline float bilinear(const std::vector<float>& p, int w, int h, float x, float y)
{
    x = std::min(w - 1.001f, std::max(0.f, x));
    y = std::min(h - 1.001f, std::max(0.f, y));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const float fx = x - x0;
    const float fy = y - y0;
    const float* q = p.data() + static_cast<size_t>(y0) * w + x0;
    const float top = q[0] + (q[1] - q[0]) * fx;
    const float bottom = q[w] + (q[w + 1] - q[w]) * fx;
    return top + (bottom - top) * fy;
}

// Image -> image averaged down by an integer factor
void shrinkRgb(const std::vector<unsigned char>& src, int w, int h, int f, std::vector<unsigned char>& dst, int& dw, int& dh)
{
    dw = std::max(1, w / f);
    dh = std::max(1, h / f);
    dst.resize(static_cast<size_t>(dw) * dh * 3);
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            unsigned sum[3] = { 0, 0, 0 };
            for (int j = 0; j < f; ++j) {
                const unsigned char* p = src.data() + (static_cast<size_t>(y * f + j) * w + static_cast<size_t>(x) * f) * 3;
                for (int i = 0; i < f; ++i, p += 3) {
                    sum[0] += p[0];
                    sum[1] += p[1];
                    sum[2] += p[2];
                }
            }
            unsigned char* d = dst.data() + (static_cast<size_t>(y) * dw + x) * 3;
            d[0] = static_cast<unsigned char>(sum[0] / (f * f));
            d[1] = static_cast<unsigned char>(sum[1] / (f * f));
            d[2] = static_cast<unsigned char>(sum[2] / (f * f));
        }
    }
}

// ---- hand-made mask parts, drawn into a float buffer (0..1) of size w x h.
// An image point (nx, ny in 0..1) lies at pixel (nx * sx + ox, ny * sy + oy).

struct Xform {
    double sx, sy, ox, oy;
};

void drawBrush(std::vector<float>& m, int w, int h, const MaskOp& op, const Xform& t)
{
    const size_t n = op.xy.size() / 2;
    if (!n) {
        return;
    }
    const double R = std::max(0.5, op.radius * t.sx);
    for (size_t i = 0; i < n; ++i) {
        const double ax = op.xy[2 * i] * t.sx + t.ox, ay = op.xy[2 * i + 1] * t.sy + t.oy;
        const size_t j = i + 1 < n ? i + 1 : i;
        const double bx = op.xy[2 * j] * t.sx + t.ox, by = op.xy[2 * j + 1] * t.sy + t.oy;
        const int x0 = std::max(0, static_cast<int>(std::floor(std::min(ax, bx) - R - 1)));
        const int x1 = std::min(w - 1, static_cast<int>(std::ceil(std::max(ax, bx) + R + 1)));
        const int y0 = std::max(0, static_cast<int>(std::floor(std::min(ay, by) - R - 1)));
        const int y1 = std::min(h - 1, static_cast<int>(std::ceil(std::max(ay, by) + R + 1)));
        const double dx = bx - ax, dy = by - ay;
        const double len2 = dx * dx + dy * dy;
        for (int y = y0; y <= y1; ++y) {
            float* row = m.data() + static_cast<size_t>(y) * w;
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5 - ax, py = y + 0.5 - ay;
                double u = len2 > 0.0 ? (px * dx + py * dy) / len2 : 0.0;
                u = std::min(1.0, std::max(0.0, u));
                const double ex = px - u * dx, ey = py - u * dy;
                const float cov = static_cast<float>(std::min(1.0, std::max(0.0, R + 0.5 - std::sqrt(ex * ex + ey * ey))));
                if (cov > 0.f) {
                    row[x] = op.add ? std::max(row[x], cov) : std::min(row[x], 1.f - cov);
                }
            }
        }
    }
}

void drawPolygon(std::vector<float>& m, int w, int h, const MaskOp& op, const Xform& t)
{
    const size_t n = op.xy.size() / 2;
    if (n < 3) {
        return;
    }
    std::vector<double> px(n), py(n);
    double ymin = 1e30, ymax = -1e30;
    for (size_t i = 0; i < n; ++i) {
        px[i] = op.xy[2 * i] * t.sx + t.ox;
        py[i] = op.xy[2 * i + 1] * t.sy + t.oy;
        ymin = std::min(ymin, py[i]);
        ymax = std::max(ymax, py[i]);
    }
    const int y0 = std::max(0, static_cast<int>(std::floor(ymin)));
    const int y1 = std::min(h - 1, static_cast<int>(std::ceil(ymax)));
    std::vector<double> xs;
    for (int y = y0; y <= y1; ++y) {
        const double yc = y + 0.5;
        xs.clear();
        for (size_t i = 0; i < n; ++i) {
            const size_t j = (i + 1) % n;
            if ((py[i] <= yc && py[j] > yc) || (py[j] <= yc && py[i] > yc)) {
                xs.push_back(px[i] + (yc - py[i]) * (px[j] - px[i]) / (py[j] - py[i]));
            }
        }
        std::sort(xs.begin(), xs.end());
        float* row = m.data() + static_cast<size_t>(y) * w;
        for (size_t k = 0; k + 1 < xs.size(); k += 2) {
            const int a = std::max(0, static_cast<int>(std::ceil(xs[k] - 0.5)));
            const int b = std::min(w - 1, static_cast<int>(std::floor(xs[k + 1] - 0.5)));
            for (int x = a; x <= b; ++x) {
                row[x] = op.add ? 1.f : 0.f;
            }
        }
    }
}

void drawGradient(std::vector<float>& m, int w, int h, const MaskOp& op, const Xform& t)
{
    if (op.xy.size() < 4) {
        return;
    }
    const double ax = op.xy[0] * t.sx + t.ox, ay = op.xy[1] * t.sy + t.oy;
    const double bx = op.xy[2] * t.sx + t.ox, by = op.xy[3] * t.sy + t.oy;
    const double dx = bx - ax, dy = by - ay;
    const double len2 = dx * dx + dy * dy;
    if (len2 < 1.0) {
        return;
    }
    for (int y = 0; y < h; ++y) {
        float* row = m.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            double u = ((x + 0.5 - ax) * dx + (y + 0.5 - ay) * dy) / len2;
            u = std::min(1.0, std::max(0.0, u));
            const float ramp = static_cast<float>(1.0 - u * u * (3.0 - 2.0 * u));   // full at the start, none at the end
            row[x] = op.add ? std::max(row[x], ramp) : row[x] * (1.f - ramp);
        }
    }
}

void drawOps(std::vector<float>& m, int w, int h, const std::vector<MaskOp>& ops, const Xform& t)
{
    for (const MaskOp& op : ops) {
        if (op.type == 0) {
            drawBrush(m, w, h, op, t);
        } else if (op.type == 1) {
            drawPolygon(m, w, h, op, t);
        } else if (op.type == 2) {
            drawGradient(m, w, h, op, t);
        }
    }
}

// Running maximum (grow = true) or minimum over a window of radius r along one line.
void slideLine(const float* src, float* dst, int n, int stride, int r, bool grow, std::vector<int>& dq)
{
    dq.resize(n);
    int head = 0, tail = 0;   // indices into src, kept so that their values are monotonic
    int next = 0;
    for (int i = 0; i < n; ++i) {
        const int hi = std::min(n - 1, i + r);
        while (next <= hi) {
            const float v = src[static_cast<size_t>(next) * stride];
            while (tail > head && (grow ? src[static_cast<size_t>(dq[tail - 1]) * stride] <= v : src[static_cast<size_t>(dq[tail - 1]) * stride] >= v)) {
                --tail;
            }
            dq[tail++] = next++;
        }
        while (dq[head] < i - r) {
            ++head;
        }
        dst[static_cast<size_t>(i) * stride] = src[static_cast<size_t>(dq[head]) * stride];
    }
}

// Grows (r > 0) or shrinks (r < 0) a soft mask by |r| pixels (square structuring element,
// two passes for a rounder shape).
void growMask(std::vector<float>& m, int w, int h, int r)
{
    if (r == 0) {
        return;
    }
    const bool grow = r > 0;
    const int rad = std::abs(r);
    std::vector<float> tmp(m.size());
    std::vector<int> dq;
    for (int y = 0; y < h; ++y) {
        slideLine(m.data() + static_cast<size_t>(y) * w, tmp.data() + static_cast<size_t>(y) * w, w, 1, rad, grow, dq);
    }
    for (int x = 0; x < w; ++x) {
        slideLine(tmp.data() + x, m.data() + x, h, w, rad, grow, dq);
    }
}

// Ops, grow/shrink, feather and invert, in that order. px = pixels per unit of image width.
// rgb: the photo on the same grid as the mask (8 bit, 3 per pixel), for the brightness range.
void finishTouches(std::vector<float>& m, int w, int h, const MaskEdit& edit, const Xform& t, const unsigned char* rgb)
{
    drawOps(m, w, h, edit.ops, t);
    growMask(m, w, h, static_cast<int>(std::lround(edit.grow * t.sx)));
    const int f = static_cast<int>(std::lround(edit.feather * t.sx));
    if (f > 0) {
        boxBlur(m, w, h, f);
        boxBlur(m, w, h, f);
    }
    if (edit.invert) {
        for (float& v : m) {
            v = 1.f - v;
        }
    }
    if (rgb && (edit.lumaLo > 0.0 || edit.lumaHi < 1.0)) {
        // soft limits, so that the mask does not get a hard, noisy border
        const float lo = static_cast<float>(edit.lumaLo), hi = static_cast<float>(edit.lumaHi), soft = 0.06f;
        float lut[256];
        for (int i = 0; i < 256; ++i) {
            const float l = i / 255.f;
            float f = 1.f;
            if (lo > 0.f) {
                f = std::min(f, std::min(1.f, std::max(0.f, (l - lo) / soft + 0.5f)));
            }
            if (hi < 1.f) {
                f = std::min(f, std::min(1.f, std::max(0.f, (hi - l) / soft + 0.5f)));
            }
            lut[i] = f;
        }
        const size_t n = static_cast<size_t>(w) * h;
        for (size_t i = 0; i < n; ++i) {
            const unsigned char* p = rgb + i * 3;
            m[i] *= lut[(p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8];
        }
    }
}

void putNumber(std::string& s, double v)
{
    char buf[48];
    snprintf(buf, sizeof(buf), "%.6f", v);
    for (char* c = buf; *c; ++c) {
        if (*c == ',') {
            *c = '.';   // some locales print a decimal comma
        }
    }
    s += buf;
    s += ' ';
}

// Locale-independent number reader. Returns false at the end of the line or text.
bool getNumber(const char*& p, double& v)
{
    while (*p == ' ' || *p == '\t') {
        ++p;
    }
    const char* start = p;
    double sign = 1.0, val = 0.0, frac = 0.1;
    if (*p == '-') {
        sign = -1.0;
        ++p;
    }
    bool digits = false;
    while (*p >= '0' && *p <= '9') {
        val = val * 10.0 + (*p++ - '0');
        digits = true;
    }
    if (*p == '.') {
        ++p;
        while (*p >= '0' && *p <= '9') {
            val += (*p++ - '0') * frac;
            frac *= 0.1;
            digits = true;
        }
    }
    if (!digits) {
        p = start;
        return false;
    }
    v = sign * val;
    return true;
}

}

std::string editToText(const MaskEdit& edit)
{
    std::string s;
    for (const SelectPoint& p : edit.points) {
        s += "P ";
        putNumber(s, p.x);
        putNumber(s, p.y);
        s += p.add ? "1\n" : "0\n";
    }
    for (const MaskOp& op : edit.ops) {
        s += "O ";
        putNumber(s, op.type);
        putNumber(s, op.add ? 1 : 0);
        putNumber(s, op.radius);
        for (double v : op.xy) {
            putNumber(s, v);
        }
        s += "\n";
    }
    s += "F ";
    putNumber(s, edit.feather);
    putNumber(s, edit.grow);
    s += edit.invert ? "1\n" : "0\n";
    s += "L ";
    putNumber(s, edit.lumaLo);
    putNumber(s, edit.lumaHi);
    s += "\n";
    return s;
}

bool editFromText(const std::string& text, MaskEdit& edit)
{
    edit = MaskEdit();
    const char* p = text.c_str();
    bool any = false;
    while (*p) {
        const char kind = *p;
        if (kind == 'P' || kind == 'O' || kind == 'F' || kind == 'L') {
            ++p;
            std::vector<double> v;
            double d;
            while (getNumber(p, d)) {
                v.push_back(d);
            }
            if (kind == 'P' && v.size() >= 3 && v[0] >= 0.0 && v[0] <= 1.0 && v[1] >= 0.0 && v[1] <= 1.0) {
                edit.points.push_back({ v[0], v[1], v[2] > 0.5 });
                any = true;
            } else if (kind == 'O' && v.size() >= 5) {
                MaskOp op;
                op.type = static_cast<int>(v[0] + 0.5);
                op.add = v[1] > 0.5;
                op.radius = v[2];
                op.xy.assign(v.begin() + 3, v.end());
                if (op.xy.size() % 2) {
                    op.xy.pop_back();
                }
                if (op.type >= 0 && op.type <= 2) {
                    edit.ops.push_back(op);
                    any = true;
                }
            } else if (kind == 'F' && v.size() >= 3) {
                edit.feather = std::min(0.2, std::max(0.0, v[0]));
                edit.grow = std::min(0.1, std::max(-0.1, v[1]));
                edit.invert = v[2] > 0.5;
            } else if (kind == 'L' && v.size() >= 2) {
                edit.lumaLo = std::min(1.0, std::max(0.0, v[0]));
                edit.lumaHi = std::min(1.0, std::max(edit.lumaLo, v[1]));
            }
        } else if (kind >= '0' && kind <= '9') {
            // old format: "x y add" per line
            double x, y, a;
            if (getNumber(p, x) && getNumber(p, y) && getNumber(p, a) && x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0) {
                edit.points.push_back({ x, y, a > 0.5 });
                any = true;
            }
        }
        while (*p && *p != '\n') {
            ++p;
        }
        if (*p == '\n') {
            ++p;
        }
    }
    return any;
}

#ifdef HAVE_ONNXRT

namespace
{

constexpr int SAM_SIZE = 1024;
constexpr int SAM_MASK = 256;

struct SamPoint {
    float x, y;   // pixels of the encoded image
    int label;    // 1 = on the object, 0 = not on the object
};

struct SamQuery {
    std::vector<SamPoint> points;
    bool useBox = false;
    float bx0 = 0.f, by0 = 0.f, bx1 = 0.f, by1 = 0.f;
    const std::vector<float>* prevLogits = nullptr;   // outW*outH logits of an earlier selection (optional)
    const unsigned char* reference = nullptr;         // outW x outH binary: prefer the candidate closest to it (optional)
    int referenceStride = 0;
};

std::unique_ptr<Net> g_samEncoder;
std::unique_ptr<Ort::Session> g_samDecoder;
Net::PathString g_samEncoderPath, g_samDecoderPath;
int g_samW = 0, g_samH = 0;      // size of the image whose features are loaded
float g_samScale = 1.f;

void samRelease()
{
    g_samDecoder.reset();
    g_samEncoder.reset();
    g_samW = g_samH = 0;
}

// Computes the features of one image; they stay loaded for samDecode.
std::string samEncode(const Net::PathString& encoderPath, const Net::PathString& decoderPath, const unsigned char* rgb, int w, int h)
{
    if (!rgb || w < 16 || h < 16) {
        return "image too small";
    }
    if (!g_samEncoder || g_samEncoderPath != encoderPath || g_samDecoderPath != decoderPath) {
        samRelease();
        g_samEncoder.reset(new Net(encoderPath, 3, SAM_SIZE, "sam21-encoder"));
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.SetIntraOpNumThreads(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
        g_samDecoder.reset(new Ort::Session(g_samEncoder->env(), decoderPath.c_str(), so));
        g_samEncoderPath = encoderPath;
        g_samDecoderPath = decoderPath;
    }
    g_samW = g_samH = 0;

    // image -> 1024x1024, longest side fitted, rest padded with zeros
    const float scale = static_cast<float>(SAM_SIZE) / std::max(w, h);
    const int nw = std::min(SAM_SIZE, static_cast<int>(w * scale + 0.5f));
    const int nh = std::min(SAM_SIZE, static_cast<int>(h * scale + 0.5f));
    const size_t plane = static_cast<size_t>(SAM_SIZE) * SAM_SIZE;
    std::vector<float> in(plane * 3, 0.f);
    static const float mean[3] = { 123.675f, 116.28f, 103.53f };
    static const float stdv[3] = { 58.395f, 57.12f, 57.375f };
    const float inv = 1.f / scale;
    for (int y = 0; y < nh; ++y) {
        const int sy0 = std::min(h - 1, static_cast<int>(y * inv));
        const int sy1 = std::min(h, std::max(sy0 + 1, static_cast<int>((y + 1) * inv)));
        for (int x = 0; x < nw; ++x) {
            const int sx0 = std::min(w - 1, static_cast<int>(x * inv));
            const int sx1 = std::min(w, std::max(sx0 + 1, static_cast<int>((x + 1) * inv)));
            unsigned sum[3] = { 0, 0, 0 };
            for (int sy = sy0; sy < sy1; ++sy) {
                const unsigned char* p = rgb + (static_cast<size_t>(sy) * w + sx0) * 3;
                for (int sx = sx0; sx < sx1; ++sx, p += 3) {
                    sum[0] += p[0];
                    sum[1] += p[1];
                    sum[2] += p[2];
                }
            }
            const float cnt = static_cast<float>((sy1 - sy0) * (sx1 - sx0));
            const size_t o = static_cast<size_t>(y) * SAM_SIZE + x;
            for (int c = 0; c < 3; ++c) {
                in[c * plane + o] = (sum[c] / cnt - mean[c]) / stdv[c];
            }
        }
    }

    g_samEncoder->run(in);
    g_samW = w;
    g_samH = h;
    g_samScale = scale;
    return {};
}

// Selects on the encoded image. logits receives outW*outH values covering the whole
// image (> 0: selected). agreement receives the overlap (IoU) with the reference, or -1.
std::string samDecode(const SamQuery& query, int outW, int outH, std::vector<float>& logits, float& quality, float& agreement)
{
    quality = 0.f;
    agreement = -1.f;
    if (!g_samEncoder || !g_samDecoder || g_samW <= 0 || outW < 2 || outH < 2) {
        return "no image is loaded for selection";
    }
    const int w = g_samW, h = g_samH;
    const float scale = g_samScale;

    std::vector<float> coords, labels;
    for (const SamPoint& pt : query.points) {
        coords.push_back(pt.x * scale);
        coords.push_back(pt.y * scale);
        labels.push_back(static_cast<float>(pt.label));
    }
    if (query.useBox) {
        coords.push_back(query.bx0 * scale);
        coords.push_back(query.by0 * scale);
        labels.push_back(2.f);
        coords.push_back(query.bx1 * scale);
        coords.push_back(query.by1 * scale);
        labels.push_back(3.f);
    } else {
        coords.push_back(0.f);
        coords.push_back(0.f);
        labels.push_back(-1.f);
    }
    const int64_t npts = static_cast<int64_t>(labels.size());

    // 256x256 grid over the padded 1024 square
    const float toMask = scale * SAM_MASK / SAM_SIZE;   // mask cells per image pixel
    const float sxOut = static_cast<float>(outW) / w;   // output samples per image pixel
    const float syOut = static_cast<float>(outH) / h;
    std::vector<float> maskIn(static_cast<size_t>(SAM_MASK) * SAM_MASK, 0.f);
    float hasMask = 0.f;
    if (query.prevLogits && query.prevLogits->size() == static_cast<size_t>(outW) * outH) {
        hasMask = 1.f;
        for (int my = 0; my < SAM_MASK; ++my) {
            const float fy = (my + 0.5f) / toMask;   // image pixels
            for (int mx = 0; mx < SAM_MASK; ++mx) {
                const float fx = (mx + 0.5f) / toMask;
                float v = -8.f;   // padding: not selected
                if (fx < w && fy < h) {
                    v = std::min(8.f, std::max(-8.f, bilinear(*query.prevLogits, outW, outH, fx * sxOut - 0.5f, fy * syOut - 0.5f)));
                }
                maskIn[static_cast<size_t>(my) * SAM_MASK + mx] = v;
            }
        }
    }

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions alloc;
    const int64_t coordShape[3] = { 1, npts, 2 };
    const int64_t labelShape[2] = { 1, npts };
    const int64_t maskShape[4] = { 1, 1, SAM_MASK, SAM_MASK };
    const int64_t oneShape[1] = { 1 };

    std::vector<std::string> inNames;
    std::vector<Ort::Value> inValues;
    const size_t nIn = g_samDecoder->GetInputCount();
    for (size_t i = 0; i < nIn; ++i) {
        const std::string name = g_samDecoder->GetInputNameAllocated(i, alloc).get();
        inNames.push_back(name);
        if (name == "point_coords") {
            inValues.push_back(Ort::Value::CreateTensor<float>(mem, coords.data(), coords.size(), coordShape, 3));
        } else if (name == "point_labels") {
            inValues.push_back(Ort::Value::CreateTensor<float>(mem, labels.data(), labels.size(), labelShape, 2));
        } else if (name == "mask_input") {
            inValues.push_back(Ort::Value::CreateTensor<float>(mem, maskIn.data(), maskIn.size(), maskShape, 4));
        } else if (name == "has_mask_input") {
            inValues.push_back(Ort::Value::CreateTensor<float>(mem, &hasMask, 1, oneShape, 1));
        } else {
            // a feature map: take the encoder output with the same channel count and size
            const std::vector<int64_t> want = g_samDecoder->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            bool found = false;
            for (size_t k = 0; k < 3 && !found; ++k) {
                Ort::Value& ev = g_samEncoder->output(k);
                const std::vector<int64_t> have = ev.GetTensorTypeAndShapeInfo().GetShape();
                if (have.size() == 4 && want.size() == 4 && have[1] == want[1] && have[2] == want[2]) {
                    size_t count = 1;
                    for (int64_t d : have) {
                        count *= static_cast<size_t>(d);
                    }
                    inValues.push_back(Ort::Value::CreateTensor<float>(mem, ev.GetTensorMutableData<float>(), count, have.data(), have.size()));
                    found = true;
                }
            }
            if (!found) {
                return "the selection networks do not fit together";
            }
        }
    }
    std::vector<const char*> inPtrs;
    for (const std::string& n : inNames) {
        inPtrs.push_back(n.c_str());
    }
    std::vector<std::string> outNames;
    for (size_t i = 0; i < g_samDecoder->GetOutputCount(); ++i) {
        outNames.push_back(g_samDecoder->GetOutputNameAllocated(i, alloc).get());
    }
    std::vector<const char*> outPtrs;
    for (const std::string& n : outNames) {
        outPtrs.push_back(n.c_str());
    }
    std::vector<Ort::Value> outs = g_samDecoder->Run(Ort::RunOptions{nullptr}, inPtrs.data(), inValues.data(), inValues.size(), outPtrs.data(), outPtrs.size());
    if (outs.size() < 2) {
        return "unexpected output of the selection network";
    }
    const std::vector<int64_t> ms = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    const size_t nIou = outs[1].GetTensorTypeAndShapeInfo().GetElementCount();
    if (ms.size() != 4 || ms[2] != SAM_MASK || ms[3] != SAM_MASK || ms[1] < 1 || nIou < static_cast<size_t>(ms[1])) {
        return "unexpected output of the selection network";
    }
    const float* iou = outs[1].GetTensorData<float>();
    const float* all = outs[0].GetTensorData<float>();
    const int ncand = static_cast<int>(ms[1]);

    // 256x256 logits -> outW x outH, for one candidate
    const auto expand = [&](int cand, std::vector<float>& dstv) {
        const float* logit = all + static_cast<size_t>(cand) * SAM_MASK * SAM_MASK;
        dstv.resize(static_cast<size_t>(outW) * outH);
        for (int y = 0; y < outH; ++y) {
            float fy = (y + 0.5f) / syOut * toMask - 0.5f;
            fy = std::min(static_cast<float>(SAM_MASK) - 1.001f, std::max(0.f, fy));
            const int y0 = static_cast<int>(fy);
            const float wy = fy - y0;
            float* dst = dstv.data() + static_cast<size_t>(y) * outW;
            for (int x = 0; x < outW; ++x) {
                float fx = (x + 0.5f) / sxOut * toMask - 0.5f;
                fx = std::min(static_cast<float>(SAM_MASK) - 1.001f, std::max(0.f, fx));
                const int x0 = static_cast<int>(fx);
                const float wx = fx - x0;
                const float* p = logit + static_cast<size_t>(y0) * SAM_MASK + x0;
                const float top = p[0] + (p[1] - p[0]) * wx;
                const float bottom = p[SAM_MASK] + (p[SAM_MASK + 1] - p[SAM_MASK]) * wx;
                dst[x] = top + (bottom - top) * wy;
            }
        }
    };

    int best = 0;
    if (query.reference) {
        // the candidate that agrees best with the earlier, coarser selection
        std::vector<float> tmp;
        for (int c = 0; c < ncand; ++c) {
            expand(c, tmp);
            size_t inter = 0, uni = 0;
            for (int y = 0; y < outH; ++y) {
                const unsigned char* ref = query.reference + static_cast<size_t>(y) * query.referenceStride;
                const float* row = tmp.data() + static_cast<size_t>(y) * outW;
                for (int x = 0; x < outW; ++x) {
                    const bool a = row[x] > 0.f;
                    const bool b = ref[x] != 0;
                    inter += a && b;
                    uni += a || b;
                }
            }
            const float agree = uni ? static_cast<float>(inter) / uni : 0.f;
            if (agree > agreement) {
                agreement = agree;
                best = c;
                logits.swap(tmp);
            }
        }
    } else {
        for (int c = 1; c < ncand; ++c) {
            if (iou[c] > iou[best]) {
                best = c;
            }
        }
        expand(best, logits);
    }
    quality = iou[best];
    return {};
}

inline float sigmoid(float v)
{
    return 1.f / (1.f + std::exp(-std::min(30.f, std::max(-30.f, v))));
}

// ---- one selection session: the developed frame stays loaded while points are edited

struct Session {
    Net::PathString enc, dec;
    RenderFn render;
    std::vector<unsigned char> rgb;   // the frame, longest side <= 4200
    int rw = 0, rh = 0;               // its size
    int W = 0, H = 0;                 // full image size
    bool encoded = false;             // the loaded features belong to the frame
    std::vector<unsigned char> small; // cached preview-size copy
    int sf = 0, sw = 0, sh = 0;
    std::vector<unsigned char> aiBin; // rough AI selection at the size of `small`
    std::vector<SelectPoint> aiPoints; // the points aiBin was made from
    bool aiValid = false;
};

std::unique_ptr<Session> g_session;

std::string encodeFrame()
{
    if (!g_session->encoded) {
        const std::string err = samEncode(g_session->enc, g_session->dec, g_session->rgb.data(), g_session->rw, g_session->rh);
        if (!err.empty()) {
            return err;
        }
        g_session->encoded = true;
    }
    return {};
}

void toQuery(const std::vector<SelectPoint>& points, float sx, float sy, SamQuery& q, std::vector<PointPx>& px, int w, int h)
{
    for (const SelectPoint& p : points) {
        const float x = static_cast<float>(p.x) * sx;
        const float y = static_cast<float>(p.y) * sy;
        q.points.push_back({ x, y, p.add ? 1 : 0 });
        px.push_back({ std::min(w - 1, std::max(0, static_cast<int>(static_cast<float>(p.x) * w))), std::min(h - 1, std::max(0, static_cast<int>(static_cast<float>(p.y) * h))), p.add });
    }
}

std::string describe(const std::exception_ptr& e)
{
    try {
        std::rethrow_exception(e);
    } catch (const Ort::Exception& ex) {
        return std::string("ONNX Runtime: ") + ex.what();
    } catch (const std::bad_alloc&) {
        return "out of memory";
    } catch (const std::exception& ex) {
        return ex.what();
    } catch (...) {
        return "unknown error";
    }
}

}

void selectEnd()
{
    g_session.reset();
    samRelease();
}

#ifdef _WIN32
std::string selectBegin(const wchar_t* encoderPath, const wchar_t* decoderPath, const RenderFn& render, const std::function<void(double)>& progress)
#else
std::string selectBegin(const char* encoderPath, const char* decoderPath, const RenderFn& render, const std::function<void(double)>& progress)
#endif
{
    try {
        g_session.reset(new Session());
        g_session->enc = encoderPath;
        g_session->dec = decoderPath;
        g_session->render = render;
        if (progress) {
            progress(0.05);
        }
        int W = 0, H = 0;
        if (!render(0, 0, 0, 0, 4200, g_session->rgb, g_session->rw, g_session->rh, W, H) || W < 16 || H < 16 || g_session->rw < 16 || g_session->rh < 16) {
            selectEnd();
            return "cannot develop the image";
        }
        g_session->W = W;
        g_session->H = H;
        if (progress) {
            progress(0.5);
        }
        const std::string err = encodeFrame();
        if (!err.empty()) {
            selectEnd();
            return err;
        }
        if (progress) {
            progress(1.0);
        }
    } catch (...) {
        const std::string err = describe(std::current_exception());
        selectEnd();
        return err;
    }
    return {};
}

namespace
{

bool samePoints(const std::vector<SelectPoint>& a, const std::vector<SelectPoint>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].add != b[i].add) {
            return false;
        }
    }
    return true;
}

bool anyAddPoint(const std::vector<SelectPoint>& points)
{
    for (const SelectPoint& p : points) {
        if (p.add) {
            return true;
        }
    }
    return false;
}

// Brings the rough AI selection (at a third of the frame size) up to date.
std::string updateRoughSelection(const std::vector<SelectPoint>& points)
{
    Session& s = *g_session;
    if (s.sf != 3) {
        shrinkRgb(s.rgb, s.rw, s.rh, 3, s.small, s.sw, s.sh);
        s.sf = 3;
        s.aiValid = false;
    }
    if (s.aiValid && samePoints(points, s.aiPoints)) {
        return {};
    }
    const int pw = s.sw, ph = s.sh;
    const size_t n = static_cast<size_t>(pw) * ph;
    s.aiBin.assign(n, 0);
    if (anyAddPoint(points)) {
        std::string err = encodeFrame();
        if (!err.empty()) {
            return err;
        }
        std::vector<PointPx> px;
        SamQuery q;
        toQuery(points, static_cast<float>(s.rw), static_cast<float>(s.rh), q, px, pw, ph);
        std::vector<float> logits;
        float quality = 0.f, agree = 0.f;
        err = samDecode(q, pw, ph, logits, quality, agree);
        if (!err.empty()) {
            return err;
        }
        for (float& v : logits) {
            v = std::min(8.f, std::max(-8.f, v));
        }
        guidedFilter(s.small.data(), pw, ph, logits, std::max(2, std::max(pw, ph) / 85), 1e-3f, false);
        for (size_t i = 0; i < n; ++i) {
            s.aiBin[i] = logits[i] > 0.f;
        }
        cleanupSelection(s.aiBin, pw, ph, px);
    }
    s.aiPoints = points;
    s.aiValid = true;
    return {};
}

}

std::string selectView(const MaskEdit& edit, double cx, double cy, double zoom, int outW, int outH,
                       std::vector<unsigned char>& rgbOut, double& scaleX, double& scaleY, double& offX, double& offY)
{
    if (!g_session) {
        return "no selection is in progress";
    }
    if (outW < 16 || outH < 16) {
        return "the view is too small";
    }
    try {
        const std::string err = updateRoughSelection(edit.points);
        if (!err.empty()) {
            return err;
        }
        Session& s = *g_session;
        // frame pixels per output pixel
        const double fit = std::max(static_cast<double>(s.rw) / outW, static_cast<double>(s.rh) / outH);
        const double z = std::max(0.25, fit / std::max(1.0, zoom));
        const double vx = cx * s.rw - outW * z / 2.0;   // frame position of the view's left edge
        const double vy = cy * s.rh - outH * z / 2.0;
        scaleX = s.rw / z;
        scaleY = s.rh / z;
        offX = -vx / z;
        offY = -vy / z;
        const Xform t = { s.rw / z, s.rh / z, offX, offY };

        const size_t n = static_cast<size_t>(outW) * outH;
        std::vector<float> m(n, 0.f);
        std::vector<unsigned char> inside(n, 0);
        rgbOut.resize(n * 3);
        const int step = std::max(1, static_cast<int>(z));
        for (int y = 0; y < outH; ++y) {
            const double fy = vy + (y + 0.5) * z;
            for (int x = 0; x < outW; ++x) {
                const double fx = vx + (x + 0.5) * z;
                const size_t i = static_cast<size_t>(y) * outW + x;
                unsigned char* d = rgbOut.data() + i * 3;
                if (fx < 0.0 || fy < 0.0 || fx >= s.rw || fy >= s.rh) {
                    d[0] = d[1] = d[2] = 48;
                    continue;
                }
                inside[i] = 1;
                const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
                if (step <= 1) {
                    const unsigned char* p = s.rgb.data() + (static_cast<size_t>(iy) * s.rw + ix) * 3;
                    d[0] = p[0];
                    d[1] = p[1];
                    d[2] = p[2];
                } else {
                    // average a few samples of the footprint
                    const int x0 = std::max(0, ix - step / 2), y0 = std::max(0, iy - step / 2);
                    const int x1 = std::min(s.rw, x0 + step), y1 = std::min(s.rh, y0 + step);
                    unsigned sum[3] = { 0, 0, 0 }, cnt = 0;
                    for (int yy = y0; yy < y1; ++yy) {
                        const unsigned char* p = s.rgb.data() + (static_cast<size_t>(yy) * s.rw + x0) * 3;
                        for (int xx = x0; xx < x1; ++xx, p += 3) {
                            sum[0] += p[0];
                            sum[1] += p[1];
                            sum[2] += p[2];
                            ++cnt;
                        }
                    }
                    d[0] = static_cast<unsigned char>(sum[0] / cnt);
                    d[1] = static_cast<unsigned char>(sum[1] / cnt);
                    d[2] = static_cast<unsigned char>(sum[2] / cnt);
                }
                const int ax = std::min(s.sw - 1, ix / 3), ay = std::min(s.sh - 1, iy / 3);
                m[i] = s.aiBin[static_cast<size_t>(ay) * s.sw + ax];
            }
        }

        finishTouches(m, outW, outH, edit, t, rgbOut.data());

        for (size_t i = 0; i < n; ++i) {
            if (!inside[i]) {
                continue;
            }
            unsigned char* d = rgbOut.data() + i * 3;
            const float a = std::min(1.f, std::max(0.f, m[i])) * 0.55f;
            // brightened a little so that dark photos remain readable
            d[0] = static_cast<unsigned char>(std::min(255.f, d[0] * 1.3f) * (1.f - a) + 255.f * a);
            d[1] = static_cast<unsigned char>(std::min(255.f, d[1] * 1.3f) * (1.f - a) + 40.f * a);
            d[2] = static_cast<unsigned char>(std::min(255.f, d[2] * 1.3f) * (1.f - a) + 40.f * a);
        }
        // AI points: green = add, blue = subtract, with a white ring
        for (const SelectPoint& p : edit.points) {
            const int px = static_cast<int>(p.x * t.sx + t.ox), py = static_cast<int>(p.y * t.sy + t.oy);
            for (int dy = -8; dy <= 8; ++dy) {
                for (int dx = -8; dx <= 8; ++dx) {
                    const int d2 = dx * dx + dy * dy;
                    const int x = px + dx, y = py + dy;
                    if (d2 > 64 || x < 0 || y < 0 || x >= outW || y >= outH) {
                        continue;
                    }
                    unsigned char* d = rgbOut.data() + (static_cast<size_t>(y) * outW + x) * 3;
                    if (d2 > 30) {
                        d[0] = d[1] = d[2] = 255;
                    } else if (p.add) {
                        d[0] = 40;
                        d[1] = 220;
                        d[2] = 60;
                    } else {
                        d[0] = 40;
                        d[1] = 110;
                        d[2] = 255;
                    }
                }
            }
        }
    } catch (...) {
        return describe(std::current_exception());
    }
    return {};
}

std::string selectFinish(const MaskEdit& edit, SelectResult& result, const std::function<void(double)>& progress)
{
    if (!g_session) {
        return "no selection is in progress";
    }
    const auto report = [&progress](double v) {
        if (progress) {
            progress(v);
        }
    };

    try {
        Session& s = *g_session;
        const std::vector<unsigned char>& rgb = s.rgb;
        const int rw = s.rw, rh = s.rh, W = s.W, H = s.H;
        const float k = static_cast<float>(rw) / W;   // frame pixels per full-image pixel

        const std::vector<SelectPoint>& points = edit.points;
        const size_t rn = static_cast<size_t>(rw) * rh;
        std::vector<float> P(rn, 0.f);   // final probabilities at frame size
        float quality = 0.f, agree = 0.f;
        result.zoomed = false;
        result.tiles = 0;
        std::vector<unsigned char> M1;
        std::string err;

        if (!anyAddPoint(points) && edit.ops.empty() && !edit.invert) {
            return "nothing";
        }

        if (anyAddPoint(points)) {
        // ---- pass 1: the whole frame
        report(0.03);
        err = encodeFrame();
        if (!err.empty()) {
            return err;
        }
        report(0.20);
        SamQuery q1;
        std::vector<PointPx> px;
        toQuery(points, static_cast<float>(rw), static_cast<float>(rh), q1, px, rw, rh);
        std::vector<float> L1;
        err = samDecode(q1, rw, rh, L1, quality, agree);
        if (!err.empty()) {
            return err;
        }
        report(0.30);

        // Decide uncertain areas as a whole: smooth the logits along the image structure,
        // then drop specks and fill pinholes.
        std::vector<float> S1(rn);
        for (size_t i = 0; i < rn; ++i) {
            S1[i] = std::min(8.f, std::max(-8.f, L1[i]));
        }
        guidedFilter(rgb.data(), rw, rh, S1, std::max(4, std::max(rw, rh) / 85), 1e-3f, false);
        M1.resize(rn);
        for (size_t i = 0; i < rn; ++i) {
            M1[i] = S1[i] > 0.f;
        }
        S1.clear();
        S1.shrink_to_fit();
        cleanupSelection(M1, rw, rh, px);

        int bx0 = rw, by0 = rh, bx1 = -1, by1 = -1;
        for (int y = 0; y < rh; ++y) {
            const unsigned char* row = M1.data() + static_cast<size_t>(y) * rw;
            for (int x = 0; x < rw; ++x) {
                if (row[x]) {
                    bx0 = std::min(bx0, x);
                    bx1 = std::max(bx1, x);
                    by0 = std::min(by0, y);
                    by1 = std::max(by1, y);
                }
            }
        }
        const bool found = bx1 >= bx0 && by1 >= by0;
        ++bx1;
        ++by1;
        if (found) {

        const double fbx0 = bx0 / k, fby0 = by0 / k, fbx1 = bx1 / k, fby1 = by1 / k;   // full-image pixels
        const double want = std::max(fbx1 - fbx0, fby1 - fby0) * 1.6 + 64.0;

        if (want < 0.5 * std::max(W, H)) {
            // ---- small object: zoom in on it, developed at full resolution
            const int cropW = std::min(W, static_cast<int>(want));
            const int cropH = std::min(H, static_cast<int>(want));
            const double ccx = 0.5 * (fbx0 + fbx1), ccy = 0.5 * (fby0 + fby1);
            const int cropX = std::max(0, std::min(W - cropW, static_cast<int>(ccx - cropW / 2.0)));
            const int cropY = std::max(0, std::min(H - cropH, static_cast<int>(ccy - cropH / 2.0)));
            std::vector<unsigned char> rgb2;
            int w2 = 0, h2 = 0, nW = 0, nH = 0;
            if (s.render(cropX, cropY, cropW, cropH, 2048, rgb2, w2, h2, nW, nH) && nW > 0 && w2 >= 16 && h2 >= 16) {
                report(0.55);
                const float k2 = static_cast<float>(w2) / nW;
                SamQuery q2;
                bool addInside = false;
                for (const SelectPoint& p : points) {
                    const float x = static_cast<float>(p.x * W - cropX) * k2;
                    const float y = static_cast<float>(p.y * H - cropY) * k2;
                    if (x >= 0.f && y >= 0.f && x < w2 && y < h2) {
                        q2.points.push_back({ x, y, p.add ? 1 : 0 });
                        addInside = addInside || p.add;
                    }
                }
                if (addInside) {
                    q2.useBox = true;
                    q2.bx0 = static_cast<float>(std::max(0.0, fbx0 - cropX)) * k2;
                    q2.by0 = static_cast<float>(std::max(0.0, fby0 - cropY)) * k2;
                    q2.bx1 = static_cast<float>(std::min<double>(cropW, fbx1 - cropX)) * k2;
                    q2.by1 = static_cast<float>(std::min<double>(cropH, fby1 - cropY)) * k2;
                    std::vector<float> L2;
                    float q = 0.f;
                    s.encoded = false;
                    err = samEncode(s.enc, s.dec, rgb2.data(), w2, h2);
                    if (err.empty()) {
                        err = samDecode(q2, w2, h2, L2, q, agree);
                    }
                    if (err.empty()) {
                        for (float& v : L2) {
                            v = sigmoid(v);
                        }
                        guidedFilter(rgb2.data(), w2, h2, L2, std::max(2, w2 / 256), 1e-4f, true);
                        for (int y = 0; y < rh; ++y) {
                            const float fy = (y + 0.5f) / k;
                            if (fy < cropY || fy >= cropY + cropH) {
                                continue;
                            }
                            float* row = P.data() + static_cast<size_t>(y) * rw;
                            for (int x = 0; x < rw; ++x) {
                                const float fx = (x + 0.5f) / k;
                                if (fx >= cropX && fx < cropX + cropW) {
                                    row[x] = bilinear(L2, w2, h2, (fx - cropX) * k2 - 0.5f, (fy - cropY) * k2 - 0.5f);
                                }
                            }
                        }
                        quality = q;
                        result.zoomed = true;
                    }
                }
            }
        }

        if (!result.zoomed) {
            // ---- large region: go over it tile by tile at higher resolution
            const int T = std::max(256, std::max(rw, rh) / 3);
            const int overlap = std::max(16, T * 15 / 100);
            const int tw = std::min(T, rw), th = std::min(T, rh);
            std::vector<float> wsum(rn, 0.f);
            std::vector<int> txs, tys;
            for (int ty = std::max(0, by0 - 64); ty < std::min(rh, by1 + 64); ty += T - overlap) {
                tys.push_back(std::max(0, std::min(rh, ty + th) - th));
            }
            for (int tx = std::max(0, bx0 - 64); tx < std::min(rw, bx1 + 64); tx += T - overlap) {
                txs.push_back(std::max(0, std::min(rw, tx + tw) - tw));
            }
            const int total = static_cast<int>(txs.size() * tys.size());
            int done = 0;
            std::vector<unsigned char> tile(static_cast<size_t>(tw) * th * 3);
            std::vector<float> tilePrev(static_cast<size_t>(tw) * th), tileLogits;
            for (int sy : tys) {
                for (int sx : txs) {
                    size_t ones = 0;
                    for (int y = 0; y < th; ++y) {
                        const unsigned char* row = M1.data() + static_cast<size_t>(sy + y) * rw + sx;
                        for (int x = 0; x < tw; ++x) {
                            ones += row[x];
                        }
                    }
                    const double frac = static_cast<double>(ones) / (static_cast<double>(tw) * th);
                    bool useTile = false;
                    if (frac > 0.003 && frac < 0.997) {
                        int ix = 0, iy = 0, ox = 0, oy = 0;
                        const unsigned char* sub = M1.data() + static_cast<size_t>(sy) * rw + sx;
                        if (farthestInside(sub, rw, tw, th, 1, ix, iy) && farthestInside(sub, rw, tw, th, 0, ox, oy)) {
                            for (int y = 0; y < th; ++y) {
                                memcpy(tile.data() + static_cast<size_t>(y) * tw * 3, rgb.data() + (static_cast<size_t>(sy + y) * rw + sx) * 3, static_cast<size_t>(tw) * 3);
                                memcpy(tilePrev.data() + static_cast<size_t>(y) * tw, L1.data() + static_cast<size_t>(sy + y) * rw + sx, static_cast<size_t>(tw) * sizeof(float));
                            }
                            SamQuery qt;
                            qt.points.push_back({ static_cast<float>(ix), static_cast<float>(iy), 1 });
                            qt.points.push_back({ static_cast<float>(ox), static_cast<float>(oy), 0 });
                            // the user's own points inside this tile
                            for (const PointPx& p : px) {
                                if (p.x >= sx && p.y >= sy && p.x < sx + tw && p.y < sy + th && static_cast<bool>(M1[static_cast<size_t>(p.y) * rw + p.x]) == p.add) {
                                    qt.points.push_back({ static_cast<float>(p.x - sx), static_cast<float>(p.y - sy), p.add ? 1 : 0 });
                                }
                            }
                            qt.prevLogits = &tilePrev;
                            qt.reference = sub;
                            qt.referenceStride = rw;
                            float q = 0.f, a = 0.f;
                            s.encoded = false;
                            if (samEncode(s.enc, s.dec, tile.data(), tw, th).empty() && samDecode(qt, tw, th, tileLogits, q, a).empty() && a >= 0.6f) {
                                useTile = true;
                                ++result.tiles;
                            }
                        }
                    }
                    for (int y = 0; y < th; ++y) {
                        const float wy = std::min(1.f, std::min(y + 1, th - y) / static_cast<float>(overlap));
                        const size_t o = static_cast<size_t>(sy + y) * rw + sx;
                        for (int x = 0; x < tw; ++x) {
                            const float wgt = wy * std::min(1.f, std::min(x + 1, tw - x) / static_cast<float>(overlap));
                            const float v = useTile ? sigmoid(tileLogits[static_cast<size_t>(y) * tw + x]) : static_cast<float>(M1[o + x]);
                            P[o + x] += v * wgt;
                            wsum[o + x] += wgt;
                        }
                    }
                    ++done;
                    report(0.30 + 0.55 * done / std::max(1, total));
                }
            }
            for (size_t i = 0; i < rn; ++i) {
                P[i] = wsum[i] > 0.f ? P[i] / wsum[i] : 0.f;
            }
            guidedFilter(rgb.data(), rw, rh, P, 3, 1e-4f, true);
        }
        L1.clear();
        L1.shrink_to_fit();
        report(0.88);

        // ---- clean-up of the AI part: no specks, no pinholes
        for (size_t i = 0; i < rn; ++i) {
            M1[i] = P[i] > 0.5f;
        }
        std::vector<unsigned char> before = M1;
        cleanupSelection(M1, rw, rh, px);
        for (size_t i = 0; i < rn; ++i) {
            if (before[i] && !M1[i]) {
                P[i] = 0.f;
            } else if (!before[i] && M1[i]) {
                P[i] = 1.f;
            }
        }
        before.clear();
        }   // found
        }   // AI points
        M1.clear();
        M1.shrink_to_fit();

        // ---- hand-made parts, grow/shrink, feather, invert
        report(0.92);
        {
            const Xform t = { static_cast<double>(rw), static_cast<double>(rh), 0.0, 0.0 };
            finishTouches(P, rw, rh, edit, t, rgb.data());
        }

        result.w = rw;
        result.h = rh;
        result.mask.resize(rn);
        int mx0 = rw, my0 = rh, mx1 = -1, my1 = -1;
        size_t selected = 0;
        for (int y = 0; y < rh; ++y) {
            for (int x = 0; x < rw; ++x) {
                const size_t i = static_cast<size_t>(y) * rw + x;
                const unsigned char u = static_cast<unsigned char>(std::min(1.f, std::max(0.f, P[i])) * 255.f + 0.5f);
                result.mask[i] = u;
                if (u > 127) {
                    ++selected;
                    mx0 = std::min(mx0, x);
                    mx1 = std::max(mx1, x);
                    my0 = std::min(my0, y);
                    my1 = std::max(my1, y);
                }
            }
        }
        if (mx1 < mx0 || my1 < my0) {
            return "nothing";
        }
        result.x0 = static_cast<double>(mx0) / rw;
        result.y0 = static_cast<double>(my0) / rh;
        result.x1 = static_cast<double>(mx1 + 1) / rw;
        result.y1 = static_cast<double>(my1 + 1) / rh;
        result.area = static_cast<double>(selected) / rn;
        result.quality = quality;

        // ---- preview: the frame with the selection tinted red
        const int pf = std::max(1, (rw + 899) / 900);
        result.previewW = rw / pf;
        result.previewH = rh / pf;
        result.preview.resize(static_cast<size_t>(result.previewW) * result.previewH * 3);
        for (int y = 0; y < result.previewH; ++y) {
            for (int x = 0; x < result.previewW; ++x) {
                const size_t si = static_cast<size_t>(y) * pf * rw + static_cast<size_t>(x) * pf;
                const unsigned char* src = rgb.data() + si * 3;
                const float a = result.mask[si] * (0.6f / 255.f);
                unsigned char* d = result.preview.data() + (static_cast<size_t>(y) * result.previewW + x) * 3;
                d[0] = static_cast<unsigned char>(std::min(255.f, src[0] * 1.3f) * (1.f - a) + 255.f * a);
                d[1] = static_cast<unsigned char>(std::min(255.f, src[1] * 1.3f) * (1.f - a) + 40.f * a);
                d[2] = static_cast<unsigned char>(std::min(255.f, src[2] * 1.3f) * (1.f - a) + 40.f * a);
            }
        }
        report(1.0);
    } catch (...) {
        return describe(std::current_exception());
    }

    return {};
}

#ifdef _WIN32
std::string selectObject(const wchar_t* encoderPath, const wchar_t* decoderPath, const RenderFn& render,
                         const SelectRequest& request, SelectResult& result, const std::function<void(double)>& progress)
#else
std::string selectObject(const char* encoderPath, const char* decoderPath, const RenderFn& render,
                         const SelectRequest& request, SelectResult& result, const std::function<void(double)>& progress)
#endif
{
    std::string err = selectBegin(encoderPath, decoderPath, render, nullptr);
    if (err.empty()) {
        MaskEdit edit;
        edit.points.push_back({ request.cx, request.cy, true });
        for (const SelectPoint& p : request.more) {
            edit.points.push_back(p);
        }
        err = selectFinish(edit, result, progress);
    }
    selectEnd();
    return err;
}

#else

void selectEnd()
{
}

#ifdef _WIN32
std::string selectBegin(const wchar_t*, const wchar_t*, const RenderFn&, const std::function<void(double)>&)
#else
std::string selectBegin(const char*, const char*, const RenderFn&, const std::function<void(double)>&)
#endif
{
    return "built without ONNX Runtime";
}

std::string selectView(const MaskEdit&, double, double, double, int, int, std::vector<unsigned char>&, double&, double&, double&, double&)
{
    return "built without ONNX Runtime";
}

std::string selectFinish(const MaskEdit&, SelectResult&, const std::function<void(double)>&)
{
    return "built without ONNX Runtime";
}

#ifdef _WIN32
std::string selectObject(const wchar_t*, const wchar_t*, const RenderFn&, const SelectRequest&, SelectResult&, const std::function<void(double)>&)
#else
std::string selectObject(const char*, const char*, const RenderFn&, const SelectRequest&, SelectResult&, const std::function<void(double)>&)
#endif
{
    return "built without ONNX Runtime";
}

#endif

}
