#include "ainoise.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <vector>
#include <glibmm/fileutils.h>
#include <glibmm/miscutils.h>

#include "options.h"
#include "rtengine/iimage.h"

#ifdef HAVE_ONNXRT
// The runtime is loaded on demand from the program folder, so RawTherapee itself
// never depends on it at start-up.
#define NOMINMAX
#include <windows.h>
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#endif

namespace AIDenoise
{

bool compiledIn()
{
#ifdef HAVE_ONNXRT
    return true;
#else
    return false;
#endif
}

Glib::ustring modelPath()
{
    return Glib::build_filename(App::get().argv0(), "models", "denoise-nafnet.onnx");
}

Glib::ustring runtimePath()
{
    return Glib::build_filename(App::get().argv0(), "onnxruntime.dll");
}

#ifdef HAVE_ONNXRT

namespace
{
constexpr int TILE = 768;      // fixed input size of the model
constexpr int OVERLAP = 64;    // overlap between neighbouring tiles
constexpr int STEP = TILE - OVERLAP;

std::wstring toWide(const Glib::ustring& s)
{
    gunichar2* w = g_utf8_to_utf16(s.c_str(), -1, nullptr, nullptr, nullptr);
    if (!w) {
        return {};
    }
    std::wstring res(reinterpret_cast<wchar_t*>(w));
    g_free(w);
    return res;
}

// Loads onnxruntime.dll by its full path (Windows has an older copy of its own in System32
// that must not be picked up) and initialises the C++ wrapper. Returns an error text or "".
Glib::ustring loadRuntime()
{
    static bool loaded = false;
    if (loaded) {
        return {};
    }
    const Glib::ustring path = runtimePath();
    const std::wstring wpath = toWide(path);
    HMODULE h = wpath.empty() ? nullptr : LoadLibraryExW(wpath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {
        return Glib::ustring::compose("无法加载 AI 运行库（错误 %1）：%2\n如果文件存在，请安装 Microsoft Visual C++ 运行库。", static_cast<unsigned long>(GetLastError()), path);
    }
    typedef const OrtApiBase* (ORT_API_CALL * GetBaseFn)(void);
    GetBaseFn getBase = reinterpret_cast<GetBaseFn>(reinterpret_cast<void*>(GetProcAddress(h, "OrtGetApiBase")));
    if (!getBase) {
        return "AI 运行库文件不正确：" + path;
    }
    const OrtApi* api = getBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        return "AI 运行库版本不匹配：" + path;
    }
    Ort::InitApi(api);
    loaded = true;
    return {};
}

inline int mirror(int v, int n)
{
    if (n <= 1) {
        return 0;
    }
    const int period = 2 * (n - 1);
    v %= period;
    if (v < 0) {
        v += period;
    }
    return v < n ? v : period - v;
}
}

Glib::ustring run(rtengine::IImagefloat* img, const std::function<void(double)>& progress)
{
    const int W = img->getWidth();
    const int H = img->getHeight();
    const Glib::ustring path = modelPath();

    if (!Glib::file_test(path, Glib::FILE_TEST_IS_REGULAR)) {
        return "找不到降噪模型文件：" + path;
    }

    const Glib::ustring loadErr = loadRuntime();
    if (!loadErr.empty()) {
        return loadErr;
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "rawtherapee");
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(std::max(1u, std::thread::hardware_concurrency()));
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.DisableMemPattern();
        const std::wstring wmodel = toWide(path);
        Ort::Session session(env, wmodel.c_str(), so);
        Ort::AllocatorWithDefaultOptions alloc;
        auto inName = session.GetInputNameAllocated(0, alloc);
        auto outName = session.GetOutputNameAllocated(0, alloc);
        const char* inNames[] = { inName.get() };
        const char* outNames[] = { outName.get() };

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int64_t shape[4] = { 1, 3, TILE, TILE };
        const size_t plane = static_cast<size_t>(TILE) * TILE;

        std::vector<float> acc(static_cast<size_t>(W) * H * 3, 0.f);
        std::vector<float> wsum(static_cast<size_t>(W) * H, 0.f);
        std::vector<float> in(plane * 3);

        // 1D feather weights for a tile
        std::vector<float> ramp(TILE);
        for (int i = 0; i < TILE; ++i) {
            const float d = static_cast<float>(std::min(i + 1, TILE - i));
            ramp[i] = std::min(1.f, d / OVERLAP);
        }

        const int nx = std::max(1, (W + STEP - 1) / STEP);
        const int ny = std::max(1, (H + STEP - 1) / STEP);
        const int total = nx * ny;
        int done = 0;

        for (int ty = 0; ty < ny; ++ty) {
            for (int tx = 0; tx < nx; ++tx) {
                // top-left of the tile; shift the last tiles back so that they stay inside the image
                int x0 = tx * STEP;
                int y0 = ty * STEP;
                if (W > TILE) {
                    x0 = std::min(x0, W - TILE);
                }
                if (H > TILE) {
                    y0 = std::min(y0, H - TILE);
                }

                for (int y = 0; y < TILE; ++y) {
                    const int sy = mirror(y0 + y, H);
                    for (int x = 0; x < TILE; ++x) {
                        const int sx = mirror(x0 + x, W);
                        const size_t o = static_cast<size_t>(y) * TILE + x;
                        in[o] = std::min(1.f, std::max(0.f, img->r(sy, sx) / 65535.f));
                        in[plane + o] = std::min(1.f, std::max(0.f, img->g(sy, sx) / 65535.f));
                        in[2 * plane + o] = std::min(1.f, std::max(0.f, img->b(sy, sx) / 65535.f));
                    }
                }

                Ort::Value input = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape, 4);
                auto outs = session.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 1);
                const float* out = outs[0].GetTensorData<float>();

                const int wy = std::min(TILE, H - y0);
                const int wx = std::min(TILE, W - x0);
                for (int y = 0; y < wy; ++y) {
                    for (int x = 0; x < wx; ++x) {
                        const float wt = ramp[x] * ramp[y];
                        const size_t o = static_cast<size_t>(y) * TILE + x;
                        const size_t d = static_cast<size_t>(y0 + y) * W + (x0 + x);
                        acc[d * 3] += out[o] * wt;
                        acc[d * 3 + 1] += out[plane + o] * wt;
                        acc[d * 3 + 2] += out[2 * plane + o] * wt;
                        wsum[d] += wt;
                    }
                }

                ++done;
                if (progress) {
                    progress(static_cast<double>(done) / total);
                }
            }
        }

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const size_t d = static_cast<size_t>(y) * W + x;
                const float inv = wsum[d] > 0.f ? 65535.f / wsum[d] : 0.f;
                img->r(y, x) = std::min(65535.f, std::max(0.f, acc[d * 3] * inv));
                img->g(y, x) = std::min(65535.f, std::max(0.f, acc[d * 3 + 1] * inv));
                img->b(y, x) = std::min(65535.f, std::max(0.f, acc[d * 3 + 2] * inv));
            }
        }
    } catch (const Ort::Exception& e) {
        return Glib::ustring("AI 降噪运行失败：") + e.what();
    } catch (const std::bad_alloc&) {
        return "内存不足，无法完成 AI 降噪（请确认已开启系统页面文件）";
    } catch (const std::exception& e) {
        return Glib::ustring("AI 降噪出错：") + e.what();
    }

    return {};
}

#else

Glib::ustring run(rtengine::IImagefloat*, const std::function<void(double)>&)
{
    return "此版本未包含 AI 降噪组件";
}

#endif

}
