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
#include <onnxruntime_cxx_api.h>
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

#ifdef HAVE_ONNXRT

namespace
{
constexpr int TILE = 768;      // fixed input size of the model
constexpr int OVERLAP = 64;    // overlap between neighbouring tiles
constexpr int STEP = TILE - OVERLAP;

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

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "rawtherapee");
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(std::max(1u, std::thread::hardware_concurrency()));
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.DisableMemPattern();
#ifdef _WIN32
        gunichar2* w = g_utf8_to_utf16(path.c_str(), -1, nullptr, nullptr, nullptr);
        if (!w) {
            return "模型路径无法转换：" + path;
        }
        std::wstring wpath(reinterpret_cast<wchar_t*>(w));
        g_free(w);
        Ort::Session session(env, wpath.c_str(), so);
#else
        Ort::Session session(env, path.c_str(), so);
#endif
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
