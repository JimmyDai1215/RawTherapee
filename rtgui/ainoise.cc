#include "ainoise.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <glib/gstdio.h>
#include <glibmm/fileutils.h>
#include <glibmm/miscutils.h>

#include "airawcore.h"
#include "options.h"
#include "rtengine/aimask.h"
#include "rtengine/imagefloat.h"
#include "rtengine/procparams.h"
#include "rtengine/rawimage.h"
#include "rtengine/rtengine.h"

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
    return Glib::build_filename(App::get().argv0(), "models", "rawdenoise-nind-bayer.onnx");
}

Glib::ustring runtimePath()
{
    return Glib::build_filename(App::get().argv0(), "onnxruntime.dll");
}

bool available()
{
    return compiledIn()
           && Glib::file_test(modelPath(), Glib::FILE_TEST_IS_REGULAR)
           && Glib::file_test(runtimePath(), Glib::FILE_TEST_IS_REGULAR);
}

void setGpuAllowed(bool allowed)
{
    AIRawCore::setGpuAllowed(allowed);
}

Glib::ustring lastDevice()
{
    return AIRawCore::lastDevice();
}

double lastGpuDeviation()
{
    return AIRawCore::lastGpuDeviation();
}

namespace
{

#ifdef _WIN32
std::wstring nativePath(const Glib::ustring& s)
{
    gunichar2* w = g_utf8_to_utf16(s.c_str(), -1, nullptr, nullptr, nullptr);
    if (!w) {
        return std::wstring();
    }
    std::wstring res(reinterpret_cast<wchar_t*>(w));
    g_free(w);
    return res;
}
#else
std::string nativePath(const Glib::ustring& s)
{
    return s.raw();
}
#endif

void invert3(const double m[3][3], double inv[3][3])
{
    const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                     - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                     + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const double d = std::fabs(det) > 1e-12 ? 1.0 / det : 0.0;
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d;
}

}

Glib::ustring rawToDng(const Glib::ustring& rawFile, const Glib::ustring& dngFile, float strength, const std::function<void(double)>& progress)
{
    if (!available()) {
        return "此安装中缺少 AI 降噪组件或模型文件";
    }

    const std::string initErr = AIRawCore::initRuntime(nativePath(runtimePath()).c_str());
    if (!initErr.empty()) {
        return Glib::ustring("无法加载 AI 运行库：") + initErr + "\n如果文件存在，请安装 Microsoft Visual C++ 运行库。";
    }

    rtengine::RawImage ri(rawFile);
    if (ri.loadRaw(true, 0, true) != 0) {
        return "无法读取这个文件的 RAW 数据（AI 降噪只能用于 RAW 文件）";
    }
    if (!ri.isBayer() || ri.get_colors() != 3 || ri.isFloat()) {
        return "AI 降噪目前只支持拜耳阵列传感器的 RAW 文件";
    }
    ri.compress_image(0);
    if (!ri.data) {
        return "无法读取 RAW 数据";
    }

    AIRawCore::BayerInfo info;
    info.width = ri.get_width();
    info.height = ri.get_height();

    float black[4] = {0.f, 0.f, 0.f, 0.f};
    ri.get_colorsCoeff(nullptr, nullptr, black, false);
    float white = 65535.f;
    for (int c = 0; c < 3; ++c) {
        const float w = ri.get_white(c);
        if (w > 0.f && w < white) {
            white = w;
        }
    }
    info.white = white;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            const unsigned fc = ri.FC(y, x);
            info.cfa[y][x] = fc == 3 ? 1 : static_cast<int>(fc);
            info.black[y][x] = black[fc];
        }
    }

    const std::string err = AIRawCore::denoiseBayer(nativePath(modelPath()).c_str(), ri.data, info, strength, progress);
    if (!err.empty()) {
        if (err == "out of memory") {
            return "内存不足，无法完成 AI 降噪（请确认已开启系统页面文件）";
        }
        return Glib::ustring("AI 降噪失败：") + err;
    }

    AIRawCore::DngMeta meta;
    meta.make = ri.get_maker();
    meta.model = ri.get_model();

    switch (ri.get_rotateDegree()) {
        case 90:
            meta.orientation = 6;
            break;
        case 180:
            meta.orientation = 3;
            break;
        case 270:
            meta.orientation = 8;
            break;
        default:
            meta.orientation = 1;
    }

    // as-shot white balance
    double mul[3];
    bool mulOk = true;
    for (int c = 0; c < 3; ++c) {
        mul[c] = ri.get_cam_mul(c);
        if (!(mul[c] > 0.0) || !std::isfinite(mul[c])) {
            mulOk = false;
        }
    }
    if (!mulOk) {
        for (int c = 0; c < 3; ++c) {
            mul[c] = ri.get_pre_mul(c);
            if (!(mul[c] > 0.0)) {
                mul[c] = 1.0;
            }
        }
    }
    for (int c = 0; c < 3; ++c) {
        meta.asShotNeutral[c] = mul[1] / mul[c];
    }

    // XYZ -> camera matrix, rebuilt from the camera -> sRGB matrix and the daylight multipliers
    {
        float rc[3][4];
        ri.getRgbCam(rc);
        double rgbCam[3][3], camRgb[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                rgbCam[i][j] = rc[i][j];
            }
        }
        invert3(rgbCam, camRgb);
        static const double rgbXyz[3][3] = {
            { 3.2404542, -1.5371385, -0.4985314 },
            { -0.9692660, 1.8760108, 0.0415560 },
            { 0.0556434, -0.2040259, 1.0572252 }
        };
        for (int i = 0; i < 3; ++i) {
            double pm = ri.get_pre_mul(i);
            if (!(pm > 0.0) || !std::isfinite(pm)) {
                pm = 1.0;
            }
            for (int j = 0; j < 3; ++j) {
                double v = 0.0;
                for (int k = 0; k < 3; ++k) {
                    v += camRgb[i][k] * rgbXyz[k][j];
                }
                meta.colorMatrix[i * 3 + j] = v / pm;
            }
        }
    }

    {
        std::unique_ptr<rtengine::FramesMetaData> md(rtengine::FramesMetaData::fromFile(rawFile));
        if (md) {
            meta.iso = md->getISOSpeed();
            meta.fnumber = md->getFNumber();
            meta.shutter = md->getShutterSpeed();
            meta.focal = md->getFocalLen();
            meta.lens = md->getLens();
            const tm t = md->getDateTime();
            if (t.tm_year > 0) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%04d:%02d:%02d %02d:%02d:%02d",
                         (t.tm_year + 1900) % 10000, (t.tm_mon + 1) % 100, t.tm_mday % 100, t.tm_hour % 100, t.tm_min % 100, t.tm_sec % 100);
                meta.dateTime = buf;
            }
        }
    }

    FILE* f = g_fopen(dngFile.c_str(), "wb");
    if (!f) {
        return "无法创建文件：" + dngFile;
    }
    const bool ok = AIRawCore::writeDng(f, ri.data, info, meta);
    if (fclose(f) != 0 || !ok) {
        g_remove(dngFile.c_str());
        return "写入文件失败：" + dngFile;
    }

    return Glib::ustring();
}

Glib::ustring upscaleModelPath(int model)
{
    return Glib::build_filename(App::get().argv0(), "models", model == 1 ? "upscale-bsrgan-x2.onnx" : "upscale-realplksr-x2.onnx");
}

bool upscaleAvailable(int model)
{
    return compiledIn()
           && Glib::file_test(upscaleModelPath(model), Glib::FILE_TEST_IS_REGULAR)
           && Glib::file_test(runtimePath(), Glib::FILE_TEST_IS_REGULAR);
}

Glib::ustring upscaleToTiff(rtengine::IImagefloat* img, const Glib::ustring& tifFile, int model, const std::function<void(double)>& progress)
{
    if (!img) {
        return "没有可放大的图像";
    }
    if (!upscaleAvailable(model)) {
        return "此安装中缺少 AI 放大组件或模型文件";
    }

    const std::string initErr = AIRawCore::initRuntime(nativePath(runtimePath()).c_str());
    if (!initErr.empty()) {
        return Glib::ustring("无法加载 AI 运行库：") + initErr + "\n如果文件存在，请安装 Microsoft Visual C++ 运行库。";
    }

    const int W = img->getWidth();
    const int H = img->getHeight();
    std::vector<const float*> r(H), g(H), b(H);
    for (int y = 0; y < H; ++y) {
        r[y] = img->r(y);
        g[y] = img->g(y);
        b[y] = img->b(y);
    }

    std::string icc;
    if (const rtengine::Imagefloat* imf = dynamic_cast<const rtengine::Imagefloat*>(img)) {
        icc = imf->getOutputProfile();
    }

    FILE* f = g_fopen(tifFile.c_str(), "wb");
    if (!f) {
        return "无法创建文件：" + tifFile;
    }
    const std::string err = AIRawCore::upscale2x(nativePath(upscaleModelPath(model)).c_str(), r.data(), g.data(), b.data(), W, H, f, icc, model == 1 ? "upscale-bsrgan-x2" : "upscale-realplksr-x2", progress);
    const bool closed = fclose(f) == 0;
    if (!err.empty() || !closed) {
        g_remove(tifFile.c_str());
        if (err == "out of memory") {
            return "内存不足，无法完成 AI 放大（请确认已开启系统页面文件）";
        }
        return Glib::ustring("AI 放大失败：") + (err.empty() ? std::string("cannot write the file") : err);
    }

    return Glib::ustring();
}

namespace
{

Glib::ustring samEncoderPath()
{
    return Glib::build_filename(App::get().argv0(), "models", "sam21-base-plus-encoder.onnx");
}

Glib::ustring samDecoderPath()
{
    return Glib::build_filename(App::get().argv0(), "models", "sam21-base-plus-decoder.onnx");
}

// Developed image (0..65535) -> interleaved 8 bit RGB, averaged down by an integer factor
void toRgb8(rtengine::IImagefloat* img, int maxSide, std::vector<unsigned char>& rgb, int& ow, int& oh)
{
    const int W = img->getWidth();
    const int H = img->getHeight();
    const int f = std::max(1, (std::max(W, H) + maxSide - 1) / std::max(1, maxSide));
    ow = std::max(1, W / f);
    oh = std::max(1, H / f);
    rgb.resize(static_cast<size_t>(ow) * oh * 3);
    const float norm = 255.f / (65535.f * f * f);
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            float sr = 0.f, sg = 0.f, sb = 0.f;
            for (int j = 0; j < f; ++j) {
                const int sy = std::min(H - 1, y * f + j);
                const float* rr = img->r(sy);
                const float* gg = img->g(sy);
                const float* bb = img->b(sy);
                for (int i = 0; i < f; ++i) {
                    const int sx = std::min(W - 1, x * f + i);
                    sr += std::min(65535.f, std::max(0.f, rr[sx]));
                    sg += std::min(65535.f, std::max(0.f, gg[sx]));
                    sb += std::min(65535.f, std::max(0.f, bb[sx]));
                }
            }
            unsigned char* d = rgb.data() + (static_cast<size_t>(y) * ow + x) * 3;
            d[0] = static_cast<unsigned char>(sr * norm + 0.5f);
            d[1] = static_cast<unsigned char>(sg * norm + 0.5f);
            d[2] = static_cast<unsigned char>(sb * norm + 0.5f);
        }
    }
}

inline float sampleProb(const std::vector<float>& p, int w, int h, float x, float y)
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

}

MaskRender makeRenderer(rtengine::InitialImage* image, const rtengine::procparams::ProcParams& params)
{
    const rtengine::procparams::ProcParams base = params;
    return [image, base](int x, int y, int w, int h, int maxSide, std::vector<unsigned char>& rgb, int& outW, int& outH, int& nativeW, int& nativeH) -> bool {
        rtengine::procparams::ProcParams p = base;
        p.locallab.enabled = false;
        p.resize.enabled = false;
        p.sharpening.enabled = false;
        p.prsharpening.enabled = false;
        p.crop.enabled = w > 0 && h > 0;
        if (p.crop.enabled) {
            p.crop.x = x;
            p.crop.y = y;
            p.crop.w = w;
            p.crop.h = h;
        }
        int errorCode = 0;
        rtengine::ProcessingJob* job = rtengine::ProcessingJob::create(image, p);
        rtengine::IImagefloat* img = rtengine::processImage(job, errorCode, nullptr, false);
        if (!img) {
            return false;
        }
        nativeW = img->getWidth();
        nativeH = img->getHeight();
        toRgb8(img, maxSide, rgb, outW, outH);
        delete img;
        return outW >= 16 && outH >= 16;
    };
}

bool maskAvailable()
{
    return compiledIn()
           && Glib::file_test(samEncoderPath(), Glib::FILE_TEST_IS_REGULAR)
           && Glib::file_test(samDecoderPath(), Glib::FILE_TEST_IS_REGULAR)
           && Glib::file_test(runtimePath(), Glib::FILE_TEST_IS_REGULAR);
}

Glib::ustring makeMask(const MaskRender& render, const MaskRequest& request, const Glib::ustring& maskFile, MaskResult& result, const std::function<void(double)>& progress)
{
    if (!maskAvailable()) {
        return "此安装中缺少 AI 蒙版组件或模型文件";
    }
    const std::string initErr = AIRawCore::initRuntime(nativePath(runtimePath()).c_str());
    if (!initErr.empty()) {
        return Glib::ustring("无法加载 AI 运行库：") + initErr;
    }
    const auto report = [&progress](double v) {
        if (progress) {
            progress(v);
        }
    };
    const auto enc = nativePath(samEncoderPath());
    const auto dec = nativePath(samDecoderPath());

    // ---- pass 1: the whole frame
    report(0.05);
    std::vector<unsigned char> rgb1;
    int w1 = 0, h1 = 0, W = 0, H = 0;
    if (!render(0, 0, 0, 0, 2048, rgb1, w1, h1, W, H) || W < 16 || H < 16) {
        return "显影失败";
    }
    report(0.30);
    const float k1 = static_cast<float>(w1) / W;

    AIRawCore::SegPrompt prompt;
    prompt.px = static_cast<float>(request.centerX * W) * k1;
    prompt.py = static_cast<float>(request.centerY * H) * k1;
    if (request.useBox) {
        prompt.useBox = true;
        prompt.bx0 = static_cast<float>(request.boxX0 * W) * k1;
        prompt.by0 = static_cast<float>(request.boxY0 * H) * k1;
        prompt.bx1 = static_cast<float>(request.boxX1 * W) * k1;
        prompt.by1 = static_cast<float>(request.boxY1 * H) * k1;
    }
    std::vector<float> prob1;
    float q1 = 0.f;
    std::string err = AIRawCore::samSegment(enc.c_str(), dec.c_str(), rgb1.data(), w1, h1, prompt, prob1, q1);
    if (!err.empty()) {
        AIRawCore::samRelease();
        return err == "out of memory" ? Glib::ustring("内存不足，无法生成 AI 蒙版") : Glib::ustring("AI 蒙版失败：") + err;
    }
    report(0.50);

    int bx0 = w1, by0 = h1, bx1 = -1, by1 = -1;
    for (int y = 0; y < h1; ++y) {
        const float* row = prob1.data() + static_cast<size_t>(y) * w1;
        for (int x = 0; x < w1; ++x) {
            if (row[x] > 0.5f) {
                bx0 = std::min(bx0, x);
                bx1 = std::max(bx1, x);
                by0 = std::min(by0, y);
                by1 = std::max(by1, y);
            }
        }
    }
    if (bx1 < bx0 || by1 < by0) {
        AIRawCore::samRelease();
        return "没有在这个位置识别出物体，请把点移到物体上再试";
    }

    // ---- pass 2: zoom in on the object when it is small enough to gain from it
    const double fbx0 = bx0 / k1, fby0 = by0 / k1, fbx1 = (bx1 + 1) / k1, fby1 = (by1 + 1) / k1;   // full-image pixels
    const double want = std::max(fbx1 - fbx0, fby1 - fby0) * 1.6 + 64.0;
    std::vector<unsigned char> rgb2;
    std::vector<float> prob2;
    int w2 = 0, h2 = 0, cropX = 0, cropY = 0, cropW = 0, cropH = 0;
    float k2 = 1.f, q2 = 0.f;
    bool refined = false;
    if (want < 0.7 * std::max(W, H)) {
        cropW = std::min(W, static_cast<int>(want));
        cropH = std::min(H, static_cast<int>(want));
        const double cx = 0.5 * (fbx0 + fbx1), cy = 0.5 * (fby0 + fby1);
        cropX = std::max(0, std::min(W - cropW, static_cast<int>(cx - cropW / 2.0)));
        cropY = std::max(0, std::min(H - cropH, static_cast<int>(cy - cropH / 2.0)));
        int nW = 0, nH = 0;
        if (render(cropX, cropY, cropW, cropH, 2048, rgb2, w2, h2, nW, nH) && nW > 0) {
            report(0.70);
            k2 = static_cast<float>(w2) / nW;
            AIRawCore::SegPrompt p2;
            p2.px = static_cast<float>(request.centerX * W - cropX) * k2;
            p2.py = static_cast<float>(request.centerY * H - cropY) * k2;
            p2.useBox = true;
            p2.bx0 = static_cast<float>(std::max(0.0, fbx0 - cropX)) * k2;
            p2.by0 = static_cast<float>(std::max(0.0, fby0 - cropY)) * k2;
            p2.bx1 = static_cast<float>(std::min<double>(cropW, fbx1 - cropX)) * k2;
            p2.by1 = static_cast<float>(std::min<double>(cropH, fby1 - cropY)) * k2;
            err = AIRawCore::samSegment(enc.c_str(), dec.c_str(), rgb2.data(), w2, h2, p2, prob2, q2);
            if (err.empty()) {
                AIRawCore::guidedRefine(rgb2.data(), w2, h2, prob2, std::max(2, w2 / 256), 1e-4f);
                refined = true;
            }
        }
    }
    if (!refined) {
        AIRawCore::guidedRefine(rgb1.data(), w1, h1, prob1, 4, 1e-4f);
    }
    AIRawCore::samRelease();
    report(0.85);

    // ---- assemble the mask over the full frame
    const double ms = std::min(1.0, 4096.0 / std::max(W, H));
    const int mw = std::max(2, static_cast<int>(W * ms + 0.5));
    const int mh = std::max(2, static_cast<int>(H * ms + 0.5));
    std::vector<unsigned char> mask(static_cast<size_t>(mw) * mh, 0);
    int mx0 = mw, my0 = mh, mx1 = -1, my1 = -1;
    size_t selected = 0;
    for (int y = 0; y < mh; ++y) {
        const float fy = (y + 0.5f) * H / mh;   // full-image pixel centre
        unsigned char* row = mask.data() + static_cast<size_t>(y) * mw;
        for (int x = 0; x < mw; ++x) {
            const float fx = (x + 0.5f) * W / mw;
            float v;
            if (refined) {
                if (fx < cropX || fy < cropY || fx >= cropX + cropW || fy >= cropY + cropH) {
                    v = 0.f;
                } else {
                    v = sampleProb(prob2, w2, h2, (fx - cropX) * k2 - 0.5f, (fy - cropY) * k2 - 0.5f);
                }
            } else {
                v = sampleProb(prob1, w1, h1, fx * k1 - 0.5f, fy * k1 - 0.5f);
            }
            const unsigned char u = static_cast<unsigned char>(std::min(1.f, std::max(0.f, v)) * 255.f + 0.5f);
            row[x] = u;
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
        return "没有在这个位置识别出物体，请把点移到物体上再试";
    }
    if (!rtengine::saveAiMask(maskFile, mw, mh, mask.data())) {
        return "无法写入蒙版文件：" + maskFile;
    }

    result.x0 = static_cast<double>(mx0) / mw;
    result.y0 = static_cast<double>(my0) / mh;
    result.x1 = static_cast<double>(mx1 + 1) / mw;
    result.y1 = static_cast<double>(my1 + 1) / mh;
    result.area = static_cast<double>(selected) / mask.size();
    result.quality = refined ? q2 : q1;
    result.refined = refined;

    // ---- preview: the frame with the selection tinted red
    const int pf = std::max(1, (w1 + 899) / 900);
    result.previewW = w1 / pf;
    result.previewH = h1 / pf;
    result.preview.resize(static_cast<size_t>(result.previewW) * result.previewH * 3);
    for (int y = 0; y < result.previewH; ++y) {
        for (int x = 0; x < result.previewW; ++x) {
            const unsigned char* s = rgb1.data() + (static_cast<size_t>(y) * pf * w1 + static_cast<size_t>(x) * pf) * 3;
            const int mx = std::min(mw - 1, static_cast<int>((x * pf + 0.5f) / w1 * mw));
            const int my = std::min(mh - 1, static_cast<int>((y * pf + 0.5f) / h1 * mh));
            const float a = mask[static_cast<size_t>(my) * mw + mx] * (0.6f / 255.f);
            unsigned char* d = result.preview.data() + (static_cast<size_t>(y) * result.previewW + x) * 3;
            // brighten a little so that dark photos remain readable
            d[0] = static_cast<unsigned char>(std::min(255.f, s[0] * 1.3f) * (1.f - a) + 255.f * a);
            d[1] = static_cast<unsigned char>(std::min(255.f, s[1] * 1.3f) * (1.f - a) + 40.f * a);
            d[2] = static_cast<unsigned char>(std::min(255.f, s[2] * 1.3f) * (1.f - a) + 40.f * a);
        }
    }
    report(1.0);
    return Glib::ustring();
}

}
