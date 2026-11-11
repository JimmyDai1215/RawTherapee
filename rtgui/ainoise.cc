#include "ainoise.h"

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
#include "rtengine/imagefloat.h"
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

}
