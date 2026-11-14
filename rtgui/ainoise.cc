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

namespace
{

Glib::ustring maskError(const std::string& err)
{
    if (err.empty()) {
        return Glib::ustring();
    }
    if (err == "nothing") {
        return "没有在这个位置识别出物体，请换个位置再试";
    }
    if (err == "out of memory") {
        return "内存不足，无法生成 AI 蒙版";
    }
    return Glib::ustring("AI 蒙版失败：") + err;
}

std::vector<AIRawCore::SelectPoint> toCore(const std::vector<MaskPoint>& points)
{
    std::vector<AIRawCore::SelectPoint> res;
    for (const MaskPoint& p : points) {
        res.push_back({ p.x, p.y, p.add });
    }
    return res;
}

Glib::ustring storeMask(AIRawCore::SelectResult& sel, const std::vector<MaskPoint>& points, const Glib::ustring& maskFile, MaskResult& result)
{
    if (!rtengine::saveAiMask(maskFile, sel.w, sel.h, sel.mask.data())) {
        return "无法写入蒙版文件：" + maskFile;
    }
    // the points, so that the selection can be edited again later
    std::string text;
    for (const MaskPoint& p : points) {
        char bx[G_ASCII_DTOSTR_BUF_SIZE], by[G_ASCII_DTOSTR_BUF_SIZE];
        g_ascii_formatd(bx, sizeof(bx), "%.6f", p.x);
        g_ascii_formatd(by, sizeof(by), "%.6f", p.y);
        text += std::string(bx) + " " + by + " " + (p.add ? "1" : "0") + "\n";
    }
    g_file_set_contents((maskFile + ".pts").c_str(), text.c_str(), text.size(), nullptr);

    result.x0 = sel.x0;
    result.y0 = sel.y0;
    result.x1 = sel.x1;
    result.y1 = sel.y1;
    result.area = sel.area;
    result.quality = sel.quality;
    result.refined = sel.zoomed || sel.tiles > 0;
    result.preview.swap(sel.preview);
    result.previewW = sel.previewW;
    result.previewH = sel.previewH;
    return Glib::ustring();
}

}

std::vector<MaskPoint> loadMaskPoints(const Glib::ustring& maskFile)
{
    std::vector<MaskPoint> res;
    gchar* text = nullptr;
    if (maskFile.empty() || !g_file_get_contents((maskFile + ".pts").c_str(), &text, nullptr, nullptr) || !text) {
        return res;
    }
    const char* p = text;
    while (*p) {
        char* end = nullptr;
        const double x = g_ascii_strtod(p, &end);
        if (end == p) {
            break;
        }
        p = end;
        const double y = g_ascii_strtod(p, &end);
        if (end == p) {
            break;
        }
        p = end;
        const double a = g_ascii_strtod(p, &end);
        if (end == p) {
            break;
        }
        p = end;
        if (x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0) {
            res.push_back({ x, y, a > 0.5 });
        }
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
            ++p;
        }
    }
    g_free(text);
    return res;
}

void setMaskOverlay(bool show)
{
    rtengine::aiMaskOverlayEnabled = show;
}

bool maskOverlay()
{
    return rtengine::aiMaskOverlayEnabled.load();
}

Glib::ustring maskBegin(const MaskRender& render, const std::function<void(double)>& progress)
{
    if (!maskAvailable()) {
        return "此安装中缺少 AI 蒙版组件或模型文件";
    }
    const std::string initErr = AIRawCore::initRuntime(nativePath(runtimePath()).c_str());
    if (!initErr.empty()) {
        return Glib::ustring("无法加载 AI 运行库：") + initErr;
    }
    const std::string err = AIRawCore::selectBegin(nativePath(samEncoderPath()).c_str(), nativePath(samDecoderPath()).c_str(), render, progress);
    if (err == "cannot develop the image") {
        return "显影失败";
    }
    return maskError(err);
}

Glib::ustring maskPreview(const std::vector<MaskPoint>& points, int maxW, int maxH, std::vector<unsigned char>& rgb, int& w, int& h, double& area)
{
    return maskError(AIRawCore::selectPreview(toCore(points), maxW, maxH, rgb, w, h, area));
}

Glib::ustring maskFinish(const std::vector<MaskPoint>& points, const Glib::ustring& maskFile, MaskResult& result, const std::function<void(double)>& progress)
{
    AIRawCore::SelectResult sel;
    const std::string err = AIRawCore::selectFinish(toCore(points), nullptr, sel, progress);
    if (!err.empty()) {
        return maskError(err);
    }
    return storeMask(sel, points, maskFile, result);
}

void maskEnd()
{
    AIRawCore::selectEnd();
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

    AIRawCore::SelectRequest req;
    req.cx = request.centerX;
    req.cy = request.centerY;
    req.useBox = request.useBox;
    req.bx0 = request.boxX0;
    req.by0 = request.boxY0;
    req.bx1 = request.boxX1;
    req.by1 = request.boxY1;
    req.more = toCore(request.more);

    AIRawCore::SelectResult sel;
    const std::string err = AIRawCore::selectObject(nativePath(samEncoderPath()).c_str(), nativePath(samDecoderPath()).c_str(), render, req, sel, progress);
    if (!err.empty()) {
        return maskError(err);
    }
    std::vector<MaskPoint> points;
    points.push_back({ request.centerX, request.centerY, true });
    points.insert(points.end(), request.more.begin(), request.more.end());
    return storeMask(sel, points, maskFile, result);
}

}
