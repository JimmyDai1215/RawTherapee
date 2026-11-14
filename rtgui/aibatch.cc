#include "aibatch.h"

#include <atomic>
#include <iomanip>
#include <memory>
#include <mutex>
#include <thread>

#include <glib/gstdio.h>
#include <gtkmm.h>

#include "ainoise.h"
#include "multilangmgr.h"
#include "rtengine/rtengine.h"

namespace
{

double batchStrength = 100.0;
int batchModel = 1;
bool batchNoSharpen = true;
bool batchGpu = true;

struct Shared {
    std::mutex mutex;
    size_t index = 0;            // photo being worked on
    double fraction = 0.0;       // progress within that photo
    bool finished = false;
    std::vector<Glib::ustring> done;      // output files
    std::vector<Glib::ustring> failed;    // "file: reason"
    std::atomic<bool> cancel { false };
};

Glib::ustring freeName(const Glib::ustring& base, const Glib::ustring& suffix, const Glib::ustring& ext)
{
    Glib::ustring out = base + suffix + ext;
    for (int i = 2; Glib::file_test(out, Glib::FILE_TEST_EXISTS) && i < 1000; ++i) {
        out = Glib::ustring::compose("%1%2-%3%4", base, suffix, i, ext);
    }
    return out;
}

Glib::ustring baseName(const Glib::ustring& file)
{
    const Glib::ustring::size_type dot = file.find_last_of('.');
    const Glib::ustring::size_type sep = file.find_last_of("/\\");
    if (dot == Glib::ustring::npos || (sep != Glib::ustring::npos && dot < sep)) {
        return file;
    }
    return file.substr(0, dot);
}

class Progress final : public rtengine::ProgressListener
{
public:
    explicit Progress(Shared& s) : s_(s) {}
    void setProgress(double p) override
    {
        std::lock_guard<std::mutex> lock(s_.mutex);
        s_.fraction = 0.3 * p;   // developing is the first part of an upscale
    }
    void setProgressStr(const Glib::ustring&) override {}
    void setProgressState(bool) override {}
    void error(const Glib::ustring&) override {}

private:
    Shared& s_;
};

void work(std::vector<AIBatch::Item> items, AIBatch::Mode mode, Shared* s, Glib::Dispatcher* notify)
{
    for (size_t i = 0; i < items.size() && !s->cancel; ++i) {
        {
            std::lock_guard<std::mutex> lock(s->mutex);
            s->index = i;
            s->fraction = 0.0;
        }
        notify->emit();
        const AIBatch::Item& it = items[i];
        const Glib::ustring base = baseName(it.file);
        Glib::ustring out, msg;

        if (mode == AIBatch::Mode::DENOISE) {
            out = freeName(base, "_AI", ".dng");
            msg = AIDenoise::rawToDng(it.file, out, static_cast<float>(batchStrength / 100.0), [s, notify](double p) {
                {
                    std::lock_guard<std::mutex> lock(s->mutex);
                    s->fraction = p;
                }
                notify->emit();
            });
            if (msg.empty() && it.hasParams) {
                rtengine::procparams::ProcParams copy = it.params;
                copy.save(out + ".pp3");   // carry the edits over to the new raw file
            }
        } else {
            out = freeName(base, "_2x", ".tif");
            int err = 0;
            rtengine::InitialImage* ii = rtengine::InitialImage::load(it.file, true, &err, nullptr);
            if (!ii) {
                // not a raw file: try it as an ordinary image
                ii = rtengine::InitialImage::load(it.file, false, &err, nullptr);
            }
            if (!ii) {
                msg = "无法读取文件";
            } else {
                rtengine::procparams::ProcParams params = it.params;
                if (batchNoSharpen) {
                    params.sharpening.enabled = false;
                    params.prsharpening.enabled = false;
                    params.pdsharpening.enabled = false;
                }
                params.resize.enabled = false;
                Progress pl(*s);
                rtengine::ProcessingJob* job = rtengine::ProcessingJob::create(ii, params);
                rtengine::IImagefloat* img = rtengine::processImage(job, err, &pl, false);
                if (!img) {
                    msg = "显影失败";
                } else {
                    msg = AIDenoise::upscaleToTiff(img, out, batchModel, [s, notify](double p) {
                        {
                            std::lock_guard<std::mutex> lock(s->mutex);
                            s->fraction = 0.3 + 0.7 * p;
                        }
                        notify->emit();
                    }, it.file);
                    delete img;
                }
                ii->decreaseRef();
            }
        }

        {
            std::lock_guard<std::mutex> lock(s->mutex);
            if (msg.empty()) {
                s->done.push_back(out);
            } else {
                s->failed.push_back(Glib::path_get_basename(it.file) + "：" + msg);
            }
        }
        notify->emit();
    }
    {
        std::lock_guard<std::mutex> lock(s->mutex);
        s->finished = true;
    }
    notify->emit();
}

}

namespace AIBatch
{

void run(Gtk::Window& parent, const std::vector<Item>& items, Mode mode)
{
    if (items.empty()) {
        return;
    }
    const bool denoise = mode == Mode::DENOISE;
    const Glib::ustring title = M(denoise ? "MAIN_BUTTON_AIDENOISE" : "MAIN_BUTTON_AIUPSCALE");
    if (denoise ? !AIDenoise::available() : !(AIDenoise::upscaleAvailable(0) && AIDenoise::upscaleAvailable(1))) {
        Gtk::MessageDialog msgd(parent, M("MAIN_MSG_AIDENOISE_MISSING"), false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
        msgd.run();
        return;
    }

    // ---- settings
    {
        Gtk::Dialog dlg(title, parent, true);
        Gtk::Label lbl(Glib::ustring::compose(M(denoise ? "AIBATCH_DENOISE_INTRO" : "AIBATCH_UPSCALE_INTRO"), Glib::ustring::format(items.size())));
        lbl.set_line_wrap(true);
        lbl.set_max_width_chars(60);
        lbl.set_xalign(0.0);
        Gtk::Scale strength(Gtk::ORIENTATION_HORIZONTAL);
        strength.set_range(0.0, 100.0);
        strength.set_digits(0);
        strength.set_value(batchStrength);
        Gtk::Label strengthLbl(M("MAIN_MSG_AIDENOISE_STRENGTH"));
        strengthLbl.set_xalign(0.0);
        Gtk::ComboBoxText modelBox;
        modelBox.append(M("MAIN_MSG_AIUPSCALE_MODEL_FAITHFUL"));
        modelBox.append(M("MAIN_MSG_AIUPSCALE_MODEL_CLEAN"));
        modelBox.set_active(batchModel == 0 ? 0 : 1);
        Gtk::CheckButton sharpChk(M("MAIN_MSG_AIUPSCALE_NOSHARPEN"));
        sharpChk.set_active(batchNoSharpen);
        Gtk::CheckButton gpuChk(M("MAIN_MSG_AI_USEGPU"));
        gpuChk.set_active(batchGpu);
        Gtk::Box* const box = dlg.get_content_area();
        box->set_spacing(8);
        box->set_border_width(12);
        box->pack_start(lbl, Gtk::PACK_SHRINK);
        if (denoise) {
            box->pack_start(strengthLbl, Gtk::PACK_SHRINK);
            box->pack_start(strength, Gtk::PACK_SHRINK);
        } else {
            box->pack_start(modelBox, Gtk::PACK_SHRINK);
            box->pack_start(sharpChk, Gtk::PACK_SHRINK);
        }
        box->pack_start(gpuChk, Gtk::PACK_SHRINK);
        dlg.add_button(M("GENERAL_CANCEL"), Gtk::RESPONSE_CANCEL);
        dlg.add_button(M("GENERAL_OK"), Gtk::RESPONSE_OK);
        dlg.set_default_response(Gtk::RESPONSE_OK);
        dlg.show_all();
        if (dlg.run() != Gtk::RESPONSE_OK) {
            return;
        }
        batchStrength = strength.get_value();
        batchModel = modelBox.get_active_row_number() == 0 ? 0 : 1;
        batchNoSharpen = sharpChk.get_active();
        batchGpu = gpuChk.get_active();
        AIDenoise::setGpuAllowed(batchGpu);
    }

    // ---- work, with a progress window
    Shared shared;
    Glib::Dispatcher notify;
    Gtk::Dialog dlg(title, parent, true);
    dlg.set_deletable(false);
    Gtk::Label what;
    what.set_xalign(0.0);
    Gtk::ProgressBar bar;
    bar.set_size_request(420, -1);
    dlg.get_content_area()->set_spacing(8);
    dlg.get_content_area()->set_border_width(12);
    dlg.get_content_area()->pack_start(what, Gtk::PACK_SHRINK);
    dlg.get_content_area()->pack_start(bar, Gtk::PACK_SHRINK);
    Gtk::Button* const cancelBtn = dlg.add_button(M("AIBATCH_STOP"), Gtk::RESPONSE_CANCEL);
    const size_t total = items.size();

    notify.connect([&]() {
        std::lock_guard<std::mutex> lock(shared.mutex);
        if (shared.finished) {
            dlg.response(Gtk::RESPONSE_OK);
            return;
        }
        const size_t i = std::min(shared.index, total - 1);
        what.set_text(Glib::ustring::compose("%1 / %2　%3", Glib::ustring::format(i + 1), Glib::ustring::format(total), Glib::path_get_basename(items[i].file))
                      + (shared.cancel ? Glib::ustring("　") + M("AIBATCH_STOPPING") : Glib::ustring()));
        bar.set_fraction(std::min(1.0, (i + std::min(1.0, std::max(0.0, shared.fraction))) / total));
    });
    cancelBtn->signal_clicked().connect([&]() {
        shared.cancel = true;
        cancelBtn->set_sensitive(false);
    });
    // "stop" only asks the worker to stop after the current photo; the window closes when it has
    dlg.signal_response().connect([&](int response) {
        if (response != Gtk::RESPONSE_OK) {
            dlg.signal_response().emission_stop();
        }
    }, false);

    std::thread worker(work, items, mode, &shared, &notify);
    dlg.show_all();
    while (true) {
        const int r = dlg.run();
        std::lock_guard<std::mutex> lock(shared.mutex);
        if (r == Gtk::RESPONSE_OK || shared.finished) {
            break;
        }
        shared.cancel = true;
    }
    worker.join();
    dlg.hide();

    // ---- summary
    Glib::ustring text = Glib::ustring::compose(M("AIBATCH_DONE"), Glib::ustring::format(shared.done.size()), Glib::ustring::format(total));
    if (!shared.done.empty()) {
        text += Glib::ustring("\n") + Glib::ustring::compose(M("AIBATCH_FOLDER"), Glib::path_get_dirname(shared.done.front()));
    }
    for (size_t i = 0; i < shared.failed.size() && i < 8; ++i) {
        text += "\n" + shared.failed[i];
    }
    if (AIDenoise::lastDevice() == "GPU" || !shared.done.empty()) {
        text += Glib::ustring("\n\n") + Glib::ustring::compose(M("MAIN_MSG_AI_DEVICE"), AIDenoise::lastDevice() == "GPU" ? M("MAIN_MSG_AI_DEVICE_GPU") : M("MAIN_MSG_AI_DEVICE_CPU"));
    }
    Gtk::MessageDialog msgd(parent, text, false, shared.failed.empty() ? Gtk::MESSAGE_INFO : Gtk::MESSAGE_WARNING, Gtk::BUTTONS_OK, true);
    msgd.run();
}

}
