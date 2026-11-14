#include "aimaskeditor.h"

#include <algorithm>
#include <cmath>

#include "ainoise.h"
#include "multilangmgr.h"

namespace
{
const double ZOOM_MAX = 16.0;
}

AiMaskEditor::AiMaskEditor(Gtk::Window& parent, const AIRawCore::MaskEdit& initial) :
    Gtk::Dialog(M("MASKEDIT_TITLE"), parent, true),
    edit_(initial),
    aiBtn_(M("MASKEDIT_TOOL_AI")),
    brushBtn_(M("MASKEDIT_TOOL_BRUSH")),
    polyBtn_(M("MASKEDIT_TOOL_POLYGON")),
    gradBtn_(M("MASKEDIT_TOOL_GRADIENT")),
    sizeLbl_(M("MASKEDIT_BRUSHSIZE")),
    featherLbl_(M("MASKEDIT_FEATHER")),
    growLbl_(M("MASKEDIT_GROW")),
    lumaLbl_(M("MASKEDIT_LUMA")),
    size_(Gtk::ORIENTATION_HORIZONTAL),
    feather_(Gtk::ORIENTATION_HORIZONTAL),
    grow_(Gtk::ORIENTATION_HORIZONTAL),
    lumaLo_(Gtk::ORIENTATION_HORIZONTAL),
    lumaHi_(Gtk::ORIENTATION_HORIZONTAL),
    invert_(M("MASKEDIT_INVERT")),
    undo_(M("MASKEDIT_UNDO")),
    reset_(M("MASKEDIT_RESET")),
    fit_(M("MASKEDIT_FIT"))
{
    set_default_size(std::max(900, parent.get_width() - 60), std::max(620, parent.get_height() - 60));

    // tools: radio buttons drawn as toggle buttons
    Gtk::RadioButton::Group group = aiBtn_.get_group();
    brushBtn_.set_group(group);
    polyBtn_.set_group(group);
    gradBtn_.set_group(group);
    for (Gtk::RadioButton* b : { &aiBtn_, &brushBtn_, &polyBtn_, &gradBtn_ }) {
        b->set_mode(false);
        b->signal_toggled().connect(sigc::mem_fun(*this, &AiMaskEditor::onToolChanged));
        tools_.pack_start(*b, Gtk::PACK_SHRINK);
    }
    tools_.set_spacing(4);
    aiBtn_.set_tooltip_text(M("MASKEDIT_TOOL_AI_TOOLTIP"));
    brushBtn_.set_tooltip_text(M("MASKEDIT_TOOL_BRUSH_TOOLTIP"));
    polyBtn_.set_tooltip_text(M("MASKEDIT_TOOL_POLYGON_TOOLTIP"));
    gradBtn_.set_tooltip_text(M("MASKEDIT_TOOL_GRADIENT_TOOLTIP"));

    size_.set_range(1.0, 150.0);
    size_.set_value(20.0);
    feather_.set_range(0.0, 100.0);
    grow_.set_range(-50.0, 50.0);
    lumaLo_.set_range(0.0, 100.0);
    lumaHi_.set_range(0.0, 100.0);
    lumaHi_.set_value(100.0);
    lumaLbl_.set_tooltip_text(M("MASKEDIT_LUMA_TOOLTIP"));
    lumaLo_.set_tooltip_text(M("MASKEDIT_LUMA_TOOLTIP"));
    lumaHi_.set_tooltip_text(M("MASKEDIT_LUMA_TOOLTIP"));
    for (Gtk::Scale* s : { &size_, &feather_, &grow_, &lumaLo_, &lumaHi_ }) {
        s->set_digits(0);
        s->set_size_request(150, -1);
        s->set_value_pos(Gtk::POS_RIGHT);
        s->signal_value_changed().connect(sigc::mem_fun(*this, &AiMaskEditor::onSliderChanged));
    }
    invert_.signal_toggled().connect(sigc::mem_fun(*this, &AiMaskEditor::onSliderChanged));
    sliders_.set_spacing(6);
    sliders_.pack_start(sizeLbl_, Gtk::PACK_SHRINK);
    sliders_.pack_start(size_, Gtk::PACK_SHRINK);
    sliders_.pack_start(featherLbl_, Gtk::PACK_SHRINK);
    sliders_.pack_start(feather_, Gtk::PACK_SHRINK);
    sliders_.pack_start(growLbl_, Gtk::PACK_SHRINK);
    sliders_.pack_start(grow_, Gtk::PACK_SHRINK);
    sliders_.pack_start(invert_, Gtk::PACK_SHRINK);
    tools_.pack_end(sliders_, Gtk::PACK_SHRINK);

    help_.set_xalign(0.0);
    help_.set_line_wrap(true);
    second_.set_spacing(6);
    second_.pack_start(help_, Gtk::PACK_EXPAND_WIDGET);
    second_.pack_end(lumaHi_, Gtk::PACK_SHRINK);
    second_.pack_end(lumaLo_, Gtk::PACK_SHRINK);
    second_.pack_end(lumaLbl_, Gtk::PACK_SHRINK);
    info_.set_xalign(1.0);

    area_.set_hexpand(true);
    area_.set_vexpand(true);
    area_.add_events(Gdk::BUTTON_PRESS_MASK | Gdk::BUTTON_RELEASE_MASK | Gdk::POINTER_MOTION_MASK | Gdk::SCROLL_MASK | Gdk::LEAVE_NOTIFY_MASK);
    area_.signal_draw().connect(sigc::mem_fun(*this, &AiMaskEditor::onDraw));
    area_.signal_button_press_event().connect(sigc::mem_fun(*this, &AiMaskEditor::onPress));
    area_.signal_button_release_event().connect(sigc::mem_fun(*this, &AiMaskEditor::onRelease));
    area_.signal_motion_notify_event().connect(sigc::mem_fun(*this, &AiMaskEditor::onMotion));
    area_.signal_scroll_event().connect(sigc::mem_fun(*this, &AiMaskEditor::onScroll));
    area_.signal_leave_notify_event().connect([this](GdkEventCrossing*) -> bool {
        mouseIn_ = false;
        area_.queue_draw();
        return false;
    });

    undo_.signal_clicked().connect(sigc::mem_fun(*this, &AiMaskEditor::onUndo));
    reset_.signal_clicked().connect(sigc::mem_fun(*this, &AiMaskEditor::onReset));
    fit_.signal_clicked().connect(sigc::mem_fun(*this, &AiMaskEditor::onFit));
    bottom_.set_spacing(6);
    bottom_.pack_start(undo_, Gtk::PACK_SHRINK);
    bottom_.pack_start(reset_, Gtk::PACK_SHRINK);
    bottom_.pack_start(fit_, Gtk::PACK_SHRINK);
    bottom_.pack_start(info_, Gtk::PACK_EXPAND_WIDGET);

    get_content_area()->set_spacing(6);
    get_content_area()->set_border_width(10);
    get_content_area()->pack_start(tools_, Gtk::PACK_SHRINK);
    get_content_area()->pack_start(second_, Gtk::PACK_SHRINK);
    get_content_area()->pack_start(area_, Gtk::PACK_EXPAND_WIDGET);
    get_content_area()->pack_start(bottom_, Gtk::PACK_SHRINK);
    add_button(M("GENERAL_CANCEL"), Gtk::RESPONSE_CANCEL);
    add_button(M("MASKEDIT_APPLY"), Gtk::RESPONSE_OK);

    syncWidgets();
    onToolChanged();
    show_all();
}

void AiMaskEditor::toImage(double mx, double my, double& nx, double& ny) const
{
    nx = std::min(1.0, std::max(0.0, (mx - offX_) / std::max(1e-6, scaleX_)));
    ny = std::min(1.0, std::max(0.0, (my - offY_) / std::max(1e-6, scaleY_)));
}

void AiMaskEditor::pushHistory()
{
    history_.push_back(edit_);
    if (history_.size() > 60) {
        history_.erase(history_.begin());
    }
}

void AiMaskEditor::syncWidgets()
{
    syncing_ = true;
    feather_.set_value(edit_.feather * 1000.0 * 5.0);   // slider 100 = 2 % of the width
    grow_.set_value(edit_.grow * 1000.0 * 5.0);         // slider 50 = 1 % of the width
    invert_.set_active(edit_.invert);
    lumaLo_.set_value(edit_.lumaLo * 100.0);
    lumaHi_.set_value(edit_.lumaHi * 100.0);
    syncing_ = false;
}

void AiMaskEditor::changed()
{
    dirty_ = true;
    updateInfo();
    area_.queue_draw();
}

void AiMaskEditor::updateInfo()
{
    info_.set_text(Glib::ustring::compose(M("MASKEDIT_INFO"), Glib::ustring::format(edit_.points.size()), Glib::ustring::format(edit_.ops.size()),
                                          Glib::ustring::format(static_cast<int>(zoom_ * 100.0 + 0.5))));
}

void AiMaskEditor::onToolChanged()
{
    const Tool before = tool_;
    tool_ = brushBtn_.get_active() ? TOOL_BRUSH : polyBtn_.get_active() ? TOOL_POLYGON : gradBtn_.get_active() ? TOOL_GRADIENT : TOOL_AI;
    if (before == TOOL_POLYGON && tool_ != TOOL_POLYGON) {
        pending_.clear();
    }
    switch (tool_) {
        case TOOL_BRUSH:
            help_.set_text(M("MASKEDIT_HELP_BRUSH"));
            break;
        case TOOL_POLYGON:
            help_.set_text(M("MASKEDIT_HELP_POLYGON"));
            break;
        case TOOL_GRADIENT:
            help_.set_text(M("MASKEDIT_HELP_GRADIENT"));
            break;
        default:
            help_.set_text(M("MASKEDIT_HELP_AI"));
    }
    updateInfo();
    area_.queue_draw();
}

void AiMaskEditor::onSliderChanged()
{
    if (syncing_) {
        return;
    }
    const double f = feather_.get_value() / 5000.0;
    const double g = grow_.get_value() / 5000.0;
    const bool inv = invert_.get_active();
    // the two ends of the brightness range push each other along
    if (lumaLo_.get_value() > lumaHi_.get_value()) {
        syncing_ = true;
        if (edit_.lumaLo * 100.0 != lumaLo_.get_value()) {
            lumaHi_.set_value(lumaLo_.get_value());
        } else {
            lumaLo_.set_value(lumaHi_.get_value());
        }
        syncing_ = false;
    }
    const double lo = lumaLo_.get_value() / 100.0, hi = lumaHi_.get_value() / 100.0;
    if (f != edit_.feather || g != edit_.grow || inv != edit_.invert || lo != edit_.lumaLo || hi != edit_.lumaHi) {
        edit_.lumaLo = lo;
        edit_.lumaHi = hi;
        edit_.feather = f;
        edit_.grow = g;
        edit_.invert = inv;
        changed();
    } else {
        area_.queue_draw();   // brush size: only the cursor changes
    }
}

void AiMaskEditor::onUndo()
{
    if (!pending_.empty()) {
        pending_.resize(pending_.size() - 2);
        area_.queue_draw();
        return;
    }
    if (history_.empty()) {
        return;
    }
    edit_ = history_.back();
    history_.pop_back();
    syncWidgets();
    changed();
}

void AiMaskEditor::onReset()
{
    pushHistory();
    pending_.clear();
    edit_ = AIRawCore::MaskEdit();
    syncWidgets();
    changed();
}

void AiMaskEditor::onFit()
{
    cx_ = cy_ = 0.5;
    zoom_ = 1.0;
    changed();
}

void AiMaskEditor::closePolygon(bool add)
{
    if (pending_.size() >= 6) {
        pushHistory();
        AIRawCore::MaskOp op;
        op.type = 1;
        op.add = add;
        op.xy = pending_;
        edit_.ops.push_back(op);
    }
    pending_.clear();
    changed();
}

bool AiMaskEditor::onPress(GdkEventButton* ev)
{
    lastX_ = ev->x;
    lastY_ = ev->y;
    // moving the picture: middle button, or Ctrl + left button
    if (ev->button == 2 || (ev->button == 1 && (ev->state & GDK_CONTROL_MASK))) {
        drag_ = 3;
        return true;
    }
    if (ev->button != 1 && ev->button != 3) {
        return true;
    }
    const bool add = ev->button == 1 && !(ev->state & GDK_SHIFT_MASK);
    double nx, ny;
    toImage(ev->x, ev->y, nx, ny);

    switch (tool_) {
        case TOOL_AI:
            if (ev->type == GDK_BUTTON_PRESS) {
                pushHistory();
                edit_.points.push_back({ nx, ny, add });
                changed();
            }
            break;

        case TOOL_BRUSH:
            if (ev->type == GDK_BUTTON_PRESS) {
                pushHistory();
                AIRawCore::MaskOp op;
                op.type = 0;
                op.add = add;
                // the slider is the radius in thousandths of the image width
                op.radius = size_.get_value() / 1000.0;
                op.xy = { nx, ny };
                edit_.ops.push_back(op);
                drag_ = 1;
                changed();
            }
            break;

        case TOOL_POLYGON:
            if (ev->button == 3) {
                closePolygon(false);
            } else if (ev->type == GDK_2BUTTON_PRESS) {
                closePolygon(true);
            } else if (ev->type == GDK_BUTTON_PRESS) {
                pending_.push_back(nx);
                pending_.push_back(ny);
                area_.queue_draw();
            }
            break;

        case TOOL_GRADIENT:
            if (ev->type == GDK_BUTTON_PRESS) {
                pushHistory();
                AIRawCore::MaskOp op;
                op.type = 2;
                op.add = add;
                op.xy = { nx, ny, nx, ny };
                edit_.ops.push_back(op);
                drag_ = 2;
            }
            break;
    }
    return true;
}

bool AiMaskEditor::onMotion(GdkEventMotion* ev)
{
    mouseIn_ = true;
    const double dx = ev->x - lastX_, dy = ev->y - lastY_;
    double nx, ny;
    toImage(ev->x, ev->y, nx, ny);

    if (drag_ == 3) {
        cx_ = std::min(1.0, std::max(0.0, cx_ - dx / std::max(1e-6, scaleX_)));
        cy_ = std::min(1.0, std::max(0.0, cy_ - dy / std::max(1e-6, scaleY_)));
        lastX_ = ev->x;
        lastY_ = ev->y;
        changed();
    } else if (drag_ == 1 && !edit_.ops.empty()) {
        if (dx * dx + dy * dy >= 2.25) {
            edit_.ops.back().xy.push_back(nx);
            edit_.ops.back().xy.push_back(ny);
            lastX_ = ev->x;
            lastY_ = ev->y;
            changed();
        }
    } else if (drag_ == 2 && !edit_.ops.empty()) {
        edit_.ops.back().xy[2] = nx;
        edit_.ops.back().xy[3] = ny;
        lastX_ = ev->x;
        lastY_ = ev->y;
        changed();
    } else {
        lastX_ = ev->x;
        lastY_ = ev->y;
        if (tool_ == TOOL_BRUSH || !pending_.empty()) {
            area_.queue_draw();   // cursor circle / rubber band
        }
    }
    return true;
}

bool AiMaskEditor::onRelease(GdkEventButton*)
{
    if (drag_ == 2 && !edit_.ops.empty()) {
        // a gradient that was not dragged out is dropped again
        const std::vector<double>& xy = edit_.ops.back().xy;
        const double px = (xy[2] - xy[0]) * scaleX_, py = (xy[3] - xy[1]) * scaleY_;
        if (px * px + py * py < 25.0) {
            edit_.ops.pop_back();
            if (!history_.empty()) {
                history_.pop_back();
            }
        }
        changed();
    }
    drag_ = 0;
    return true;
}

bool AiMaskEditor::onScroll(GdkEventScroll* ev)
{
    double factor = 1.0;
    if (ev->direction == GDK_SCROLL_UP) {
        factor = 1.25;
    } else if (ev->direction == GDK_SCROLL_DOWN) {
        factor = 0.8;
    } else if (ev->direction == GDK_SCROLL_SMOOTH) {
        factor = ev->delta_y < 0 ? 1.25 : ev->delta_y > 0 ? 0.8 : 1.0;
    }
    const double newZoom = std::min(ZOOM_MAX, std::max(1.0, zoom_ * factor));
    if (newZoom == zoom_) {
        return true;
    }
    // keep the image point under the mouse where it is
    const double nx = (ev->x - offX_) / std::max(1e-6, scaleX_);
    const double ny = (ev->y - offY_) / std::max(1e-6, scaleY_);
    const double ratio = newZoom / zoom_;
    const double sx = scaleX_ * ratio, sy = scaleY_ * ratio;
    zoom_ = newZoom;
    cx_ = std::min(1.0, std::max(0.0, (viewW_ / 2.0 - ev->x) / sx + nx));
    cy_ = std::min(1.0, std::max(0.0, (viewH_ / 2.0 - ev->y) / sy + ny));
    if (zoom_ <= 1.0) {
        cx_ = cy_ = 0.5;
    }
    changed();
    return true;
}

bool AiMaskEditor::onDraw(const Cairo::RefPtr<Cairo::Context>& cr)
{
    const int w = area_.get_allocated_width();
    const int h = area_.get_allocated_height();
    if (w < 16 || h < 16) {
        return true;
    }
    if (dirty_ || w != viewW_ || h != viewH_ || !pix_) {
        viewW_ = w;
        viewH_ = h;
        error_ = AIDenoise::maskView(edit_, cx_, cy_, zoom_, w, h, rgb_, scaleX_, scaleY_, offX_, offY_);
        if (error_.empty()) {
            // copy(): the pixbuf must not depend on rgb_, which is reused
            pix_ = Gdk::Pixbuf::create_from_data(rgb_.data(), Gdk::COLORSPACE_RGB, false, 8, w, h, w * 3)->copy();
        }
        dirty_ = false;
    }
    if (pix_) {
        Gdk::Cairo::set_source_pixbuf(cr, pix_, 0, 0);
        cr->paint();
    }
    if (!error_.empty()) {
        cr->set_source_rgb(1.0, 0.9, 0.3);
        cr->move_to(12, 24);
        cr->show_text(error_);
    }

    // polygon being drawn, with a line to the mouse
    if (!pending_.empty()) {
        cr->set_line_width(1.5);
        cr->set_source_rgb(1.0, 1.0, 0.2);
        for (size_t i = 0; i + 1 < pending_.size(); i += 2) {
            const double x = pending_[i] * scaleX_ + offX_, y = pending_[i + 1] * scaleY_ + offY_;
            if (i == 0) {
                cr->move_to(x, y);
            } else {
                cr->line_to(x, y);
            }
        }
        if (mouseIn_) {
            cr->line_to(lastX_, lastY_);
        }
        cr->stroke();
        for (size_t i = 0; i + 1 < pending_.size(); i += 2) {
            cr->arc(pending_[i] * scaleX_ + offX_, pending_[i + 1] * scaleY_ + offY_, 3.0, 0.0, 2.0 * G_PI);
            cr->fill();
        }
    }
    // gradient being dragged: its axis
    if (drag_ == 2 && !edit_.ops.empty()) {
        const std::vector<double>& xy = edit_.ops.back().xy;
        cr->set_line_width(1.5);
        cr->set_source_rgb(1.0, 1.0, 0.2);
        cr->move_to(xy[0] * scaleX_ + offX_, xy[1] * scaleY_ + offY_);
        cr->line_to(xy[2] * scaleX_ + offX_, xy[3] * scaleY_ + offY_);
        cr->stroke();
    }
    // brush outline at the mouse
    if (tool_ == TOOL_BRUSH && mouseIn_) {
        const double r = std::max(1.0, size_.get_value() / 1000.0 * scaleX_);
        cr->set_line_width(1.0);
        cr->set_source_rgb(0.0, 0.0, 0.0);
        cr->arc(lastX_, lastY_, r + 1.0, 0.0, 2.0 * G_PI);
        cr->stroke();
        cr->set_source_rgb(1.0, 1.0, 1.0);
        cr->arc(lastX_, lastY_, r, 0.0, 2.0 * G_PI);
        cr->stroke();
    }
    return true;
}
