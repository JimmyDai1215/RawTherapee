/*
 *  Mask editor: the photo with the mask in red; the mask is built from AI selections
 *  (clicks), brush strokes, polygons and gradients, then grown/shrunk, feathered or inverted.
 *  Part of the personal Capture-One-like fork; GPL-3.0-or-later like RawTherapee.
 */
#pragma once

#include <vector>
#include <gtkmm.h>

#include "airawcore.h"

class AiMaskEditor final : public Gtk::Dialog
{
public:
    AiMaskEditor(Gtk::Window& parent, const AIRawCore::MaskEdit& initial);

    const AIRawCore::MaskEdit& edit() const
    {
        return edit_;
    }

private:
    enum Tool { TOOL_AI = 0, TOOL_BRUSH, TOOL_POLYGON, TOOL_GRADIENT };

    bool onDraw(const Cairo::RefPtr<Cairo::Context>& cr);
    bool onPress(GdkEventButton* ev);
    bool onRelease(GdkEventButton* ev);
    bool onMotion(GdkEventMotion* ev);
    bool onScroll(GdkEventScroll* ev);
    void onToolChanged();
    void onSliderChanged();
    void onUndo();
    void onReset();
    void onFit();

    void toImage(double mx, double my, double& nx, double& ny) const;
    void pushHistory();
    void syncWidgets();
    void changed();
    void updateInfo();
    void closePolygon(bool add);

    AIRawCore::MaskEdit edit_;
    std::vector<AIRawCore::MaskEdit> history_;
    std::vector<double> pending_;          // vertices of the polygon being drawn (image coordinates)

    Tool tool_ = TOOL_AI;
    double cx_ = 0.5, cy_ = 0.5, zoom_ = 1.0;
    double scaleX_ = 1.0, scaleY_ = 1.0, offX_ = 0.0, offY_ = 0.0;
    bool dirty_ = true;
    int viewW_ = 0, viewH_ = 0;
    std::vector<unsigned char> rgb_;
    Glib::RefPtr<Gdk::Pixbuf> pix_;
    Glib::ustring error_;

    int drag_ = 0;                         // 0 none, 1 brush, 2 gradient, 3 pan
    double lastX_ = 0.0, lastY_ = 0.0;     // last mouse position (widget pixels)
    bool mouseIn_ = false;
    bool syncing_ = false;

    Gtk::DrawingArea area_;
    Gtk::Box tools_, sliders_, second_, bottom_;
    Gtk::RadioButton aiBtn_, brushBtn_, polyBtn_, gradBtn_;
    Gtk::Label help_, info_, sizeLbl_, featherLbl_, growLbl_, lumaLbl_;
    Gtk::Scale size_, feather_, grow_, lumaLo_, lumaHi_;
    Gtk::CheckButton invert_;
    Gtk::Button undo_, reset_, fit_;
};
