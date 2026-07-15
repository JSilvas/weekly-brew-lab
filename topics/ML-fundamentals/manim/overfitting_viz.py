"""The overfitting dial — companion animation for Section 6 of
ml-fundamentals.ipynb. Same experiment as the notebook: polynomials of
degree 1..15 fit to 30 noisy samples of sin(2.5x). As the degree dial
turns, the fit morphs from stiff to hysterical while train and validation
error part ways.
"""
import numpy as np
from numpy.polynomial import Polynomial
from manim import *


class OverfittingDial(Scene):
    def construct(self):
        # ---- the exact experiment from the notebook ----------------------
        def true_fn(x):
            return np.sin(2.5 * x)

        rng = np.random.default_rng(6)
        x_tr = rng.uniform(-1, 1, 30)
        y_tr = true_fn(x_tr) + rng.normal(0, 0.25, 30)
        x_va = rng.uniform(-1, 1, 100)
        y_va = true_fn(x_va) + rng.normal(0, 0.25, 100)

        xg = np.linspace(-1, 1, 220)
        degs = list(range(1, 16))
        fits, tr_err, va_err = [], [], []
        for d in degs:
            p = Polynomial.fit(x_tr, y_tr, d)
            fits.append(p(xg))
            tr_err.append(np.mean((p(x_tr) - y_tr) ** 2))
            va_err.append(np.mean((p(x_va) - y_va) ** 2))
        fits = np.array(fits)

        # ---- left panel: the fit -----------------------------------------
        axL = Axes(
            x_range=[-1.1, 1.1, 0.5], y_range=[-2.2, 2.2, 1],
            x_length=5.6, y_length=4.6,
            axis_config={"color": GRAY, "stroke_width": 2,
                         "include_ticks": False, "tip_width": 0.15,
                         "tip_height": 0.15},
        ).to_edge(LEFT, buff=0.7).shift(DOWN * 0.5)
        truth = axL.plot(true_fn, x_range=[-1.05, 1.05],
                         color=GRAY_B, stroke_width=2.5)
        truth = DashedVMobject(truth, num_dashes=40)
        dots = VGroup(*[
            Dot(axL.c2p(xi, yi), radius=0.045, color=BLUE_D, fill_opacity=0.85)
            for xi, yi in zip(x_tr, y_tr)
        ])
        lblL = Text("30 noisy points, one polynomial", font_size=26,
                    color=BLUE_B).next_to(axL, UP, buff=0.3)

        # ---- right panel: train vs validation error ----------------------
        axR = Axes(
            x_range=[0, 16, 5], y_range=[0, 0.65, 0.2],
            x_length=5.6, y_length=4.6,
            axis_config={"color": GRAY, "stroke_width": 2,
                         "include_ticks": False, "tip_width": 0.15,
                         "tip_height": 0.15},
        ).to_edge(RIGHT, buff=0.7).shift(DOWN * 0.5)
        lblR = Text("error vs degree", font_size=26,
                    color=TEAL).next_to(axR, UP, buff=0.3)
        deg_axis_lbl = Text("degree", font_size=22, color=GRAY_B)
        deg_axis_lbl.next_to(axR.x_axis.get_end(), DOWN, buff=0.15)

        # ---- the dial ------------------------------------------------------
        deg = ValueTracker(1.0)

        def fit_at(dv):
            lo = int(np.clip(np.floor(dv), 1, 15))
            hi = int(np.clip(lo + 1, 1, 15))
            f = dv - lo
            return np.clip(fits[lo - 1] * (1 - f) + fits[hi - 1] * f,
                           -2.15, 2.15)

        fit_curve = always_redraw(lambda: VMobject(
            stroke_color=YELLOW, stroke_width=4,
        ).set_points_as_corners(
            [axL.c2p(xx, yy) for xx, yy in zip(xg, fit_at(deg.get_value()))]
        ))

        def err_path(errs, color):
            def build():
                dv = deg.get_value()
                pts = [axR.c2p(d, min(errs[d - 1], 0.63))
                       for d in degs if d <= dv]
                pts.append(axR.c2p(dv, min(np.interp(dv, degs, errs), 0.63)))
                return VMobject(stroke_color=color,
                                stroke_width=3.5).set_points_as_corners(pts)
            return always_redraw(build)

        tr_curve = err_path(tr_err, BLUE)
        va_curve = err_path(va_err, RED)
        tr_key = Text("train", font_size=22, color=BLUE)
        va_key = Text("validation", font_size=22, color=RED)
        keys = VGroup(tr_key, va_key).arrange(RIGHT, buff=0.5)
        keys.move_to(axR.c2p(11, 0.58))

        readout = always_redraw(lambda: Text(
            f"degree = {deg.get_value():.0f}",
            font_size=30, color=YELLOW, font="Menlo",
        ).to_edge(UP, buff=0.35))

        # ---- narrative ----------------------------------------------------
        self.play(Create(axL), Create(axR), Write(lblL), Write(lblR),
                  FadeIn(deg_axis_lbl), FadeIn(keys), run_time=1.2)
        self.play(Create(truth),
                  LaggedStart(*[FadeIn(d, scale=0.5) for d in dots],
                              lag_ratio=0.02), run_time=1.4)
        self.add(fit_curve, tr_curve, va_curve)
        self.play(FadeIn(readout), run_time=0.5)
        self.wait(0.6)

        # Sweep the dial: linger in the sweet spot, then let it go feral.
        self.play(deg.animate.set_value(4), run_time=2.5,
                  rate_func=smooth)
        self.wait(0.8)
        self.play(deg.animate.set_value(15), run_time=5,
                  rate_func=smooth)
        self.wait(0.5)

        # Freeze the error history so it stays put when the dial turns back.
        tr_curve.clear_updaters()
        va_curve.clear_updaters()

        # Mark the U-turn.
        best = degs[int(np.argmin(va_err))]
        vline = DashedLine(axR.c2p(best, 0), axR.c2p(best, 0.45),
                           color=GREEN, stroke_width=3)
        best_lbl = Text(f"sweet spot: degree {best}", font_size=24,
                        color=GREEN).next_to(vline.get_end(), UR, buff=0.1)
        punch = Text("train error falls forever — validation calls the bluff",
                     font_size=26, color=YELLOW)
        punch.to_edge(DOWN, buff=0.75)   # clear of the video player's control bar
        self.play(Create(vline), Write(best_lbl), Write(punch), run_time=1.2)
        self.play(deg.animate.set_value(best), run_time=1.5,
                  rate_func=smooth)
        self.wait(2)
