"""Gradient descent, two views at once — companion animation for Section 1
of ml-fundamentals.ipynb. Same data-generating process as the notebook:
y = 3x + 2 + noise, MSE loss, lr = 0.1.

Left: data space (the fit line improving).
Right: parameter space (the (w, b) point rolling down the loss bowl).
"""
import numpy as np
from manim import *


class GradientDescentTwoViews(Scene):
    def construct(self):
        # ---- the exact experiment from the notebook -------------------
        rng = np.random.default_rng(42)
        TRUE_W, TRUE_B, N = 3.0, 2.0, 60
        x = rng.uniform(-2, 2, size=N)
        y = TRUE_W * x + TRUE_B + rng.normal(0, 1.0, size=N)

        def loss(w, b):
            return np.mean((w * x + b - y) ** 2)

        w, b, lr = 0.0, 0.0, 0.1
        history = []
        for _ in range(60):
            history.append((w, b, loss(w, b)))
            r = w * x + b - y
            w -= lr * 2 * np.mean(r * x)
            b -= lr * 2 * np.mean(r)
        history.append((w, b, loss(w, b)))
        history = np.array(history)                      # (61, 3)

        # ---- left panel: data space ------------------------------------
        ax_data = Axes(
            x_range=[-2.2, 2.2, 1], y_range=[-6, 10, 4],
            x_length=5.4, y_length=5.0,
            axis_config={"color": GRAY, "stroke_width": 2,
                         "include_ticks": False, "tip_width": 0.15,
                         "tip_height": 0.15},
        ).to_edge(LEFT, buff=0.7).shift(DOWN * 0.4)
        data_title = Text("data space", font_size=28, color=BLUE_B)
        data_title.next_to(ax_data, UP, buff=0.3)

        dots = VGroup(*[
            Dot(ax_data.c2p(xi, yi), radius=0.035, color=BLUE_D,
                fill_opacity=0.8)
            for xi, yi in zip(x, y)
        ])

        # ---- right panel: parameter space ------------------------------
        ax_par = Axes(
            x_range=[-0.8, 4.3, 1], y_range=[-0.8, 4.3, 1],
            x_length=5.4, y_length=5.0,
            axis_config={"color": GRAY, "stroke_width": 2,
                         "include_ticks": False, "tip_width": 0.15,
                         "tip_height": 0.15},
        ).to_edge(RIGHT, buff=0.7).shift(DOWN * 0.4)
        par_title = Text("parameter space", font_size=28, color=TEAL)
        par_title.next_to(ax_par, UP, buff=0.3)
        w_label = Text("w", font_size=24, color=GRAY_B, slant=ITALIC)
        w_label.next_to(ax_par.x_axis.get_end(), DOWN, buff=0.15)
        b_label = Text("b", font_size=24, color=GRAY_B, slant=ITALIC)
        b_label.next_to(ax_par.y_axis.get_end(), LEFT, buff=0.15)

        # Loss contours: level sets of the (quadratic) MSE bowl.
        # The bowl bottoms out at the least-squares solution, not the truth.
        w_star, b_star = np.linalg.solve(
            np.array([[np.mean(x**2), np.mean(x)], [np.mean(x), 1.0]]),
            np.array([np.mean(x * y), np.mean(y)]),
        )
        l_min = loss(w_star, b_star)
        levels = [l_min + d for d in (0.4, 1.5, 4, 9, 16, 25)]
        contours = VGroup(*[
            ax_par.plot_implicit_curve(
                lambda W, B, lv=lv: loss(W, B) - lv,
                color=interpolate_color(TEAL, DARK_GRAY, i / len(levels)),
                stroke_width=2.5,
            )
            for i, lv in enumerate(levels)
        ])
        target = Dot(ax_par.c2p(w_star, b_star), radius=0.05, color=GREEN)
        target_label = Text("minimum", font_size=20, color=GREEN)
        target_label.next_to(target, UR, buff=0.1)

        # ---- moving pieces, both driven by one tracker ------------------
        t = ValueTracker(0.0)

        def current():
            i = t.get_value() * (len(history) - 1)
            lo = int(np.floor(i))
            hi = min(lo + 1, len(history) - 1)
            frac = i - lo
            return history[lo] * (1 - frac) + history[hi] * frac

        fit_line = always_redraw(lambda: Line(
            ax_data.c2p(-2.1, current()[0] * -2.1 + current()[1]),
            ax_data.c2p(2.1, current()[0] * 2.1 + current()[1]),
            color=YELLOW, stroke_width=4,
        ))
        gd_dot = always_redraw(lambda: Dot(
            ax_par.c2p(current()[0], current()[1]), radius=0.07, color=RED,
        ))
        trail = TracedPath(gd_dot.get_center, stroke_color=GOLD,
                           stroke_width=3)

        step_text = always_redraw(lambda: Text(
            f"step {int(t.get_value() * (len(history) - 1)):>2d}    "
            f"w = {current()[0]:.2f}   b = {current()[1]:.2f}   "
            f"loss = {current()[2]:.2f}",
            font_size=26, color=GRAY_A, font="Menlo",
        ).to_edge(UP, buff=0.35))

        # ---- narrative ---------------------------------------------------
        self.play(Create(ax_data), Create(ax_par),
                  Write(data_title), Write(par_title),
                  FadeIn(w_label), FadeIn(b_label), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots],
                              lag_ratio=0.01, run_time=1.2))
        self.play(Create(contours, lag_ratio=0.1),
                  FadeIn(target), FadeIn(target_label), run_time=1.8)
        self.wait(0.4)

        self.add(trail, fit_line, gd_dot)
        self.play(FadeIn(step_text), run_time=0.4)
        self.wait(0.6)
        # Ease the index: GD's big moves are all in the early steps, so give
        # them most of the screen time (t^2.5 lingers early, rushes the tail).
        self.play(t.animate.set_value(1.0), run_time=7,
                  rate_func=lambda a: a ** 2.5)
        self.wait(0.5)

        punch = Text("one walk, two views — every step downhill in the bowl "
                     "is the line fitting better",
                     font_size=25, color=YELLOW)
        punch.to_edge(DOWN, buff=0.75)   # clear of the video player's control bar
        self.play(Write(punch), Flash(gd_dot, color=GOLD, flash_radius=0.3),
                  run_time=1.2)
        self.wait(2)
