"""The vanishing gradient, watched live — companion animation for Section 8
of ml-fundamentals.ipynb. An unrolled RNN processes a sequence; the error
signal then travels backward through time, shrinking at every hop, and the
early timesteps end up with almost nothing to learn from.
"""
import numpy as np
from manim import *


class VanishingGradient(Scene):
    def construct(self):
        n_cells = 8
        decay = 0.55                      # per-step shrink factor of the error

        # ---- the unrolled network -----------------------------------------
        xs = np.linspace(-5.4, 5.4, n_cells)
        cells = VGroup(*[
            RoundedRectangle(corner_radius=0.12, width=1.0, height=0.8,
                             color=TEAL, stroke_width=3,
                             fill_color=BLACK, fill_opacity=1
                             ).move_to([x, 0, 0])
            for x in xs
        ])
        cell_lbls = VGroup(*[
            Text(f"h{chr(0x2080 + i + 1)}", font_size=24,
                 color=TEAL).move_to(c)
            for i, c in enumerate(cells)
        ])
        h_arrows = VGroup(*[
            Arrow(cells[i].get_right(), cells[i + 1].get_left(), buff=0.08,
                  color=GRAY_A, stroke_width=3, tip_length=0.16)
            for i in range(n_cells - 1)
        ])
        in_lbls = VGroup(*[
            Text(f"x{chr(0x2080 + i + 1)}", font_size=22,
                 color=BLUE).move_to([x, -1.5, 0])
            for i, x in enumerate(xs)
        ])
        in_arrows = VGroup(*[
            Arrow([x, -1.2, 0], [x, -0.5, 0], buff=0.05, color=BLUE,
                  stroke_width=2.5, tip_length=0.14)
            for x in xs
        ])
        shared = Text("same Wh at every step — the network is a loop, unrolled",
                      font_size=24, color=GRAY_A).to_edge(UP, buff=0.35)

        self.play(FadeIn(cells), FadeIn(cell_lbls), Write(shared),
                  run_time=1.0)
        self.play(LaggedStart(*[GrowArrow(a) for a in in_arrows],
                              lag_ratio=0.05),
                  LaggedStart(*[FadeIn(l) for l in in_lbls], lag_ratio=0.05),
                  LaggedStart(*[GrowArrow(a) for a in h_arrows],
                              lag_ratio=0.08),
                  run_time=1.5)

        # ---- forward pass ---------------------------------------------------
        self.play(LaggedStart(*[
            ShowPassingFlash(a.copy().set_color(BLUE).set_stroke(width=5),
                             time_width=0.7)
            for a in h_arrows
        ], lag_ratio=0.25), run_time=2.0)

        loss = Text("loss", font_size=26, color=YELLOW)
        loss.move_to([xs[-1], 2.15, 0])
        loss_arrow = Arrow(cells[-1].get_top(), loss.get_bottom(), buff=0.08,
                           color=YELLOW, stroke_width=3, tip_length=0.16)
        self.play(GrowArrow(loss_arrow), Write(loss), run_time=0.8)
        self.wait(0.5)

        # ---- backward through time -----------------------------------------
        bptt = Text("backpropagation through time: ×(Whᵀ · tanh′) per hop, and it's < 1",
                    font_size=24, color=RED).to_edge(UP, buff=0.35)
        self.play(ReplacementTransform(shared, bptt), run_time=0.8)

        # Error packet hops right-to-left, shrinking; a bar records what's left.
        packet = Dot(radius=0.22, color=RED, fill_opacity=0.95)
        packet.move_to(cells[-1].get_center() + UP * 0.75)
        bars = VGroup()
        self.play(FadeIn(packet, scale=0.4), run_time=0.5)

        for i in range(n_cells - 1, -1, -1):
            mag = decay ** (n_cells - 1 - i)
            bar = Rectangle(width=0.34, height=max(1.1 * mag, 0.02),
                            fill_color=RED, fill_opacity=0.85,
                            stroke_width=0)
            bar.next_to(cells[i], UP, buff=0.12)
            bars.add(bar)
            anims = [FadeIn(bar)]
            if i < n_cells - 1:
                anims.append(packet.animate
                             .move_to(cells[i].get_center() + UP * 0.75)
                             .scale(decay))
            self.play(*anims, run_time=0.45 if i > 2 else 0.7)

        gone = Text("≈ 0", font_size=24, color=RED)
        gone.next_to(cells[0], UP, buff=0.35)
        self.play(Transform(packet, gone), run_time=0.8)
        self.wait(0.5)

        punch = Text("the error fades before it reaches the start — "
                     "early inputs can't be learned (LSTMs fix this)",
                     font_size=25, color=YELLOW)
        punch.to_edge(DOWN, buff=0.75)   # clear of the video player's control bar
        self.play(Write(punch), run_time=1.2)
        self.wait(2)
