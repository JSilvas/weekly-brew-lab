"""Backprop, watched live — companion animation for Section 4 of
ml-fundamentals.ipynb. Forward pass pulses left-to-right through a small
network; the error signal then flows backward, dropping off a gradient at
every layer it passes: (input to layer)T x (error at layer).
"""
import numpy as np
from manim import *


class BackpropFlow(Scene):
    def construct(self):
        # ---- the network: 2 -> 5 -> 1 -----------------------------------
        in_pos = [LEFT * 5.2 + UP * 0.9, LEFT * 5.2 + DOWN * 0.9]
        hid_pos = [UP * (1.8 - i * 0.9) for i in range(5)]
        out_pos = [RIGHT * 4.2]

        def node(p, color):
            return Circle(radius=0.28, color=color, fill_color=BLACK,
                          fill_opacity=1, stroke_width=3).move_to(p)

        in_nodes = VGroup(*[node(p, BLUE) for p in in_pos])
        hid_nodes = VGroup(*[node(p, TEAL) for p in hid_pos])
        out_nodes = VGroup(*[node(p, GREEN) for p in out_pos])

        def connect(src, dst):
            return VGroup(*[
                Line(a.get_center(), b.get_center(), buff=0.28,
                     stroke_width=1.5, color=GRAY, stroke_opacity=0.6)
                for a in src for b in dst
            ])

        w1_edges = connect(in_nodes, hid_nodes)
        w2_edges = connect(hid_nodes, out_nodes)

        lbl_x = Text("x", font_size=30, slant=ITALIC,
                     color=BLUE).next_to(in_nodes, DOWN, buff=0.4)
        lbl_h = Text("A₁ = tanh(XW₁+b₁)", font_size=24,
                     color=TEAL).next_to(hid_nodes, DOWN, buff=0.4)
        lbl_p = Text("p = σ(z₂)", font_size=24,
                     color=GREEN).next_to(out_nodes, DOWN, buff=0.4)
        lbl_w1 = Text("W₁", font_size=26, color=GRAY_A).move_to(
            (in_nodes.get_center() + hid_nodes.get_center()) / 2 + UP * 2.4)
        lbl_w2 = Text("W₂", font_size=26, color=GRAY_A).move_to(
            (hid_nodes.get_center() + out_nodes.get_center()) / 2 + UP * 2.4)

        self.play(FadeIn(in_nodes), FadeIn(hid_nodes), FadeIn(out_nodes),
                  Create(w1_edges), Create(w2_edges),
                  Write(lbl_x), Write(lbl_h), Write(lbl_p),
                  FadeIn(lbl_w1), FadeIn(lbl_w2), run_time=1.6)

        # ---- forward pass ------------------------------------------------
        fwd = Text("forward: compute and remember every layer",
                   font_size=26, color=BLUE_B).to_edge(UP, buff=0.3)
        self.play(Write(fwd), run_time=0.7)

        def pulse(edges, color):
            return LaggedStart(*[
                ShowPassingFlash(e.copy().set_stroke(color=color, width=4,
                                                     opacity=1),
                                 time_width=0.6)
                for e in edges
            ], lag_ratio=0.02)

        self.play(pulse(w1_edges, BLUE),
                  *[n.animate.set_fill(TEAL, opacity=0.5) for n in hid_nodes],
                  run_time=1.4)
        self.play(pulse(w2_edges, BLUE),
                  out_nodes[0].animate.set_fill(GREEN, opacity=0.6),
                  run_time=1.1)

        loss_txt = Text("loss = BCE(p, y)", font_size=26, color=YELLOW)
        loss_txt.move_to(out_pos[0] + UP * 2.0)
        arrow_loss = Arrow(out_nodes.get_top(), loss_txt.get_bottom(),
                           buff=0.1, color=YELLOW, stroke_width=3)
        self.play(GrowArrow(arrow_loss), Write(loss_txt), run_time=0.9)
        self.wait(0.8)

        # ---- backward pass -------------------------------------------------
        bwd = Text("backward: the error signal retraces the path",
                   font_size=26, color=RED)
        bwd.to_edge(UP, buff=0.3)
        self.play(ReplacementTransform(fwd, bwd), run_time=0.7)

        d2 = Text("δ₂ = p − y", font_size=26, color=RED)
        d2.next_to(out_nodes, RIGHT, buff=0.3)
        self.play(Flash(out_nodes[0], color=RED, flash_radius=0.4),
                  Write(d2), run_time=0.9)
        self.wait(0.4)

        g2 = Text("∇W₂ = A₁ᵀ δ₂", font_size=26, color=GOLD)
        g2.move_to(lbl_w2).shift(DOWN * 0.65)
        self.play(pulse([Line(e.get_end(), e.get_start()).match_style(e)
                         for e in w2_edges], RED),
                  FadeIn(g2, shift=DOWN * 0.2), run_time=1.2)

        d1 = Text("δ₁ = (δ₂W₂ᵀ) ⊙ tanh′", font_size=24, color=RED)
        d1.next_to(hid_nodes, UP, buff=0.35)
        self.play(Write(d1), run_time=0.8)
        self.wait(0.4)

        g1 = Text("∇W₁ = Xᵀ δ₁", font_size=26, color=GOLD)
        g1.move_to(lbl_w1).shift(DOWN * 0.65)
        self.play(pulse([Line(e.get_end(), e.get_start()).match_style(e)
                         for e in w1_edges], RED),
                  FadeIn(g1, shift=DOWN * 0.2), run_time=1.4)
        self.wait(0.6)

        # ---- punchline ----------------------------------------------------
        boxes = VGroup(SurroundingRectangle(g1, color=GOLD, buff=0.12),
                       SurroundingRectangle(g2, color=GOLD, buff=0.12))
        punch = Text("every layer, the same move:  (input to layer)ᵀ × (error at layer)",
                     font_size=26, color=GOLD)
        punch.to_edge(DOWN, buff=0.75)   # clear of the video player's control bar
        self.play(Create(boxes), Write(punch), run_time=1.3)
        self.wait(2)
