"""Tube Center of Gravity — standalone Streamlit app.

Run: python -m streamlit run tube_cg_calculator.py
Units: inches and pound-force. Coordinates are measured from the left tube end.
Each added weight represents an omitted component at its own CG. Positions and
uniform intervals may extend beyond either tube end. All weights are positive.
The solver uses only the standard library and is independent of the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import csv
import io
import math


@dataclass(frozen=True)
class Tube:
    length: float
    od: float
    id: float
    weight: float = 0.0

    @property
    def area(self):
        return math.pi * (self.od - self.id) * (self.od + self.id) / 4.0


@dataclass(frozen=True)
class Weight:
    name: str
    kind: str  # "point" or "uniform"
    weight: float  # TOTAL weight, lbf; never a signed external force.
    start: float
    end: float | None = None

    @property
    def finish(self):
        return self.start if self.kind == "point" else float(self.end)

    @property
    def center(self):
        return self.start / 2.0 + self.finish / 2.0

    @property
    def moment(self):
        return self.weight * self.center

    @property
    def intensity(self):
        return self.weight / (self.finish - self.start) if self.kind == "uniform" else None


@dataclass(frozen=True)
class Solution:
    tube: Tube
    components: tuple[Weight, ...]  # Includes the tube self-weight exactly once.
    total_weight: float
    first_moment: float
    cg: float | None
    balance_residual: float | None
    balance_tolerance: float | None

    @property
    def added_weight(self):
        return math.fsum(w.weight for w in self.components[1:])

    @property
    def location(self):
        if self.cg is None:
            return "Undefined"
        if self.cg < 0:
            return "Beyond left end"
        if self.cg > self.tube.length:
            return "Beyond right end"
        return "Within tube length"


def validate(tube: Tube, weights: list[Weight] | tuple[Weight, ...]):
    if not all(math.isfinite(v) for v in (tube.length, tube.od, tube.id, tube.weight)):
        raise ValueError("All tube inputs must be finite numbers.")
    if tube.length <= 0:
        raise ValueError("Tube length must be greater than zero.")
    if not 0 <= tube.id < tube.od:
        raise ValueError("Diameters must satisfy 0 ≤ ID < OD.")
    if tube.weight < 0:
        raise ValueError("Tube weight must be zero or positive.")
    if not math.isfinite(tube.area) or tube.area <= 0:
        raise ValueError("The section dimensions exceed the numerical range.")
    for w in weights:
        if w.kind not in ("point", "uniform"):
            raise ValueError(f"{w.name}: choose a point or uniform weight.")
        if not all(math.isfinite(v) for v in (w.weight, w.start)):
            raise ValueError(f"{w.name}: enter finite weights and positions.")
        if w.weight < 0:
            raise ValueError(f"{w.name}: weight must be zero or positive.")
        if w.kind == "uniform":
            if w.end is None or not math.isfinite(w.end):
                raise ValueError(f"{w.name}: enter a finite end position.")
            if w.end <= w.start:
                raise ValueError(f"{w.name}: end must be greater than start.")
            span = w.end - w.start
            if not math.isfinite(span) or not math.isfinite(w.intensity):
                raise ValueError(f"{w.name}: interval or intensity exceeds the numerical range.")
    positions = [0.0, tube.length] + [x for w in weights for x in (w.start, w.finish)]
    if not math.isfinite(max(positions) - min(positions)):
        raise ValueError("The position range exceeds the numerical range.")


def solve_cg(tube: Tube, weights: list[Weight] | tuple[Weight, ...] = ()) -> Solution:
    """Exact first-moment calculation; no support conditions or sampling.

    Uniform weights act at their interval midpoints. A zero total weight has
    no defined CG and is returned as cg=None. Input geometry does not infer
    weight: the entered tube weight is used directly.
    """
    validate(tube, weights)
    components = (Weight("Tube self-weight", "uniform", tube.weight, 0.0, tube.length), *weights)
    try:
        total = math.fsum(w.weight for w in components)
        moment = math.fsum(w.moment for w in components)
        if not math.isfinite(total) or not math.isfinite(moment):
            raise ValueError("The weight and position combination exceeds the numerical range.")
        if total == 0:
            return Solution(tube, components, 0.0, 0.0, None, None, None)
        cg = moment / total
        offsets = [w.weight * (w.center - cg) for w in components if w.weight > 0]
        residual = math.fsum(offsets)
        scale = max(math.fsum(abs(w.moment) for w in components), abs(total * cg), 1.0)
        tolerance = scale * 1e-10
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("The weight and position combination exceeds the numerical range.") from error
    if not all(math.isfinite(v) for v in (cg, residual, tolerance)):
        raise ValueError("The weight and position combination exceeds the numerical range.")
    return Solution(tube, components, total, moment, cg, residual, tolerance)


def number(value, digits=5):
    if value == 0:
        return "0"
    if abs(value) < 10 ** (-digits) or abs(value) >= 1e10:
        return f"{value:.5g}"
    return f"{value:,.{digits}f}".rstrip("0").rstrip(".")


def plot_extent(solution):
    active = [w for w in solution.components if w.weight > 0]
    values = [0.0, solution.tube.length] + [x for w in active for x in (w.start, w.finish)]
    return min(values), max(values)


def export_csv(solution):
    """Include units, zero-weight entries, signed positions and moment checks."""
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["Component", "Type", "Weight (lbf)", "Start x (in)", "End x (in)",
                     "Component CG x (in)", "First moment (lbf*in)", "Weight share (%)"])
    for w in solution.components:
        share = w.weight / solution.total_weight * 100 if solution.total_weight else ""
        writer.writerow([w.name, w.kind, w.weight, w.start, w.finish, w.center, w.moment, share])
    writer.writerow([])
    writer.writerow(["Total weight (lbf)", solution.total_weight])
    writer.writerow(["Total first moment (lbf*in)", solution.first_moment])
    writer.writerow(["Assembly CG from left end (in)", "Undefined" if solution.cg is None else solution.cg])
    writer.writerow(["Assembly CG from right end, positive left (in)",
                     "Undefined" if solution.cg is None else solution.tube.length - solution.cg])
    writer.writerow(["Moment balance residual about CG (lbf*in)",
                     "Undefined" if solution.cg is None else solution.balance_residual])
    return out.getvalue()


def tube_svg(solution):
    """Scale the coordinate range to include all active components and the tube."""
    tube = solution.tube
    low, high = plot_extent(solution)
    x0, x1, width = 80.0, 920.0, 1000.0
    xpos = lambda x: x0 + ((x - low) / (high - low)) * (x1 - x0)
    active = [w for w in solution.components if w.weight > 0]
    occupied, arranged = [], []
    for index, w in enumerate(active):
        start, end = xpos(w.start), xpos(w.finish)
        center = min(865.0, max(135.0, (start + end) / 2))
        left, right = min(start - 8, center - 120), max(end + 8, center + 120)
        lane = 0
        while lane < len(occupied) and any(left < e + 14 and right > s - 14 for s, e in occupied[lane]):
            lane += 1
        if lane == len(occupied):
            occupied.append([])
        occupied[lane].append((left, right))
        arranged.append((w, lane, center, index))
    beam_y = 45 + max(1, len(occupied)) * 65 if active else 65
    height = beam_y + 220
    ink, muted, blue, red, green = "#23354b", "#6e7e91", "#2869ae", "#bf493c", "#168063"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" width="100%" role="img" aria-label="Tube, component weights and assembly center of gravity">',
             '<defs><marker id="cg-dim-head" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse"><path d="M0,0 L6,3 L0,6 Z" fill="#6e7e91"/></marker>',
             '<marker id="cg-point-head" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#bf493c"/></marker>',
             '<marker id="cg-uniform-head" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#2869ae"/></marker></defs>',
             '<style>text{font-family:Arial,sans-serif;font-size:14px;fill:#23354b}.dim{fill:#6e7e91;font-size:13px}.load{font-size:13px;font-weight:500}.cg{fill:#168063;font-weight:600}</style>',
             '<rect width="100%" height="100%" fill="white"/>']

    def line(xa, ya, xb, yb, color=muted, stroke=1, extra=""):
        parts.append(f'<line x1="{xa:.5f}" y1="{ya:.5f}" x2="{xb:.5f}" y2="{yb:.5f}" stroke="{color}" stroke-width="{stroke}" {extra}/>')

    # Dashed coordinate extensions are not physical tube or support members.
    line(x0, beam_y, x1, beam_y, "#c5ced9", 1.5, 'stroke-dasharray="6 5"')
    for w, lane, center, index in arranged:
        start, end = xpos(w.start), xpos(w.finish)
        y = beam_y - 43 - lane * 65
        color, marker = (red, "cg-point-head") if w.kind == "point" else (blue, "cg-uniform-head")
        label_name = w.name if len(w.name) <= 24 else w.name[:21] + "…"
        title = f"{label_name} · {number(w.weight, 4)} lbf"
        position = f"x = {number(w.start)} in" if w.kind == "point" else f"{number(w.start)} to {number(w.finish)} in"
        parts.append(f'<g><title>{escape(w.name)}: {number(w.weight, 6)} lbf; {escape(position)}</title>')
        if w.kind == "uniform":
            line(start, y, end, y, color, 1.6)
            count = max(2, int((end - start) / 42) + 1)
            arrows = [start + (end - start) * i / (count - 1) for i in range(count)]
        else:
            arrows = [start]
        for x in arrows:
            line(x, y, x, beam_y - 9, color, 1.8, f'marker-end="url(#{marker})"')
        parts.append(f'<rect x="{center-125}" y="{y-43}" width="250" height="40" rx="4" fill="white" fill-opacity=".94"/>')
        parts.append(f'<text x="{center}" y="{y-25}" text-anchor="middle" class="load">{escape(title)}</text>')
        parts.append(f'<text x="{center}" y="{y-8}" text-anchor="middle" class="dim">{escape(position)}</text></g>')
    left, right = xpos(0.0), xpos(tube.length)
    line(left, beam_y, right, beam_y, ink, 8)
    for x in (left, right):
        line(x, beam_y - 8, x, beam_y + 8, ink, 1.6)
        line(x, beam_y + 15, x, beam_y + 103, "#c2cad4", .8)
    # End labels move to separate rows if the tube is small relative to the full range.
    parts.append(f'<text x="{min(870,max(130,left))}" y="{beam_y+35}" text-anchor="middle" class="dim">Left · x = 0</text>')
    parts.append(f'<text x="{min(870,max(130,right))}" y="{beam_y+(57 if right-left<210 else 35)}" text-anchor="middle" class="dim">Right · x = {number(tube.length)} in</text>')
    line(left, beam_y + 96, right, beam_y + 96, muted, 1,
         'marker-start="url(#cg-dim-head)" marker-end="url(#cg-dim-head)"')
    parts.append(f'<text x="{min(820,max(180,(left+right)/2))}" y="{beam_y+83}" text-anchor="middle" class="dim">Tube length · {number(tube.length)} in</text>')
    if solution.cg is not None:
        cgx = xpos(solution.cg)
        line(cgx, beam_y - 15, cgx, beam_y + 158, green, 1.8, 'stroke-dasharray="5 4"')
        parts.append(f'<circle cx="{cgx}" cy="{beam_y}" r="10" fill="white" stroke="{green}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{cgx}" cy="{beam_y}" r="3.5" fill="{green}"/>')
        label_x = min(840, max(160, cgx))
        parts.append(f'<rect x="{label_x-160}" y="{beam_y+150}" width="320" height="46" rx="6" fill="#edf7f2"/>')
        parts.append(f'<text x="{label_x}" y="{beam_y+170}" text-anchor="middle" class="cg">Assembly CG · x = {number(solution.cg)} in</text>')
        parts.append(f'<text x="{label_x}" y="{beam_y+188}" text-anchor="middle" class="dim">{solution.location}</text>')
    else:
        parts.append(f'<text x="500" y="{beam_y+160}" text-anchor="middle" class="dim">Enter a positive weight to locate the center of gravity.</text>')
    parts.append('</svg>')
    return "".join(parts)


def section_svg(tube):
    inner = 80.0 * (tube.id / tube.od)
    bore = f'<circle cx="130" cy="145" r="{inner}" fill="white" stroke="#23354b" stroke-width="2"/>' if tube.id > 0 else ""
    inner_dim = f'<path d="M{130-inner} 246H{130+inner}"/>' if tube.id > 0 else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 290" width="100%" role="img" aria-label="Tube cross-section, OD {tube.od:g} inches and ID {tube.id:g} inches">
    <rect width="260" height="290" fill="white"/>
    <defs><marker id="cg-section-arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse"><path d="M0 0 L6 3 L0 6Z" fill="#607186"/></marker></defs>
    <circle cx="130" cy="145" r="80" fill="#e4ecf5" stroke="#23354b" stroke-width="2"/>{bore}
    <g stroke="#8693a4" stroke-width="1"><path d="M50 145V40M210 145V40"/><path d="M{130-inner} 145V251 M{130+inner} 145V251"/></g>
    <g stroke="#607186" marker-start="url(#cg-section-arrow)" marker-end="url(#cg-section-arrow)"><path d="M50 45H210"/>{inner_dim}</g>
    <g font-family="Arial,sans-serif" font-size="14" fill="#23354b" text-anchor="middle"><text x="130" y="30">OD {number(tube.od)} in</text><text x="130" y="271">ID {number(tube.id)} in</text></g></svg>'''


def make_distribution(solution):
    """Separate point weights (lbf) from line intensities (lbf/in)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.13,
                        subplot_titles=("Point component weights", "Uniform weight per unit length"))
    active = [w for w in solution.components if w.weight > 0]
    points = [w for w in active if w.kind == "point"]
    uniforms = [w for w in active if w.kind == "uniform"]
    for w in points:
        fig.add_trace(go.Scatter(x=[w.start, w.start], y=[0, w.weight], mode="lines+markers",
                                line=dict(color="#bf493c", width=2), marker=dict(size=[0, 8]),
                                name=w.name, text=[w.name] * 2,
                                hovertemplate="%{text}<br>x = %{x:.6g} in<br>Weight = %{y:.6g} lbf<extra></extra>"), row=1, col=1)
    low, high = plot_extent(solution)
    edges = sorted(set([low, high] + [x for w in uniforms for x in (w.start, w.finish)]))
    xs, ys = [], []
    for a, b in zip(edges, edges[1:]):
        mid = a / 2 + b / 2
        intensity = math.fsum(w.intensity for w in uniforms if w.start <= mid < w.finish)
        xs.extend([a, b])
        ys.extend([intensity, intensity])
    if xs:
        xs, ys = [xs[0], *xs, xs[-1]], [0, *ys, 0]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy", name="Combined uniform weight",
                                line=dict(color="#2869ae", width=2), fillcolor="rgba(40,105,174,.14)",
                                hovertemplate="x = %{x:.6g} in<br>Intensity = %{y:.6g} lbf/in<extra></extra>"), row=2, col=1)
    for row in (1, 2):
        fig.add_vrect(x0=0, x1=solution.tube.length, fillcolor="#23354b", opacity=.035,
                      layer="below", line_width=0, row=row, col=1)
        fig.add_hline(y=0, line_color="#aeb9c7", line_width=1, row=row, col=1)
        if solution.cg is not None:
            fig.add_vline(x=solution.cg, line_color="#168063", line_width=2, line_dash="dash", row=row, col=1)
    if not points:
        fig.add_annotation(text="No point weights", x=.5, y=.85, xref="paper", yref="paper", showarrow=False, font_color="#6e7e91")
    if not uniforms:
        fig.add_annotation(text="No uniform weights", x=.5, y=.2, xref="paper", yref="paper", showarrow=False, font_color="#6e7e91")
    padding = (high - low) * .05
    fig.update_xaxes(range=[low - padding, high + padding], showgrid=True, gridcolor="#edf0f4", zeroline=False)
    fig.update_xaxes(title_text="Position from left tube end (in)", row=2, col=1)
    fig.update_yaxes(title_text="Weight (lbf)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Intensity (lbf/in)", rangemode="tozero", row=2, col=1)
    fig.update_layout(height=510, template="plotly_white", showlegend=False, margin=dict(l=65, r=25, t=45, b=50),
                      font=dict(family="Arial, sans-serif", color="#23354b"), hovermode="closest")
    return fig


def latex_number(value):
    s = f"{value:.9g}"
    if "e" in s:
        base, exponent = s.split("e")
        return rf"{base}\times10^{{{int(exponent)}}}"
    return s


def render_calculations(solution):
    import pandas as pd
    import streamlit as st

    st.subheader("Calculation method")
    st.write("Use the first moment of every component's weight about the left tube end. "
             "All weights are nonnegative; a component left of that end has a negative coordinate and first moment.")
    st.latex(r"x_i=\begin{cases}x_{\mathrm{point}},&\text{point weight}\\(a_i+b_i)/2,&\text{uniform weight}\end{cases}")
    st.latex(r"W_{\mathrm{total}}=\sum_i W_i,\qquad M_0=\sum_i W_i x_i,\qquad x_{\mathrm{CG}}=\frac{M_0}{W_{\mathrm{total}}}")
    st.write("The tube is a uniform weight over its full length and is included once.")
    st.latex(rf"x_{{\mathrm{{tube}}}}=\frac{{L}}{{2}}=\frac{{{latex_number(solution.tube.length)}}}{{2}}={latex_number(solution.tube.length/2)}\;\mathrm{{in}}")
    rows = [{"Component": w.name, "Type": w.kind, "Weight (lbf)": w.weight,
             "Start x (in)": w.start, "End x (in)": w.finish, "Component CG x (in)": w.center,
             "Intensity (lbf/in)": w.intensity, "First moment (lbf·in)": w.moment}
            for w in solution.components]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.latex(rf"W_{{\mathrm{{total}}}}={latex_number(solution.total_weight)}\;\mathrm{{lbf}},\qquad M_0={latex_number(solution.first_moment)}\;\mathrm{{lbf\cdot in}}")
    if solution.cg is None:
        st.info("The total weight is zero. The center of gravity is undefined because division by zero is not permitted.")
    else:
        st.latex(rf"x_{{\mathrm{{CG}}}}=\frac{{{latex_number(solution.first_moment)}}}{{{latex_number(solution.total_weight)}}}={latex_number(solution.cg)}\;\mathrm{{in}}")
        st.latex(rf"d_{{\mathrm{{right}}}}=L-x_{{\mathrm{{CG}}}}={latex_number(solution.tube.length-solution.cg)}\;\mathrm{{in}}")
        st.subheader("Moment balance about the calculated CG")
        st.latex(r"\sum_i W_i(x_i-x_{\mathrm{CG}})=0")
        st.dataframe(pd.DataFrame([{"Component": w.name,
                                    "Offset from assembly CG (in)": w.center - solution.cg,
                                    "Moment about assembly CG (lbf·in)": w.weight * (w.center - solution.cg)}
                                   for w in solution.components]), hide_index=True, width="stretch")
        st.write(f"Moment balance residual: {number(solution.balance_residual, 9)} lbf·in. "
                 f"Floating-point tolerance: {number(solution.balance_tolerance, 9)} lbf·in.")
        if abs(solution.balance_residual) <= solution.balance_tolerance:
            st.success("The component moments balance about the calculated CG.")
        else:
            st.warning("The moment residual exceeds the numerical tolerance. Check the scale and precision of the inputs.")
    with st.expander("Individual component substitutions"):
        for i, w in enumerate(solution.components, 1):
            st.write(f"{i}. {w.name}")
            if w.kind == "uniform":
                st.latex(rf"x_{{{i}}}=\frac{{({latex_number(w.start)})+({latex_number(w.finish)})}}{{2}}={latex_number(w.center)}\;\mathrm{{in}}")
                st.latex(rf"q_{{{i}}}=\frac{{{latex_number(w.weight)}}}{{({latex_number(w.finish)})-({latex_number(w.start)})}}={latex_number(w.intensity)}\;\mathrm{{lbf/in}}")
            else:
                st.latex(rf"x_{{{i}}}={latex_number(w.start)}\;\mathrm{{in}}")
            st.latex(rf"M_{{0,{i}}}=({latex_number(w.weight)})({latex_number(w.center)})={latex_number(w.moment)}\;\mathrm{{lbf\cdot in}}")
    with st.expander("Cross-section and model assumptions"):
        st.latex(rf"A=\frac{{\pi}}{{4}}(OD^2-ID^2)={latex_number(solution.tube.area)}\;\mathrm{{in^2}}")
        st.write("The entered tube weight is used directly. Changing OD or ID updates the section drawing; "
                 "it does not infer a different weight. The tube self-weight is uniformly distributed. "
                 "Uniform component weights are constant over their entered intervals. The result is the "
                 "longitudinal CG of the tube and all represented parts under uniform gravity. "
                 "Transverse offsets and deformation are outside this one-dimensional model.")
        st.markdown("Reference: [NASA Glenn Research Center — Center of Gravity](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/center-of-gravity/).")


def weight_defaults(length, ident):
    return {"name": f"Weight {ident}", "kind": "Point weight", "magnitude": 0.0,
            "position": length / 2.0, "full": True, "start": 0.0, "end": length}


def initialize_state():
    import streamlit as st

    defaults = {"cg_tube_length": 305.625, "cg_tube_od": 3.96875, "cg_tube_id": 3.219,
                "cg_tube_weight": 0.0, "cg_weight_ids": [], "cg_next_weight_id": 1}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Preserve temporarily hidden point and interval values when type/full-span
    # controls change. All session keys are namespaced for a future shared app.
    for ident in st.session_state.cg_weight_ids:
        for suffix, default in weight_defaults(st.session_state.cg_tube_length, ident).items():
            key = f"cg_weight_{ident}_{suffix}"
            st.session_state[key] = st.session_state.get(key, default)


def add_weight():
    import streamlit as st

    ident = st.session_state.cg_next_weight_id
    st.session_state.cg_next_weight_id = ident + 1
    st.session_state.cg_weight_ids = [*st.session_state.cg_weight_ids, ident]
    for suffix, value in weight_defaults(st.session_state.cg_tube_length, ident).items():
        st.session_state[f"cg_weight_{ident}_{suffix}"] = value


def remove_weight(ident):
    import streamlit as st

    st.session_state.cg_weight_ids = [i for i in st.session_state.cg_weight_ids if i != ident]
    for key in list(st.session_state.keys()):
        if key.startswith(f"cg_weight_{ident}_"):
            del st.session_state[key]


def load_example():
    import streamlit as st

    for key in list(st.session_state.keys()):
        if key.startswith("cg_"):
            del st.session_state[key]
    initialize_state()
    st.session_state.cg_tube_weight = 370.0
    for name, magnitude, position in (("Left assembly", 1050.0, -12.0),
                                       ("Right assembly", 46.0, 323.625)):
        add_weight()
        prefix = f"cg_weight_{st.session_state.cg_weight_ids[-1]}_"
        st.session_state[prefix + "name"] = name
        st.session_state[prefix + "magnitude"] = magnitude
        st.session_state[prefix + "position"] = position


def input_panel():
    import streamlit as st

    weights = []
    with st.sidebar:
        st.header("Inputs")
        st.caption("Schematic and results update when you commit an input: press Enter or leave the field.")
        st.subheader("Geometry")
        length = st.number_input("Tube length (in)", min_value=.001, step=1.0, format="%.5f", key="cg_tube_length")
        st.subheader("Cross-section")
        c1, c2 = st.columns(2)
        od = c1.number_input("OD (in)", min_value=.001, step=.125, format="%.5f", key="cg_tube_od")
        id_ = c2.number_input("ID (in)", min_value=0.0, step=.125, format="%.5f", key="cg_tube_id")
        st.subheader("Tube self-weight")
        tube_weight = st.number_input("Total tube weight (lbf)", min_value=0.0, step=10.0, format="%.4f", key="cg_tube_weight")
        st.caption("Uniform over the full tube. Use the actual tube weight; OD and ID do not calculate it automatically.")
        st.subheader("Additional parts")
        st.caption("Each entry represents a part's weight at its own CG. Positions can be negative or beyond the tube length.")
        st.button("+ Add weight", key="cg_add_weight", on_click=add_weight, width="stretch")
        for ident in st.session_state.cg_weight_ids:
            prefix = f"cg_weight_{ident}_"
            with st.expander(f"Weight {ident}", expanded=True):
                name = st.text_input("Component name", key=prefix + "name", max_chars=80)
                kind = st.radio("Weight type", ["Point weight", "Uniform weight"], horizontal=True, key=prefix + "kind")
                magnitude = st.number_input("Total weight (lbf)", min_value=0.0, step=10.0, format="%.4f", key=prefix + "magnitude")
                if kind == "Point weight":
                    position = st.number_input("Position from left (in)", step=1.0, format="%.5f", key=prefix + "position",
                                               help="Use the omitted part's CG position. Negative is left of the tube; values greater than L are right of it.")
                    weights.append(Weight(name.strip() or f"Weight {ident}", "point", magnitude, position))
                else:
                    full = st.checkbox("Full tube length", key=prefix + "full")
                    c1, c2 = st.columns(2)
                    if full:
                        st.session_state[prefix + "full_start"] = 0.0
                        st.session_state[prefix + "full_end"] = float(length)
                        c1.number_input("Start from left (in)", disabled=True, format="%.5f", key=prefix + "full_start")
                        c2.number_input("End from left (in)", disabled=True, format="%.5f", key=prefix + "full_end")
                        start, end = 0.0, length
                    else:
                        start = c1.number_input("Start from left (in)", step=1.0, format="%.5f", key=prefix + "start")
                        end = c2.number_input("End from left (in)", step=1.0, format="%.5f", key=prefix + "end")
                    st.caption("Enter the total weight over this interval, not weight per inch.")
                    weights.append(Weight(name.strip() or f"Weight {ident}", "uniform", magnitude, start, end))
                st.button("Remove weight", key=prefix + "remove", on_click=remove_weight, args=(ident,), width="stretch")
        calculate = st.button("Calculate", type="primary", key="cg_calculate", width="stretch")
        st.button("Load example", key="cg_load_example", on_click=load_example, width="stretch",
                  help="Replace the current inputs with an editable example containing weights beyond both ends.")
        st.caption("Units: inches and pound-force. Do not include the same part in more than one weight entry.")
    return Tube(length, od, id_, tube_weight), weights, calculate


def render_app():
    """Render with cg_ widget keys; no page configuration or solver side effects."""
    import pandas as pd
    import streamlit as st

    st.markdown("""<style>
    .block-container{max-width:1550px;padding-top:1.5rem;padding-bottom:2rem}
    [data-testid="stSidebar"]{min-width:350px;max-width:370px}
    [data-testid="stSidebar"] .block-container{padding-top:1.2rem}
    [data-testid="stSidebar"] h3{font-size:1.04rem;padding-bottom:.2rem}
    [data-testid="stMetricValue"]{font-size:1.5rem}
    @media(max-width:700px){[data-testid="stSidebar"]{min-width:0;max-width:90vw}}
    </style>""", unsafe_allow_html=True)
    initialize_state()
    tube, weights, calculate = input_panel()
    st.title("Tube Center of Gravity")
    st.caption("Straight circular tube · component weights · inches · pound-force")
    try:
        solution = solve_cg(tube, weights)
    except (ValueError, OverflowError) as error:
        st.error(str(error))
        st.stop()
    left, right = st.columns([3.4, 1.0], gap="medium")
    with left:
        st.subheader("Tube schematic")
        st.caption("Horizontal positions to scale. Dashed extensions show coordinates beyond the tube; arrow lengths do not encode weight.")
        st.image(tube_svg(solution), width="stretch")
    with right:
        st.subheader("Cross-section")
        st.image(section_svg(tube), width="stretch")
    if calculate:
        st.toast("Center of gravity recalculated from the current inputs.")
    st.subheader("Assembly results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total assembly weight", f"{number(solution.total_weight, 4)} lbf")
    c2.metric("Added parts weight", f"{number(solution.added_weight, 4)} lbf")
    c3.metric("CG from left end", "—" if solution.cg is None else f"{number(solution.cg)} in")
    c4.metric("CG from right end", "—" if solution.cg is None else f"{number(tube.length-solution.cg)} in")
    if solution.cg is None:
        st.info("Enter a positive tube weight or add a positive component weight. The CG is undefined while total weight is zero.")
    else:
        if solution.cg < 0:
            location = f"The assembly CG is {number(-solution.cg)} in beyond the tube's left end."
        elif solution.cg > tube.length:
            location = f"The assembly CG is {number(solution.cg-tube.length)} in beyond the tube's right end."
        else:
            location = "The assembly CG is within the tube's length."
        st.success(location)
        st.caption("From left: positive toward the right, negative beyond the left end. "
                   "From right: positive toward the left, negative beyond the right end. Green markers show the assembly CG.")
    tabs = st.tabs(["Weight distribution", "Component summary", "Calculations"])
    with tabs[0]:
        if solution.total_weight > 0:
            st.plotly_chart(make_distribution(solution), width="stretch", theme=None, key="cg_distribution",
                            config={"displaylogo": False, "scrollZoom": True,
                                    "toImageButtonOptions": {"format": "png", "filename": "tube_weight_distribution", "scale": 2}})
            st.caption("Point weights and uniform intensities use separate vertical scales. Overlapping uniform weights are added. "
                       "The shaded band marks the tube length; the dashed green line marks the assembly CG.")
        else:
            st.info("The weight distribution appears when a positive weight is entered.")
    with tabs[1]:
        rows = [{"Component": w.name, "Type": "Point" if w.kind == "point" else "Uniform",
                 "Weight (lbf)": w.weight, "Start x (in)": w.start, "End x (in)": w.finish,
                 "Component CG x (in)": w.center,
                 "Assembly weight (%)": w.weight / solution.total_weight * 100 if solution.total_weight else 0.0}
                for w in solution.components]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Tube self-weight is included once. Zero-weight entries are retained here and do not affect the result.")
    with tabs[2]:
        render_calculations(solution)
    st.download_button("Download results as CSV", data=export_csv(solution), file_name="tube_cg_results.csv",
                       mime="text/csv", key="cg_download_csv", width="content")


def main():
    import streamlit as st

    st.set_page_config(page_title="Tube Center of Gravity", layout="wide", initial_sidebar_state="expanded")
    render_app()


if __name__ == "__main__":
    main()
