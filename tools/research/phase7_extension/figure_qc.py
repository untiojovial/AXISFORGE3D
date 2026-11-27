"""Quality control of the figure PDFs (canvas margins, fonts, printed type size, thinnest stroke, fit, colour).

    python -I tools/research/phase7_extension/figure_qc.py publication/intelligent_service_robotics/figures OUT_DIR

Writes OUT_DIR/figure_qc.json and a 300-dpi PNG of each figure for visual inspection."""
import json, re, subprocess, sys, zlib
from pathlib import Path
import numpy as np
from PIL import Image

FIG = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
COL, TW = 216.24, 372.0   # article column width (iicol) and supplement text width, pt
USE = {'fig_overview': ('article', None), 'fig_effects': ('article', None),
       'fig_dev_torsion': ('supplement', 1.0), 'fig_dev_articulated': ('supplement', 1.0),
       'fig_dev_representation': ('supplement', 1.0), 'fig_conf_success_at_budget': ('supplement', 0.75),
       'fig_conf_rates': ('supplement', 1.0), 'fig_conf_primary': ('supplement', 0.9)}

def page_size(pdf):
    info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
    w, h = re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', info).groups(); return float(w), float(h)

def fonts(pdf):
    out = subprocess.run(['pdffonts', str(pdf)], capture_output=True, text=True).stdout.splitlines()[2:]
    return [(l.split()[0], 'yes' in l.split()[-5:-2][0:1] or l.split()[-5] == 'yes') for l in out if l.strip()]

def min_text_height(pdf):
    html = subprocess.run(['pdftotext', '-bbox', str(pdf), '-'], capture_output=True, text=True).stdout
    hs = [float(b) - float(a) for a, b in re.findall(r'yMin="([\d.]+)" xMax="[\d.]+" yMax="([\d.]+)"', html)]
    return min(hs) if hs else None, len(hs)

def line_widths(pdf):
    raw = Path(pdf).read_bytes(); ws = []
    for m in re.finditer(rb'stream\r?\n', raw):
        end = raw.find(b'endstream', m.end())
        try: data = zlib.decompress(raw[m.end():end])
        except Exception: continue
        ws += [float(x) for x in re.findall(rb'(?<![\w.])(\d*\.?\d+) w(?![\w])', data)]
    return sorted(set(ws))

def colour_spaces(pdf):
    raw = Path(pdf).read_bytes()
    return sorted({c.decode() for c in re.findall(rb'/(DeviceRGB|DeviceCMYK|DeviceGray|ICCBased)', raw)})

def margins(png):
    a = np.asarray(Image.open(png).convert('L')).astype(int); nonwhite = a < 245
    rows, cols = np.where(nonwhite)
    h, w = a.shape
    return dict(top=int(rows.min()), bottom=int(h - 1 - rows.max()), left=int(cols.min()), right=int(w - 1 - cols.max()),
                touches_edge=bool(nonwhite[0].any() or nonwhite[-1].any() or nonwhite[:, 0].any() or nonwhite[:, -1].any()))

report = {}
for name, (where, frac) in USE.items():
    pdf = FIG/f'{name}.pdf'; w, h = page_size(pdf)
    scale = 1.0 if frac is None else frac*TW/w
    subprocess.run(['pdftoppm', '-r', '300', '-png', '-singlefile', str(pdf), str(OUT/name)], check=True)
    mh, nwords = min_text_height(pdf); lw = line_widths(pdf)
    report[name] = dict(used_in=where, size_pt=[round(w, 1), round(h, 1)], size_mm=[round(w/72*25.4, 1), round(h/72*25.4, 1)],
                        scale_in_document=round(scale, 3), width_in_document_pt=round(w*scale, 1),
                        fits_width=bool(w*scale <= (COL if where == 'article' else TW) + 0.5),
                        fonts=subprocess.run(['pdffonts', str(pdf)], capture_output=True, text=True).stdout.splitlines()[2:],
                        min_text_box_height_pt=None if mh is None else round(mh*scale, 2), words=nwords,
                        min_line_width_pt=round(min(w for w in lw if w > 0)*scale, 3) if any(w > 0 for w in lw) else None,  # 0 w = fill only line_widths=lw[:8],
                        colour=colour_spaces(pdf), margins_px_300dpi=margins(OUT/f'{name}.png'))
Path(OUT/'figure_qc.json').write_text(json.dumps(report, indent=1))
for k, v in report.items():
    f = [l.split()[0] + ('' if ' yes ' in l else ' NOT-EMBEDDED') for l in v['fonts']]
    print(f"{k:28s} {v['used_in']:10s} {v['size_mm'][0]:5.1f}x{v['size_mm'][1]:5.1f} mm scale {v['scale_in_document']:.2f} fits {v['fits_width']} "
          f"textH {v['min_text_box_height_pt']} lw {v['min_line_width_pt']} {v['colour']} margins {v['margins_px_300dpi']}\n   fonts {sorted(set(f))}")
