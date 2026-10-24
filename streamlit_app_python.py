import bisect, json
from pathlib import Path
import joblib, pandas as pd, streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="Window Sticker · MPG Predictor", page_icon="⛽", layout="wide")

@st.cache_resource
def load():
    try:  # saved artefacts; if scikit-learn versions differ on the server, retrain
        m, meta = joblib.load(HERE / "model.joblib"), json.load(open(HERE / "metrics.json"))
        m.predict(pd.DataFrame([{"displacement": 150, "horsepower": None, "weight": 2500, "acceleration": 15,
                                 "model year": 75, "cylinders": 4, "origin": 1}]))  # sanity check
        return m, meta
    except Exception:
        from train import train
        return train()

model, meta = load()
R = meta["ranges"]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Anybody:wdth,wght@100..150,300..900&family=Hanken+Grotesk:wght@400;500;700&display=swap');
.stApp{background:#f5d90a;color:#14110b}
html,body,[class*="st-"],.stMarkdown,label,p{font-family:'Hanken Grotesk',system-ui,sans-serif}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1200px;padding-top:2rem}
h1.big{font-family:Anybody,Impact,sans-serif;font-stretch:130%;font-weight:900;font-size:clamp(2.4rem,7vw,5.4rem);line-height:.92;max-width:11ch;margin:0 0 10px}
.stSlider label p,.stRadio label p,.stCheckbox label p{font-weight:700;color:#14110b}
div[data-testid="stSlider"]{border-top:3px solid #14110b;padding-top:6px}
div[data-testid="stSlider"] [role="slider"]{background:#fffef8;border:4px solid #14110b}
.sticker{background:#fffef8;border:4px solid #14110b;padding:20px;box-shadow:10px 10px 0 #14110b;color:#14110b}
.sticker h2{font:800 1.05rem Anybody;font-stretch:130%;border-bottom:4px solid #14110b;padding:0 0 8px;margin:0;display:flex;justify-content:space-between}
.odo{display:flex;justify-content:center;gap:4px;margin:18px 0 4px;font-size:clamp(4.5rem,9vw,7rem);line-height:1;height:1em}
.col{height:1em;overflow:hidden;width:.62em;background:#14110b;color:#fffef8;border-radius:4px}
.col>div{animation:roll .9s cubic-bezier(.2,.9,.2,1) both}
.col span{display:block;height:1em;text-align:center;font-family:Anybody;font-weight:900;font-stretch:130%}
@keyframes roll{from{transform:translateY(0)}to{transform:translateY(var(--d))}}
.dot{font-family:Anybody;font-weight:900;line-height:1.1}
.unit{text-align:center;font:800 1.1rem Anybody;font-stretch:130%}
.alt{display:grid;grid-template-columns:1fr 1fr;border-top:2px solid #14110b;border-bottom:2px solid #14110b;margin-top:16px}
.alt div{padding:10px 0;text-align:center}.alt div+div{border-left:2px solid #14110b}
.alt b{display:block;font:800 1.6rem Anybody;font-stretch:130%}.alt span{font-size:.8rem;color:#6d6a5e}
.track{position:relative;height:16px;margin-top:30px;background:linear-gradient(90deg,#c0321f,#f5d90a 50%,#0b6b3a);border:2px solid #14110b}
.pin{position:absolute;top:-22px;border:9px solid transparent;border-top:14px solid #14110b;transform:translateX(-9px);transition:left .9s}
.ends{display:flex;justify-content:space-between;font-size:.75rem;color:#6d6a5e;margin-top:4px}
h2.dy{font:900 clamp(1.8rem,4vw,3rem) Anybody;font-stretch:130%;margin:2rem 0 .3rem}
table.dyno{width:100%;border-collapse:collapse;background:#fffef8;border:4px solid #14110b}
.dyno th,.dyno td{text-align:left;padding:12px 14px;border-bottom:2px solid #14110b;color:#14110b}
.dyno th{background:#14110b;color:#f5d90a;font-size:.85rem}
.dyno tr.win td{background:#fff3a0;font-weight:700}
.dyno .bar div{height:12px;background:#14110b}
.badge{background:#14110b;color:#f5d90a;padding:2px 8px;font-size:.75rem;margin-left:8px}
</style>""", unsafe_allow_html=True)

st.markdown(f'<h1 class="big">What will it sip?</h1><p style="max-width:52ch">Fill in the spec sheet like you’re reading it off a car’s door jamb. A machine-learning model trained on {meta["n"]} real cars prints the window sticker for you, live.</p>', unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")
with left:
    def sl(label, key, unit, step=1):
        lo, hi, med = R[key]
        if step == 1: lo, hi, med = int(lo), int(hi), int(med)
        else: step = 0.1
        return st.slider(f"{label} ({unit})" if unit else label, lo, hi, med, step=step if step != 1 else 1)
    weight = sl("Curb weight", "weight", "lb")
    disp = sl("Engine displacement", "displacement", "cu in")
    hp_unknown = st.checkbox("I don’t know the horsepower (the pipeline fills it in)")
    hp = sl("Horsepower", "horsepower", "hp")
    acc = sl("0–60 mph time", "acceleration", "s", step=0.1)
    year = sl("Model year (19xx)", "model year", "")
    cyl = st.radio("Cylinders", [3, 4, 5, 6, 8], index=1, horizontal=True)
    origin = st.radio("Built in", [1, 2, 3], format_func={1: "USA", 2: "Europe", 3: "Japan"}.get, horizontal=True)

row = pd.DataFrame([{"displacement": disp, "horsepower": None if hp_unknown else hp, "weight": weight,
                     "acceleration": acc, "model year": year, "cylinders": cyl, "origin": origin}])
mpg = float(model.predict(row)[0])
pct = round(bisect.bisect_left(meta["mpg_sorted"], mpg) / len(meta["mpg_sorted"]) * 100)

digits = f"{mpg:04.1f}"
odo = "".join('<span class="dot">.</span>' if c == "." else
              f'<div class="col"><div style="--d:-{int(c) * 10}%">' + "".join(f"<span>{i}</span>" for i in range(10)) + "</div></div>"
              for c in digits)
with right:
    st.markdown(f'<div class="sticker"><h2><span>Estimated fuel economy</span><span>ML</span></h2>'
                f'<div class="odo">{odo}</div><div class="unit">MILES PER GALLON</div>'
                f'<div class="alt"><div><b>{235.215 / mpg:.1f}</b><span>litres per 100 km</span></div>'
                f'<div><b>{mpg * 0.4251:.1f}</b><span>km per litre</span></div></div>'
                f'<p style="margin:18px 0 0;font-size:.9rem">Thriftier than {pct}% of the cars the model learned from.</p>'
                f'<div class="track"><div class="pin" style="left:{pct}%"></div></div>'
                f'<div class="ends"><span>thirstiest car in data</span><span>sippiest</span></div></div>', unsafe_allow_html=True)

rows = "".join(
    f'<tr class="{"win" if r["name"] == meta["best"] else ""}"><td>{r["name"]}'
    + ('<span class="badge">DEPLOYED</span>' if r["name"] == meta["best"] else "")
    + f'</td><td>{r["cv_rmse"]}</td><td>{r["rmse"]}</td><td>{r["mae"]}</td><td>{r["r2"]}</td>'
    f'<td class="bar"><div style="width:{round(r["r2"] * 100)}%"></div></td></tr>' for r in meta["results"])
st.markdown('<h2 class="dy">Dyno results</h2><p style="max-width:62ch">Four algorithms went through the same preprocessing pipeline '
            '(median imputation, standard scaling, one-hot encoding) and 5-fold cross-validation. The lowest CV error is deployed.</p>'
            '<table class="dyno"><thead><tr><th>Algorithm</th><th>CV RMSE</th><th>Test RMSE</th><th>Test MAE</th><th>Test R²</th><th style="width:22%">R² bar</th></tr></thead><tbody>'
            + rows + '</tbody></table>'
            f'<p style="font-size:.9rem;max-width:80ch;margin-top:16px">{meta["n_missing_hp"]} rows had a missing horsepower value (“?”). They are imputed inside the pipeline, the same safeguard used for live predictions.</p>',
            unsafe_allow_html=True)
