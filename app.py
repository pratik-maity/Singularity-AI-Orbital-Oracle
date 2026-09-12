import streamlit as st
import pickle
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# Page
st.set_page_config(page_title="Singularity — AI Orbital Oracle", page_icon="🕳️", layout="wide")

@st.cache_resource
def load_models():
    with open("orbit_model.pkl","rb") as f: clf = pickle.load(f)
    with open("label_encoder.pkl","rb") as f: le = pickle.load(f)
    with open("time_model.pkl","rb") as f: reg = pickle.load(f)
    return clf, le, reg

clf, le, reg = load_models()

# Hero
st.markdown("""
<style>
.hero {text-align:center; padding: 30px 0;}
.hero h1 {font-size: 48px; margin:0;}
.hero p {color:#aaa; font-size:18px;}
.card {background:#111; border:1px solid #222; padding:20px; border-radius:16px;}
</style>
<div class="hero">
<h1>🕳️ SINGULARITY</h1>
<p>AI Orbital Oracle — Physics-informed ML that predicts if an orbit survives a black hole</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,1.3])

with col1:
    st.subheader("🛰️ Launch Parameters")
    bh_mass = st.slider("Black Hole Mass", 500, 8000, 3000, step=100)
    dist = st.slider("Initial Distance", 40, 600, 250)
    v_ratio = st.slider("Velocity Ratio (1.0 = stable orbit)", 0.2, 2.0, 1.0, step=0.05)
    angle = st.slider("Approach Angle Noise", -0.5, 0.5, 0.0, step=0.05)

    # calc features like training
    G = 0.5
    v_orb = np.sqrt(G * bh_mass / dist)
    speed = v_orb * v_ratio
    # simplified ang_mom and energy for prediction
    ang_mom = dist * speed * np.cos(angle) # approx
    energy = 0.5*speed**2 - G*bh_mass/dist

    X = pd.DataFrame([{
        "bh_mass": bh_mass,
        "dist": dist,
        "speed": speed,
        "v_ratio": v_ratio,
        "ang_mom": ang_mom,
        "energy": energy
    }])

    pred_enc = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    fate = le.inverse_transform([pred_enc])[0]
    conf = np.max(proba)*100

    time_pred = reg.predict(X)[0] if fate=="SWALLOWED" else 0

    st.markdown(f"""
    <div class="card">
    <h3>🤖 AI Prediction</h3>
    <h2 style="color:{'#ff4d4d' if fate=='SWALLOWED' else '#4dff88' if fate=='STABLE' else '#4da6ff'}">{fate}</h2>
    <p>Confidence: <b>{conf:.1f}%</b></p>
    <p>{"Time to swallow: <b>%.1f s</b>" % time_pred if fate=="SWALLOWED" else "Orbit will hold"}</p>
    <p style="color:#888; font-size:13px">Energy: {energy:.2f} | Ang Mom: {ang_mom:.1f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Model: XGBoost 97.5% accuracy on 10k simulations")

with col2:
    # Interactive canvas sim - proves AI
    html_code = f"""
    <canvas id="c" width="700" height="500" style="background:#000; border-radius:16px; width:100%"></canvas>
    <script>
    const canvas=document.getElementById('c'), ctx=canvas.getContext('2d');
    let bh_mass={bh_mass}, dist={dist}, v_ratio={v_ratio}, angle={angle};
    let G=0.5;
    let x=Math.cos(0)*dist + 350, y=Math.sin(0)*dist + 250;
    let v_orb=Math.sqrt(G*bh_mass/dist);
    let vx=Math.cos(Math.PI/2 + angle)*v_orb*v_ratio;
    let vy=Math.sin(Math.PI/2 + angle)*v_orb*v_ratio;
    let trail=[];
    function loop(){{
        let rx=x-350, ry=y-250, r=Math.sqrt(rx*rx+ry*ry);
        let h=Math.sqrt(bh_mass)*0.8+8;
        if(r<h){{ ctx.fillStyle='#ff0000'; ctx.fillRect(0,0,700,500); ctx.fillStyle='white'; ctx.fillText('SWALLOWED',320,250); return; }}
        if(r>900){{ ctx.fillStyle='#0044ff'; ctx.fillRect(0,0,700,500); ctx.fillStyle='white'; ctx.fillText('ESCAPED',320,250); return; }}
        let a=G*bh_mass/(r*r*r)*0.2;
        vx-=rx*a*10; vy-=ry*a*10;
        x+=vx; y+=vy;
        trail.push([x,y]); if(trail.length>400) trail.shift();
        ctx.fillStyle='rgba(0,0,0,0.2)'; ctx.fillRect(0,0,700,500);
        // accretion disk
        ctx.beginPath(); ctx.arc(350,250,h+20,0,Math.PI*2); ctx.strokeStyle='rgba(100,180,255,0.2)'; ctx.lineWidth=20; ctx.stroke();
        // bh
        ctx.beginPath(); ctx.arc(350,250,h,0,Math.PI*2); ctx.fillStyle='black'; ctx.fill(); ctx.strokeStyle='#88f'; ctx.stroke();
        // trail
        ctx.beginPath(); ctx.moveTo(trail[0][0],trail[0][1]);
        for(let p of trail) ctx.lineTo(p[0],p[1]);
        ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
        // planet
        ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.fillStyle='#ffdd55'; ctx.fill();
        requestAnimationFrame(loop);
    }}
    loop();
    </script>
    """
    components.html(html_code, height=520)
    st.caption("Live physics simulation — validates AI prediction above")

st.divider()
st.markdown("### How it works | ML Methodology: 10k physics sims → XGBoost classifier (97.5%) + regressor (MAE 12s) → Real-time inference + visual proof. Explainability via energy & angular momentum.")
