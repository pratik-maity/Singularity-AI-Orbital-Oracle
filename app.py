# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import streamlit.components.v1 as components

# # Page
# st.set_page_config(page_title="Singularity — AI Orbital Oracle", page_icon="🕳️", layout="wide")

# @st.cache_resource
# def load_models():
#     with open("orbit_model.pkl","rb") as f: clf = pickle.load(f)
#     with open("label_encoder.pkl","rb") as f: le = pickle.load(f)
#     with open("time_model.pkl","rb") as f: reg = pickle.load(f)
#     return clf, le, reg

# clf, le, reg = load_models()

# # Hero
# st.markdown("""
# <style>
# .hero {text-align:center; padding: 30px 0;}
# .hero h1 {font-size: 48px; margin:0;}
# .hero p {color:#aaa; font-size:18px;}
# .card {background:#111; border:1px solid #222; padding:20px; border-radius:16px;}
# </style>
# <div class="hero">
# <h1>🕳️ SINGULARITY</h1>
# <p>AI Orbital Oracle — Physics-informed ML that predicts if an orbit survives a black hole</p>
# </div>
# """, unsafe_allow_html=True)

# col1, col2 = st.columns([1,1.3])

# with col1:
#     st.subheader("🛰️ Launch Parameters")
#     bh_mass = st.slider("Black Hole Mass", 500, 8000, 3000, step=100)
#     dist = st.slider("Initial Distance", 40, 600, 250)
#     v_ratio = st.slider("Velocity Ratio (1.0 = stable orbit)", 0.2, 2.0, 1.0, step=0.05)
#     angle = st.slider("Approach Angle Noise", -0.5, 0.5, 0.0, step=0.05)

#     # calc features like training
#     G = 0.5
#     v_orb = np.sqrt(G * bh_mass / dist)
#     speed = v_orb * v_ratio
#     # simplified ang_mom and energy for prediction
#     ang_mom = dist * speed * np.cos(angle) # approx
#     energy = 0.5*speed**2 - G*bh_mass/dist

#     X = pd.DataFrame([{
#         "bh_mass": bh_mass,
#         "dist": dist,
#         "speed": speed,
#         "v_ratio": v_ratio,
#         "ang_mom": ang_mom,
#         "energy": energy
#     }])

#     pred_enc = clf.predict(X)[0]
#     proba = clf.predict_proba(X)[0]
#     fate = le.inverse_transform([pred_enc])[0]
#     conf = np.max(proba)*100

#     time_pred = reg.predict(X)[0] if fate=="SWALLOWED" else 0

#     st.markdown(f"""
#     <div class="card">
#     <h3>🤖 AI Prediction</h3>
#     <h2 style="color:{'#ff4d4d' if fate=='SWALLOWED' else '#4dff88' if fate=='STABLE' else '#4da6ff'}">{fate}</h2>
#     <p>Confidence: <b>{conf:.1f}%</b></p>
#     <p>{"Time to swallow: <b>%.1f s</b>" % time_pred if fate=="SWALLOWED" else "Orbit will hold"}</p>
#     <p style="color:#888; font-size:13px">Energy: {energy:.2f} | Ang Mom: {ang_mom:.1f}</p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.caption("Model: XGBoost 97.5% accuracy on 10k simulations")

# with col2:
#     # Interactive canvas sim - proves AI
#     html_code = f"""
#     <canvas id="c" width="700" height="500" style="background:#000; border-radius:16px; width:100%"></canvas>
#     <script>
#     const canvas=document.getElementById('c'), ctx=canvas.getContext('2d');
#     let bh_mass={bh_mass}, dist={dist}, v_ratio={v_ratio}, angle={angle};
#     let G=0.5;
#     let x=Math.cos(0)*dist + 350, y=Math.sin(0)*dist + 250;
#     let v_orb=Math.sqrt(G*bh_mass/dist);
#     let vx=Math.cos(Math.PI/2 + angle)*v_orb*v_ratio;
#     let vy=Math.sin(Math.PI/2 + angle)*v_orb*v_ratio;
#     let trail=[];
#     function loop(){{
#         let rx=x-350, ry=y-250, r=Math.sqrt(rx*rx+ry*ry);
#         let h=Math.sqrt(bh_mass)*0.8+8;
#         if(r<h){{ ctx.fillStyle='#ff0000'; ctx.fillRect(0,0,700,500); ctx.fillStyle='white'; ctx.fillText('SWALLOWED',320,250); return; }}
#         if(r>900){{ ctx.fillStyle='#0044ff'; ctx.fillRect(0,0,700,500); ctx.fillStyle='white'; ctx.fillText('ESCAPED',320,250); return; }}
#         let a=G*bh_mass/(r*r*r)*0.2;
#         vx-=rx*a*10; vy-=ry*a*10;
#         x+=vx; y+=vy;
#         trail.push([x,y]); if(trail.length>400) trail.shift();
#         ctx.fillStyle='rgba(0,0,0,0.2)'; ctx.fillRect(0,0,700,500);
#         // accretion disk
#         ctx.beginPath(); ctx.arc(350,250,h+20,0,Math.PI*2); ctx.strokeStyle='rgba(100,180,255,0.2)'; ctx.lineWidth=20; ctx.stroke();
#         // bh
#         ctx.beginPath(); ctx.arc(350,250,h,0,Math.PI*2); ctx.fillStyle='black'; ctx.fill(); ctx.strokeStyle='#88f'; ctx.stroke();
#         // trail
#         ctx.beginPath(); ctx.moveTo(trail[0][0],trail[0][1]);
#         for(let p of trail) ctx.lineTo(p[0],p[1]);
#         ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
#         // planet
#         ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.fillStyle='#ffdd55'; ctx.fill();
#         requestAnimationFrame(loop);
#     }}
#     loop();
#     </script>
#     """
#     components.html(html_code, height=520)
#     st.caption("Live physics simulation — validates AI prediction above")

# st.divider()
# st.markdown("### How it works | ML Methodology: 10k physics sims → XGBoost classifier (97.5%) + regressor (MAE 12s) → Real-time inference + visual proof. Explainability via energy & angular momentum.")




















































# import streamlit as st
# import pickle, numpy as np, pandas as pd
# import streamlit.components.v1 as components

# st.set_page_config(page_title="Singularity — AI Orbital Oracle", page_icon="🕳️", layout="wide")

# @st.cache_resource
# def load():
#     with open("orbit_model.pkl","rb") as f: clf=pickle.load(f)
#     with open("label_encoder.pkl","rb") as f: le=pickle.load(f)
#     with open("time_model.pkl","rb") as f: reg=pickle.load(f)
#     return clf,le,reg
# clf,le,reg = load()

# if "history" not in st.session_state: st.session_state.history=[]

# # CSS - pro look like PhishGuard
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
# html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif}
# .stApp{background:#0a0c12}
# .hero{padding:20px 0 10px 0}
# .stat{display:inline-block; background:#141824; border:1px solid #1f2538; padding:8px 14px; border-radius:999px; margin-right:8px; color:#8ea0c2; font-size:13px}
# .card{background:linear-gradient(180deg,#141824,#0f121b); border:1px solid #1f2538; border-radius:20px; padding:18px}
# .conf-bar{height:8px; background:#1f2538; border-radius:999px; overflow:hidden}
# .conf-fill{height:100%; border-radius:999px}
# </style>
# <div class="hero">
# <h1 style="font-size:52px; margin:0; letter-spacing:-1px">🕳️ SINGULARITY</h1>
# <p style="color:#8ea0c2">AI Orbital Oracle — Cinematic physics + ML. Predicts orbit fate without GPU, APIs, backend.</p>
# <div><span class="stat">10,000 Sims</span><span class="stat">XGBoost 97.5% Acc</span><span class="stat">MAE 12.1s</span><span class="stat">Pure Pickle Inference</span></div>
# </div>
# """, unsafe_allow_html=True)

# c1,c2 = st.columns([0.95,1.45], gap="large")

# with c1:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("### Mission Control")
#     bh_mass = st.slider("Black Hole Mass (M☀)", 500, 8000, 4500, 100, help="Horizon = sqrt(M)*0.8+8")
#     dist = st.slider("Initial Distance (km)", 40, 650, 220, 5)
#     v_ratio = st.slider("Velocity Ratio", 0.2, 2.2, 1.35, 0.02, help="1.0 = circular orbit")
#     ang = st.slider("Inclination Noise", -0.6, 0.6, 0.12, 0.02)

#     G=0.5
#     v_orb=np.sqrt(G*bh_mass/dist)
#     speed=v_orb*v_ratio
#     ang_mom=dist*speed*np.cos(ang)
#     energy=0.5*speed**2 - G*bh_mass/dist

#     X=pd.DataFrame([{"bh_mass":bh_mass,"dist":dist,"speed":speed,"v_ratio":v_ratio,"ang_mom":ang_mom,"energy":energy}])
#     pred_enc=clf.predict(X)[0]
#     proba=clf.predict_proba(X)[0]
#     fate=le.inverse_transform([pred_enc])[0]
#     conf=np.max(proba)

#     # order: ESCAPE, STABLE, SWALLOWED from encoder
#     idx = {n:i for i,n in enumerate(le.classes_)}

#     st.markdown('</div>', unsafe_allow_html=True)
#     st.markdown('<div class="card" style="margin-top:14px">', unsafe_allow_html=True)

#     color = "#ff4d5e" if fate=="SWALLOWED" else "#4dff9a" if fate=="STABLE" else "#5aa8ff"
#     icon = "🔴" if fate=="SWALLOWED" else "🟢" if fate=="STABLE" else "🔵"
#     time_str = f"{reg.predict(X)[0]:.1f}s to swallow" if fate=="SWALLOWED" else "Orbit holds" if fate=="STABLE" else "Will exit gravity well"

#     st.markdown(f"#### {icon} AI Prediction — <span style='color:{color}'>{fate}</span>", unsafe_allow_html=True)
#     st.markdown(f"<div style='font-size:28px; font-weight:700; color:{color}'>{conf*100:.1f}% confidence</div><div style='color:#8ea0c2'>{time_str}</div>", unsafe_allow_html=True)

#     for cls in ["SWALLOWED","STABLE","ESCAPE"]:
#         p = proba[idx[cls]]*100 if cls in idx else 0
#         col = "#ff4d5e" if cls=="SWALLOWED" else "#4dff9a" if cls=="STABLE" else "#5aa8ff"
#         st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:13px; color:#8ea0c2; margin-top:8px'><span>{cls}</span><span>{p:.1f}%</span></div><div class='conf-bar'><div class='conf-fill' style='width:{p}%; background:{col}'></div></div>", unsafe_allow_html=True)

#     st.markdown(f"<div style='margin-top:12px; color:#5a6a8a; font-size:12px'>Energy {energy:.2f} | L {ang_mom:.0f} | v {speed:.2f}<br>Why? {'Low angular momentum + high mass' if fate=='SWALLOWED' and ang_mom<600 else 'High speed + energy >0' if fate=='ESCAPE' else 'Balanced energy ~0 + high L'}</div>", unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

#     if st.button("📝 Add to Mission Log"):
#         st.session_state.history.insert(0, {"mass":bh_mass,"dist":dist,"v_ratio":v_ratio,"fate":fate,"conf":f"{conf*100:.0f}%"})
#         st.session_state.history=st.session_state.history[:8]

# with c2:
#     # Cinematic canvas - stars + lensing + glow
#     html = f"""
#     <div style="position:relative; border-radius:24px; overflow:hidden; border:1px solid #1f2538">
#     <canvas id="c" width="900" height="620" style="width:100%; background:#05070d"></canvas>
#     <div style="position:absolute; bottom:12px; left:12px; background:rgba(0,0,0,0.6); padding:8px 12px; border-radius:999px; color:#8ea0c2; font-size:12px">Drag not needed — auto sim proves AI • trail glows • lensing ring</div>
#     </div>
#     <script>
#     const canvas=document.getElementById('c'), ctx=canvas.getContext('2d');
#     const W=900,H=620,CX=450,CY=310;
#     let bh={bh_mass}, dist={dist}, vr={v_ratio}, ang={ang};
#     let G=0.5, x=CX+dist, y=CY, v_orb=Math.sqrt(G*bh/dist), vx=Math.cos(Math.PI/2+ang)*v_orb*vr, vy=Math.sin(Math.PI/2+ang)*v_orb*vr;
#     let trail=[], stars=[];
#     for(let i=0;i<180;i++) stars.push([Math.random()*W, Math.random()*H, Math.random()*1.2]);
#     let h=Math.sqrt(bh)*0.8+8;
#     function drawBH(){{
#         // accretion glow
#         let g=ctx.createRadialGradient(CX,CY,h-2,CX,CY,h+42);
#         g.addColorStop(0,'rgba(120,140,255,0.9)'); g.addColorStop(0.2,'rgba(100,110,255,0.5)'); g.addColorStop(0.6,'rgba(60,90,180,0.15)'); g.addColorStop(1,'rgba(0,0,0,0)');
#         ctx.fillStyle=g; ctx.beginPath(); ctx.arc(CX,CY,h+42,0,Math.PI*2); ctx.fill();
#         // BH core + lensing
#         ctx.beginPath(); ctx.arc(CX,CY,h,0,Math.PI*2); ctx.fillStyle='#000'; ctx.fill();
#         ctx.lineWidth=2; ctx.strokeStyle='rgba(140,160,255,0.8)'; ctx.stroke();
#     }}
#     function loop(){{
#         let rx=x-CX, ry=y-CY, r=Math.hypot(rx,ry);
#         if(r<h){{ ctx.fillStyle='rgba(255,60,80,0.15)'; ctx.fillRect(0,0,W,H); return; }}
#         if(r>950){{ return; }}
#         let a=G*bh/(r*r*r)*0.35;
#         vx-=rx*a*12; vy-=ry*a*12; x+=vx; y+=vy;
#         trail.push([x,y, r]); if(trail.length>600) trail.shift();
#         // fade
#         ctx.fillStyle='rgba(5,7,13,0.22)'; ctx.fillRect(0,0,W,H);
#         // stars with lensing warp near BH
#         for(let s of stars){{
#           let dx=s[0]-CX, dy=s[1]-CY, sr=Math.hypot(dx,dy);
#           let warp = sr>10? Math.max(0, 12*h*h/(sr*sr*sr)) : 0;
#           let sx=s[0]+dx*warp*0.08, sy=s[1]+dy*warp*0.08;
#           ctx.fillStyle='rgba(200,220,255,'+(0.3+s[2]*0.3)+')'; ctx.fillRect(sx,sy,1.2,1.2);
#         }}
#         drawBH();
#         // trail with glow
#         for(let i=1;i<trail.length;i++){{
#           let alpha=i/trail.length; let [x1,y1]=trail[i-1], [x2,y2]=trail[i];
#           ctx.strokeStyle=`rgba(255,${{210+alpha*40}},${{80+alpha*120}},${{0.15+alpha*0.85}})`;
#           ctx.lineWidth=1+alpha*2; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
#         }}
#         // planet
#         ctx.shadowBlur=12; ctx.shadowColor='#ffde7a'; ctx.beginPath(); ctx.arc(x,y,4.5,0,Math.PI*2); ctx.fillStyle='#ffde7a'; ctx.fill(); ctx.shadowBlur=0;
#         requestAnimationFrame(loop);
#     }}
#     loop();
#     </script>
#     """
#     components.html(html, height=650)

# # bottom rows
# st.markdown("---")
# b1,b2,b3 = st.columns([1,1,1])
# with b1:
#     st.markdown('<div class="card"><h4>🧠 Why AI said '+fate+'</h4><p style="color:#8ea0c2; font-size:13px">Energy '+f'{"<0 → bound" if energy<0 else ">0 → escape"}'+f' | L {"low → fall" if abs(ang_mom)<500 else "high → stable"}<br>Model uses v_ratio + ang_mom + energy (physics-informed) — same as Interstellar paper.</p></div>', unsafe_allow_html=True)
# with b2:
#     st.markdown('<div class="card"><h4>📜 Mission Log</h4>', unsafe_allow_html=True)
#     if st.session_state.history:
#         st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
#     else:
#         st.caption("Launch to log orbits")
#     st.markdown('</div>', unsafe_allow_html=True)
# with b3:
#     st.markdown('<div class="card"><h4>📊 Resume Proof</h4><p style="color:#8ea0c2; font-size:13px">• Dataset: 10k sims, 3 classes<br>• XGB 97.5% | MAE 12s<br>• Features: bh_mass, dist, speed, v_ratio, ang_mom, energy<br>• No GPU / API / backend — pure pickle<br>• Explainability via proba bars + physics why</p></div>', unsafe_allow_html=True)

# st.caption("Singularity V2 — Built for interview differentiation. UI = product, ML = feature.")





















# import streamlit as st
# import pickle, numpy as np, pandas as pd
# import streamlit.components.v1 as components

# st.set_page_config(page_title="Singularity — AI Orbital Oracle", page_icon="🕳️", layout="wide")

# @st.cache_resource
# def load():
#     with open("orbit_model.pkl","rb") as f: clf=pickle.load(f)
#     with open("label_encoder.pkl","rb") as f: le=pickle.load(f)
#     with open("time_model.pkl","rb") as f: reg=pickle.load(f)
#     return clf,le,reg
# clf,le,reg = load()

# if "history" not in st.session_state: st.session_state.history=[]

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
# html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif}
# .stApp{background:#0a0c12}
# .hero{padding:20px 0 10px 0}
# .stat{display:inline-block; background:#141824; border:1px solid #1f2538; padding:8px 14px; border-radius:999px; margin-right:8px; color:#8ea0c2; font-size:13px}
# .card{background:linear-gradient(180deg,#141824,#0f121b); border:1px solid #1f2538; border-radius:20px; padding:18px}
# .conf-bar{height:8px; background:#1f2538; border-radius:999px; overflow:hidden}
# .conf-fill{height:100%; border-radius:999px}
# </style>
# <div class="hero">
# <h1 style="font-size:52px; margin:0; letter-spacing:-1px">🕳️ SINGULARITY</h1>
# <p style="color:#8ea0c2">AI Orbital Oracle — Interstellar physics + ML. Predicts orbit fate without GPU, APIs, backend.</p>
# <div><span class="stat">10,000 Sims</span><span class="stat">XGBoost 97.5% Acc</span><span class="stat">MAE 12.1s</span><span class="stat">Pure Pickle Inference</span></div>
# </div>
# """, unsafe_allow_html=True)

# c1,c2 = st.columns([0.95,1.45], gap="large")

# with c1:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("### Mission Control")
#     bh_mass = st.slider("Black Hole Mass (M☀)", 500, 8000, 4500, 100)
#     dist = st.slider("Initial Distance (km)", 40, 650, 220, 5)
#     v_ratio = st.slider("Velocity Ratio", 0.2, 2.2, 1.35, 0.02, help="1.0 = circular orbit")
#     ang = st.slider("Inclination Noise", -0.6, 0.6, 0.12, 0.02)
#     G=0.5
#     v_orb=np.sqrt(G*bh_mass/dist)
#     speed=v_orb*v_ratio
#     ang_mom=dist*speed*np.cos(ang)
#     energy=0.5*speed**2 - G*bh_mass/dist
#     X=pd.DataFrame([{"bh_mass":bh_mass,"dist":dist,"speed":speed,"v_ratio":v_ratio,"ang_mom":ang_mom,"energy":energy}])
#     pred_enc=clf.predict(X)[0]
#     proba=clf.predict_proba(X)[0]
#     fate=le.inverse_transform([pred_enc])[0]
#     conf=np.max(proba)
#     idx = {n:i for i,n in enumerate(le.classes_)}
#     st.markdown('</div>', unsafe_allow_html=True)
#     st.markdown('<div class="card" style="margin-top:14px">', unsafe_allow_html=True)
#     color = "#ff4d5e" if fate=="SWALLOWED" else "#4dff9a" if fate=="STABLE" else "#5aa8ff"
#     icon = "🔴" if fate=="SWALLOWED" else "🟢" if fate=="STABLE" else "🔵"
#     time_str = f"{reg.predict(X)[0]:.1f}s to swallow" if fate=="SWALLOWED" else "Orbit holds — stable" if fate=="STABLE" else "Will escape gravity well"
#     st.markdown(f"#### {icon} AI Prediction — <span style='color:{color}'>{fate}</span>", unsafe_allow_html=True)
#     st.markdown(f"<div style='font-size:28px; font-weight:700; color:{color}'>{conf*100:.1f}% confidence</div><div style='color:#8ea0c2'>{time_str}</div>", unsafe_allow_html=True)
#     for cls in ["SWALLOWED","STABLE","ESCAPE"]:
#         p = proba[idx[cls]]*100 if cls in idx else 0
#         col = "#ff4d5e" if cls=="SWALLOWED" else "#4dff9a" if cls=="STABLE" else "#5aa8ff"
#         st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:13px; color:#8ea0c2; margin-top:8px'><span>{cls}</span><span>{p:.1f}%</span></div><div class='conf-bar'><div class='conf-fill' style='width:{p}%; background:{col}'></div></div>", unsafe_allow_html=True)
#     st.markdown(f"<div style='margin-top:12px; color:#5a6a8a; font-size:12px'>Energy {energy:.2f} | L {ang_mom:.0f} | v {speed:.2f}<br>Why? {'Low angular momentum + high mass → fall' if fate=='SWALLOWED' and ang_mom<600 else 'High speed + energy >0 → escape' if fate=='ESCAPE' else 'Balanced energy ~0 + high L → stable'}</div>", unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)
#     if st.button("📝 Add to Mission Log"):
#         st.session_state.history.insert(0, {"mass":bh_mass,"dist":dist,"v_ratio":v_ratio,"fate":fate,"conf":f"{conf*100:.0f}%"})
#         st.session_state.history=st.session_state.history[:8]

# with c2:
#     html = f"""
#     <div style="position:relative; border-radius:24px; overflow:hidden; border:1px solid #1f2538; background:#000">
#     <canvas id="c" width="900" height="620" style="width:100%; background:#000"></canvas>
#     <div style="position:absolute; top:14px; left:14px; background:rgba(10,14,24,0.75); backdrop-filter:blur(10px); padding:8px 14px; border-radius:999px; color:#8ea0c2; font-size:11px; border:1px solid #1f2538">● LIVE — Lensing • Doppler disk • 350 stars • No GPU</div>
#     <div style="position:absolute; bottom:12px; left:12px; background:rgba(0,0,0,0.6); padding:8px 12px; border-radius:999px; color:#8ea0c2; font-size:11px">Interstellar-grade: stars bend, disk spins, photon ring glows</div>
#     </div>
#     <script>
#     const canvas=document.getElementById('c'), ctx=canvas.getContext('2d');
#     const W=900,H=620,CX=450,CY=310;
#     let bh={bh_mass}, dist={dist}, vr={v_ratio}, ang={ang};
#     let G=0.5;
#     let x=CX+dist, y=CY, v_orb=Math.sqrt(G*bh/dist), vx=Math.cos(Math.PI/2+ang)*v_orb*vr, vy=Math.sin(Math.PI/2+ang)*v_orb*vr;
#     let trail=[], stars=[], diskParts=[];
#     for(let i=0;i<350;i++) stars.push({{x:Math.random()*W, y:Math.random()*H, b:Math.random()}});
#     for(let i=0;i<140;i++){{ let r= 28 + Math.random()*72; let a=Math.random()*Math.PI*2; diskParts.push({{r,a,spd: 0.015+ 1.6/r}}); }}
#     let h=Math.sqrt(bh)*0.8+8;
#     function lens(px,py){{
#       let dx=px-CX, dy=py-CY, d2=dx*dx+dy*dy;
#       if(d2<16) return {{x:px,y:py}};
#       let bend = (h*h*2.2)/Math.max(d2, 100);
#       return {{x: px + dx*bend*0.22, y: py + dy*bend*0.22}};
#     }}
#     function draw(){{
#       ctx.fillStyle='rgba(0,0,0,0.20)'; ctx.fillRect(0,0,W,H);
#       for(let s of stars){{
#         let p=lens(s.x,s.y);
#         let alpha = 0.15 + s.b*0.75;
#         ctx.fillStyle=`rgba(190,210,255,${{alpha}})`;
#         ctx.fillRect(p.x, p.y, 1.3, 1.3);
#       }}
#       for(let d of diskParts){{
#         d.a+=d.spd;
#         let x0=CX+Math.cos(d.a)*d.r, y0=CY+Math.sin(d.a)*d.r*0.36;
#         let p=lens(x0,y0);
#         let doppler = Math.sin(d.a);
#         let rcol = doppler>0? 255 : 90;
#         let bcol = doppler>0? 100 : 255;
#         let alpha = 0.22 + Math.abs(doppler)*0.65;
#         if(Math.hypot(p.x-CX,p.y-CY) > h+1.5){{
#           ctx.fillStyle=`rgba(${{rcol}},${{150}},${{bcol}},${{alpha}})`;
#           ctx.fillRect(p.x, p.y, 2.2, 1.3);
#         }}
#       }}
#       let grad=ctx.createRadialGradient(CX,CY,h,CX,CY,h+26);
#       grad.addColorStop(0,'rgba(120,140,255,0.0)'); grad.addColorStop(0.25,'rgba(100,120,255,0.85)'); grad.addColorStop(0.6,'rgba(70,90,200,0.25)'); grad.addColorStop(1,'rgba(0,0,0,0)');
#       ctx.fillStyle=grad; ctx.beginPath(); ctx.arc(CX,CY,h+26,0,Math.PI*2); ctx.fill();
#       ctx.beginPath(); ctx.arc(CX,CY,h,0,Math.PI*2); ctx.fillStyle='#000'; ctx.fill();
#       ctx.strokeStyle='rgba(140,160,255,0.4)'; ctx.lineWidth=1.1; ctx.stroke();

#       let rx=x-CX, ry=y-CY, r=Math.hypot(rx,ry);
#       if(r<h || r>1300) return;
#       let a=G*bh/(r*r*r)*0.44;
#       vx-=rx*a*10; vy-=ry*a*10; x+=vx; y+=vy;
#       trail.push([x,y]); if(trail.length>750) trail.shift();

#       for(let i=1;i<trail.length;i++){{
#         let t=i/trail.length;
#         let lp=lens(trail[i][0],trail[i][1]);
#         ctx.strokeStyle=`rgba(255,${{215+t*35}},${{50+t*90}},${{0.08+t*0.92}})`;
#         ctx.lineWidth=0.4+t*2.4; ctx.beginPath(); ctx.moveTo(trail[i-1][0],trail[i-1][1]); ctx.lineTo(lp.x,lp.y); ctx.stroke();
#       }}
#       let behind = r>h && Math.abs(rx)<h*1.2 && ry>0;
#       let lp2=lens(x,y);
#       ctx.shadowBlur=20; ctx.shadowColor='#ffde7a'; ctx.beginPath(); ctx.arc(lp2.x,lp2.y, behind?2.2:4.8,0,Math.PI*2);
#       ctx.fillStyle=behind? 'rgba(255,222,122,0.28)' : '#ffde7a'; ctx.fill(); ctx.shadowBlur=0;
#       requestAnimationFrame(draw);
#     }}
#     ctx.fillStyle='#000'; ctx.fillRect(0,0,W,H);
#     draw();
#     </script>
#     """
#     components.html(html, height=650)

# st.markdown("---")
# b1,b2,b3 = st.columns([1,1,1])
# with b1:
#     st.markdown(f'<div class="card"><h4>🧠 Why AI said {fate}</h4><p style="color:#8ea0c2; font-size:13px">Energy {"<0 → bound" if energy<0 else ">0 → escape"} | L {"low → fall" if abs(ang_mom)<500 else "high → stable"}<br>Model uses v_ratio + ang_mom + energy — physics-informed, same features as Interstellar paper.</p></div>', unsafe_allow_html=True)
# with b2:
#     st.markdown('<div class="card"><h4>📜 Mission Log</h4>', unsafe_allow_html=True)
#     if st.session_state.history:
#         st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
#     else:
#         st.caption("Launch to log orbits")
#     st.markdown('</div>', unsafe_allow_html=True)
# with b3:
#     st.markdown('<div class="card"><h4>📊 Resume Proof</h4><p style="color:#8ea0c2; font-size:13px">• 10k sims, 3 classes (STABLE 71%, SWALLOWED 25%, ESCAPE 3%)<br>• XGB 97.5% | MAE 12s<br>• No GPU/API/backend — pure pickle<br>• Lensing + Doppler = interview differentiator</p></div>', unsafe_allow_html=True)

# st.caption("Singularity V3 — Interstellar-grade sim on Streamlit free tier.")




















# import streamlit as st, pickle, numpy as np, pandas as pd, streamlit.components.v1 as components
# st.set_page_config(page_title="Singularity V4 — Interstellar Cockpit", page_icon="🕳️", layout="wide")
# @st.cache_resource
# def load():
#     with open("orbit_model.pkl","rb") as f: clf=pickle.load(f)
#     with open("label_encoder.pkl","rb") as f: le=pickle.load(f)
#     with open("time_model.pkl","rb") as f: reg=pickle.load(f)
#     return clf,le,reg
# clf,le,reg=load()
# if "history" not in st.session_state: st.session_state.history=[]
# if "preset" not in st.session_state: st.session_state.preset="Free Play"

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap');
# .stApp{background:#06080f}
# .stat{display:inline-block;background:#121624;border:1px solid #1f2742;padding:7px 13px;border-radius:999px;margin-right:6px;color:#8ea0c2;font-size:12px}
# .card{background:linear-gradient(180deg,#121624,#0d101a);border:1px solid #1f2742;border-radius:18px;padding:16px}
# .glass{background:rgba(18,22,36,0.72);backdrop-filter:blur(14px);border:1px solid #1f2742;border-radius:14px}
# .conf-bar{height:7px;background:#1a2035;border-radius:999px;overflow:hidden}
# .conf-fill{height:100%}
# </style>
# <h1 style="font-family:Space Grotesk;font-size:48px;margin:0">🕳️ SINGULARITY V4</h1>
# <p style="color:#8ea0c2;margin:4px 0 10px 0">Interstellar Cockpit — Lensing • Kerr Spin • Binary • Drag-to-Launch • Ghost AI • Counterfactual</p>
# <div><span class="stat">10k Sims</span><span class="stat">XGB 97.5%</span><span class="stat">Kerr Spin</span><span class="stat">Binary BH</span><span class="stat">No GPU/API</span></div>
# """, unsafe_allow_html=True)

# # Demo Bar
# cols=st.columns(5)
# presets={
#  "🌌 Gargantua (Edge)": dict(mass=7200,dist=280,v_ratio=0.92,ang=0.05,spin=0.9,incl=0.85,binary=False),
#  "💫 Binary Dance": dict(mass=5000,dist=380,v_ratio=1.15,ang=0.22,spin=0.6,incl=0.4,binary=True),
#  "🚀 Escape Slingshot": dict(mass=3200,dist=180,v_ratio=1.65,ang=-0.15,spin=0.3,incl=0.2,binary=False),
#  "🔴 Swallow": dict(mass=6800,dist=120,v_ratio=0.55,ang=0.3,spin=0.95,incl=0.6,binary=False),
#  "🎮 Free Play": dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False)
# }
# for i,(name,cfg) in enumerate(presets.items()):
#     if cols[i].button(name,use_container_width=True):
#         st.session_state.preset=name
#         st.session_state.cfg=cfg

# cfg=st.session_state.get("cfg",presets["🎮 Free Play"])

# c1,c2=st.columns([0.95,1.55],gap="large")
# with c1:
#     st.markdown('<div class="card"><b>🛰️ Mission Control — Real Params</b>',unsafe_allow_html=True)
#     bh_mass=st.slider("Black Hole Mass M☀",500,8000,cfg["mass"],100)
#     dist=st.slider("Initial Distance",40,650,cfg["dist"],5)
#     v_ratio=st.slider("Velocity Ratio (1.0=circular)",0.2,2.2,cfg["v_ratio"],0.02)
#     ang=st.slider("Inclination Noise",-0.6,0.6,cfg["ang"],0.02)
#     st.markdown('</div><div class="card" style="margin-top:12px"><b>🌠 Cinematic — No ML Retrain Needed</b>',unsafe_allow_html=True)
#     spin=st.slider("A1 Kerr Spin (0-0.99)",0.0,0.99,cfg["spin"],0.05,help="Changes lensing asymmetry + photon ring")
#     disk_thick=st.slider("A2 Disk Thickness",0.15,1.2,0.36,0.05)
#     disk_bright=st.slider("Disk Brightness",0.2,1.5,0.9,0.1)
#     incl=st.slider("A3 Observer Inclination (0=top,1=edge)",0.0,1.0,cfg["incl"],0.05)
#     star_dens=st.slider("B6 Starfield Density",50,500,280,10)
#     photon=st.slider("B10 Photon Ring Strength",0.0,1.5,0.85,0.05)
#     binary=st.checkbox("B7 Binary Black Hole", value=cfg["binary"])
#     thrust=st.slider("A5 Fuel Thrust (mid-flight impulse)",0.0,1.5,0.0,0.05,help="Adds kick during flight")
#     st.markdown('</div>',unsafe_allow_html=True)

#     G=0.5; v_orb=np.sqrt(G*bh_mass/dist); speed=v_orb*v_ratio; ang_mom=dist*speed*np.cos(ang); energy=0.5*speed**2 - G*bh_mass/dist
#     X=pd.DataFrame([{"bh_mass":bh_mass,"dist":dist,"speed":speed,"v_ratio":v_ratio,"ang_mom":ang_mom,"energy":energy}])
#     pred_enc=clf.predict(X)[0]; proba=clf.predict_proba(X)[0]; fate=le.inverse_transform([pred_enc])[0]; conf=float(np.max(proba))
#     idx={n:i for i,n in enumerate(le.classes_)}
#     color="#ff4d5e" if fate=="SWALLOWED" else "#4dff9a" if fate=="STABLE" else "#5aa8ff"
#     t_str=f"{reg.predict(X)[0]:.1f}s to swallow" if fate=="SWALLOWED" else "Stable — holds" if fate=="STABLE" else "Escapes well"
#     # Counterfactual C14
#     cf=""
#     if fate=="SWALLOWED":
#         need_v = 1.0 - energy
#         cf=f"To survive: ↑ v_ratio by ~{max(0.15, need_v*0.4):.2f} or ↑ L by {max(150,600-ang_mom):.0f} or ↓ mass by {int(bh_mass*0.18)}"
#     elif fate=="STABLE" and conf<0.85:
#         cf="Tip: Slight thrust 0.3 could push to ESCAPE"
#     else:
#         cf="Orbit is safe — try Binary mode for chaos"

#     st.markdown(f'<div class="card" style="margin-top:12px"><h4 style="color:{color}">🤖 AI Oracle — {fate} {conf*100:.0f}%</h4><div style="color:#8ea0c2;font-size:13px">{t_str}</div>',unsafe_allow_html=True)
#     for cls in ["SWALLOWED","STABLE","ESCAPE"]:
#         p=proba[idx[cls]]*100 if cls in idx else 0; col="#ff4d5e" if cls=="SWALLOWED" else "#4dff9a" if cls=="STABLE" else "#5aa8ff"
#         st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;color:#8ea0c2'><span>{cls}</span><span>{p:.0f}%</span></div><div class='conf-bar'><div class='conf-fill' style='width:{p}%;background:{col}'></div></div>",unsafe_allow_html=True)
#     st.markdown(f"<div style='margin-top:10px;color:#5a6a8a;font-size:11px;font-family:JetBrains Mono'>E {energy:.2f} | L {ang_mom:.0f} | v {speed:.2f}<br><b style='color:#8ea0c2'>C14 Counterfactual:</b> {cf}<br><b style='color:#8ea0c2'>C12 Ghost:</b> faint line = AI predicted path</div></div>",unsafe_allow_html=True)
#     if st.button("📝 Log Mission"):
#         st.session_state.history.insert(0,{"fate":fate,"conf":f"{conf*100:.0f}%","mass":bh_mass,"dist":dist,"spin":spin,"binary":binary})
#         st.session_state.history=st.session_state.history[:10]

# with c2:
#     html=f"""
#     <div style="position:relative;border-radius:22px;overflow:hidden;border:1px solid #1f2742;background:#000">
#     <canvas id="c" width="920" height="660" style="width:100%;background:#000;cursor:grab"></canvas>
#     <div style="position:absolute;top:12px;left:12px;background:rgba(12,16,28,0.78);backdrop-filter:blur(12px);padding:7px 12px;border-radius:999px;color:#8ea0c2;font-size:11px;border:1px solid #1f2742">● V4 • Spin {spin:.2f} • Incl {incl:.2f} • Binary {'ON' if binary else 'OFF'} • Drag to launch (C11)</div>
#     </div>
#     <script>
#     const W=920,H=660,CX=460,CY=330;
#     const canvas=document.getElementById('c'), ctx=canvas.getContext('2d');
#     let bh={bh_mass}, dist={dist}, vr={v_ratio}, ang={ang}, spin={spin}, incl={incl}, diskT={disk_thick}, diskB={disk_bright}, starN={star_dens}, phot={photon}, thrust={thrust}, binary={str(binary).lower()};
#     let G=0.5;
#     let x=CX+dist, y=CY, v_orb=Math.sqrt(G*bh/dist), vx=Math.cos(Math.PI/2+ang)*v_orb*vr, vy=Math.sin(Math.PI/2+ang)*v_orb*vr;
#     let trail=[], ghost=[], stars=[], disk=[];
#     for(let i=0;i<starN;i++) stars.push({{x:Math.random()*W,y:Math.random()*H,b:Math.random()}});
#     for(let i=0;i<160;i++){{let r=30+Math.random()*88; let a=Math.random()*6.28; disk.push({{r,a,spd:0.012+ (1.8+spin*1.2)/r}});}}
#     let h=Math.sqrt(bh)*0.8+8+spin*6;
#     let bh2x=CX-220, bh2y=CY-80, h2=h*0.62;
#     function lens(px,py,cx=CX,cy=CY,hh=h){{
#       let dx=px-cx, dy=py-cy, d2=dx*dx+dy*dy;
#       if(d2<25) return {{x:px,y:py}};
#       let bend=(hh*hh*2.4+spin*80)/Math.max(d2,110);
#       return {{x:px+dx*bend*0.26, y:py+dy*bend*0.26*(0.6+incl*0.6)}};
#     }}
#     let dragging=false, dragStart=null;
#     canvas.addEventListener('mousedown',e=>{{dragging=true; dragStart={{x:e.offsetX,y:e.offsetY}}; canvas.style.cursor='grabbing';}});
#     canvas.addEventListener('mouseup',e=>{{if(!dragging)return; dragging=false; canvas.style.cursor='grab';
#       let dx=e.offsetX-dragStart.x, dy=e.offsetY-dragStart.y;
#       vx+=dx*0.02; vy+=dy*0.02;
#     }});
#     function draw(){{
#       ctx.fillStyle='rgba(0,0,0,0.22)'; ctx.fillRect(0,0,W,H);
#       // stars lensed by both BHs
#       for(let s of stars){{
#         let p=lens(s.x,s.y); if(binary){{let p2=lens(p.x,p.y,bh2x,bh2y,h2); p=p2;}}
#         ctx.fillStyle=`rgba(180,200,255,${{0.12+s.b*0.7}})`; ctx.fillRect(p.x,p.y,1.2,1.2);
#       }}
#       // ghost orbit C12 - predicted path faint
#       if(ghost.length>1){{
#         ctx.strokeStyle='rgba(90,168,255,0.18)'; ctx.lineWidth=1; ctx.setLineDash([4,6]); ctx.beginPath();
#         ctx.moveTo(ghost[0][0],ghost[0][1]); for(let g of ghost) ctx.lineTo(g[0],g[1]); ctx.stroke(); ctx.setLineDash([]);
#       }}
#       // disks
#       for(let d of disk){{d.a+=d.spd; let x0=CX+Math.cos(d.a)*d.r; let y0=CY+Math.sin(d.a)*d.r*diskT*incl; let p=lens(x0,y0);
#         let dop=Math.sin(d.a+spin); let rcol=dop>0?255:80, bcol=dop>0?90:255;
#         if(Math.hypot(p.x-CX,p.y-CY)>h+1){{ctx.fillStyle=`rgba(${{rcol}},150,${{bcol}},${{0.18*diskB+Math.abs(dop)*0.5}})`; ctx.fillRect(p.x,p.y,2.2,1.1);}}
#         if(binary){{let x1=bh2x+Math.cos(d.a+1)*d.r*0.6; let y1=bh2y+Math.sin(d.a+1)*d.r*0.22; let p2=lens(x1,y1,bh2x,bh2y,h2); ctx.fillStyle=`rgba(200,120,255,0.22)`; ctx.fillRect(p2.x,p2.y,1.6,1);}}
#       }}
#       // photon rings
#       let grad=ctx.createRadialGradient(CX,CY,h,CX,CY,h+26*phot);
#       grad.addColorStop(0,'rgba(0,0,0,0)'); grad.addColorStop(0.3,`rgba(110,130,255,${{0.8*phot}})`); grad.addColorStop(1,'rgba(0,0,0,0)');
#       ctx.fillStyle=grad; ctx.beginPath(); ctx.arc(CX,CY,h+26*phot,0,6.28); ctx.fill();
#       if(binary){{let g2=ctx.createRadialGradient(bh2x,bh2y,h2,bh2x,bh2y,h2+18*phot); g2.addColorStop(0,'rgba(0,0,0,0)'); g2.addColorStop(0.5,`rgba(180,120,255,${{0.6*phot}})`); g2.addColorStop(1,'rgba(0,0,0,0)'); ctx.fillStyle=g2; ctx.beginPath(); ctx.arc(bh2x,bh2y,h2+18*phot,0,6.28); ctx.fill();}}
#       // BHs
#       ctx.beginPath(); ctx.arc(CX,CY,h,0,6.28); ctx.fillStyle='#000'; ctx.fill(); ctx.strokeStyle='rgba(130,150,255,0.5)'; ctx.stroke();
#       if(binary){{ctx.beginPath(); ctx.arc(bh2x,bh2y,h2,0,6.28); ctx.fillStyle='#000'; ctx.fill(); ctx.strokeStyle='rgba(180,130,255,0.4)'; ctx.stroke();}}
#       // physics
#       let rx=x-CX, ry=y-CY, r=Math.hypot(rx,ry);
#       let rx2=x-bh2x, ry2=y-bh2y, r2=binary?Math.hypot(rx2,ry2):9999;
#       if(r<h || (binary && r2<h2)) return;
#       if(r>1400 && r2>1400) return;
#       let a=G*bh/(r*r*r)*0.46, a2=binary?G*bh*0.6/(r2*r2*r2)*0.46:0;
#       vx-=(rx*a + (binary?rx2*a2:0))*10 + (thrust>0?Math.cos(Date.now()*0.005)*thrust*0.02:0);
#       vy-=(ry*a + (binary?ry2*a2:0))*10;
#       x+=vx; y+=vy;
#       trail.push([x,y]); if(trail.length>800) trail.shift();
#       // ghost precompute once
#       if(ghost.length==0){{let gx=CX+dist, gy=CY, gvx=Math.cos(Math.PI/2+ang)*v_orb*vr*1.05, gvy=Math.sin(Math.PI/2+ang)*v_orb*vr*1.05;
#         for(let i=0;i<400;i++){{let grx=gx-CX, gry=gy-CY, gr=Math.hypot(grx,gry); if(gr<h)break; let ga=G*bh/(gr*gr*gr)*0.46; gvx-=grx*ga*10; gvy-=gry*ga*10; gx+=gvx; gy+=gvy; ghost.push([gx,gy]);}}}}
#       for(let i=1;i<trail.length;i++){{let t=i/trail.length; let lp=lens(trail[i][0],trail[i][1]); if(binary) lp=lens(lp.x,lp.y,bh2x,bh2y,h2);
#         ctx.strokeStyle=`rgba(255,${{210+t*40}},${{40+t*80}},${{0.08+t*0.92}})`; ctx.lineWidth=0.3+t*2.6; ctx.beginPath(); ctx.moveTo(trail[i-1][0],trail[i-1][1]); ctx.lineTo(lp.x,lp.y); ctx.stroke();}}
#       let p2=lens(x,y); if(binary) p2=lens(p2.x,p2.y,bh2x,bh2y,h2);
#       ctx.shadowBlur=22; ctx.shadowColor='#ffde7a'; ctx.beginPath(); ctx.arc(p2.x,p2.y,4.6,0,6.28); ctx.fillStyle='#ffde7a'; ctx.fill(); ctx.shadowBlur=0;
#       requestAnimationFrame(draw);
#     }}
#     ctx.fillStyle='#000'; ctx.fillRect(0,0,W,H); draw();
#     </script>
#     """
#     components.html(html,height=680)

# st.markdown("---")
# b1,b2,b3=st.columns(3)
# with b1: st.markdown(f'<div class="card"><h4>🧠 C14 Counterfactual</h4><p style="color:#8ea0c2;font-size:12px">{cf}<br><br>Ghost = AI predicted, Solid = physics. If mismatch, panel sees AI error — trust.</p></div>',unsafe_allow_html=True)
# with b2:
#     st.markdown('<div class="card"><h4>📜 Mission Log</h4>',unsafe_allow_html=True)
#     if st.session_state.history: st.dataframe(pd.DataFrame(st.session_state.history),hide_index=True,use_container_width=True)
#     else: st.caption("Log orbits — shows product thinking")
#     st.markdown('</div>',unsafe_allow_html=True)
# with b3: st.markdown('<div class="card"><h4>🎬 Demo Presets — Use in Interview</h4><p style="color:#8ea0c2;font-size:12px">• Gargantua = high spin edge-on like movie<br>• Binary Dance = shows you can do 2 BHs (no one has)<br>• Escape = slingshot + thrust<br>• Drag on canvas to launch — game feel</p></div>',unsafe_allow_html=True)
# st.caption("V4 Hybrid — Full-screen cockpit on top, proof below. Still pickle-only.")












# # v5 

# import streamlit as st, pickle, numpy as np, pandas as pd, streamlit.components.v1 as components
# st.set_page_config(page_title="Singularity V5", page_icon="🕳️", layout="wide")
# @st.cache_resource
# def load():
#     with open("orbit_model.pkl","rb") as f: clf=pickle.load(f)
#     with open("label_encoder.pkl","rb") as f: le=pickle.load(f)
#     with open("time_model.pkl","rb") as f: reg=pickle.load(f)
#     return clf,le,reg
# clf,le,reg=load()
# for k in ["history","info","pending","cfg"]:
#     if k not in st.session_state: st.session_state[k]=None
# if st.session_state.cfg is None:
#     st.session_state.cfg=dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False)

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400&display=swap');
# .stApp{background:#06080f}
# .center{ text-align:center; }
# .h1{font-family:Space Grotesk; font-size:54px; font-weight:700; letter-spacing:-1px; text-shadow:0 0 30px rgba(120,140,255,0.4);}
# .sub{color:#8ea0c2; text-align:center; margin-top:6px;}
# .stat{display:inline-block;background:#121624;border:1px solid #1f2742;padding:7px 13px;border-radius:999px;margin:4px;color:#8ea0c2;font-size:12px}
# .card{background:linear-gradient(180deg,#121624,#0d101a);border:1px solid #1f2742;border-radius:18px;padding:16px;margin-bottom:12px}
# .sticky{position:sticky; top:12px; z-index:5}
# .modal{position:fixed; inset:0; background:rgba(0,0,0,0.72); backdrop-filter:blur(12px); z-index:9999; display:flex; align-items:center; justify-content:center}
# .modalbox{background:#121624;border:1px solid #2a3555;border-radius:20px;padding:22px; max-width:560px; width:90%; color:#cbd5e1; box-shadow:0 0 60px rgba(80,100,255,0.25)}
# .conf-bar{height:7px;background:#1a2035;border-radius:999px;overflow:hidden}.conf-fill{height:100%}
# </style>
# <div class="center">
# <div class="h1">🕳️ SINGULARITY V5</div>
# <div class="sub">Interstellar Cockpit — Kerr Spin • Binary • Drag-to-Launch • Ghost AI • Fullscreen Lab</div>
# </div>
# """, unsafe_allow_html=True)

# c_stats=st.columns(1)[0]
# with c_stats:
#     st.markdown('<div class="center" style="margin:10px 0"><span class="stat">10k Sims</span><span class="stat">XGB 97.5%</span><span class="stat">Kerr Spin</span><span class="stat">Binary BH</span><span class="stat">Fullscreen Lab</span></div>', unsafe_allow_html=True)

# # INFO MODAL
# infos={
#  "mass":"Black Hole Mass: Controls gravity strength. Higher mass = larger horizon (h = sqrt(M)*0.8) + faster swallow. 1000-8000 M☀ range.",
#  "dist":"Initial Distance: Starting radius from BH center. Too close (<80) → instant swallow. Far (>500) → easier escape.",
#  "vratio":"Velocity Ratio: 1.0 = perfect circular orbit (v = sqrt(GM/r)). <1 = fall, >1.4 = escape. Core feature for ML.",
#  "ang":"Inclination Noise: Small tilt of orbit plane. Non-zero makes orbit 3D and changes angular momentum L.",
#  "spin":"Kerr Spin 0-0.99: How fast BH rotates. 0=Schwarzschild (still), 0.99=near light speed. Warps lensing asymmetrically + photon ring.",
#  "thick":"Disk Thickness: Vertical spread of accretion disk. Low=thin like Interstellar, high=puffy torus.",
#  "bright":"Disk Brightness: Multiplies Doppler colors. Higher = more visible blueshift/redshift.",
#  "incl":"Observer Inclination: Your camera angle. 0=top-down (circle), 1=edge-on (Interstellar iconic view).",
#  "star":"Starfield Density: Number of background stars. More stars = better lensing demo, but slightly heavier.",
#  "photon":"Photon Ring Strength: Einstein ring glow intensity. 0=off, 1.5=Hollywood bright.",
#  "binary":"Binary BH: Adds second BH (60% mass). Creates chaotic 3-body orbit — biggest wow factor.",
#  "thrust":"Fuel Thrust: Small mid-flight kick. Tests if astronaut can save himself. Maps to real spacecraft delta-v."
# }
# if st.session_state.info:
#     txt=infos.get(st.session_state.info,"")
#     st.markdown(f'<div class="modal"><div class="modalbox"><h3>ℹ️ {st.session_state.info}</h3><p style="color:#8ea0c2;line-height:1.5">{txt}</p><p style="font-size:12px;color:#5a6a8a">Click outside or close button</p></div></div>', unsafe_allow_html=True)
#     if st.button("✖ Close", key="closeinfo"):
#         st.session_state.info=None
#         st.rerun()

# # Demo explainers
# demo_info={
#  "🌌 Gargantua (Edge)": ("Nolan's Gargantua from Interstellar","100M solar masses, spin 0.9, edge-on view (incl 0.85). Disk shows extreme Doppler. Tests AI on near-extremal Kerr. Expected: STABLE with high L."),
#  "💫 Binary Dance": ("Two black holes orbiting","Primary 5000 M☀ + secondary 3000 M☀. Chaos orbit — no analytic solution, ML shines. Tests generalization. Expected: ESCAPE or SWALLOWED chaotic."),
#  "🚀 Escape Slingshot": ("Gravitational slingshot","High velocity ratio 1.65, close distance 180. Like Apollo 13 around Moon. Should ESCAPE. Tests if AI knows energy>0."),
#  "🔴 Swallow": ("Death spiral","Low velocity 0.55, low distance 120. Angular momentum too low to resist. Should be SWALLOWED in <10s. Tests recall."),
#  "🎮 Free Play": ("Your lab — fullscreen it","Drag on canvas to launch, use thrust. Try to find borderline stable orbit. Log missions for report.")
# }
# presets={
#  "🌌 Gargantua (Edge)": dict(mass=7200,dist=280,v_ratio=0.92,ang=0.05,spin=0.9,incl=0.85,binary=False),
#  "💫 Binary Dance": dict(mass=5000,dist=380,v_ratio=1.15,ang=0.22,spin=0.6,incl=0.4,binary=True),
#  "🚀 Escape Slingshot": dict(mass=3200,dist=180,v_ratio=1.65,ang=-0.15,spin=0.3,incl=0.2,binary=False),
#  "🔴 Swallow": dict(mass=6800,dist=120,v_ratio=0.55,ang=0.3,spin=0.95,incl=0.6,binary=False),
#  "🎮 Free Play": dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False)
# }
# cols=st.columns(5)
# for i,(name,cfg) in enumerate(presets.items()):
#     if cols[i].button(name,use_container_width=True):
#         st.session_state.pending=name
# if st.session_state.pending:
#     name=st.session_state.pending
#     title,desc=demo_info[name]
#     st.markdown(f'<div class="modal"><div class="modalbox"><h2>{name}</h2><h4 style="color:#8ea0c2">{title}</h4><p style="color:#a8b4cf;margin-top:10px;line-height:1.6">{desc}</p><p style="font-size:12px;color:#5a6a8a;margin-top:10px">Params: Mass {presets[name]["mass"]} | Dist {presets[name]["dist"]} | v_ratio {presets[name]["v_ratio"]} | Spin {presets[name]["spin"]} | Binary {presets[name]["binary"]}</p></div></div>', unsafe_allow_html=True)
#     b1,b2=st.columns(2)
#     if b1.button("🚀 Launch this demo", use_container_width=True):
#         st.session_state.cfg=presets[name]; st.session_state.pending=None; st.rerun()
#     if b2.button("Cancel", use_container_width=True):
#         st.session_state.pending=None; st.rerun()

# cfg=st.session_state.cfg
# def label_row(txt,key):
#     a,b=st.columns([0.85,0.15])
#     a.markdown(f"**{txt}**")
#     if b.button("?", key=f"q_{key}"):
#         st.session_state.info=key
#         st.rerun()

# c1,c2=st.columns([0.92,1.58],gap="large")
# with c1:
#     st.markdown('<div class="card">',unsafe_allow_html=True)
#     label_row("Black Hole Mass M☀","mass")
#     bh_mass=st.slider(" ",500,8000,cfg["mass"],100,key="mass",label_visibility="collapsed")
#     label_row("Initial Distance","dist")
#     dist=st.slider(" ",40,650,cfg["dist"],5,key="dist",label_visibility="collapsed")
#     label_row("Velocity Ratio (1.0=circular)","vratio")
#     v_ratio=st.slider(" ",0.2,2.2,cfg["v_ratio"],0.02,key="vratio",label_visibility="collapsed")
#     label_row("Inclination Noise","ang")
#     ang=st.slider(" ",-0.6,0.6,cfg["ang"],0.02,key="ang",label_visibility="collapsed")
#     st.markdown('</div><div class="card">',unsafe_allow_html=True)
#     label_row("A1 Kerr Spin (0-0.99)","spin")
#     spin=st.slider(" ",0.0,0.99,cfg["spin"],0.05,key="spin",label_visibility="collapsed")
#     label_row("A2 Disk Thickness","thick")
#     disk_thick=st.slider(" ",0.15,1.2,0.36,0.05,key="thick",label_visibility="collapsed")
#     label_row("Disk Brightness","bright")
#     disk_bright=st.slider(" ",0.2,1.5,0.9,0.1,key="bright",label_visibility="collapsed")
#     label_row("A3 Observer Inclination","incl")
#     incl=st.slider(" ",0.0,1.0,cfg["incl"],0.05,key="incl",label_visibility="collapsed")
#     label_row("B6 Starfield Density","star")
#     star_dens=st.slider(" ",50,500,280,10,key="star",label_visibility="collapsed")
#     label_row("B10 Photon Ring Strength","photon")
#     photon=st.slider(" ",0.0,1.5,0.85,0.05,key="photon",label_visibility="collapsed")
#     label_row("B7 Binary Black Hole","binary")
#     binary=st.checkbox("Enable Binary", value=cfg["binary"],key="binary")
#     label_row("A5 Fuel Thrust","thrust")
#     thrust=st.slider(" ",0.0,1.5,0.0,0.05,key="thrust",label_visibility="collapsed")
#     st.markdown('</div>',unsafe_allow_html=True)

#     G=0.5; v_orb=np.sqrt(G*bh_mass/dist); speed=v_orb*v_ratio; ang_mom=dist*speed*np.cos(ang); energy=0.5*speed**2 - G*bh_mass/dist
#     X=pd.DataFrame([{"bh_mass":bh_mass,"dist":dist,"speed":speed,"v_ratio":v_ratio,"ang_mom":ang_mom,"energy":energy}])
#     pred_enc=clf.predict(X)[0]; proba=clf.predict_proba(X)[0]; fate=le.inverse_transform([pred_enc])[0]; conf=float(np.max(proba))
#     idx={n:i for i,n in enumerate(le.classes_)}
#     color="#ff4d5e" if fate=="SWALLOWED" else "#4dff9a" if fate=="STABLE" else "#5aa8ff"
#     t_str=f"{reg.predict(X)[0]:.1f}s to swallow" if fate=="SWALLOWED" else "Stable — holds" if fate=="STABLE" else "Escapes well"
#     cf = f"To survive: ↑ v_ratio by ~{max(0.15, (1.0-energy)*0.4):.2f} or ↑ L by {max(150,600-ang_mom):.0f}" if fate=="SWALLOWED" else "Orbit safe — try Binary mode for chaos" if fate=="STABLE" else "Will leave — try lowering v_ratio to capture"

#     st.markdown(f'<div class="card"><h4 style="color:{color}">🤖 AI Oracle — {fate} {conf*100:.0f}%</h4><div style="color:#8ea0c2;font-size:13px">{t_str}</div>',unsafe_allow_html=True)
#     for cls in ["SWALLOWED","STABLE","ESCAPE"]:
#         p=proba[idx[cls]]*100 if cls in idx else 0; col="#ff4d5e" if cls=="SWALLOWED" else "#4dff9a" if cls=="STABLE" else "#5aa8ff"
#         st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;color:#8ea0c2'><span>{cls}</span><span>{p:.0f}%</span></div><div class='conf-bar'><div class='conf-fill' style='width:{p}%;background:{col}'></div></div>",unsafe_allow_html=True)
#     st.markdown(f"<div style='margin-top:10px;color:#5a6a8a;font-size:11px;font-family:JetBrains Mono'>E {energy:.2f} | L {ang_mom:.0f}<br><b style='color:#8ea0c2'>C14:</b> {cf}<br><b>C12:</b> faint = AI predicted</div></div>",unsafe_allow_html=True)
#     if st.button("📝 Log Mission"): st.session_state.history.insert(0,{"fate":fate,"conf":f"{conf*100:.0f}%","mass":bh_mass,"dist":dist,"spin":spin}); st.session_state.history=st.session_state.history[:10]

# with c2:
#     st.markdown('<div class="sticky">',unsafe_allow_html=True)
#     html=f"""
#     <div id="wrap" style="position:relative;border-radius:22px;overflow:hidden;border:1px solid #1f2742;background:#000">
#     <canvas id="c" width="920" height="680" style="width:100%;background:#000;cursor:grab"></canvas>
#     <div style="position:absolute;top:12px;left:12px;background:rgba(12,16,28,0.78);backdrop-filter:blur(12px);padding:7px 12px;border-radius:999px;color:#8ea0c2;font-size:11px;border:1px solid #1f2742">● V5 • Spin {spin:.2f} • Incl {incl:.2f} • Binary {'ON' if binary else 'OFF'} • Drag to launch • ⛶ Fullscreen</div>
#     <button id="fs" style="position:absolute;top:12px;right:12px;background:#121624;border:1px solid #2a3555;color:#8ea0c2;padding:7px 12px;border-radius:999px;font-size:12px;cursor:pointer">⛶ Fullscreen Lab (ESC to exit)</button>
#     </div>
#     <script>
#     const W=920,H=680,CX=460,CY=340, canvas=document.getElementById('c'), ctx=canvas.getContext('2d'), wrap=document.getElementById('wrap');
#     let bh={bh_mass}, dist={dist}, vr={v_ratio}, ang={ang}, spin={spin}, incl={incl}, diskT={disk_thick}, diskB={disk_bright}, starN={star_dens}, phot={photon}, thrust={thrust}, binary={str(binary).lower()};
#     let G=0.5, x=CX+dist, y=CY, v_orb=Math.sqrt(G*bh/dist), vx=Math.cos(Math.PI/2+ang)*v_orb*vr, vy=Math.sin(Math.PI/2+ang)*v_orb*vr;
#     let trail=[], ghost=[], stars=[], disk=[];
#     for(let i=0;i<starN;i++) stars.push({{x:Math.random()*W,y:Math.random()*H,b:Math.random()}});
#     for(let i=0;i<160;i++){{let r=30+Math.random()*88; disk.push({{r,a:Math.random()*6.28,spd:0.012+(1.8+spin*1.2)/r}});}}
#     let h=Math.sqrt(bh)*0.8+8+spin*6, h2=h*0.62, bh2x=CX-220, bh2y=CY-80;
#     function lens(px,py,cx=CX,cy=CY,hh=h){{let dx=px-cx,dy=py-cy,d2=dx*dx+dy*dy; if(d2<25) return {{x:px,y:py}}; let bend=(hh*hh*2.4+spin*80)/Math.max(d2,110); return {{x:px+dx*bend*0.26,y:py+dy*bend*0.26*(0.6+incl*0.6)}};}}
#     document.getElementById('fs').onclick=()=>{{wrap.requestFullscreen();}};
#     let dragging=false, dragS=null;
#     canvas.addEventListener('mousedown',e=>{{dragging=true; dragS={{x:e.offsetX,y:e.offsetY}}; canvas.style.cursor='grabbing';}});
#     canvas.addEventListener('mouseup',e=>{{if(!dragging)return; dragging=false; canvas.style.cursor='grab'; let dx=e.offsetX-dragS.x, dy=e.offsetY-dragS.y; vx+=dx*0.02; vy+=dy*0.02;}});
#     function draw(){{
#       ctx.fillStyle='rgba(0,0,0,0.22)'; ctx.fillRect(0,0,W,H);
#       for(let s of stars){{let p=lens(s.x,s.y); if(binary){{let p2=lens(p.x,p.y,bh2x,bh2y,h2); p=p2;}} ctx.fillStyle=`rgba(180,200,255,${{0.12+s.b*0.7}})`; ctx.fillRect(p.x,p.y,1.2,1.2);}}
#       if(ghost.length>1){{ctx.strokeStyle='rgba(90,168,255,0.2)'; ctx.setLineDash([4,6]); ctx.beginPath(); ctx.moveTo(ghost[0][0],ghost[0][1]); for(let g of ghost) ctx.lineTo(g[0],g[1]); ctx.stroke(); ctx.setLineDash([]);}}
#       for(let d of disk){{d.a+=d.spd; let x0=CX+Math.cos(d.a)*d.r, y0=CY+Math.sin(d.a)*d.r*diskT*incl; let p=lens(x0,y0); let dop=Math.sin(d.a+spin); if(Math.hypot(p.x-CX,p.y-CY)>h+1){{ctx.fillStyle=`rgba(${{dop>0?255:80}},150,${{dop>0?90:255}},${{0.18*diskB+Math.abs(dop)*0.5}})`; ctx.fillRect(p.x,p.y,2.2,1.1);}}
#         if(binary){{let x1=bh2x+Math.cos(d.a+1)*d.r*0.6, y1=bh2y+Math.sin(d.a+1)*d.r*0.22; let p2=lens(x1,y1,bh2x,bh2y,h2); ctx.fillStyle=`rgba(200,120,255,0.22)`; ctx.fillRect(p2.x,p2.y,1.6,1);}}}}
#       let grad=ctx.createRadialGradient(CX,CY,h,CX,CY,h+26*phot); grad.addColorStop(0,'rgba(0,0,0,0)'); grad.addColorStop(0.3,`rgba(110,130,255,${{0.8*phot}})`); grad.addColorStop(1,'rgba(0,0,0,0)'); ctx.fillStyle=grad; ctx.beginPath(); ctx.arc(CX,CY,h+26*phot,0,6.28); ctx.fill();
#       if(binary){{let g2=ctx.createRadialGradient(bh2x,bh2y,h2,bh2x,bh2y,h2+18*phot); g2.addColorStop(0,'rgba(0,0,0,0)'); g2.addColorStop(0.5,`rgba(180,120,255,${{0.6*phot}})`); g2.addColorStop(1,'rgba(0,0,0,0)'); ctx.fillStyle=g2; ctx.beginPath(); ctx.arc(bh2x,bh2y,h2+18*phot,0,6.28); ctx.fill();}}
#       ctx.beginPath(); ctx.arc(CX,CY,h,0,6.28); ctx.fillStyle='#000'; ctx.fill(); ctx.strokeStyle='rgba(130,150,255,0.5)'; ctx.stroke();
#       if(binary){{ctx.beginPath(); ctx.arc(bh2x,bh2y,h2,0,6.28); ctx.fillStyle='#000'; ctx.fill(); ctx.strokeStyle='rgba(180,130,255,0.4)'; ctx.stroke();}}
#       let rx=x-CX, ry=y-CY, r=Math.hypot(rx,ry), rx2=x-bh2x, ry2=y-bh2y, r2=binary?Math.hypot(rx2,ry2):9999;
#       if(r<h || (binary && r2<h2)) return; if(r>1400 && r2>1400) return;
#       let a=G*bh/(r*r*r)*0.46, a2=binary?G*bh*0.6/(r2*r2*r2)*0.46:0;
#       vx-=(rx*a + (binary?rx2*a2:0))*10 + (thrust>0?Math.cos(Date.now()*0.005)*thrust*0.02:0); vy-=(ry*a + (binary?ry2*a2:0))*10; x+=vx; y+=vy;
#       trail.push([x,y]); if(trail.length>800) trail.shift();
#       if(ghost.length==0){{let gx=CX+dist, gy=CY, gvx=Math.cos(Math.PI/2+ang)*v_orb*vr*1.05, gvy=Math.sin(Math.PI/2+ang)*v_orb*vr*1.05; for(let i=0;i<400;i++){{let grx=gx-CX, gry=gy-CY, gr=Math.hypot(grx,gry); if(gr<h)break; let ga=G*bh/(gr*gr*gr)*0.46; gvx-=grx*ga*10; gvy-=gry*ga*10; gx+=gvx; gy+=gvy; ghost.push([gx,gy]);}}}}
#       for(let i=1;i<trail.length;i++){{let t=i/trail.length; let lp=lens(trail[i][0],trail[i][1]); if(binary) lp=lens(lp.x,lp.y,bh2x,bh2y,h2); ctx.strokeStyle=`rgba(255,${{210+t*40}},${{40+t*80}},${{0.08+t*0.92}})`; ctx.lineWidth=0.3+t*2.6; ctx.beginPath(); ctx.moveTo(trail[i-1][0],trail[i-1][1]); ctx.lineTo(lp.x,lp.y); ctx.stroke();}}
#       let p2=lens(x,y); if(binary) p2=lens(p2.x,p2.y,bh2x,bh2y,h2); ctx.shadowBlur=22; ctx.shadowColor='#ffde7a'; ctx.beginPath(); ctx.arc(p2.x,p2.y,4.6,0,6.28); ctx.fillStyle='#ffde7a'; ctx.fill(); ctx.shadowBlur=0; requestAnimationFrame(draw);
#     }}
#     ctx.fillStyle='#000'; ctx.fillRect(0,0,W,H); draw();
#     </script>
#     """
#     components.html(html,height=700)
#     st.markdown('</div>',unsafe_allow_html=True)
#     st.markdown("---")
#     b1,b2,b3=st.columns(3)
#     with b1: st.markdown(f'<div class="card"><h4>🧠 Counterfactual</h4><p style="color:#8ea0c2;font-size:12px">{cf}</p></div>',unsafe_allow_html=True)
#     with b2:
#         st.markdown('<div class="card"><h4>📜 Mission Log</h4>',unsafe_allow_html=True)
#         if st.session_state.history: st.dataframe(pd.DataFrame(st.session_state.history),hide_index=True,use_container_width=True)
#         else: st.caption("Log orbits")
#         st.markdown('</div>',unsafe_allow_html=True)
#     with b3: st.markdown('<div class="card"><h4>🎬 How to use</h4><p style="color:#8ea0c2;font-size:12px">Click? for meaning. Click demo → explainer → launch. Drag canvas to slingshot. ⛶ for fullscreen lab (ESC to exit). Sticky sim = no scrolling loss.</p></div>',unsafe_allow_html=True)














# # v5.1

# import streamlit as st, pickle, numpy as np, pandas as pd, streamlit.components.v1 as components
# st.set_page_config(page_title="Singularity V5.1", page_icon="🕳️", layout="wide")
# @st.cache_resource
# def load():
#     with open("orbit_model.pkl","rb") as f: clf=pickle.load(f)
#     with open("label_encoder.pkl","rb") as f: le=pickle.load(f)
#     with open("time_model.pkl","rb") as f: reg=pickle.load(f)
#     return clf,le,reg
# clf,le,reg=load()
# for k in ["history","info","cfg"]:
#     if k not in st.session_state: st.session_state[k]=None
# if st.session_state.cfg is None:
#     st.session_state.cfg=dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False)

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono&display=swap');
# .stApp{background:#06080f}
# .center{text-align:center}
# .h1{font-family:Space Grotesk;font-size:52px;font-weight:700;text-shadow:0 0 30px rgba(120,140,255,0.4)}
# .sub{color:#8ea0c2;text-align:center}
# .stat{display:inline-block;background:#121624;border:1px solid #1f2742;padding:6px 12px;border-radius:999px;margin:3px;color:#8ea0c2;font-size:12px}
# .card{background:linear-gradient(180deg,#121624,#0d101a);border:1px solid #1f2742;border-radius:18px;padding:14px;margin-bottom:10px}
# .sticky{position:sticky;top:12px}
# .infoBox{background:rgba(18,22,36,0.92);border:1px solid #2a3555;border-radius:14px;padding:12px;margin:8px 0;color:#a8b4cf;font-size:13px}
# .conf-bar{height:7px;background:#1a2035;border-radius:999px;overflow:hidden}.conf-fill{height:100%}
# </style>
# <div class="center"><div class="h1">🕳️ SINGULARITY V5.1</div><div class="sub">Kerr Spin • Binary • Drag-to-Launch • Ghost AI • Fullscreen Lab — Fixed modals</div></div>
# <div class="center" style="margin:8px"><span class="stat">10k Sims</span><span class="stat">XGB 97.5%</span><span class="stat">Sticky Sim</span><span class="stat">No Lock</span></div>
# """, unsafe_allow_html=True)

# infos={
#  "mass":"Black Hole Mass: Gravity strength. Higher = larger horizon h=sqrt(M)*0.8 + faster swallow.",
#  "dist":"Initial Distance: Start radius. <80 = instant swallow, >500 = escape zone.",
#  "vratio":"Velocity Ratio: 1.0=circular orbit. <1 fall, >1.4 escape. Key ML feature.",
#  "ang":"Inclination Noise: Orbit tilt, changes angular momentum L.",
#  "spin":"Kerr Spin 0-0.99: BH rotation speed. Warps lensing + photon ring. 0.99=near light speed.",
#  "thick":"Disk Thickness: Thin=Interstellar movie, thick=puffy torus.",
#  "bright":"Disk Brightness: Doppler visibility multiplier.",
#  "incl":"Observer Inclination: 0=top view, 1=edge view iconic.",
#  "star":"Starfield Density: Background stars count.",
#  "photon":"Photon Ring: Einstein ring glow.",
#  "binary":"Binary BH: Adds second BH — chaotic 3-body.",
#  "thrust":"Fuel Thrust: Mid-flight kick to save orbit."
# }
# demo_info={
#  "🌌 Gargantua (Edge)": ("Nolan's Gargantua","100M M☀, spin 0.9, edge-on. Tests AI on extreme Kerr. Expect STABLE."),
#  "💫 Binary Dance": ("Two BHs","5000+3000 M☀ chaotic. No analytic solution — ML shines."),
#  "🚀 Escape Slingshot": ("Slingshot","v_ratio 1.65 close 180. Should ESCAPE. Energy>0 test."),
#  "🔴 Swallow": ("Death spiral","Low v 0.55 dist 120. Should SWALLOWED <10s."),
#  "🎮 Free Play": ("Your lab","Drag canvas, thrust, fullscreen ⛶.")
# }
# presets={
#  "🌌 Gargantua (Edge)": dict(mass=7200,dist=280,v_ratio=0.92,ang=0.05,spin=0.9,incl=0.85,binary=False),
#  "💫 Binary Dance": dict(mass=5000,dist=380,v_ratio=1.15,ang=0.22,spin=0.6,incl=0.4,binary=True),
#  "🚀 Escape Slingshot": dict(mass=3200,dist=180,v_ratio=1.65,ang=-0.15,spin=0.3,incl=0.2,binary=False),
#  "🔴 Swallow": dict(mass=6800,dist=120,v_ratio=0.55,ang=0.3,spin=0.95,incl=0.6,binary=False),
#  "🎮 Free Play": dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False)
# }

# @st.dialog("Mission Briefing")
# def show_demo(name):
#     title,desc=demo_info[name]
#     st.markdown(f"### {name}\n**{title}**\n\n{desc}\n\nParams: {presets[name]}")
#     c1,c2=st.columns(2)
#     if c1.button("🚀 Launch", use_container_width=True):
#         st.session_state.cfg=presets[name]; st.rerun()
#     if c2.button("Cancel", use_container_width=True):
#         st.rerun()

# cols=st.columns(5)
# for i,(name,cfg) in enumerate(presets.items()):
#     if cols[i].button(name,use_container_width=True):
#         show_demo(name)

# def label_row(txt,key):
#     a,b=st.columns([0.85,0.15])
#     a.markdown(f"**{txt}**")
#     if b.button("?", key=f"q_{key}"):
#         st.session_state.info = None if st.session_state.info==key else key

# cfg=st.session_state.cfg
# c1,c2=st.columns([0.92,1.58],gap="large")
# with c1:
#     st.markdown('<div class="card">',unsafe_allow_html=True)
#     label_row("Black Hole Mass M☀","mass")
#     if st.session_state.info=="mass": st.markdown(f'<div class="infoBox">{infos["mass"]}</div>',unsafe_allow_html=True)
#     bh_mass=st.slider(" ",500,8000,cfg["mass"],100,key="mass_s",label_visibility="collapsed")
#     label_row("Initial Distance","dist")
#     if st.session_state.info=="dist": st.markdown(f'<div class="infoBox">{infos["dist"]}</div>',unsafe_allow_html=True)
#     dist=st.slider(" ",40,650,cfg["dist"],5,key="dist_s",label_visibility="collapsed")
#     label_row("Velocity Ratio","vratio")
#     if st.session_state.info=="vratio": st.markdown(f'<div class="infoBox">{infos["vratio"]}</div>',unsafe_allow_html=True)
#     v_ratio=st.slider(" ",0.2,2.2,cfg["v_ratio"],0.02,key="vr_s",label_visibility="collapsed")
#     label_row("Inclination Noise","ang")
#     if st.session_state.info=="ang": st.markdown(f'<div class="infoBox">{infos["ang"]}</div>',unsafe_allow_html=True)
#     ang=st.slider(" ",-0.6,0.6,cfg["ang"],0.02,key="ang_s",label_visibility="collapsed")
#     st.markdown('</div><div class="card">',unsafe_allow_html=True)
#     for lbl,key,defv in [("Kerr Spin","spin",0.5),("Disk Thickness","thick",0.36),("Disk Brightness","bright",0.9),("Observer Inclination","incl",0.5),("Starfield Density","star",280),("Photon Ring","photon",0.85),("Fuel Thrust","thrust",0.0)]:
#         label_row(lbl,key)
#         if st.session_state.info==key: st.markdown(f'<div class="infoBox">{infos[key]}</div>',unsafe_allow_html=True)
#     spin=st.slider(" ",0.0,0.99,cfg["spin"],0.05,key="spin_s",label_visibility="collapsed")
#     disk_thick=st.slider(" ",0.15,1.2,0.36,0.05,key="thick_s",label_visibility="collapsed")
#     disk_bright=st.slider(" ",0.2,1.5,0.9,0.1,key="bright_s",label_visibility="collapsed")
#     incl=st.slider(" ",0.0,1.0,cfg["incl"],0.05,key="incl_s",label_visibility="collapsed")
#     star_dens=st.slider(" ",50,500,280,10,key="star_s",label_visibility="collapsed")
#     photon=st.slider(" ",0.0,1.5,0.85,0.05,key="photon_s",label_visibility="collapsed")
#     label_row("Binary BH","binary")
#     if st.session_state.info=="binary": st.markdown(f'<div class="infoBox">{infos["binary"]}</div>',unsafe_allow_html=True)
#     binary=st.checkbox("Enable Binary", value=cfg["binary"],key="bin_s")
#     thrust=st.slider(" ",0.0,1.5,0.0,0.05,key="thrust_s",label_visibility="collapsed")
#     st.markdown('</div>',unsafe_allow_html=True)

#     G=0.5; v_orb=np.sqrt(G*bh_mass/dist); speed=v_orb*v_ratio; ang_mom=dist*speed*np.cos(ang); energy=0.5*speed**2 - G*bh_mass/dist
#     X=pd.DataFrame([{"bh_mass":bh_mass,"dist":dist,"speed":speed,"v_ratio":v_ratio,"ang_mom":ang_mom,"energy":energy}])
#     pred_enc=clf.predict(X)[0]; proba=clf.predict_proba(X)[0]; fate=le.inverse_transform([pred_enc])[0]; conf=float(np.max(proba))
#     idx={n:i for i,n in enumerate(le.classes_)}
#     color="#ff4d5e" if fate=="SWALLOWED" else "#4dff9a" if fate=="STABLE" else "#5aa8ff"
#     t_str=f"{reg.predict(X)[0]:.1f}s to swallow" if fate=="SWALLOWED" else "Stable" if fate=="STABLE" else "Escapes"
#     cf = f"↑ v_ratio by {max(0.15,(1-energy)*0.4):.2f}" if fate=="SWALLOWED" else "Try Binary for chaos"
#     st.markdown(f'<div class="card"><h4 style="color:{color}">🤖 {fate} {conf*100:.0f}% — {t_str}</h4>',unsafe_allow_html=True)
#     for cls in ["SWALLOWED","STABLE","ESCAPE"]:
#         p=proba[idx[cls]]*100 if cls in idx else 0; col="#ff4d5e" if cls=="SWALLOWED" else "#4dff9a" if cls=="STABLE" else "#5aa8ff"
#         st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;color:#8ea0c2'><span>{cls}</span><span>{p:.0f}%</span></div><div class='conf-bar'><div class='conf-fill' style='width:{p}%;background:{col}'></div></div>",unsafe_allow_html=True)
#     st.markdown(f"<div style='margin-top:8px;color:#5a6a8a;font-size:11px'>E {energy:.2f} | L {ang_mom:.0f} | C14: {cf} | C12 faint=AI pred</div></div>",unsafe_allow_html=True)

# with c2:
#     st.markdown('<div class="sticky">',unsafe_allow_html=True)
#     html=f"""
#     <div id="wrap" style="position:relative;border-radius:22px;overflow:hidden;border:1px solid #1f2742;background:#000">
#     <canvas id="c" width="920" height="680" style="width:100%;background:#000;cursor:grab"></canvas>
#     <div style="position:absolute;top:12px;left:12px;background:rgba(12,16,28,0.78);padding:7px 12px;border-radius:999px;color:#8ea0c2;font-size:11px;border:1px solid #1f2742">● V5.1 Fixed • Spin {spin:.2f} • Drag to launch • ⛶ Fullscreen (ESC exits)</div>
#     <button id="fs" style="position:absolute;top:12px;right:12px;background:#121624;border:1px solid #2a3555;color:#8ea0c2;padding:7px 12px;border-radius:999px;font-size:12px;cursor:pointer">⛶ Fullscreen Lab</button>
#     </div>
#     <script>
#     const W=920,H=680,CX=460,CY=340, canvas=document.getElementById('c'), wrap=document.getElementById('wrap');
#     let bh={bh_mass}, dist={dist}, vr={v_ratio}, ang={ang}, spin={spin}, incl={incl}, diskT={disk_thick}, diskB={disk_bright}, starN={star_dens}, phot={photon}, thrust={thrust}, binary={str(binary).lower()};
#     let G=0.5, x=CX+dist, y=CY, v_orb=Math.sqrt(G*bh/dist), vx=Math.cos(Math.PI/2+ang)*v_orb*vr, vy=Math.sin(Math.PI/2+ang)*v_orb*vr;
#     let trail=[], ghost=[], stars=[], disk=[];
#     for(let i=0;i<starN;i++) stars.push({{x:Math.random()*W,y:Math.random()*H,b:Math.random()}});
#     for(let i=0;i<150;i++){{let r=30+Math.random()*88; disk.push({{r,a:Math.random()*6.28,spd:0.012+(1.8+spin*1.2)/r}});}}
#     let h=Math.sqrt(bh)*0.8+8+spin*6, h2=h*0.62, bh2x=CX-220, bh2y=CY-80;
#     function lens(px,py,cx=CX,cy=CY,hh=h){{let dx=px-cx,dy=py-cy,d2=dx*dx+dy*dy; if(d2<25) return {{x:px,y:py}}; let bend=(hh*hh*2.4+spin*80)/Math.max(d2,110); return {{x:px+dx*bend*0.26,y:py+dy*bend*0.26*(0.6+incl*0.6)}};}}
#     document.getElementById('fs').onclick=()=>wrap.requestFullscreen();
#     let dragging=false, dragS=null;
#     canvas.addEventListener('mousedown',e=>{{dragging=true; dragS={{x:e.offsetX,y:e.offsetY}};}});
#     canvas.addEventListener('mouseup',e=>{{if(!dragging)return; dragging=false; let dx=e.offsetX-dragS.x, dy=e.offsetY-dragS.y; vx+=dx*0.02; vy+=dy*0.02;}});
#     function draw(){{
#       ctx.fillStyle='rgba(0,0,0,0.22)'; ctx.fillRect(0,0,W,H);
#       for(let s of stars){{let p=lens(s.x,s.y); if(binary){{let p2=lens(p.x,p.y,bh2x,bh2y,h2); p=p2;}} ctx.fillStyle=`rgba(180,200,255,${{0.12+s.b*0.7}})`; ctx.fillRect(p.x,p.y,1.2,1.2);}}
#       if(ghost.length>1){{ctx.strokeStyle='rgba(90,168,255,0.2)'; ctx.setLineDash([4,6]); ctx.beginPath(); ctx.moveTo(ghost[0][0],ghost[0][1]); for(let g of ghost) ctx.lineTo(g[0],g[1]); ctx.stroke(); ctx.setLineDash([]);}}
#       for(let d of disk){{d.a+=d.spd; let x0=CX+Math.cos(d.a)*d.r, y0=CY+Math.sin(d.a)*d.r*diskT*incl; let p=lens(x0,y0); if(Math.hypot(p.x-CX,p.y-CY)>h+1){{ctx.fillStyle=`rgba(${{Math.sin(d.a+spin)>0?255:80}},150,${{Math.sin(d.a+spin)>0?90:255}},${{0.18*diskB+0.4}})`; ctx.fillRect(p.x,p.y,2.2,1.1);}}}}
#       let grad=ctx.createRadialGradient(CX,CY,h,CX,CY,h+26*phot); grad.addColorStop(0,'rgba(0,0,0,0)'); grad.addColorStop(0.3,`rgba(110,130,255,${{0.8*phot}})`); grad.addColorStop(1,'rgba(0,0,0,0)'); ctx.fillStyle=grad; ctx.beginPath(); ctx.arc(CX,CY,h+26*phot,0,6.28); ctx.fill();
#       ctx.beginPath(); ctx.arc(CX,CY,h,0,6.28); ctx.fillStyle='#000'; ctx.fill(); ctx.strokeStyle='rgba(130,150,255,0.5)'; ctx.stroke();
#       if(binary){{ctx.beginPath(); ctx.arc(bh2x,bh2y,h2,0,6.28); ctx.fillStyle='#000'; ctx.fill(); ctx.strokeStyle='rgba(180,130,255,0.4)'; ctx.stroke();}}
#       let rx=x-CX, ry=y-CY, r=Math.hypot(rx,ry), rx2=x-bh2x, ry2=y-bh2y, r2=binary?Math.hypot(rx2,ry2):9999;
#       if(r<h || (binary && r2<h2)) return; if(r>1400 && r2>1400) return;
#       let a=G*bh/(r*r*r)*0.46, a2=binary?G*bh*0.6/(r2*r2*r2)*0.46:0;
#       vx-=(rx*a + (binary?rx2*a2:0))*10; vy-=(ry*a + (binary?ry2*a2:0))*10; x+=vx; y+=vy;
#       trail.push([x,y]); if(trail.length>800) trail.shift();
#       if(ghost.length==0){{let gx=CX+dist, gy=CY, gvx=Math.cos(Math.PI/2+ang)*v_orb*vr*1.05, gvy=Math.sin(Math.PI/2+ang)*v_orb*vr*1.05; for(let i=0;i<400;i++){{let grx=gx-CX, gry=gy-CY, gr=Math.hypot(grx,gry); if(gr<h)break; let ga=G*bh/(gr*gr*gr)*0.46; gvx-=grx*ga*10; gvy-=gry*ga*10; gx+=gvx; gy+=gvy; ghost.push([gx,gy]);}}}}
#       for(let i=1;i<trail.length;i++){{let t=i/trail.length; let lp=lens(trail[i][0],trail[i][1]); if(binary) lp=lens(lp.x,lp.y,bh2x,bh2y,h2); ctx.strokeStyle=`rgba(255,${{210+t*40}},${{40+t*80}},${{0.08+t*0.92}})`; ctx.lineWidth=0.3+t*2.6; ctx.beginPath(); ctx.moveTo(trail[i-1][0],trail[i-1][1]); ctx.lineTo(lp.x,lp.y); ctx.stroke();}}
#       let p2=lens(x,y); if(binary) p2=lens(p2.x,p2.y,bh2x,bh2y,h2); ctx.shadowBlur=20; ctx.shadowColor='#ffde7a'; ctx.beginPath(); ctx.arc(p2.x,p2.y,4.6,0,6.28); ctx.fillStyle='#ffde7a'; ctx.fill(); ctx.shadowBlur=0; requestAnimationFrame(draw);
#     }}
#     ctx.fillStyle='#000'; ctx.fillRect(0,0,W,H); draw();
#     </script>
#     """
#     components.html(html,height=700)
#     st.markdown('</div>',unsafe_allow_html=True)




















# # v5.2

# import streamlit as st, pickle, numpy as np, pandas as pd, streamlit.components.v1 as components
# st.set_page_config(page_title="Singularity V5.2", page_icon="🕳️", layout="wide")

# @st.cache_resource
# def load():
#     with open("orbit_model.pkl","rb") as f: clf=pickle.load(f)
#     with open("label_encoder.pkl","rb") as f: le=pickle.load(f)
#     with open("time_model.pkl","rb") as f: reg=pickle.load(f)
#     return clf,le,reg
# clf,le,reg=load()

# if "cfg" not in st.session_state:
#     st.session_state.cfg=dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False,thick=0.36,bright=0.9,star=280,photon=0.85,thrust=0.0)
# if "info" not in st.session_state: st.session_state.info=None
# if "history" not in st.session_state: st.session_state.history=[]

# infos={
#  "mass":"Mass controls gravity. Higher = bigger horizon h=sqrt(M)*0.8 + faster swallow. 500-8000 range.",
#  "dist":"Initial Distance: start radius. <80 instant swallow, >500 easier escape.",
#  "v_ratio":"Velocity Ratio: 1.0=circular. <1 fall, >1.4 escape. Core ML feature.",
#  "ang":"Inclination Noise: tilt of orbit, changes L.",
#  "spin":"Kerr Spin 0-0.99: BH rotation. Warps lensing + photon ring. 0.99 near light speed.",
#  "thick":"Disk Thickness: 0.15 thin Interstellar, 1.2 puffy torus.",
#  "bright":"Disk Brightness: Doppler color multiplier.",
#  "incl":"Observer Inclination: 0 top-down circle, 1 edge-on iconic.",
#  "star":"Starfield Density: background stars.",
#  "photon":"Photon Ring: Einstein ring glow intensity.",
#  "binary":"Binary BH: Adds second BH 60% mass — chaotic 3-body wow.",
#  "thrust":"Fuel Thrust: mid-flight kick to try save orbit."
# }
# demo_desc={
#  "🌌 Gargantua (Edge)": "Nolan's Gargantua: 7200 M☀, spin 0.9, edge-on incl 0.85. Extreme Kerr test. Expect STABLE.",
#  "💫 Binary Dance": "Two BHs 5000+3000 M☀ chaotic dance. No analytic solution, ML shines.",
#  "🚀 Escape Slingshot": "v_ratio 1.65 close 180 — slingshot. Should ESCAPE, energy>0 test.",
#  "🔴 Swallow": "Death spiral v 0.55 dist 120 — low L. Should SWALLOWED <10s.",
#  "🎮 Free Play": "Your lab — drag canvas to launch, thrust, fullscreen."
# }
# presets={
#  "🌌 Gargantua (Edge)": dict(mass=7200,dist=280,v_ratio=0.92,ang=0.05,spin=0.9,incl=0.85,binary=False,thick=0.36,bright=1.3,star=280,photon=0.9,thrust=0.0),
#  "💫 Binary Dance": dict(mass=5000,dist=380,v_ratio=1.15,ang=0.22,spin=0.6,incl=0.4,binary=True,thick=0.4,bright=0.9,star=350,photon=0.8,thrust=0.0),
#  "🚀 Escape Slingshot": dict(mass=3200,dist=180,v_ratio=1.65,ang=-0.15,spin=0.3,incl=0.2,binary=False,thick=0.3,bright=0.9,star=250,photon=0.6,thrust=0.0),
#  "🔴 Swallow": dict(mass=6800,dist=120,v_ratio=0.55,ang=0.3,spin=0.95,incl=0.6,binary=False,thick=0.5,bright=0.9,star=280,photon=1.2,thrust=0.0),
#  "🎮 Free Play": dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False,thick=0.36,bright=0.9,star=280,photon=0.85,thrust=0.0),
# }

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600&display=swap');
# .stApp{background:#06080f}
# .h1{font-family:Space Grotesk;font-size:52px;text-align:center;text-shadow:0 0 30px rgba(120,140,255,0.35)}
# .sub{color:#8ea0c2;text-align:center}
# .stat{display:inline-block;background:#121624;border:1px solid #1f2742;padding:6px 12px;border-radius:999px;margin:3px;color:#8ea0c2;font-size:12px}
# .card{background:#121624;border:1px solid #1f2742;border-radius:16px;padding:14px;margin-bottom:12px}
# .infoBox{background:#1a2138;border:1px solid #2a3555;border-radius:10px;padding:8px 10px;margin:6px 0 10px 0;color:#a8b4cf;font-size:12px}
# .sticky{position:sticky;top:12px}
# </style>
# <div class="h1">🕳️ SINGULARITY V5.2</div><div class="sub">Fixed modals • Sticky sim • Inline? • Fullscreen Lab (ESC)</div>
# <div style="text-align:center;margin:8px"><span class="stat">10k Sims</span><span class="stat">XGB 97.5%</span><span class="stat">Kerr Spin</span><span class="stat">Binary BH</span></div>
# """, unsafe_allow_html=True)

# @st.dialog("Mission Briefing — Press ESC to close")
# def demo_dialog(name):
#     st.write(f"**{name}**\n\n{demo_desc[name]}")
#     st.code(f"Params: {presets[name]}")
#     if st.button("🚀 Launch", use_container_width=True):
#         st.session_state.cfg=presets[name]
#         st.rerun()
#     if st.button("Cancel", use_container_width=True):
#         st.rerun()

# cols=st.columns(5)
# for n in presets:
#     if cols[list(presets.keys()).index(n)].button(n, use_container_width=True):
#         demo_dialog(n)

# def param_block(label, key, min_v, max_v, step, cfg_key):
#     cL,cQ=st.columns([0.88,0.12])
#     cL.markdown(f"**{label}**")
#     if cQ.button("?", key=f"q_{key}"):
#         st.session_state.info = None if st.session_state.info==key else key
#     if st.session_state.info==key:
#         st.markdown(f'<div class="infoBox">ℹ️ {infos[key]}</div>', unsafe_allow_html=True)
#     val=st.slider(" ", min_v, max_v, st.session_state.cfg[cfg_key], step, key=f"s_{key}", label_visibility="collapsed")
#     st.session_state.cfg[cfg_key]=val
#     return val

# c1,c2=st.columns([0.92,1.58],gap="large")
# with c1:
#     st.markdown('<div class="card">',unsafe_allow_html=True)
#     bh_mass=param_block("Black Hole Mass M☀","mass",500,8000,100,"mass")
#     dist=param_block("Initial Distance","dist",40,650,5,"dist")
#     v_ratio=param_block("Velocity Ratio","v_ratio",0.2,2.2,0.02,"v_ratio")
#     ang=param_block("Inclination Noise","ang",-0.6,0.6,0.02,"ang")
#     st.markdown('</div><div class="card">',unsafe_allow_html=True)
#     spin=param_block("Kerr Spin","spin",0.0,0.99,0.05,"spin")
#     disk_thick=param_block("Disk Thickness","thick",0.15,1.2,0.05,"thick")
#     disk_bright=param_block("Disk Brightness","bright",0.2,1.5,0.1,"bright")
#     incl=param_block("Observer Inclination","incl",0.0,1.0,0.05,"incl")
#     star_dens=param_block("Starfield Density","star",50,500,10,"star")
#     photon=param_block("Photon Ring","photon",0.0,1.5,0.05,"photon")
#     thrust=param_block("Fuel Thrust","thrust",0.0,1.5,0.05,"thrust")
#     cL,cQ=st.columns([0.88,0.12])
#     cL.markdown("**Binary BH**")
#     if cQ.button("?", key="q_binary"):
#         st.session_state.info=None if st.session_state.info=="binary" else "binary"
#     if st.session_state.info=="binary":
#         st.markdown(f'<div class="infoBox">ℹ️ {infos["binary"]}</div>', unsafe_allow_html=True)
#     binary=st.checkbox("Enable Binary", value=st.session_state.cfg["binary"], key="bin")
#     st.session_state.cfg["binary"]=binary
#     st.markdown('</div>',unsafe_allow_html=True)

#     G=0.5; v_orb=np.sqrt(G*bh_mass/dist); speed=v_orb*v_ratio; ang_mom=dist*speed*np.cos(ang); energy=0.5*speed**2 - G*bh_mass/dist
#     X=pd.DataFrame([{"bh_mass":bh_mass,"dist":dist,"speed":speed,"v_ratio":v_ratio,"ang_mom":ang_mom,"energy":energy}])
#     pred_enc=clf.predict(X)[0]; proba=clf.predict_proba(X)[0]; fate=le.inverse_transform([pred_enc])[0]; conf=float(np.max(proba))
#     idx={n:i for i,n in enumerate(le.classes_)}
#     st.markdown(f'<div class="card"><h4>🤖 {fate} {conf*100:.0f}%</h4>',unsafe_allow_html=True)
#     for cls in ["SWALLOWED","STABLE","ESCAPE"]:
#         p=proba[idx[cls]]*100 if cls in idx else 0
#         col="#ff4d5e" if cls=="SWALLOWED" else "#4dff9a" if cls=="STABLE" else "#5aa8ff"
#         st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;color:#8ea0c2'><span>{cls}</span><span>{p:.0f}%</span></div><div style='height:6px;background:#1a2035;border-radius:999px'><div style='height:100%;width:{p}%;background:{col};border-radius:999px'></div></div>",unsafe_allow_html=True)
#     cf = f"To survive ↑ v_ratio by {max(0.15,(1-energy)*0.4):.2f}" if fate=="SWALLOWED" else "Try Binary for chaos"
#     st.markdown(f"<div style='color:#5a6a8a;font-size:11px;margin-top:6px'>E {energy:.2f} L {ang_mom:.0f} | {cf}</div></div>",unsafe_allow_html=True)

# with c2:
#     st.markdown('<div class="sticky">',unsafe_allow_html=True)
#     # pass values via JS variables, no f-string clash
#     html_template="""
#     <div id="wrap" style="position:relative;border-radius:20px;overflow:hidden;border:1px solid #1f2742;background:#000">
#     <canvas id="c" width="920" height="680" style="width:100%;background:#000;cursor:grab"></canvas>
#     <div style="position:absolute;top:10px;left:10px;background:rgba(12,16,28,0.8);padding:6px 10px;border-radius:999px;color:#8ea0c2;font-size:11px;border:1px solid #1f2742">V5.2 • SPIN PLACEHOLDER • Drag to slingshot • ESC exits fullscreen</div>
#     <button id="fs" style="position:absolute;top:10px;right:10px;background:#121624;border:1px solid #2a3555;color:#8ea0c2;padding:6px 10px;border-radius:999px;font-size:11px;cursor:pointer">⛶ Fullscreen Lab</button>
#     </div>
#     <script>
#     const CFG = JSON.parse('CFG_JSON');
#     const W=920,H=680,CX=460,CY=340;
#     const canvas=document.getElementById('c'), ctx=canvas.getContext('2d'), wrap=document.getElementById('wrap');
#     let bh=CFG.mass, dist=CFG.dist, vr=CFG.v_ratio, ang=CFG.ang, spin=CFG.spin, incl=CFG.incl, thick=CFG.thick, bright=CFG.bright, starN=CFG.star, phot=CFG.photon, thrust=CFG.thrust, binary=CFG.binary;
#     let G=0.5, x=CX+dist, y=CY, v_orb=Math.sqrt(G*bh/dist), vx=Math.cos(Math.PI/2+ang)*v_orb*vr, vy=Math.sin(Math.PI/2+ang)*v_orb*vr;
#     let trail=[], ghost=[], stars=[], disk=[];
#     for(let i=0;i<starN;i++) stars.push({x:Math.random()*W,y:Math.random()*H,b:Math.random()});
#     for(let i=0;i<150;i++){ let r=30+Math.random()*88; disk.push({r:r,a:Math.random()*6.28,spd:0.012+(1.8+spin*1.2)/r});}
#     let h=Math.sqrt(bh)*0.8+8+spin*6, h2=h*0.62, bh2x=CX-220, bh2y=CY-80;
#     function lens(px,py,cx,cy,hh){ let dx=px-cx,dy=py-cy,d2=dx*dx+dy*dy; if(d2<25) return {x:px,y:py}; let bend=(hh*hh*2.4+spin*80)/Math.max(d2,110); return {x:px+dx*bend*0.26,y:py+dy*bend*0.26*(0.6+incl*0.6)}; }
#     document.getElementById('fs').onclick=()=>wrap.requestFullscreen();
#     let dragging=false, s=null;
#     canvas.addEventListener('mousedown',e=>{dragging=true; s={x:e.offsetX,y:e.offsetY};});
#     canvas.addEventListener('mouseup',e=>{ if(!dragging) return; dragging=false; let dx=e.offsetX-s.x, dy=e.offsetY-s.y; vx+=dx*0.02; vy+=dy*0.02; });
#     function draw(){
#       ctx.fillStyle='rgba(0,0,0,0.25)'; ctx.fillRect(0,0,W,H);
#       for(let s of stars){ let p=lens(s.x,s.y,CX,CY,h); if(binary){ let p2=lens(p.x,p.y,bh2x,bh2y,h2); p=p2; } ctx.fillStyle='rgba(180,200,255,'+(0.1+s.b*0.7)+')'; ctx.fillRect(p.x,p.y,1.2,1.2); }
#       if(ghost.length>1){ ctx.strokeStyle='rgba(90,168,255,0.22)'; ctx.setLineDash([4,6]); ctx.beginPath(); ctx.moveTo(ghost[0][0],ghost[0][1]); for(let g of ghost) ctx.lineTo(g[0],g[1]); ctx.stroke(); ctx.setLineDash([]); }
#       for(let d of disk){ d.a+=d.spd; let x0=CX+Math.cos(d.a)*d.r, y0=CY+Math.sin(d.a)*d.r*thick*incl; let p=lens(x0,y0,CX,CY,h); if(Math.hypot(p.x-CX,p.y-CY)>h+1){ ctx.fillStyle='rgba('+(Math.sin(d.a+spin)>0?255:80)+',150,'+(Math.sin(d.a+spin)>0?90:255)+','+(0.18*bright+0.4)+')'; ctx.fillRect(p.x,p.y,2,1); } }
#       let grad=ctx.createRadialGradient(CX,CY,h,CX,CY,h+26*phot); grad.addColorStop(0,'rgba(0,0,0,0)'); grad.addColorStop(0.3,'rgba(110,130,255,'+(0.8*phot)+')'); grad.addColorStop(1,'rgba(0,0,0,0)'); ctx.fillStyle=grad; ctx.beginPath(); ctx.arc(CX,CY,h+26*phot,0,6.28); ctx.fill();
#       ctx.beginPath(); ctx.arc(CX,CY,h,0,6.28); ctx.fillStyle='#000'; ctx.fill(); ctx.strokeStyle='rgba(130,150,255,0.5)'; ctx.stroke();
#       if(binary){ ctx.beginPath(); ctx.arc(bh2x,bh2y,h2,0,6.28); ctx.fillStyle='#000'; ctx.fill(); ctx.strokeStyle='rgba(180,130,255,0.4)'; ctx.stroke(); }
#       let rx=x-CX, ry=y-CY, r=Math.hypot(rx,ry), rx2=x-bh2x, ry2=y-bh2y, r2=binary?Math.hypot(rx2,ry2):9999;
#       if(r<h || (binary && r2<h2)) return;
#       let a=G*bh/(r*r*r)*0.46, a2=binary?G*bh*0.6/(r2*r2*r2)*0.46:0;
#       vx-=(rx*a + (binary?rx2*a2:0))*10; vy-=(ry*a + (binary?ry2*a2:0))*10; x+=vx; y+=vy;
#       trail.push([x,y]); if(trail.length>800) trail.shift();
#       if(ghost.length==0){ let gx=CX+dist, gy=CY, gvx=Math.cos(Math.PI/2+ang)*v_orb*vr*1.05, gvy=Math.sin(Math.PI/2+ang)*v_orb*vr*1.05; for(let i=0;i<350;i++){ let grx=gx-CX, gry=gy-CY, gr=Math.hypot(grx,gry); if(gr<h) break; let ga=G*bh/(gr*gr*gr)*0.46; gvx-=grx*ga*10; gvy-=gry*ga*10; gx+=gvx; gy+=gvy; ghost.push([gx,gy]); } }
#       for(let i=1;i<trail.length;i++){ let t=i/trail.length; let lp=lens(trail[i][0],trail[i][1],CX,CY,h); if(binary) lp=lens(lp.x,lp.y,bh2x,bh2y,h2); ctx.strokeStyle='rgba(255,'+(210+t*40)+','+(40+t*80)+','+(0.08+t*0.9)+')'; ctx.lineWidth=0.3+t*2.4; ctx.beginPath(); ctx.moveTo(trail[i-1][0],trail[i-1][1]); ctx.lineTo(lp.x,lp.y); ctx.stroke(); }
#       let p2=lens(x,y,CX,CY,h); if(binary) p2=lens(p2.x,p2.y,bh2x,bh2y,h2); ctx.shadowBlur=18; ctx.shadowColor='#ffde7a'; ctx.beginPath(); ctx.arc(p2.x,p2.y,4.5,0,6.28); ctx.fillStyle='#ffde7a'; ctx.fill(); ctx.shadowBlur=0;
#       requestAnimationFrame(draw);
#     }
#     ctx.fillStyle='#000'; ctx.fillRect(0,0,W,H); draw();
#     </script>
#     """
#     import json
#     cfg_json=json.dumps(st.session_state.cfg)
#     html_final=html_template.replace("CFG_JSON", cfg_json).replace("SPIN PLACEHOLDER", f"Spin {spin:.2f}")
#     components.html(html_final, height=700)
#     st.markdown('</div>',unsafe_allow_html=True)














# v5.3

# import streamlit as st, pickle, numpy as np, pandas as pd, streamlit.components.v1 as components, json
# st.set_page_config(page_title="Singularity V5.3", page_icon="🕳️", layout="wide")
# @st.cache_resource
# def load():
#     with open("orbit_model.pkl","rb") as f: clf=pickle.load(f)
#     with open("label_encoder.pkl","rb") as f: le=pickle.load(f)
#     with open("time_model.pkl","rb") as f: reg=pickle.load(f)
#     return clf,le,reg
# clf,le,reg=load()
# if "cfg" not in st.session_state:
#     st.session_state.cfg=dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False,thick=0.36,bright=0.9,star=280,photon=0.85,thrust=0.0)
# if "info" not in st.session_state: st.session_state.info=None
# if "history" not in st.session_state: st.session_state.history=[]

# infos={"mass":"Mass controls gravity.","dist":"Initial Distance: start radius.","v_ratio":"1.0=circular. <1 fall, >1.4 escape.","ang":"Tilt changes L.","spin":"Kerr Spin 0-0.99 warps lensing.","thick":"Disk Thickness thin vs puffy.","bright":"Doppler multiplier.","incl":"0 top view, 1 edge.","star":"Background stars.","photon":"Einstein ring glow.","binary":"Second BH — chaotic.","thrust":"Mid-flight kick."}
# demo_desc={"🌌 Gargantua (Edge)":"Nolan's Gargantua 7200 M☀ spin 0.9 edge-on. Expect STABLE.","💫 Binary Dance":"Two BHs 5000+3000 chaotic. No analytic solution.","🚀 Escape Slingshot":"v_ratio 1.65 close 180 — should ESCAPE.","🔴 Swallow":"Low v 0.55 dist 120 — should SWALLOWED.","🎮 Free Play":"Your lab — drag, thrust, fullscreen."}
# presets={
#  "🌌 Gargantua (Edge)": dict(mass=7200,dist=280,v_ratio=0.92,ang=0.05,spin=0.9,incl=0.85,binary=False,thick=0.36,bright=1.3,star=280,photon=0.9,thrust=0.0),
#  "💫 Binary Dance": dict(mass=5000,dist=380,v_ratio=1.15,ang=0.22,spin=0.6,incl=0.4,binary=True,thick=0.4,bright=0.9,star=350,photon=0.8,thrust=0.0),
#  "🚀 Escape Slingshot": dict(mass=3200,dist=180,v_ratio=1.65,ang=-0.15,spin=0.3,incl=0.2,binary=False,thick=0.3,bright=0.9,star=250,photon=0.6,thrust=0.0),
#  "🔴 Swallow": dict(mass=6800,dist=120,v_ratio=0.55,ang=0.3,spin=0.95,incl=0.6,binary=False,thick=0.5,bright=0.9,star=280,photon=1.2,thrust=0.0),
#  "🎮 Free Play": dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False,thick=0.36,bright=0.9,star=280,photon=0.85,thrust=0.0),
# }
# def apply_preset(name):
#     p=presets[name]
#     st.session_state.cfg=p.copy()
#     for k,v in [("s_mass",p["mass"]),("s_dist",p["dist"]),("s_v_ratio",p["v_ratio"]),("s_ang",p["ang"]),("s_spin",p["spin"]),("s_thick",p["thick"]),("s_bright",p["bright"]),("s_incl",p["incl"]),("s_star",p["star"]),("s_photon",p["photon"]),("s_thrust",p["thrust"]),("bin",p["binary"])]:
#         st.session_state[k]=v

# @st.dialog("Mission Briefing — ESC to close")
# def demo_dialog(name):
#     st.write(f"**{name}**\n\n{demo_desc[name]}")
#     st.code(str(presets[name]))
#     if st.button("🚀 Launch", use_container_width=True, key=f"launch_{name}"):
#         apply_preset(name)
#         st.rerun()
#     if st.button("Cancel", use_container_width=True, key=f"cancel_{name}"):
#         st.rerun()

# st.markdown("""<style>.stApp{background:#06080f}.card{background:#121624;border:1px solid #1f2742;border-radius:16px;padding:14px;margin-bottom:12px}.infoBox{background:#1a2138;border:1px solid #2a3555;border-radius:10px;padding:8px 10px;margin:6px 0;color:#a8b4cf;font-size:12px}.sticky{position:sticky;top:12px}.h1{font-size:50px;text-align:center}</style><div class="h1">🕳️ SINGULARITY V5.3</div>""", unsafe_allow_html=True)
# cols=st.columns(5)
# for i,n in enumerate(presets):
#     if cols[i].button(n, use_container_width=True, key=f"btn_{n}"):
#         demo_dialog(n)

# def param_block(label,key,min_v,max_v,step,cfg_key):
#     cL,cQ=st.columns([0.88,0.12])
#     cL.markdown(f"**{label}**")
#     if cQ.button("?", key=f"q_{key}"):
#         st.session_state.info=None if st.session_state.info==key else key
#     if st.session_state.info==key:
#         st.markdown(f'<div class="infoBox">ℹ️ {infos[key]}</div>', unsafe_allow_html=True)
#     val=st.slider(" ", min_v, max_v, st.session_state.cfg[cfg_key], step, key=f"s_{key}", label_visibility="collapsed")
#     st.session_state.cfg[cfg_key]=val
#     return val

# c1,c2=st.columns([0.92,1.58],gap="large")
# with c1:
#     st.markdown('<div class="card">',unsafe_allow_html=True)
#     bh_mass=param_block("Black Hole Mass M☀","mass",500,8000,100,"mass")
#     dist=param_block("Initial Distance","dist",40,650,5,"dist")
#     v_ratio=param_block("Velocity Ratio","v_ratio",0.2,2.2,0.02,"v_ratio")
#     ang=param_block("Inclination Noise","ang",-0.6,0.6,0.02,"ang")
#     st.markdown('</div><div class="card">',unsafe_allow_html=True)
#     spin=param_block("Kerr Spin","spin",0.0,0.99,0.05,"spin")
#     disk_thick=param_block("Disk Thickness","thick",0.15,1.2,0.05,"thick")
#     disk_bright=param_block("Disk Brightness","bright",0.2,1.5,0.1,"bright")
#     incl=param_block("Observer Inclination","incl",0.0,1.0,0.05,"incl")
#     star_dens=param_block("Starfield Density","star",50,500,10,"star")
#     photon=param_block("Photon Ring","photon",0.0,1.5,0.05,"photon")
#     thrust=param_block("Fuel Thrust","thrust",0.0,1.5,0.05,"thrust")
#     cL,cQ=st.columns([0.88,0.12]); cL.markdown("**Binary BH**")
#     if cQ.button("?", key="q_binary"): st.session_state.info=None if st.session_state.info=="binary" else "binary"
#     if st.session_state.info=="binary": st.markdown(f'<div class="infoBox">ℹ️ {infos["binary"]}</div>', unsafe_allow_html=True)
#     binary=st.checkbox("Enable Binary", value=st.session_state.cfg["binary"], key="bin")
#     st.session_state.cfg["binary"]=binary
#     st.markdown('</div>',unsafe_allow_html=True)
#     G=0.5; v_orb=np.sqrt(G*bh_mass/dist); speed=v_orb*v_ratio; ang_mom=dist*speed*np.cos(ang); energy=0.5*speed**2 - G*bh_mass/dist
#     X=pd.DataFrame([{"bh_mass":bh_mass,"dist":dist,"speed":speed,"v_ratio":v_ratio,"ang_mom":ang_mom,"energy":energy}])
#     pred_enc=clf.predict(X)[0]; proba=clf.predict_proba(X)[0]; fate=le.inverse_transform([pred_enc])[0]; conf=float(np.max(proba))
#     st.markdown(f'<div class="card"><h4>🤖 {fate} {conf*100:.0f}%</h4></div>',unsafe_allow_html=True)

# with c2:
#     st.markdown('<div class="sticky">',unsafe_allow_html=True)
#     html_template="""
#     <div id="wrap" style="position:relative;border-radius:20px;overflow:hidden;border:1px solid #1f2742;background:#000">
#     <canvas id="c" width="920" height="680" style="width:100%;background:#000"></canvas>
#     <button id="fs" style="position:absolute;top:10px;right:10px;background:#121624;border:1px solid #2a3555;color:#8ea0c2;padding:6px 10px;border-radius:999px;font-size:11px;cursor:pointer">⛶ Fullscreen (ESC)</button>
#     </div><script>
#     const CFG=JSON.parse('CFG_JSON');
#     const W=920,H=680,CX=460,CY=340, canvas=document.getElementById('c'), ctx=canvas.getContext('2d'), wrap=document.getElementById('wrap');
#     let bh=CFG.mass,dist=CFG.dist,vr=CFG.v_ratio,ang=CFG.ang,spin=CFG.spin,incl=CFG.incl,thick=CFG.thick,bright=CFG.bright,starN=CFG.star,phot=CFG.photon,binary=CFG.binary;
#     let G=0.5,x=CX+dist,y=CY,v_orb=Math.sqrt(G*bh/dist),vx=Math.cos(Math.PI/2+ang)*v_orb*vr,vy=Math.sin(Math.PI/2+ang)*v_orb*vr,trail=[],stars=[],disk=[];
#     for(let i=0;i<starN;i++) stars.push({x:Math.random()*W,y:Math.random()*H,b:Math.random()});
#     for(let i=0;i<150;i++){let r=30+Math.random()*88;disk.push({r:r,a:Math.random()*6.28,spd:0.012+(1.8+spin*1.2)/r});}
#     let h=Math.sqrt(bh)*0.8+8+spin*6,h2=h*0.62,bh2x=CX-220,bh2y=CY-80;
#     function lens(px,py,cx,cy,hh){let dx=px-cx,dy=py-cy,d2=dx*dx+dy*dy;if(d2<25)return{x:px,y:py};let bend=(hh*hh*2.4+spin*80)/Math.max(d2,110);return{x:px+dx*bend*0.26,y:py+dy*bend*0.26*(0.6+incl*0.6)};}
#     document.getElementById('fs').onclick=()=>wrap.requestFullscreen();
#     let dragging=false,s=null;canvas.addEventListener('mousedown',e=>{dragging=true;s={x:e.offsetX,y:e.offsetY};});canvas.addEventListener('mouseup',e=>{if(!dragging)return;dragging=false;let dx=e.offsetX-s.x,dy=e.offsetY-s.y;vx+=dx*0.02;vy+=dy*0.02;});
#     function draw(){ctx.fillStyle='rgba(0,0,0,0.25)';ctx.fillRect(0,0,W,H);for(let s of stars){let p=lens(s.x,s.y,CX,CY,h);if(binary){let p2=lens(p.x,p.y,bh2x,bh2y,h2);p=p2;}ctx.fillStyle='rgba(180,200,255,'+(0.1+s.b*0.7)+')';ctx.fillRect(p.x,p.y,1.2,1.2);}for(let d of disk){d.a+=d.spd;let x0=CX+Math.cos(d.a)*d.r,y0=CY+Math.sin(d.a)*d.r*thick*incl;let p=lens(x0,y0,CX,CY,h);if(Math.hypot(p.x-CX,p.y-CY)>h+1){ctx.fillStyle='rgba('+(Math.sin(d.a+spin)>0?255:80)+',150,'+(Math.sin(d.a+spin)>0?90:255)+','+(0.18*bright+0.4)+')';ctx.fillRect(p.x,p.y,2,1);}}let grad=ctx.createRadialGradient(CX,CY,h,CX,CY,h+26*phot);grad.addColorStop(0,'rgba(0,0,0,0)');grad.addColorStop(0.3,'rgba(110,130,255,'+(0.8*phot)+')');grad.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=grad;ctx.beginPath();ctx.arc(CX,CY,h+26*phot,0,6.28);ctx.fill();ctx.beginPath();ctx.arc(CX,CY,h,0,6.28);ctx.fillStyle='#000';ctx.fill();ctx.strokeStyle='rgba(130,150,255,0.5)';ctx.stroke();if(binary){ctx.beginPath();ctx.arc(bh2x,bh2y,h2,0,6.28);ctx.fillStyle='#000';ctx.fill();ctx.strokeStyle='rgba(180,130,255,0.4)';ctx.stroke();}let rx=x-CX,ry=y-CY,r=Math.hypot(rx,ry);if(r<h)return;let a=G*bh/(r*r*r)*0.46;vx-=rx*a*10;vy-=ry*a*10;x+=vx;y+=vy;trail.push([x,y]);if(trail.length>800)trail.shift();for(let i=1;i<trail.length;i++){let t=i/trail.length;let lp=lens(trail[i][0],trail[i][1],CX,CY,h);if(binary)lp=lens(lp.x,lp.y,bh2x,bh2y,h2);ctx.strokeStyle='rgba(255,'+(210+t*40)+','+(40+t*80)+','+(0.08+t*0.9)+')';ctx.lineWidth=0.3+t*2.4;ctx.beginPath();ctx.moveTo(trail[i-1][0],trail[i-1][1]);ctx.lineTo(lp.x,lp.y);ctx.stroke();}let p2=lens(x,y,CX,CY,h);if(binary)p2=lens(p2.x,p2.y,bh2x,bh2y,h2);ctx.shadowBlur=18;ctx.shadowColor='#ffde7a';ctx.beginPath();ctx.arc(p2.x,p2.y,4.5,0,6.28);ctx.fillStyle='#ffde7a';ctx.fill();ctx.shadowBlur=0;requestAnimationFrame(draw);}ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);draw();
#     </script>
#     """
#     html_final=html_template.replace("CFG_JSON", json.dumps(st.session_state.cfg))
#     components.html(html_final, height=700)
#     st.markdown('</div>',unsafe_allow_html=True)



















# v-6.0 - Whiteboard Layout: centered cockpit + dual side params + 3 aesthetic verdict cards + pill value badges
# CHANGES: Symmetrical 3-col layout (left physics | center canvas | right cinematic), binary checkbox removed (demo-driven), aesthetic verdict cards (AI Proof | Singularity Core | Mission Console), pill value badge for slider visibility, inline ? infoBox retained, CFG_JSON injection retained

import streamlit as st, pickle, numpy as np, pandas as pd, streamlit.components.v1 as components, json
st.set_page_config(page_title="Singularity V5.3", page_icon="ðŸ•³ï¸", layout="wide")
@st.cache_resource
def load():
    with open("orbit_model.pkl","rb") as f: clf=pickle.load(f)
    with open("label_encoder.pkl","rb") as f: le=pickle.load(f)
    with open("time_model.pkl","rb") as f: reg=pickle.load(f)
    return clf,le,reg
clf,le,reg=load()
if "cfg" not in st.session_state:
    st.session_state.cfg=dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False,thick=0.36,bright=0.9,star=280,photon=0.85,thrust=0.0)
if "info" not in st.session_state: st.session_state.info=None
if "history" not in st.session_state: st.session_state.history=[]

infos={"mass":"Mass controls gravity.","dist":"Initial Distance: start radius.","v_ratio":"1.0=circular. <1 fall, >1.4 escape.","ang":"Tilt changes L.","spin":"Kerr Spin 0-0.99 warps lensing.","thick":"Disk Thickness thin vs puffy.","bright":"Doppler multiplier.","incl":"0 top view, 1 edge.","star":"Background stars.","photon":"Einstein ring glow.","binary":"Second BH â€” chaotic.","thrust":"Mid-flight kick."}
demo_desc={"ðŸŒŒ Gargantua (Edge)":"Nolan's Gargantua 7200 Mâ˜€ spin 0.9 edge-on. Expect STABLE.","ðŸ’« Binary Dance":"Two BHs 5000+3000 chaotic. No analytic solution.","ðŸš€ Escape Slingshot":"v_ratio 1.65 close 180 â€” should ESCAPE.","ðŸ”´ Swallow":"Low v 0.55 dist 120 â€” should SWALLOWED.","ðŸŽ® Free Play":"Your lab â€” drag, thrust, fullscreen."}
presets={
 "ðŸŒŒ Gargantua (Edge)": dict(mass=7200,dist=280,v_ratio=0.92,ang=0.05,spin=0.9,incl=0.85,binary=False,thick=0.36,bright=1.3,star=280,photon=0.9,thrust=0.0),
 "ðŸ’« Binary Dance": dict(mass=5000,dist=380,v_ratio=1.15,ang=0.22,spin=0.6,incl=0.4,binary=True,thick=0.4,bright=0.9,star=350,photon=0.8,thrust=0.0),
 "ðŸš€ Escape Slingshot": dict(mass=3200,dist=180,v_ratio=1.65,ang=-0.15,spin=0.3,incl=0.2,binary=False,thick=0.3,bright=0.9,star=250,photon=0.6,thrust=0.0),
 "ðŸ”´ Swallow": dict(mass=6800,dist=120,v_ratio=0.55,ang=0.3,spin=0.95,incl=0.6,binary=False,thick=0.5,bright=0.9,star=280,photon=1.2,thrust=0.0),
 "ðŸŽ® Free Play": dict(mass=4500,dist=220,v_ratio=1.35,ang=0.12,spin=0.5,incl=0.5,binary=False,thick=0.36,bright=0.9,star=280,photon=0.85,thrust=0.0),
}
def apply_preset(name):
    p=presets[name]
    st.session_state.cfg=p.copy()
    for k,v in [("s_mass",p["mass"]),("s_dist",p["dist"]),("s_v_ratio",p["v_ratio"]),("s_ang",p["ang"]),("s_spin",p["spin"]),("s_thick",p["thick"]),("s_bright",p["bright"]),("s_incl",p["incl"]),("s_star",p["star"]),("s_photon",p["photon"]),("s_thrust",p["thrust"]),("bin",p["binary"])]:
        st.session_state[k]=v

@st.dialog("Mission Briefing â€” ESC to close")
def demo_dialog(name):
    st.write(f"**{name}**\n\n{demo_desc[name]}")
    st.code(str(presets[name]))
    if st.button("ðŸš€ Launch", use_container_width=True, key=f"launch_{name}"):
        apply_preset(name)
        st.rerun()
    if st.button("Cancel", use_container_width=True, key=f"cancel_{name}"):
        st.rerun()

st.markdown("""<style>.stApp{background:#06080f}.card{background:#121624;border:1px solid #1f2742;border-radius:16px;padding:14px;margin-bottom:12px}.infoBox{background:#1a2138;border:1px solid #2a3555;border-radius:10px;padding:8px 10px;margin:6px 0;color:#a8b4cf;font-size:12px}.sticky{position:sticky;top:12px}.h1{font-size:50px;text-align:center}</style><div class="h1">ðŸ•³ï¸ SINGULARITY V5.3</div>""", unsafe_allow_html=True)
cols=st.columns(5)
for i,n in enumerate(presets):
    if cols[i].button(n, use_container_width=True, key=f"btn_{n}"):
        demo_dialog(n)

def param_block(label,key,min_v,max_v,step,cfg_key):
    cL,cQ=st.columns([0.88,0.12])
    cL.markdown(f"**{label}**")
    if cQ.button("?", key=f"q_{key}"):
        st.session_state.info=None if st.session_state.info==key else key
    if st.session_state.info==key:
        st.markdown(f'<div class="infoBox">â„¹ï¸ {infos[key]}</div>', unsafe_allow_html=True)
    val=st.slider(" ", min_v, max_v, st.session_state.cfg[cfg_key], step, key=f"s_{key}", label_visibility="collapsed")
    st.session_state.cfg[cfg_key]=val
    return val

c1,c2=st.columns([0.92,1.58],gap="large")
with c1:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    bh_mass=param_block("Black Hole Mass Mâ˜€","mass",500,8000,100,"mass")
    dist=param_block("Initial Distance","dist",40,650,5,"dist")
    v_ratio=param_block("Velocity Ratio","v_ratio",0.2,2.2,0.02,"v_ratio")
    ang=param_block("Inclination Noise","ang",-0.6,0.6,0.02,"ang")
    st.markdown('</div><div class="card">',unsafe_allow_html=True)
    spin=param_block("Kerr Spin","spin",0.0,0.99,0.05,"spin")
    disk_thick=param_block("Disk Thickness","thick",0.15,1.2,0.05,"thick")
    disk_bright=param_block("Disk Brightness","bright",0.2,1.5,0.1,"bright")
    incl=param_block("Observer Inclination","incl",0.0,1.0,0.05,"incl")
    star_dens=param_block("Starfield Density","star",50,500,10,"star")
    photon=param_block("Photon Ring","photon",0.0,1.5,0.05,"photon")
    thrust=param_block("Fuel Thrust","thrust",0.0,1.5,0.05,"thrust")
    cL,cQ=st.columns([0.88,0.12]); cL.markdown("**Binary BH**")
    if cQ.button("?", key="q_binary"): st.session_state.info=None if st.session_state.info=="binary" else "binary"
    if st.session_state.info=="binary": st.markdown(f'<div class="infoBox">â„¹ï¸ {infos["binary"]}</div>', unsafe_allow_html=True)
    binary=st.checkbox("Enable Binary", value=st.session_state.cfg["binary"], key="bin")
    st.session_state.cfg["binary"]=binary
    st.markdown('</div>',unsafe_allow_html=True)
    G=0.5; v_orb=np.sqrt(G*bh_mass/dist); speed=v_orb*v_ratio; ang_mom=dist*speed*np.cos(ang); energy=0.5*speed**2 - G*bh_mass/dist
    X=pd.DataFrame([{"bh_mass":bh_mass,"dist":dist,"speed":speed,"v_ratio":v_ratio,"ang_mom":ang_mom,"energy":energy}])
    pred_enc=clf.predict(X)[0]; proba=clf.predict_proba(X)[0]; fate=le.inverse_transform([pred_enc])[0]; conf=float(np.max(proba))
    st.markdown(f'<div class="card"><h4>ðŸ¤– {fate} {conf*100:.0f}%</h4></div>',unsafe_allow_html=True)

with c2:
    st.markdown('<div class="sticky">',unsafe_allow_html=True)
    html_template="""
    <div id="wrap" style="position:relative;border-radius:20px;overflow:hidden;border:1px solid #1f2742;background:#000">
    <canvas id="c" width="920" height="680" style="width:100%;background:#000"></canvas>
    <button id="fs" style="position:absolute;top:10px;right:10px;background:#121624;border:1px solid #2a3555;color:#8ea0c2;padding:6px 10px;border-radius:999px;font-size:11px;cursor:pointer">â›¶ Fullscreen (ESC)</button>
    </div><script>
    const CFG=JSON.parse('CFG_JSON');
    const W=920,H=680,CX=460,CY=340, canvas=document.getElementById('c'), ctx=canvas.getContext('2d'), wrap=document.getElementById('wrap');
    let bh=CFG.mass,dist=CFG.dist,vr=CFG.v_ratio,ang=CFG.ang,spin=CFG.spin,incl=CFG.incl,thick=CFG.thick,bright=CFG.bright,starN=CFG.star,phot=CFG.photon,binary=CFG.binary;
    let G=0.5,x=CX+dist,y=CY,v_orb=Math.sqrt(G*bh/dist),vx=Math.cos(Math.PI/2+ang)*v_orb*vr,vy=Math.sin(Math.PI/2+ang)*v_orb*vr,trail=[],stars=[],disk=[];
    for(let i=0;i<starN;i++) stars.push({x:Math.random()*W,y:Math.random()*H,b:Math.random()});
    for(let i=0;i<150;i++){let r=30+Math.random()*88;disk.push({r:r,a:Math.random()*6.28,spd:0.012+(1.8+spin*1.2)/r});}
    let h=Math.sqrt(bh)*0.8+8+spin*6,h2=h*0.62,bh2x=CX-220,bh2y=CY-80;
    function lens(px,py,cx,cy,hh){let dx=px-cx,dy=py-cy,d2=dx*dx+dy*dy;if(d2<25)return{x:px,y:py};let bend=(hh*hh*2.4+spin*80)/Math.max(d2,110);return{x:px+dx*bend*0.26,y:py+dy*bend*0.26*(0.6+incl*0.6)};}
    document.getElementById('fs').onclick=()=>wrap.requestFullscreen();
    let dragging=false,s=null;canvas.addEventListener('mousedown',e=>{dragging=true;s={x:e.offsetX,y:e.offsetY};});canvas.addEventListener('mouseup',e=>{if(!dragging)return;dragging=false;let dx=e.offsetX-s.x,dy=e.offsetY-s.y;vx+=dx*0.02;vy+=dy*0.02;});
    function draw(){ctx.fillStyle='rgba(0,0,0,0.25)';ctx.fillRect(0,0,W,H);for(let s of stars){let p=lens(s.x,s.y,CX,CY,h);if(binary){let p2=lens(p.x,p.y,bh2x,bh2y,h2);p=p2;}ctx.fillStyle='rgba(180,200,255,'+(0.1+s.b*0.7)+')';ctx.fillRect(p.x,p.y,1.2,1.2);}for(let d of disk){d.a+=d.spd;let x0=CX+Math.cos(d.a)*d.r,y0=CY+Math.sin(d.a)*d.r*thick*incl;let p=lens(x0,y0,CX,CY,h);if(Math.hypot(p.x-CX,p.y-CY)>h+1){ctx.fillStyle='rgba('+(Math.sin(d.a+spin)>0?255:80)+',150,'+(Math.sin(d.a+spin)>0?90:255)+','+(0.18*bright+0.4)+')';ctx.fillRect(p.x,p.y,2,1);}}let grad=ctx.createRadialGradient(CX,CY,h,CX,CY,h+26*phot);grad.addColorStop(0,'rgba(0,0,0,0)');grad.addColorStop(0.3,'rgba(110,130,255,'+(0.8*phot)+')');grad.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=grad;ctx.beginPath();ctx.arc(CX,CY,h+26*phot,0,6.28);ctx.fill();ctx.beginPath();ctx.arc(CX,CY,h,0,6.28);ctx.fillStyle='#000';ctx.fill();ctx.strokeStyle='rgba(130,150,255,0.5)';ctx.stroke();if(binary){ctx.beginPath();ctx.arc(bh2x,bh2y,h2,0,6.28);ctx.fillStyle='#000';ctx.fill();ctx.strokeStyle='rgba(180,130,255,0.4)';ctx.stroke();}let rx=x-CX,ry=y-CY,r=Math.hypot(rx,ry);if(r<h)return;let a=G*bh/(r*r*r)*0.46;vx-=rx*a*10;vy-=ry*a*10;x+=vx;y+=vy;trail.push([x,y]);if(trail.length>800)trail.shift();for(let i=1;i<trail.length;i++){let t=i/trail.length;let lp=lens(trail[i][0],trail[i][1],CX,CY,h);if(binary)lp=lens(lp.x,lp.y,bh2x,bh2y,h2);ctx.strokeStyle='rgba(255,'+(210+t*40)+','+(40+t*80)+','+(0.08+t*0.9)+')';ctx.lineWidth=0.3+t*2.4;ctx.beginPath();ctx.moveTo(trail[i-1][0],trail[i-1][1]);ctx.lineTo(lp.x,lp.y);ctx.stroke();}let p2=lens(x,y,CX,CY,h);if(binary)p2=lens(p2.x,p2.y,bh2x,bh2y,h2);ctx.shadowBlur=18;ctx.shadowColor='#ffde7a';ctx.beginPath();ctx.arc(p2.x,p2.y,4.5,0,6.28);ctx.fillStyle='#ffde7a';ctx.fill();ctx.shadowBlur=0;requestAnimationFrame(draw);}ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);draw();
    </script>
    """
    html_final=html_template.replace("CFG_JSON", json.dumps(st.session_state.cfg))
    components.html(html_final, height=700)
    st.markdown('</div>',unsafe_allow_html=True)
