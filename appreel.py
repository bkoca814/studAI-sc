import os
import io
import re
import time
import random
import textwrap
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from google import genai
from google.genai import types

try:
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
except ImportError:
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from gtts import gTTS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import matplotlib.font_manager as fm
    FONT_PATH = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
except Exception:
    FONT_PATH = None

st.set_page_config(page_title="StudAI Math", page_icon="✨", layout="wide")

# ============================================================
# 🔑 API KEY — Hugging Face Spaces "Settings > Repository secrets"
# bölümünden GOOGLE_API_KEY adıyla eklenmelidir. Kod içine ASLA
# gerçek key yazma; secrets sızarsa iptal edip yenisini oluştur.
# ============================================================
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not API_KEY:
    st.error("GOOGLE_API_KEY bulunamadı. Space Settings > Repository secrets kısmına ekleyin.")
    st.stop()

# ============================================================
# 🔒 UYGULAMA ŞİFRESİ — Space Settings > Repository secrets
# bölümüne APP_PASSWORD adıyla kendi belirlediğin bir şifre ekle.
# Bu şifreyi bilmeyen kimse uygulamayı kullanamaz, dolayısıyla
# senin API key'in üzerinden istek atamaz.
# ============================================================
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
if APP_PASSWORD:
    if "yetkili" not in st.session_state:
        st.session_state.yetkili = False
    if not st.session_state.yetkili:
        st.markdown("### 🔒 Giriş")
        girilen_sifre = st.text_input("Şifre", type="password")
        if st.button("Giriş yap"):
            if girilen_sifre == APP_PASSWORD:
                st.session_state.yetkili = True
                st.rerun()
            else:
                st.error("Yanlış şifre.")
        st.stop()

# İlk iki model günde 500 istek hakkı veriyor; sonrakiler sadece ikisi de
# tükenirse (kota/429 ya da 503 hatası devam ederse) devreye giren son çare modeller.
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite"]

# ============================================================
# 🎨 TEMA / CSS
# ============================================================
LIGHT_VARS = """
:root {
  --bg-app: radial-gradient(circle at 15% 0%, #e8fff5 0%, #f5fdfb 35%, #ffffff 70%);
  --bg-sidebar: linear-gradient(190deg, #0b201a 0%, #08160f 100%);
  --sidebar-text: #cfe7de; --sidebar-text-dim: #7fa294;
  --sidebar-active-bg: rgba(255,255,255,0.09);
  --accent: #12b886; --accent-2: #6ee7c2; --accent-contrast: #04120c; --accent-soft: rgba(18,184,134,0.12);
  --text-heading: #0d1f1a; --text-body: #5b6b66;
  --card-bg: #ffffff; --card-border: rgba(13,31,26,0.09);
  --shadow: 0 10px 30px rgba(13,31,26,0.07);
  --shadow-lift: 0 16px 40px rgba(13,31,26,0.12);
}
"""

DARK_VARS = """
:root {
  --bg-app: radial-gradient(circle at top left, #0a1512 0%, #050a08 60%, #030503 100%);
  --bg-sidebar: linear-gradient(190deg, #030907 0%, #020504 100%);
  --sidebar-text: #b7f2d2; --sidebar-text-dim: #4f7a63;
  --sidebar-active-bg: rgba(57,217,138,0.10);
  --accent: #39d98a; --accent-2: #7ef4c2; --accent-contrast: #04120c; --accent-soft: rgba(57,217,138,0.14);
  --text-heading: #d9ffe9; --text-body: #83a595;
  --card-bg: #081410; --card-border: rgba(57,217,138,0.18);
  --shadow: 0 0 26px rgba(57,217,138,0.09);
  --shadow-lift: 0 0 42px rgba(57,217,138,0.16);
}
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-app); }
.block-container { padding-top: 1.1rem; max-width: 1180px; }
section[data-testid="stSidebar"] { background: var(--bg-sidebar) !important; border-right: 1px solid rgba(255,255,255,0.04); }
section[data-testid="stSidebar"] * { color: var(--sidebar-text); }
.sb-logo { display:flex; align-items:center; gap:11px; padding:6px 0 22px 0; }
.sb-logo .mark { width:38px; height:38px; border-radius:12px; background: linear-gradient(135deg, var(--accent), var(--accent-2)); display:flex; align-items:center; justify-content:center; font-size:18px; box-shadow: 0 4px 14px rgba(57,217,138,0.35); }
.sb-logo .brand { font-weight:800; font-size:17px; color:#fff; line-height:1.15; letter-spacing:-.01em; }
.sb-logo .sub { font-size:10px; letter-spacing:.14em; color:var(--sidebar-text-dim); text-transform:uppercase; }
.sb-section-title { font-size:10px; letter-spacing:.16em; color:var(--sidebar-text-dim); text-transform:uppercase; margin:18px 0 9px 2px; }
.sb-nav-item { display:flex; align-items:center; gap:9px; justify-content:space-between; padding:10px 12px; border-radius:11px; margin-bottom:4px; font-size:14px; font-weight:600; transition: background .15s ease; }
.sb-nav-item.active { background: var(--sidebar-active-bg); color:#fff; }
.sb-dot { width:6px; height:6px; border-radius:50%; background:var(--accent); box-shadow: 0 0 6px var(--accent); }
.sb-info-card { margin-top:24px; padding:15px; border-radius:14px; background:rgba(255,255,255,0.045); border:1px solid rgba(255,255,255,0.08); }
.sb-info-card .t { font-weight:700; font-size:13px; color:#fff; margin-bottom:4px; display:flex; align-items:center; gap:6px; }
.sb-info-card .d { font-size:12px; color:var(--sidebar-text-dim); line-height:1.45; }
.sb-status { display:flex; align-items:center; gap:7px; font-size:11px; letter-spacing:.1em; color:var(--sidebar-text-dim); text-transform:uppercase; margin-top:20px; }
.sb-status .dot { width:7px; height:7px; border-radius:50%; background:#22c55e; box-shadow:0 0 8px #22c55e; animation: pulse 1.8s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1);} 50% { opacity:.45; transform:scale(.75);} }
.badge-pill { display:inline-flex; align-items:center; gap:6px; font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); background:var(--accent-soft); padding:7px 15px; border-radius:999px; margin-bottom:18px; }
.hero-title { font-size:40px; font-weight:800; line-height:1.14; color:var(--text-heading); margin:0 0 12px 0; letter-spacing:-.02em; }
.hero-title .accent-text { background: linear-gradient(100deg, var(--accent), var(--accent-2)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-sub { font-size:15px; color:var(--text-body); max-width:600px; line-height:1.65; margin-bottom:8px; }
.panel-card { background:var(--card-bg); border:1px solid var(--card-border); border-radius:18px; padding:20px 22px; box-shadow:var(--shadow); transition: box-shadow .2s ease, transform .2s ease; }
.panel-card:hover { box-shadow: var(--shadow-lift); }
.panel-label { font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:var(--text-body); margin-bottom:7px; font-weight:700; }
.panel-title { font-size:18px; font-weight:800; color:var(--text-heading); margin-bottom:15px; letter-spacing:-.01em; }
.check-row { display:flex; align-items:center; gap:11px; padding:7px 0; font-size:13.5px; color:var(--text-heading); }
.check-num { width:21px; height:21px; border-radius:50%; background:var(--accent-soft); color:var(--accent); font-size:11px; font-weight:800; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.check-mark { margin-left:auto; color:var(--accent); }
div[data-testid="stButton"] > button { background: var(--card-bg) !important; color: var(--text-heading) !important; border: 1px solid var(--card-border) !important; border-radius: 999px !important; padding: 10px 20px !important; font-weight:600 !important; font-size:13.5px !important; box-shadow:none !important; transition: all .15s ease !important; }
div[data-testid="stButton"] > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; transform: translateY(-1px); }
div[data-testid="stForm"] { background: var(--card-bg); border:1px solid var(--card-border); border-radius: 24px; padding: 5px 12px; box-shadow: var(--shadow); }
div[data-testid="stForm"] input { background: transparent !important; border:none !important; box-shadow:none !important; font-size:14.5px !important; color: var(--text-heading) !important; }
div[data-testid="stFormSubmitButton"] > button { background: linear-gradient(100deg, var(--accent), var(--accent-2)) !important; color: var(--accent-contrast) !important; border-radius: 999px !important; font-weight:700 !important; border:none !important; }
div[data-testid="stFileUploader"] section { border-radius: 16px !important; border: 1.5px dashed var(--card-border) !important; background: var(--card-bg) !important; }
[data-testid="stChatMessage"] { background: var(--card-bg); border:1px solid var(--card-border); border-radius:18px; padding:8px 10px; margin-bottom:12px; box-shadow: var(--shadow); }
.video-hint { display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--text-body); background:var(--accent-soft); border-radius:12px; padding:9px 13px; margin-top:10px; }
"""

if "theme" not in st.session_state: st.session_state.theme = "light"
if "chat" not in st.session_state: st.session_state.chat = None
if "messages" not in st.session_state: st.session_state.messages = []
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
if "model_index" not in st.session_state: st.session_state.model_index = 0
if "videolar" not in st.session_state: st.session_state.videolar = {}
if "client" not in st.session_state: st.session_state.client = None
if "son_gorseller" not in st.session_state: st.session_state.son_gorseller = {}

active_css = (DARK_VARS if st.session_state.theme == "dark" else LIGHT_VARS) + BASE_CSS
st.markdown(f"<style>{active_css}</style>", unsafe_allow_html=True)

SISTEM_TALIMATI = (
    "Sen StudAI Math adlı bir ders asistanısın; matematik, fizik, kimya ve biyoloji sorularını çözersin. "
    "Öğrenciler sana bir fotoğraf, sadece yazılı bir soru ya da ikisini birden gönderebilir. Şu kurallara harfiyen uy:\n\n"
    "1. GÖRSEL VARSA KUSURSUZ ANALİZ: Fotoğraftaki tüm şekilleri, grafikleri, sayıları, işaretleri piksel piksel incele.\n"
    "2. TUZAK KONTROLÜ: Soru kökündeki 'olamaz', 'kesinlikle doğrudur', 'en az' gibi detayları analiz et.\n"
    "3. PYTHON KESİNLİĞİYLE ÇÖZ: Soruyu zihninden çözme; sana verilen kod çalıştırma aracını kullanarak hesaplamaları gerçekten çalıştır ve kesin sonucu bul.\n"
    "4. ÇİFT KONTROL: Sonucu yazmadan önce baştan sona iki kez kontrol et.\n"
    "5. NET AÇIKLAMA: Önce mantığı açıkla, formülleri göster, nihai doğru şıkkı/sonucu KALIN harflerle belirt.\n"
    "6. SOHBET: Öğrenci ek soru sorarsa önceki çözümünle çelişmeden, sabırlı ve öğretici bir dille devam et.\n"
    "7. KİMLİK: Biri 'Yapımcın kim?', 'Seni kim yaptı?', 'Sahibin kim?' gibi bir şey sorarsa yalnızca 'Beni Bartu Koca yaptı.' de. "
    "Biri 'Adın ne?' diye sorarsa yalnızca 'Benim adım StudAI Math.' de. Bu iki soru dışında kimliğinden kendiliğinden bahsetme.\n"
    "8. ŞEKİL ÇİZİMİ: Geometri (üçgen, çember, açı, grafik, fonksiyon vb.) içeren bir soru gelirse şekli TAHMİNİ KELİMELERLE ANLATMA; "
    "kod çalıştırma aracında matplotlib ile gerçek koordinatlarla çiz ve plt.show() ile göster. "
    "Şekli çizerken plt.style.use('dark_background') kullan, çizgi/nokta renklerini parlak yeşil (#39d98a) veya beyaz yap ki video temasıyla uyumlu olsun.\n"
    "9. BİÇİM: Cevaplarında ASLA LaTeX veya $ işareti kullanma (örn. $x^2$ YAZMA). Üs için ^ işareti (x^2), kesir için a/b şeklinde düz metin, "
    "çarpma için x veya *, karekök için √ sembolü kullan. Başlık için # işareti kullanma; düz metin ve numaralı liste (1. 2. 3.) ile yaz."
)

def aktif_model():
    return FALLBACK_MODELS[st.session_state.model_index]

def istemci_al():
    if st.session_state.client is None:
        st.session_state.client = genai.Client(api_key=API_KEY)
    return st.session_state.client

GECICI_HATA_ANAHTARLARI = ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "quota", "rate limit", "overloaded"]

def gecici_hata_mi(hata_metni):
    hm = (hata_metni or "").lower()
    return any(anahtar.lower() in hm for anahtar in GECICI_HATA_ANAHTARLARI)

KAPANMA_ANAHTARLARI = ["client kapat", "client has been closed", "closed client", "istemci kapat"]

def istemci_kapali_mi(hata_metni):
    hm = (hata_metni or "").lower()
    return any(anahtar.lower() in hm for anahtar in KAPANMA_ANAHTARLARI)

def _kod_calistirma_araci():
    """SDK sürümüne göre code_execution aracını uyumlu şekilde oluşturur."""
    try:
        return types.Tool(code_execution=types.ToolCodeExecution())
    except AttributeError:
        try:
            return types.Tool(code_execution={})
        except Exception:
            return None

def sohbeti_yeniden_kur(onceki_mesajlar=None):
    client = istemci_al()
    gecmis = []
    if onceki_mesajlar:
        for m in onceki_mesajlar:
            rol = "model" if m["role"] == "assistant" else "user"
            gecmis.append(types.Content(role=rol, parts=[types.Part.from_text(text=m["content"])]))

    arac = _kod_calistirma_araci()
    tools_listesi = [arac] if arac is not None else []

    try:
        return client.chats.create(
            model=aktif_model(),
            history=gecmis,
            config=types.GenerateContentConfig(
                system_instruction=SISTEM_TALIMATI,
                temperature=0.0,
                max_output_tokens=8192,
                tools=tools_listesi,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            ),
        )
    except Exception:
        # thinking_config veya tools desteklenmiyorsa daha sade konfigürasyona düş
        return client.chats.create(
            model=aktif_model(),
            history=gecmis,
            config=types.GenerateContentConfig(
                system_instruction=SISTEM_TALIMATI,
                temperature=0.0,
                max_output_tokens=8192,
                tools=tools_listesi,
            ),
        )

def gonder_guvenli(icerik, gecmis_mesajlar=None):
    son_hata = None
    while True:
        chat = st.session_state.chat
        for deneme in range(2):
            try:
                return chat.send_message(icerik)
            except Exception as e:
                hata = str(e)
                if gecici_hata_mi(hata):
                    son_hata = hata
                    time.sleep(2 * (deneme + 1))
                    continue
                if istemci_kapali_mi(hata):
                    st.session_state.client = None
                    st.session_state.chat = sohbeti_yeniden_kur(gecmis_mesajlar)
                    chat = st.session_state.chat
                    continue
                raise
        if st.session_state.model_index < len(FALLBACK_MODELS) - 1:
            st.session_state.model_index += 1
            st.session_state.chat = sohbeti_yeniden_kur(gecmis_mesajlar)
        else:
            raise Exception(son_hata or "Tüm modeller yoğun, lütfen birazdan tekrar dene.")

def yaniti_ayikla(response):
    metin_parcalari, gorseller = [], []
    try:
        parts = response.candidates[0].content.parts
    except Exception:
        parts = []
    for part in parts:
        if getattr(part, "text", None):
            metin_parcalari.append(part.text)
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None):
            gorseller.append(inline.data)
    metin = "\n".join(metin_parcalari).strip() or (response.text or "")
    return metin, gorseller

def gonder_ve_tamamla(text, img=None):
    icerik = []
    if img is not None:
        icerik.append(img)
    if text:
        icerik.append(text)
    elif img is not None:
        icerik.append("Lütfen bu görseldeki soruyu 'Sıfır Hata' ile adım adım çöz.")

    if st.session_state.chat is None:
        st.session_state.chat = sohbeti_yeniden_kur()

    onceki_gecmis = st.session_state.messages[:-1] if st.session_state.messages else []

    try:
        response = gonder_guvenli(icerik, onceki_gecmis)
    except Exception as e:
        hata = str(e)
        if gecici_hata_mi(hata):
            return "⏳ Şu an tüm modeller yoğun. Birazdan tekrar dener misin?", []
        return f"❌ Bir hata oluştu, bu 'modeller yoğun' değil, gerçek bir sorun: {hata}", []

    tam_metin, tum_gorseller = yaniti_ayikla(response)
    while response.candidates and len(response.candidates) > 0 and response.candidates[0].finish_reason in ["MAX_TOKENS", "LENGTH"]:
        try:
            response = gonder_guvenli("Lütfen yarım kalan çözümüne hiçbir adımı atlamadan, kaldığın kelimeden aynen devam et.", st.session_state.messages)
        except Exception as e:
            hata = str(e)
            if gecici_hata_mi(hata):
                return tam_metin + "\n\n⏳ (Devamı alınırken sunucu yoğunluğu nedeniyle kesildi.)", tum_gorseller
            return tam_metin + f"\n\n❌ (Devamı alınırken hata oluştu: {hata})", tum_gorseller
        ek_metin, ek_gorseller = yaniti_ayikla(response)
        tam_metin += "\n" + ek_metin
        tum_gorseller += ek_gorseller
    return tam_metin, tum_gorseller

def yeni_soru():
    st.session_state.chat = None
    st.session_state.messages = []
    st.session_state.uploader_key += 1
    st.session_state.model_index = 0
    st.session_state.videolar = {}
    st.session_state.son_gorseller = {}

# ============================================================
# METİN TEMİZLEME / BÖLME
# ============================================================
def bicimi_temizle(metin):
    if not metin:
        return metin
    t = metin
    t = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", t)
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", t)
    t = re.sub(r"\\times", "x", t)
    t = re.sub(r"\\cdot", "x", t)
    t = re.sub(r"\\div", "/", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = re.sub(r"\$\$?", "", t)
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"[#>`{}]", "", t)
    return t

def metni_adimlara_ayir(metin):
    temiz = bicimi_temizle(metin)
    parcalar = re.split(r"\n(?=\d+[\.\)]\s)", temiz.strip())
    if len(parcalar) < 2:
        parcalar = [p.strip() for p in temiz.split("\n\n") if p.strip()]
    parcalar = [p.strip() for p in parcalar if p.strip()]
    return parcalar if parcalar else [temiz.strip()]

def blogu_satirlara_ayir(blok):
    satirlar = []
    for parca in re.split(r"(?<=[\.\?\!:])\s+(?=[A-ZÇĞİÖŞÜ0-9])", blok.strip()):
        parca = parca.strip()
        if parca:
            satirlar.append(parca)
    return satirlar if satirlar else [blok.strip()]

# ============================================================
# 🖼️ GÖRSEL/GRAFİK ÜRETİMİ
# ============================================================
GORSEL_BOYUT = (720, 720)

def _yeni_eksen():
    fig = plt.figure(figsize=(GORSEL_BOYUT[0] / 100, GORSEL_BOYUT[1] / 100), dpi=100)
    fig.patch.set_facecolor("#050a08")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#050a08")
    for spine in ax.spines.values():
        spine.set_color("#183028")
    ax.tick_params(colors="#4f7a63")
    return fig, ax

def _figden_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def cizim_ucgen(etiket):
    fig, ax = _yeni_eksen()
    pts = [(0, 0), (5, 0), (random.uniform(1, 4), random.uniform(2.5, 4.5))]
    tri = plt.Polygon(pts, closed=True, fill=False, edgecolor="#39d98a", linewidth=3)
    ax.add_patch(tri)
    for i, p in enumerate(pts):
        ax.plot(*p, "o", color="#7ef4c2", markersize=7)
        ax.text(p[0], p[1] - 0.35, ["A", "B", "C"][i], color="#d9ffe9", fontsize=15, ha="center", weight="bold")
    ax.set_xlim(-1, 6); ax.set_ylim(-1.5, 5.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(etiket, color="#d9ffe9", fontsize=14, pad=14)
    return _figden_pil(fig)

def cizim_cember(etiket):
    fig, ax = _yeni_eksen()
    r = random.uniform(2, 4)
    cember = plt.Circle((0, 0), r, fill=False, edgecolor="#39d98a", linewidth=3)
    ax.add_patch(cember)
    ax.plot(0, 0, "o", color="#7ef4c2", markersize=6)
    ax.text(0, -0.3, "O", color="#d9ffe9", fontsize=14, ha="center", weight="bold")
    aci = random.uniform(20, 70)
    x, y = r * np.cos(np.radians(aci)), r * np.sin(np.radians(aci))
    ax.plot([0, x], [0, y], color="#7ef4c2", linewidth=2)
    ax.plot(x, y, "o", color="#7ef4c2", markersize=6)
    ax.text(x * 1.12, y * 1.12, "A", color="#d9ffe9", fontsize=14, ha="center", weight="bold")
    lim = r + 1.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(etiket, color="#d9ffe9", fontsize=14, pad=14)
    return _figden_pil(fig)

def cizim_fonksiyon(etiket):
    fig, ax = _yeni_eksen()
    x = np.linspace(-5, 5, 400)
    a, b, c = random.uniform(0.3, 1.2), random.uniform(-2, 2), random.uniform(-3, 3)
    y = a * x**2 + b * x + c
    ax.plot(x, y, color="#39d98a", linewidth=2.5)
    ax.axhline(0, color="#4f7a63", linewidth=1)
    ax.axvline(0, color="#4f7a63", linewidth=1)
    ax.set_xlim(-5, 5)
    ax.set_title(etiket, color="#d9ffe9", fontsize=14, pad=14)
    return _figden_pil(fig)

def metin_karti_ciz(satirlar, etiket):
    """Grafik gerektirmeyen adımlar için tahtaya yazı yazılmış gibi bir kart üretir."""
    img = Image.new("RGB", GORSEL_BOYUT, "#050a08")
    draw = ImageDraw.Draw(img)
    try:
        font_baslik = ImageFont.truetype(FONT_PATH, 26) if FONT_PATH else ImageFont.load_default()
        font_govde = ImageFont.truetype(FONT_PATH, 22) if FONT_PATH else ImageFont.load_default()
    except Exception:
        font_baslik = font_govde = ImageFont.load_default()

    draw.text((40, 40), etiket, font=font_baslik, fill="#7ef4c2")
    y = 100
    for satir in satirlar:
        for line in textwrap.wrap(satir, width=42):
            draw.text((40, y), line, font=font_govde, fill="#d9ffe9")
            y += 34
        y += 16
        if y > GORSEL_BOYUT[1] - 60:
            break
    return img

def adim_gorseli_uret(blok, adim_no):
    etiket = f"Adım {adim_no}"
    dusuk = blok.lower()
    if "üçgen" in dusuk:
        return cizim_ucgen(etiket)
    if "çember" in dusuk or "daire" in dusuk:
        return cizim_cember(etiket)
    if "fonksiyon" in dusuk or "grafi" in dusuk or "parabol" in dusuk:
        return cizim_fonksiyon(etiket)
    return metin_karti_ciz(blogu_satirlara_ayir(blok), etiket)

# ============================================================
# 🎬 VİDEO ÜRETİMİ (görsel + TTS ses -> tek klip)
# ============================================================
def video_uret(soru_id, tam_metin):
    bloklar = metni_adimlara_ayir(tam_metin)
    klipler = []
    tmp_dosyalar = []
    for i, blok in enumerate(bloklar, start=1):
        gorsel = adim_gorseli_uret(blok, i)
        gorsel_yolu = f"/tmp/{soru_id}_adim{i}.png"
        gorsel.save(gorsel_yolu)
        tmp_dosyalar.append(gorsel_yolu)

        ses_yolu = f"/tmp/{soru_id}_ses{i}.mp3"
        try:
            gTTS(text=blok, lang="tr").save(ses_yolu)
            ses = AudioFileClip(ses_yolu)
            sure = max(ses.duration, 1.5)
            klip = ImageClip(gorsel_yolu).set_duration(sure).set_audio(ses)
            tmp_dosyalar.append(ses_yolu)
        except Exception:
            klip = ImageClip(gorsel_yolu).set_duration(3)
        klipler.append(klip)

    if not klipler:
        return None

    video_yolu = f"/tmp/{soru_id}_video.mp4"
    final = concatenate_videoclips(klipler, method="compose")
    final.write_videofile(video_yolu, fps=24, codec="libx264", audio_codec="aac", logger=None)

    for dosya in tmp_dosyalar:
        try:
            os.remove(dosya)
        except Exception:
            pass

    return video_yolu

# ============================================================
# 🖥️ ARAYÜZ
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="sb-logo"><div class="mark">✨</div>'
        '<div><div class="brand">StudAI Math</div><div class="sub">Ders Asistanı</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-section-title">Görünüm</div>', unsafe_allow_html=True)
    tema_secim = st.radio("Tema", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1, label_visibility="collapsed")
    if tema_secim != st.session_state.theme:
        st.session_state.theme = tema_secim
        st.rerun()

    st.markdown('<div class="sb-section-title">Oturum</div>', unsafe_allow_html=True)
    if st.button("🆕 Yeni Soru", use_container_width=True):
        yeni_soru()
        st.rerun()

    st.markdown(
        '<div class="sb-info-card"><div class="t">ℹ️ Nasıl kullanılır?</div>'
        '<div class="d">Bir soru fotoğrafı yükle veya soruyu yazarak gönder. '
        'StudAI adım adım çözer, istersen çözümü videoya çevirir.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="sb-status"><div class="dot"></div>Aktif model: {aktif_model()}</div>', unsafe_allow_html=True)

st.markdown('<div class="badge-pill">✨ Sıfır Hata Modu</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Sorunu çöz, <span class="accent-text">adım adım anla</span>.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Matematik, fizik, kimya veya biyoloji sorunu fotoğrafla ya da yazarak gönder.</div>', unsafe_allow_html=True)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(bicimi_temizle(msg["content"]))
        if msg["role"] == "assistant":
            for gorsel_b64 in st.session_state.son_gorseller.get(i, []):
                st.image(gorsel_b64)
            video_yolu = st.session_state.videolar.get(i)
            if video_yolu and os.path.exists(video_yolu):
                st.video(video_yolu)
            else:
                if st.button("🎬 Bu çözümü videoya çevir", key=f"video_{i}"):
                    with st.spinner("Video hazırlanıyor..."):
                        yol = video_uret(f"soru{i}", msg["content"])
                    if yol:
                        st.session_state.videolar[i] = yol
                        st.rerun()

yuklenen_gorsel = st.file_uploader(
    "Soru fotoğrafı (opsiyonel)", type=["png", "jpg", "jpeg"], key=f"uploader_{st.session_state.uploader_key}"
)

with st.form("soru_formu", clear_on_submit=True):
    kullanici_metni = st.text_input("Sorunu yaz veya fotoğrafı açıkla...", label_visibility="collapsed")
    gonder = st.form_submit_button("Gönder")

if gonder and (kullanici_metni or yuklenen_gorsel is not None):
    img = None
    if yuklenen_gorsel is not None:
        img = Image.open(yuklenen_gorsel)

    kullanici_gosterim = kullanici_metni or "(fotoğraf gönderildi)"
    st.session_state.messages.append({"role": "user", "content": kullanici_gosterim})

    with st.spinner("Çözülüyor..."):
        cevap_metni, gorseller = gonder_ve_tamamla(kullanici_metni, img)

    st.session_state.messages.append({"role": "assistant", "content": cevap_metni})
    st.session_state.son_gorseller[len(st.session_state.messages) - 1] = gorseller
    st.rerun()
