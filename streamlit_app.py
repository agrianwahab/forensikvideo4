# --- START OF FILE streamlit_app.py (MODIFIED) ---

import streamlit as st
from pathlib import Path
from typing import Optional, Union
import tempfile
import ForensikVideo as fv
import sys
import io
import traceback
from datetime import datetime
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from history_manager import HistoryManager  # Import history manager

# --- KONFIGURASI HALAMAN DAN GAYA ---
st.set_page_config(
    page_title="VIFA-Pro | Dashboard Forensik Video",
    layout="wide"
)

# Kustomisasi CSS untuk tema biru-putih profesional dengan tambahan untuk Tahap 4 Enhanced
# dan komponen riwayat analisis yang diperkaya.
st.markdown("""
    <style>
    .stApp { background-color: #F0F2F6; }
    h1 { color: #0B3D91; font-weight: bold; }
    h2, h3 { color: #0056b3; }
    .stButton>button { border-radius: 8px; border: 1px solid #0c6dd6; background-color: #0c6dd6; color: white; transition: all 0.2s; }
    .stButton>button:hover { border-color: #004494; background-color: #0056b3; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
    .stDataFrame { width: 100%; }
    
    /* Tambahan untuk Tahap 3 */
    .explanation-box {
        background-color: #FFF8DC;
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .technical-box {
        background-color: #F0F8FF;
        border: 1px solid #4682B4;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        font-family: monospace;
    }
    .severity-high { color: #FF0000; font-weight: bold; }
    .severity-medium { color: #FFA500; font-weight: bold; }
    .severity-low { color: #008000; }
    
    /* Tambahan untuk Tahap 4 Enhanced */
    .integrity-score-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .pipeline-stage-card {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .localization-event-card {
        background-color: #F8F9FA;
        border-left: 5px solid #0c6dd6;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .metric-explanation {
        background-color: #E8F4F8;
        border: 1px solid #B8E0E8;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .score-breakdown-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #E0E0E0;
    }
    
    /* --- STYLE UNTUK HALAMAN RIWAYAT (SANGAT DETAIL) --- */
    .history-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        border-left: 7px solid #0056b3;
    }
    .history-card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    .history-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .history-date { color: #6c757d; font-size: 0.9em; }
    .history-score-badge {
        font-weight: bold;
        padding: 6px 12px;
        border-radius: 50px;
        color: white;
        font-size: 1.1em;
        min-width: 120px;
        text-align: center;
    }
    .history-anomaly-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 15px;
    }
    .history-anomaly-tag {
        font-size: 0.85em;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 15px;
        color: white;
    }
    .history-actions {
        display: flex;
        gap: 10px;
        margin-top: 20px;
    }
    .history-empty {
        text-align: center; padding: 50px; color: #777; background-color: #f9f9f9;
        border-radius: 10px; margin: 20px 0; border: 2px dashed #d0d0d0;
    }
    .history-toolbar {
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;
        padding: 10px; background-color: #FFFFFF; border-radius: 8px;
    }
    .history-detail-section {
        background-color: #FFFFFF; padding: 25px; border-radius: 10px; margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .history-artifact-container {
        text-align: center; margin-top: 15px; padding: 15px;
        background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;
    }
    .history-artifact-container img { max-width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔎 VIFA-Pro: Sistem Deteksi Forensik Keaslian Video")
st.markdown("Menggunakan **Metode K-Means** dan **Localization Tampering** dengan Dukungan ELA dan SIFT")

# Inisialisasi history manager
history_manager = HistoryManager()

# --- KONTROL PANEL DI SIDEBAR ---
with st.sidebar:
    selected_tab = st.radio(
        "Menu Navigasi",
        ["Analisis Baru", "Riwayat Analisis"],
        captions=["Jalankan forensik pada video baru", "Lihat, detailkan, atau hapus analisis sebelumnya"],
        format_func=lambda x: "🔍 " + x if x == "Analisis Baru" else "📜 " + x
    )
    
    st.markdown("---")

    if selected_tab == "Analisis Baru":
        st.header("Panel Kontrol Analisis")
        uploaded_video = st.file_uploader(
            "Unggah Video Bukti", type=["mp4", "avi", "mov", "mkv"]
        )
        baseline_video = st.file_uploader(
            "Unggah Video Baseline (Opsional)", type=["mp4", "avi", "mov", "mkv"]
        )
        fps = st.number_input("Frame Extraction FPS", min_value=1, max_value=30, value=10, step=1)
        run = st.button("🚀 Jalankan Analisis Forensik", use_container_width=True, type="primary")
        
        st.subheader("Pengaturan Detail")
        show_technical_details = st.checkbox("Tampilkan Detail Teknis", value=True)
        show_simple_explanations = st.checkbox("Tampilkan Penjelasan Sederhana", value=True)
        show_advanced_metrics = st.checkbox("Tampilkan Metrik Lanjutan", value=False)
    
    else: # Sidebar untuk Riwayat Analisis
        st.header("Pengaturan Riwayat")
        
        # Opsi hapus semua dengan konfirmasi yang lebih aman
        if st.button("🗑️ Hapus Semua Riwayat", use_container_width=True):
            st.session_state['confirm_delete_all_prompt'] = True

        if st.session_state.get('confirm_delete_all_prompt', False):
            st.warning("PERINGATAN: Tindakan ini tidak dapat diurungkan.")
            confirm_text = st.text_input("Ketik 'HAPUS SEMUA' untuk konfirmasi:", key="confirm_delete_all_text")
            if confirm_text == "HAPUS SEMUA":
                if st.button("Konfirmasi Hapus Semua", type="primary"):
                    count = history_manager.delete_all_history()
                    st.success(f"Berhasil menghapus {count} riwayat analisis!")
                    del st.session_state['confirm_delete_all_prompt']
                    del st.session_state['confirm_delete_all_text']
                    time.sleep(1)
                    st.rerun()
        
        st.subheader("Filter Riwayat")
        # TODO: Implement filter logic in render_history_page
        search_query = st.text_input("Cari Nama Video...")
        min_score, max_score = st.slider(
            "Filter Skor Integritas", 0, 100, (0, 100)
        )


# --- FUNGSI BANTUAN ---
def load_image_as_bytes(path_str: Optional[Union[str, Path]]) -> Optional[bytes]:
    if path_str and Path(path_str).exists():
        try:
            with open(path_str, "rb") as f: return f.read()
        except Exception: return None
    return None

def _get_metric_description(metric_name: str, value: any) -> str:
    descriptions = {
        'optical_flow_z_score': 'Z-score > 4 menunjukkan pergerakan abnormal',
        'ssim_drop': 'Penurunan > 0.25 menunjukkan perubahan drastis',
        'ssim_absolute_low': 'Nilai < 0.7 menunjukkan frame sangat berbeda',
        'color_cluster_jump': 'Perubahan nomor klaster = perubahan adegan',
        'source_frame': 'Indeks frame yang diduplikasi',
        'ssim_to_source': 'Kemiripan dengan frame sumber (1 = identik)',
        'sift_inliers': 'Jumlah titik yang cocok sempurna',
        'sift_good_matches': 'Total titik kandidat',
        'sift_inlier_ratio': 'Rasio kecocokan valid',
        'ela_max_difference': 'Perbedaan kompresi maksimal',
        'ela_suspicious_regions': 'Jumlah area mencurigakan'
    }
    return descriptions.get(metric_name, 'Metrik analisis')

def create_gauge_chart(value: int, title: str) -> go.Figure:
    value = int(value)
    if value < 50:
        bar_color = "#dc3545" # Merah
    elif value < 70:
        bar_color = "#fd7e14" # Oranye
    elif value < 80:
        bar_color = "#ffc107" # Kuning
    elif value < 90:
        bar_color = "#17a2b8" # Biru muda
    else:
        bar_color = "#28a745" # Hijau

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ced4da",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(220, 53, 69, 0.2)'},
                {'range': [50, 80], 'color': 'rgba(255, 193, 7, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(40, 167, 69, 0.2)'}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'family': "Arial"}, height=250)
    return fig

def format_timestamp(iso_timestamp):
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%d %b %Y, %H:%M:%S")
    except (ValueError, TypeError):
        return iso_timestamp

def get_score_color(score):
    if score >= 90: return "#28a745"  # Green
    elif score >= 80: return "#17a2b8" # Teal
    elif score >= 70: return "#ffc107"  # Yellow
    else: return "#dc3545" # Red

def get_anomaly_style(atype):
    styles = {
        "duplication": ("🔁 Duplikasi", "#dc3545"),
        "discontinuity": ("✂️ Diskontinuitas", "#007bff"),
        "insertion": ("➕ Penyisipan", "#fd7e14"),
    }
    return styles.get(atype, ("❓ Lainnya", "#6c757d"))

def display_history_card(entry):
    """Menampilkan kartu ringkasan untuk satu entri riwayat."""
    with st.container():
        integrity = entry.get("integrity_analysis", {})
        score = int(integrity.get("score", 0))
        description = integrity.get("description", "Status tidak diketahui")
        score_color = get_score_color(score)

        st.markdown(f'<div class="history-card" style="border-left-color: {score_color};">', unsafe_allow_html=True)

        # Header: Nama video dan tanggal
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader(entry.get("video_name", "Video Tanpa Nama"))
        with c2:
            st.markdown(f'<div class="history-date">{format_timestamp(entry.get("timestamp"))}</div>', unsafe_allow_html=True)

        # Badan: Skor dan Statistik Kunci
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="history-score-badge" style="background-color: {score_color};">Skor: {score}%</div>', unsafe_allow_html=True)
            st.caption(description)
        with c2:
            st.metric("Total Anomali", entry.get("summary", {}).get("total_anomaly", 0))
        with c3:
            st.metric("Total Peristiwa", entry.get("localizations_count", 0))

        # Tags Anomali
        st.markdown('<div class="history-anomaly-tags">', unsafe_allow_html=True)
        anomaly_types = entry.get("anomaly_types", {})
        if sum(anomaly_types.values()) > 0:
            for atype, count in anomaly_types.items():
                if count > 0:
                    label, color = get_anomaly_style(atype)
                    st.markdown(f'<div class="history-anomaly-tag" style="background-color: {color};">{label}: {count}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="history-anomaly-tag" style="background-color: #28a745;">✅ Tidak ada anomali terdeteksi</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tombol Aksi
        st.markdown('<div class="history-actions">', unsafe_allow_html=True)
        cols = st.columns(6)
        with cols[0]:
            if st.button("🔍 Lihat Detail", key=f"view_{entry['id']}", use_container_width=True, type="primary"):
                st.session_state.selected_history_id = entry["id"]
                st.rerun()
        with cols[1]:
            if st.button("🗑️ Hapus", key=f"delete_{entry['id']}", use_container_width=True):
                history_manager.delete_analysis(entry["id"])
                st.toast(f"Analisis untuk '{entry['video_name']}' telah dihapus.")
                time.sleep(0.5)
                st.rerun()

        st.markdown('</div></div>', unsafe_allow_html=True)

def display_history_detail(entry_id):
    """Menampilkan halaman detail yang komprehensif untuk satu entri riwayat."""
    entry = history_manager.get_analysis(entry_id)
    if not entry:
        st.error("Gagal memuat riwayat analisis. Mungkin sudah dihapus.")
        if st.button("Kembali ke Riwayat"): st.session_state.selected_history_id = None
        return

    # Header
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.header(f"Detail Analisis: {entry.get('video_name', 'N/A')}")
        st.caption(f"Dianalisis pada: {format_timestamp(entry.get('timestamp'))} | Hash Preservasi: {entry.get('preservation_hash', 'N/A')[:20]}...")
    with col2:
        if st.button("⬅️ Kembali ke Riwayat", use_container_width=True):
            st.session_state.selected_history_id = None
            st.rerun()

    tabs = st.tabs(["📊 Ringkasan Eksekutif", "🔍 Investigasi Anomali", "🖼️ Artefak Visual", "📝 Metadata & Laporan"])

    # === Tab 1: Ringkasan Eksekutif ===
    with tabs[0]:
        with st.container(border=True):
            st.subheader("Penilaian Integritas Video")
            integrity = entry.get("integrity_analysis", {})
            score = int(integrity.get("score", 0))
            explanation = history_manager.get_integrity_explanation(score)

            c1, c2 = st.columns([1, 2])
            with c1:
                st.plotly_chart(create_gauge_chart(score, "Skor Integritas"), use_container_width=True)

            with c2:
                st.markdown(f"<h3 style='color:{explanation['color']};'>{explanation['category']}</h3>", unsafe_allow_html=True)
                st.markdown(explanation['description'])
                st.markdown(f"**Rekomendasi:** {explanation['recommendation']}")

            if integrity.get("calculation_details"):
                with st.expander("Lihat Rincian Perhitungan Skor"):
                    details = integrity["calculation_details"]
                    st.write(f"**Metode Penilaian:** {details.get('scoring_method', 'N/A')}")
                    st.metric("Skor Dasar", f"{details.get('base_score', 0)}%", f"Berdasarkan {details.get('percentage_anomaly', 0):.1f}% anomali")
                    if details.get("adjustments"):
                        st.write("**Faktor Penyesuaian:**")
                        for name, value in details["adjustments"]:
                            emoji = "✅" if value > 0 else "❌"
                            st.text(f"- {emoji} {name}: {value:+d}%")

        with st.container(border=True):
            st.subheader("Statistik Utama")
            summary = entry.get("summary", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Frame", f"{summary.get('total_frames', 0):,}")
            c2.metric("Total Anomali", f"{summary.get('total_anomaly', 0):,}", help="Jumlah frame individual yang ditandai anomali.")
            c3.metric("Persentase Anomali", f"{summary.get('pct_anomaly', 0):.1f}%")
            c4.metric("Jumlah Peristiwa", f"{entry.get('localizations_count', 0)}", help="Jumlah kelompok anomali yang terjadi berdekatan.")
            
            # Pie Chart Anomaly
            anomaly_types = entry.get("anomaly_types", {})
            if sum(anomaly_types.values()) > 0:
                labels = [get_anomaly_style(k)[0] for k, v in anomaly_types.items() if v > 0]
                values = [v for v in anomaly_types.values() if v > 0]
                colors = [get_anomaly_style(k)[1] for k, v in anomaly_types.items() if v > 0]
                
                pie_fig = px.pie(values=values, names=labels, title="Distribusi Tipe Anomali", color_discrete_sequence=colors)
                pie_fig.update_layout(height=350, legend_title_text='Tipe Anomali')
                st.plotly_chart(pie_fig, use_container_width=True)


    # === Tab 2: Investigasi Anomali ===
    with tabs[1]:
        st.subheader("Rincian Setiap Peristiwa Anomali")
        localizations = entry.get("localizations", [])
        if not localizations:
            st.success("✅ Tidak ditemukan peristiwa anomali yang signifikan dalam analisis ini.")
        else:
            st.info(f"Ditemukan {len(localizations)} peristiwa anomali. Peristiwa adalah kelompok anomali yang terjadi berdekatan dan dianggap sebagai satu kejadian.")
            
            # Sortir berdasarkan waktu mulai
            sorted_locs = sorted(localizations, key=lambda x: x.get('start_ts', 0))

            for i, loc in enumerate(sorted_locs):
                atype = loc.get('event', '').replace('anomaly_', '')
                desc = history_manager.get_anomaly_description(atype)

                with st.expander(f"{desc['icon']} **Peristiwa #{i+1}: {desc['title']}** (Mulai: {loc['start_ts']:.2f}s, Durasi: {loc.get('duration', 0):.2f}s)", expanded=i==0):
                    st.markdown(f'<div class="localization-event-card" style="border-left-color: {desc["color"]};">', unsafe_allow_html=True)
                    
                    st.markdown("#### 📖 **Apa Artinya Ini?**")
                    st.markdown(f"**Analogi Sederhana:** *{desc['example']}*")
                    st.markdown(f"**Implikasi Forensik:** {desc['implication']}")
                    
                    st.markdown("---")
                    
                    st.markdown("#### 🔧 **Detail Teknis**")
                    st.markdown(f"**Metode Deteksi:** {desc['technical']}")
                    
                    if loc.get("metrics"):
                        st.write("**Metrik Kunci:**")
                        for key, val in loc["metrics"].items():
                            st.text(f"  - {key.replace('_', ' ').title()}: {val}")

                    if loc.get("visualizations"):
                        st.markdown("#### 🖼️ **Bukti Visual Pendukung**")
                        cols = st.columns(len(loc['visualizations']))
                        for idx, (v_key, v_path) in enumerate(loc['visualizations'].items()):
                             with cols[idx]:
                                img_data = history_manager.get_artifact_base64(v_path)
                                if img_data:
                                    st.image(img_data, caption=v_key.replace('_',' ').title(), use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)


    # === Tab 3: Artefak Visual ===
    with tabs[2]:
        st.subheader("Galeri Visualisasi Hasil Analisis")
        st.info("Ini adalah kumpulan plot dan gambar yang dihasilkan selama proses analisis, memberikan gambaran visual dari temuan.")
        saved_artifacts = entry.get("saved_artifacts", {})
        plot_artifacts = {k: v for k, v in saved_artifacts.items() if not k.startswith("anomaly_frame")}

        if not plot_artifacts:
            st.warning("Tidak ada artefak visual yang tersimpan untuk analisis ini.")
        else:
            for key, path in plot_artifacts.items():
                img_data = history_manager.get_artifact_base64(path)
                if img_data:
                    title = key.replace('_', ' ').title()
                    with st.container(border=True):
                        st.subheader(title)
                        st.markdown(f'<div class="history-artifact-container"><img src="{img_data}"></div>', unsafe_allow_html=True)

    # === Tab 4: Metadata & Laporan ===
    with tabs[3]:
        with st.container(border=True):
            st.subheader("📤 Ekspor Hasil Analisis")
            st.markdown("Anda dapat mengunduh seluruh hasil analisis ini sebagai sebuah file ZIP. File ini berisi laporan data dalam format JSON dan semua artefak visual yang tersimpan.")
            
            zip_data = history_manager.export_analysis(entry_id)
            if zip_data:
                filename = f"VIFA-Pro_Analysis_{entry_id[:8]}_{entry.get('video_name', 'export').replace(' ', '_')}.zip"
                st.download_button(
                    label="💾 Unduh Laporan ZIP",
                    data=zip_data,
                    file_name=filename,
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.error("Gagal membuat file ekspor.")
        
        with st.container(border=True):
            st.subheader("📝 Metadata Video")
            metadata = entry.get("metadata", {})
            if not metadata:
                st.warning("Tidak ada metadata yang tersimpan.")
            else:
                for category, items in metadata.items():
                    if items:
                        st.markdown(f"##### {category}")
                        df = pd.DataFrame.from_dict(items, orient='index', columns=['Nilai'])
                        st.table(df)


def render_history_page():
    """Merender halaman utama riwayat analisis, termasuk daftar, toolbar, dan filter."""
    st.title("📜 Riwayat Analisis Forensik")
    st.markdown("Telusuri, detailkan, dan kelola semua analisis video yang telah dilakukan.")
    
    # Cek jika ada detail yang harus ditampilkan
    if st.session_state.get('selected_history_id'):
        display_history_detail(st.session_state.selected_history_id)
        return

    history = history_manager.load_history()
    if not history:
        st.markdown('<div class="history-empty"><h3>Belum Ada Riwayat Analisis</h3><p>Mulai dengan menjalankan analisis baru di menu navigasi.</p></div>', unsafe_allow_html=True)
        return
        
    # Tampilkan ringkasan dan statistik riwayat
    st.subheader("Ringkasan Seluruh Riwayat")
    avg_score = np.mean([entry.get("integrity_analysis", {}).get("score", 0) for entry in history]) if history else 0
    total_anomalies = sum(entry.get("summary", {}).get("total_anomaly", 0) for entry in history)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Analisis Tersimpan", len(history))
    c2.metric("Rata-rata Skor Integritas", f"{avg_score:.1f}%")
    c3.metric("Total Anomali Ditemukan", f"{total_anomalies:,}")
    st.markdown("---")

    st.subheader("Daftar Riwayat Analisis")
    
    # Logic filter (dari sidebar)
    search_query = st.session_state.get('search_query', '').lower()
    min_score, max_score = st.session_state.get('slider_0', (0, 100)) # st.slider creates 'slider-0' etc. key
    
    filtered_history = [
        entry for entry in history
        if search_query in entry.get("video_name", "").lower() and
           min_score <= entry.get("integrity_analysis", {}).get("score", 0) <= max_score
    ]

    if not filtered_history:
        st.warning("Tidak ada riwayat yang cocok dengan kriteria filter Anda.")
    else:
        # Urutkan berdasarkan yang terbaru
        sorted_history = sorted(filtered_history, key=lambda x: x.get("timestamp", ""), reverse=True)
        for entry in sorted_history:
            display_history_card(entry)


# --- LOGIKA UTAMA APLIKASI ---
if selected_tab == "Analisis Baru":
    if run:
        if uploaded_video is None:
            st.error("⚠️ Mohon unggah video bukti terlebih dahulu di sidebar.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                sus_path = tmpdir_path / uploaded_video.name
                with open(sus_path, "wb") as f: f.write(uploaded_video.getbuffer())

                baseline_path = None
                if baseline_video is not None:
                    baseline_path = tmpdir_path / baseline_video.name
                    with open(baseline_path, "wb") as f: f.write(baseline_video.getbuffer())

                result = None; baseline_result = None
                
                # --- ALUR ANALISIS 5 TAHAP ---
                all_stages_success = True
                try:
                    with st.status("Memulai analisis forensik 5 tahap...", expanded=True) as status:
                        st.write("Tahap 1: Akuisisi & Ekstraksi Fitur Dasar...")
                        result = fv.run_tahap_1_pra_pemrosesan(sus_path, tmpdir_path, int(fps))
                        if not result: raise ValueError("Gagal pada Tahap 1.")
                        
                        if baseline_path:
                            st.write("Memproses Video Baseline...")
                            baseline_result = fv.run_tahap_1_pra_pemrosesan(baseline_path, tmpdir_path, int(fps))
                            if baseline_result:
                                fv.run_tahap_2_analisis_temporal(baseline_result)

                        st.write("Tahap 2: Menganalisis Metrik Temporal...")
                        fv.run_tahap_2_analisis_temporal(result, baseline_result)

                        st.write("Tahap 3: Sintesis Bukti & Investigasi Mendalam...")
                        fv.run_tahap_3_sintesis_bukti(result, tmpdir_path)
                        
                        st.write("Tahap 4: Visualisasi & Penilaian Integritas (Enhanced)...")
                        fv.run_tahap_4_visualisasi_dan_penilaian(result, tmpdir_path)

                        st.write("Tahap 5: Menyusun Laporan PDF & Validasi...")
                        fv.run_tahap_5_pelaporan_dan_validasi(result, tmpdir_path, baseline_result)
                        
                        status.update(label="✅ Analisis 5 Tahap Forensik Berhasil!", state="complete", expanded=False)
                
                except Exception as e:
                    all_stages_success = False
                    st.error(f"Terjadi kesalahan pada saat analisis: {e}")
                    st.code(traceback.format_exc())

                # --- PROSES SETELAH ANALISIS (TAMPILKAN HASIL) ---
                if all_stages_success and result:
                    with st.spinner("Mengemas hasil akhir untuk ditampilkan..."):
                        # Save ke riwayat DULU
                        history_manager.save_analysis(
                            result, uploaded_video.name, {"fps": fps, "has_baseline": baseline_video is not None}
                        )
                        st.toast("Hasil analisis berhasil disimpan ke riwayat!")
                        
                        # Load bytes untuk tampilan
                        if result.plots:
                            for key in list(result.plots.keys()): result.plots[key + '_bytes'] = load_image_as_bytes(result.plots.get(key))
                        if result.kmeans_artifacts:
                            if result.kmeans_artifacts.get('distribution_plot_path'): result.kmeans_artifacts['distribution_plot_bytes'] = load_image_as_bytes(result.kmeans_artifacts['distribution_plot_path'])
                            for c in result.kmeans_artifacts.get('clusters', []):
                                if c.get('palette_path'): c['palette_bytes'] = load_image_as_bytes(c['palette_path'])
                                if c.get('samples_montage_path'): c['samples_montage_bytes'] = load_image_as_bytes(c['samples_montage_path'])
                        if result.localizations:
                            for loc in result.localizations:
                                for key in ['image', 'ela_path', 'sift_path']: loc[f'{key}_bytes'] = load_image_as_bytes(loc.get(key))
                                for v_key, v_path in loc.get('visualizations', {}).items(): loc[f'{v_key}_bytes'] = load_image_as_bytes(v_path)
                        if result.frames and result.frames[0].img_path_comparison: result.frames[0].comparison_bytes = load_image_as_bytes(result.frames[0].img_path_comparison)
                        if result.pdf_report_path: result.pdf_report_data = load_image_as_bytes(result.pdf_report_path)

                    st.success("Analisis selesai. Hasil ditampilkan di bawah ini.")
                    # ... (Kode untuk menampilkan tab hasil analisis tetap sama seperti yang Anda berikan) ...
                    # Ini adalah bagian yang sangat panjang, jadi saya akan meringkasnya untuk kejelasan.
                    # Di sini, Anda akan meletakkan SEMUA kode `with tabs[...]:` Anda.

                    tab_titles = ["📄 **Tahap 1: Akuisisi & K-Means**", "📊 **Tahap 2: Analisis Temporal**", "🔬 **Tahap 3: Investigasi ELA/SIFT**", "📈 **Tahap 4: Penilaian & Lokalisasi**", "📥 **Tahap 5: Laporan**"]
                    tabs = st.tabs(tab_titles)
    
                    with tabs[0]: # Tahap 1
                        st.header("Hasil Tahap 1: Akuisisi & Ekstraksi Fitur Dasar")
                        st.info("Tujuan: Mengamankan bukti, mengekstrak metadata, menormalisasi frame, dan menerapkan **Metode Utama K-Means** untuk mengklasifikasikan adegan video.", icon="🛡️")

                        st.subheader("1.1. Identifikasi dan Preservasi Bukti")
                        c1, c2 = st.columns(2)
                        c1.metric("Total Frame Dianalisis", result.summary.get('total_frames', 'N/A'))
                        c2.write("**Hash Integritas (SHA-256)**"); c2.code(result.preservation_hash, language="bash")
                        with st.expander("Tampilkan Metadata Video Lengkap"):
                            for category, items in result.metadata.items():
                                st.write(f"**{category}**")
                                df = pd.DataFrame.from_dict(items, orient='index', columns=['Value'])
                                st.table(df)

                        st.subheader("1.2. Ekstraksi dan Normalisasi Frame")
                        st.write("Setiap frame diekstrak dan dinormalisasi untuk konsistensi analisis. Berikut adalah contoh perbandingan antara frame asli dan yang ternormalisasi.")
                        if result.frames and hasattr(result.frames[0], 'comparison_bytes') and result.frames[0].comparison_bytes:
                            st.image(result.frames[0].comparison_bytes, caption="Kiri: Original, Kanan: Normalized (Contrast-Enhanced)")

                        st.subheader("1.3. Hasil Detail Analisis K-Means")
                        st.write(f"Frame-frame dikelompokkan ke dalam **{fv.CONFIG['KMEANS_CLUSTERS']} klaster** berdasarkan kemiripan warna.")
                        if result.kmeans_artifacts.get('distribution_plot_bytes'):
                            st.image(result.kmeans_artifacts['distribution_plot_bytes'], caption="Distribusi jumlah frame untuk setiap klaster warna.")
                        
                        st.write("**Eksplorasi Setiap Klaster:**")
                        if result.kmeans_artifacts.get('clusters'):
                            cluster_tabs = st.tabs([f"Klaster {c['id']}" for c in result.kmeans_artifacts.get('clusters', [])])
                            for i, cluster_tab in enumerate(cluster_tabs):
                                with cluster_tab:
                                    cluster_data = result.kmeans_artifacts['clusters'][i]
                                    st.metric("Jumlah Frame dalam Klaster Ini", f"{cluster_data['count']}")
                                    c1, c2 = st.columns([1,2])
                                    with c1:
                                        st.write("**Palet Warna Dominan**")
                                        if cluster_data.get('palette_bytes'):
                                            st.image(cluster_data['palette_bytes'])
                                    with c2:
                                        st.write("**Contoh Frame dari Klaster Ini (Gambar Asli)**")
                                        if cluster_data.get('samples_montage_bytes'):
                                            st.image(cluster_data['samples_montage_bytes'])
                        else:
                            st.warning("Tidak ada data klaster K-Means yang dapat ditampilkan.")

                    with tabs[1]: # Tahap 2
                        st.header("Hasil Tahap 2: Analisis Anomali Temporal")
                        st.info("Tujuan: Menganalisis hubungan antar frame berurutan menggunakan versi yang sudah dinormalisasi untuk mendapatkan metrik yang stabil. Deteksi anomali temporal adalah fokus utama di sini.", icon="📈")

                        st.subheader("2.1. Visualisasi Klasterisasi Warna K-Means (Sepanjang Waktu)")
                        st.write("Plot ini menunjukkan bagaimana setiap frame dikelompokkan ke dalam klaster warna tertentu dari waktu ke waktu. Klasterisasi K-Means membantu mengidentifikasi adegan yang secara visual serupa, dan **lompatan atau perubahan klaster yang tajam sering mengindikasikan perubahan adegan yang mendadak** (misalnya, cut atau transisi yang tidak wajar).")
                        st.write("Perhatikan garis vertikal putus-putus merah: ini menandai titik-titik di mana sistem mendeteksi perubahan klaster warna yang signifikan dari frame sebelumnya, menunjukkan potensi perubahan adegan.")
                        if result.plots.get('kmeans_temporal_bytes'):
                            st.image(result.plots['kmeans_temporal_bytes'], caption="Plot ini menunjukkan hasil pengelompokan setiap frame ke dalam klaster warna. Lompatan vertikal yang tajam menandakan perubahan adegan mendadak.")
                        else:
                            st.warning("Plot klasterisasi K-Means temporal tidak tersedia.")
                        
                        st.subheader("2.2. Analisis Perubahan Gambar: Skor SSIM (Structural Similarity Index)")
                        st.write("SSIM adalah metrik yang mengukur **kemiripan struktural antara dua gambar**. Nilai SSIM mendekati 1 menunjukkan kemiripan yang sangat tinggi (frame hampir identik), sedangkan nilai mendekati 0 menunjukkan perbedaan yang besar. Dalam konteks forensik, **penurunan drastis pada skor SSIM antar frame yang berurutan seringkali merupakan indikator kuat adanya diskontinuitas atau interupsi** (misalnya, penghapusan frame, atau transisi yang tidak mulus).")
                        st.write("Pada grafik di bawah, Anda akan melihat fluktuasi skor SSIM sepanjang video. Titik-titik 'X' merah menandai lokasi di mana sistem mendeteksi penurunan SSIM yang signifikan, mengindikasikan potensi anomali.")
                        if result.plots.get('ssim_temporal_bytes'):
                            st.image(result.plots['ssim_temporal_bytes'], caption="Plot perubahan skor SSIM antar frame. Penurunan tajam mengindikasikan diskontinuitas.")
                        else:
                            st.warning("Plot SSIM temporal tidak tersedia.")

                        st.subheader("2.3. Analisis Pergerakan Gambar: Magnitudo Aliran Optik")
                        st.write("Aliran Optik (Optical Flow) mengukur **gerakan piksel antar dua frame berurutan**. Magnitudo rata-rata aliran optik merepresentasikan seberapa 'banyak' perubahan atau pergerakan yang terjadi. **Lonjakan besar pada magnitudo aliran optik dapat mengindikasikan perubahan adegan yang tiba-tiba, penyisipan objek bergerak, atau transisi paksa** yang tidak sesuai dengan aliran alami video.")
                        st.write("Grafik ini menunjukkan besaran rata-rata pergerakan yang terdeteksi. Lingkaran hijau menandai titik-titik di mana terjadi lonjakan aliran optik yang tidak wajar (melebihi ambang batas Z-score), menunjukkan potensi diskontinuitas.")
                        if result.plots.get('optical_flow_temporal_bytes'):
                            st.image(result.plots['optical_flow_temporal_bytes'], caption="Plot perubahan rata-rata magnitudo aliran optik. Lonjakan menunjukkan perubahan mendadak atau pergerakan tidak wajar.")
                        else:
                            st.warning("Plot aliran optik temporal tidak tersedia.")

                        st.subheader("2.4. Distribusi Statistik Metrik Temporal")
                        st.write("Histogram ini memberikan gambaran umum tentang sebaran nilai SSIM dan Aliran Optik di seluruh video, membantu memahami karakteristik gerakan dan kemiripan frame secara keseluruhan. Meskipun plot temporal di atas lebih menunjukkan lokasi anomali, histogram ini memberikan konteks statistik.")
                        if result.plots.get('metrics_histograms_bytes'):
                            st.image(result.plots['metrics_histograms_bytes'], caption="Kiri: Distribusi Skor SSIM. Kanan: Distribusi Aliran Optik.")
                        else:
                            st.warning("Histogram metrik temporal tidak tersedia.")
                        
                        if baseline_result:
                            st.subheader("2.5. Analisis Komparatif (dengan Video Baseline)")
                            insertion_events_count = len([loc for loc in result.localizations if loc['event'] == 'anomaly_insertion'])
                            st.info(f"Ditemukan **{insertion_events_count} peristiwa penyisipan** yang tidak ada di video baseline. Ini mengindikasikan penambahan konten baru ke video bukti.", icon="🔎")
                        else:
                            st.info("Tidak ada video baseline yang diunggah untuk analisis komparatif.", icon="ℹ️")

                    with tabs[2]: # Tahap 3
                        st.header("Hasil Tahap 3: Investigasi Detail dengan Metode Pendukung")
                        st.info("Inti analisis: mengkorelasikan temuan dan melakukan investigasi mendalam dengan **metode pendukung (ELA dan SIFT+RANSAC)** pada frame asli dari kandidat anomali terkuat.", icon="🔬")
                        
                        # Tambahkan ringkasan statistik
                        if result.statistical_summary:
                            st.subheader("📊 Ringkasan Statistik Investigasi")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Anomali", result.statistical_summary['total_anomalies'])
                                st.metric("Kluster Temporal", result.statistical_summary['temporal_clusters'])
                            with col2:
                                st.metric("Persentase Anomali", f"{result.statistical_summary.get('total_anomalies', 0)/result.statistical_summary.get('total_frames_analyzed', 1)*100:.1f}%")
                                st.metric("Rata-rata per Kluster", f"{result.statistical_summary.get('average_anomalies_per_cluster', 0):.1f}")
                            with col3:
                                # Pie chart mini untuk distribusi anomali
                                if result.statistical_summary.get('anomaly_types'):
                                    st.write("**Distribusi Jenis Anomali:**")
                                    for atype, count in result.statistical_summary['anomaly_types'].items():
                                        st.write(f"- {atype.replace('anomaly_', '').title()}: {count}")
                        
                        # Tampilkan visualisasi ringkasan jika ada
                        if result.plots.get('anomaly_summary_bytes'):
                            with st.expander("📈 Lihat Visualisasi Ringkasan Anomali", expanded=True):
                                st.image(result.plots['anomaly_summary_bytes'], caption="Ringkasan visual dari semua anomali yang terdeteksi")
                        
                        st.markdown("---")
                        
                        if not result.localizations: 
                            st.success("🎉 **Tidak Ditemukan Anomali Signifikan.**")
                        else:
                            st.warning(f"🚨 Ditemukan **{len(result.localizations)} peristiwa anomali** yang signifikan:", icon="🚨")
                            
                            # Tambahkan penjelasan umum tentang metode
                            with st.expander("📚 Panduan Membaca Hasil Investigasi", expanded=False):
                                st.markdown("""
                                ### Metode Deteksi yang Digunakan:
                                
                                **1. Error Level Analysis (ELA)**
                                - **Apa itu?** ELA mendeteksi area yang telah diedit dengan membandingkan tingkat kompresi.
                                - **Cara membaca:** Area terang = kemungkinan telah dimodifikasi. Area gelap = kemungkinan asli.
                                
                                **2. SIFT (Scale-Invariant Feature Transform)**
                                - **Apa itu?** SIFT mendeteksi dan mencocokkan titik-titik unik antar frame.
                                - **Cara membaca:** Banyak garis hijau = frame sangat mirip/duplikat. Sedikit garis = frame berbeda.
                                
                                **3. Tingkat Kepercayaan**
                                - 🟩 **RENDAH**: Anomali minor, mungkin false positive
                                - 🟨 **SEDANG**: Anomali cukup jelas, perlu investigasi lanjut
                                - 🟧 **TINGGI**: Anomali signifikan, kemungkinan besar manipulasi
                                - 🟥 **SANGAT TINGGI**: Bukti kuat manipulasi
                                """)
                            
                            # Tampilkan setiap peristiwa anomali dengan detail
                            for i, loc in enumerate(result.localizations):
                                event_type = loc['event'].replace('anomaly_', '').capitalize()
                                confidence = loc.get('confidence', 'N/A')
                                
                                # Tentukan emoji berdasarkan confidence
                                conf_emoji = "🟩" if confidence == "RENDAH" else "🟨" if confidence == "SEDANG" else "🟧" if confidence == "TINGGI" else "🟥"
                                
                                with st.expander(f"{conf_emoji} **Peristiwa #{i+1}: {event_type}** @ {loc['start_ts']:.2f} - {loc['end_ts']:.2f} detik (Keyakinan: {confidence})", expanded=(i == 0)):
                                    
                                    # Info dasar peristiwa
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Tipe Anomali", event_type)
                                    with col2:
                                        st.metric("Durasi", f"{loc['end_ts'] - loc['start_ts']:.2f} detik")
                                    with col3:
                                        st.metric("Frame Terdampak", f"{loc['end_frame'] - loc['start_frame'] + 1} frame")
                                    
                                    st.markdown("---")
                                    
                                    # Penjelasan berdasarkan pengaturan
                                    if show_simple_explanations and loc.get('explanations'):
                                        st.markdown("### 💡 Penjelasan untuk Orang Awam")
                                        for exp_type, exp_data in loc['explanations'].items():
                                            if isinstance(exp_data, dict) and exp_data.get('simple_explanation'):
                                                st.markdown(f'<div class="explanation-box">{exp_data["simple_explanation"]}</div>', unsafe_allow_html=True)
                                    
                                    if show_technical_details:
                                        st.markdown("### 🔧 Detail Teknis")
                                        
                                        # Tampilkan metrik dalam format yang lebih readable
                                        if isinstance(loc.get('metrics'), dict) and loc['metrics']:
                                            metrics_df = pd.DataFrame([
                                                {
                                                    'Metrik': k.replace('_', ' ').title(),
                                                    'Nilai': str(v),
                                                    'Keterangan': _get_metric_description(k, v)
                                                }
                                                for k, v in loc['metrics'].items()
                                            ])
                                            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                        
                                        # Penjelasan teknis dari explanations
                                        if loc.get('explanations'):
                                            for exp_type, exp_data in loc['explanations'].items():
                                                if isinstance(exp_data, dict) and exp_data.get('technical_explanation'):
                                                    st.markdown(f'<div class="technical-box">{exp_data["technical_explanation"]}</div>', unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                                    st.markdown("### 🖼️ Bukti Visual")
                                    
                                    # Layout untuk bukti visual
                                    visual_cols = st.columns(2)
                                    
                                    with visual_cols[0]:
                                        if loc.get('image_bytes'):
                                            st.write("**Frame Anomali (Asli)**")
                                            st.image(loc['image_bytes'], caption=f"Frame #{loc.get('start_frame')}")
                                        
                                        if loc.get('ela_path_bytes'):
                                            st.write("**Analisis ELA Standar**")
                                            st.image(loc['ela_path_bytes'], caption="Error Level Analysis")
                                            st.caption("Area terang = kemungkinan telah diedit")
                                    
                                    with visual_cols[1]:
                                        # Visualisasi tambahan dari Tahap 3
                                        if loc.get('ela_detailed_bytes'):
                                            st.write("**Analisis ELA Detail dengan Grid**")
                                            st.image(loc['ela_detailed_bytes'], caption="ELA dengan area mencurigakan ditandai")
                                            st.caption("Kotak merah/kuning = area dengan anomali kompresi tinggi")
                                        
                                        if loc.get('sift_heatmap_bytes'):
                                            st.write("**Heatmap SIFT**")
                                            st.image(loc['sift_heatmap_bytes'], caption="Distribusi titik kecocokan")
                                            st.caption("Area panas (merah) = konsentrasi tinggi kecocokan")
                                    
                                    # SIFT visualization full width jika ada
                                    if loc.get('sift_path_bytes'):
                                        st.write("**Bukti Pencocokan Fitur SIFT+RANSAC**")
                                        st.image(loc['sift_path_bytes'], caption="Garis hijau = titik fitur yang cocok antar frame", use_container_width=True)
                                        
                                        # Tambahkan interpretasi SIFT jika ada
                                        if loc.get('explanations', {}).get('duplication', {}).get('sift_analysis'):
                                            sift_data = loc['explanations']['duplication']['sift_analysis']
                                            if isinstance(sift_data, dict):
                                                sift_col1, sift_col2, sift_col3 = st.columns(3)
                                                with sift_col1:
                                                    st.metric("Total Kecocokan", sift_data.get('good_matches', 0))
                                                with sift_col2:
                                                    st.metric("Inliers", sift_data.get('inliers', 0))
                                                with sift_col3:
                                                    st.metric("Akurasi", f"{sift_data.get('inlier_ratio', 0)*100:.1f}%")
                                    
                                    # Implikasi forensik
                                    if loc.get('explanations'):
                                        implications = []
                                        for exp in loc['explanations'].values():
                                            if isinstance(exp, dict) and exp.get('implications'):
                                                implications.append(exp['implications'])
                                        
                                        if implications:
                                            st.markdown("### ⚖️ Implikasi Forensik")
                                            for imp in implications:
                                                st.info(imp, icon="📌")
                                    
                                    # Alasan deteksi
                                    st.markdown("### 📋 Alasan Deteksi")
                                    reasons = loc.get('reasons', 'N/A')
                                    if isinstance(reasons, str):
                                        reason_list = [r.strip() for r in reasons.split(',')]
                                        for reason in reason_list:
                                            st.write(f"• {reason}")

                    with tabs[3]: # Tahap 4 (ENHANCED)
                        st.header("Hasil Tahap 4: Penilaian & Localization Tampering (Enhanced)")
                        st.info("Tahap ini menyajikan penilaian integritas yang **realistis** dan **transparan**, dengan penjelasan detail bagaimana setiap komponen dihitung. **Metode Utama Localization Tampering** mengelompokkan anomali menjadi peristiwa yang mudah dipahami.", icon="📈")
                        
                        # Section 1: Integrity Score dengan Gauge Chart
                        st.subheader("🎯 Skor Integritas Video")
                        
                        if hasattr(result, 'integrity_analysis') and result.integrity_analysis:
                            integrity_score = result.integrity_analysis['score']
                            integrity_desc = result.integrity_analysis['description']
                            
                            # Create columns for score display
                            score_col1, score_col2 = st.columns([2, 3])
                            
                            with score_col1:
                                # Display gauge chart
                                gauge_fig = create_gauge_chart(integrity_score, "Skor Integritas")
                                st.plotly_chart(gauge_fig, use_container_width=True)
                                
                                # Score interpretation
                                st.markdown(f"""
                                <div class="integrity-score-box">
                                    <h2 style="margin: 0;">{integrity_score}%</h2>
                                    <h4 style="margin: 5px 0;">{integrity_desc}</h4>
                                    <p style="margin: 10px 0; font-size: 14px;">Target Minimal: 80%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with score_col2:
                                st.markdown("#### 📊 Bagaimana Skor Ini Dihitung?")
                                
                                # Breakdown calculation
                                calc_details = result.integrity_analysis['calculation_details']
                                
                                # Base score explanation
                                st.markdown(f"""
                                **1. Skor Dasar ({int(calc_details.get('base_score',0))}%)**
                                - Berdasarkan persentase frame anomali: {calc_details.get('percentage_anomaly',0):.1f}%
                                - Semakin sedikit anomali, semakin tinggi skor dasar
                                - Maksimal 95% (tidak pernah 100% untuk realisme)
                                """)
                                
                                # Adjustments
                                st.markdown("**2. Faktor Penyesuaian:**")
                                for name, value in calc_details['adjustments']:
                                    if value > 0:
                                        st.success(f"✅ {name}: +{value}%")
                                    else:
                                        st.error(f"❌ {name}: {value}%")
                                
                                # Final score
                                st.markdown(f"**3. Skor Akhir: {int(calc_details.get('final_score',0))}%**")
                                
                                # Method explanation
                                with st.expander("🔍 Lihat Metode Penilaian Lengkap"):
                                    st.markdown(f"""
                                    **Metode:** {calc_details.get('scoring_method', 'N/A')}
                                    
                                    **Faktor yang Dipertimbangkan:**
                                    """)
                                    for factor in calc_details.get('factors_considered', []):
                                        st.write(f"• {factor}")
                                    
                                    st.markdown("""
                                    **Interpretasi Skor:**
                                    - 90-95%: Sangat Baik - Video hampir pasti asli
                                    - 85-89%: Baik - Minor issues, kemungkinan besar asli  
                                    - 80-84%: Cukup Baik - Beberapa anomali, perlu investigasi
                                    - 70-79%: Sedang - Indikasi manipulasi
                                    - <70%: Buruk - Manipulasi signifikan terdeteksi
                                    """)
                        else:
                            # Fallback jika tidak ada integrity analysis
                            integrity_score, integrity_desc = fv.generate_integrity_score(result.summary)[:2]
                            st.metric(label="Skor Integritas Video", value=f"{integrity_score}/100", delta=integrity_desc, 
                                     delta_color="inverse" if integrity_score < 70 else "normal")
                        
                        st.markdown("---")
                        
                        # Section 2: Localization Details
                        st.subheader("🔍 Detail Localization Tampering")
                        
                        if hasattr(result, 'localization_details') and result.localization_details:
                            loc_details = result.localization_details
                            
                            # Overview metrics
                            metrics_cols = st.columns(5)
                            with metrics_cols[0]:
                                st.metric("Total Peristiwa", loc_details['total_events'])
                            with metrics_cols[1]:
                                st.metric("Frame Anomali", loc_details['total_anomalous_frames'])
                            with metrics_cols[2]:
                                st.metric("Durasi Rata-rata", f"{loc_details['average_event_duration']:.1f}s")
                            with metrics_cols[3]:
                                st.metric("Durasi Maksimal", f"{loc_details['max_event_duration']:.1f}s")
                            with metrics_cols[4]:
                                st.metric("Severity Tinggi", loc_details['high_severity_events'])
                            
                            # Event distribution
                            if loc_details['events_by_type']:
                                st.markdown("#### Distribusi Jenis Peristiwa")
                                
                                # Create pie chart for event types
                                event_types = list(loc_details['events_by_type'].keys())
                                event_counts = list(loc_details['events_by_type'].values())
                                event_labels = [t.replace('anomaly_', '').title() for t in event_types]
                                
                                fig = go.Figure(data=[go.Pie(
                                    labels=event_labels,
                                    values=event_counts,
                                    hole=0.3,
                                    marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
                                )])
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Section 3: Enhanced Visualizations
                        st.subheader("📊 Visualisasi Detail")
                        
                        viz_tabs = st.tabs(["Peta Lokalisasi", "Analisis Integritas", "Infografis Anomali"])
                        
                        with viz_tabs[0]:
                            if result.plots.get('enhanced_localization_map_bytes'):
                                st.image(result.plots['enhanced_localization_map_bytes'], 
                                        caption="Peta detail lokalisasi tampering dengan timeline visual", 
                                        use_container_width=True)
                                
                                st.markdown("""
                                **Cara Membaca Peta:**
                                - 🟩 Area hijau = Frame normal/asli
                                - 🟥 Area merah = Anomali dengan kepercayaan tinggi
                                - ⭐ Bintang = Tingkat kepercayaan anomali
                                - Grafik bawah = Kepadatan anomali sepanjang video
                                """)
                            else:
                                st.info("Peta lokalisasi tidak tersedia")
                        
                        with viz_tabs[1]:
                            if result.plots.get('integrity_breakdown_bytes'):
                                st.image(result.plots['integrity_breakdown_bytes'], 
                                        caption="Breakdown detail perhitungan skor integritas", 
                                        use_container_width=True)
                                
                                st.markdown("""
                                **Penjelasan Waterfall Chart:**
                                - Dimulai dari skor dasar berdasarkan persentase anomali
                                - Setiap batang menunjukkan penyesuaian positif (hijau) atau negatif (merah)
                                - Batang emas = Skor final
                                - Garis putus-putus = Target minimal (80%)
                                """)
                            else:
                                st.info("Breakdown integritas tidak tersedia")
                        
                        with viz_tabs[2]:
                            if result.plots.get('anomaly_infographic_bytes'):
                                st.image(result.plots['anomaly_infographic_bytes'], 
                                        caption="Infografis penjelasan jenis-jenis anomali", 
                                        use_container_width=True)
                                
                                st.markdown("""
                                **Infografis ini menjelaskan:**
                                - Setiap jenis anomali dengan analogi sederhana
                                - Cara sistem mendeteksi setiap anomali
                                - Implikasi forensik dari setiap temuan
                                - Jumlah anomali yang ditemukan per jenis
                                """)
                            else:
                                st.info("Infografis anomali tidak tersedia")
                        
                        st.markdown("---")
                        
                        # Section 4: Pipeline Assessment
                        st.subheader("⚙️ Penilaian Kualitas Pipeline Forensik")
                        
                        if hasattr(result, 'pipeline_assessment') and result.pipeline_assessment:
                            st.markdown("Setiap tahap dalam pipeline forensik dinilai untuk memastikan kualitas analisis:")
                            
                            for stage_id, assessment in result.pipeline_assessment.items():
                                with st.container():
                                    st.markdown(f"""
                                    <div class="pipeline-stage-card">
                                        <h4>{assessment['nama']}</h4>
                                        <div style="display: flex; justify-content: space-between;">
                                            <span>Status: <b>{assessment['status'].upper()}</b></span>
                                            <span>Quality Score: <b>{assessment['quality_score']}%</b></span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if assessment['issues']:
                                        for issue in assessment['issues']:
                                            st.warning(f"⚠️ {issue}")
                                    else:
                                        st.success("✅ Tidak ada masalah terdeteksi")
                                    
                                    # Show metrics if advanced metrics enabled
                                    if show_advanced_metrics and assessment['metrics']:
                                        with st.expander(f"Lihat Metrik {assessment['nama']}"):
                                            for metric, value in assessment['metrics'].items():
                                                st.write(f"• {metric}: {value}")
                            
                            # Overall pipeline quality
                            avg_quality = np.mean([a['quality_score'] for a in result.pipeline_assessment.values()])
                            st.metric("Kualitas Pipeline Keseluruhan", f"{avg_quality:.1f}%", 
                                     delta="Baik" if avg_quality >= 80 else "Perlu Perbaikan")
                        
                        st.markdown("---")
                        
                        # Section 5: Detailed Event Analysis
                        st.subheader("📋 Analisis Detail Setiap Peristiwa")
                        
                        if result.localizations:
                            # Sort by severity
                            sorted_events = sorted(result.localizations, 
                                                 key=lambda x: x.get('severity_score', 0), 
                                                 reverse=True)
                            
                            for idx, event in enumerate(sorted_events):
                                event_type = event['event'].replace('anomaly_', '').title()
                                severity = event.get('severity_score', 0)
                                duration = event.get('duration', 0)
                                
                                # Determine severity level
                                if severity > 0.7:
                                    severity_color = "🔴"
                                    severity_text = "TINGGI"
                                elif severity > 0.5:
                                    severity_color = "🟡"
                                    severity_text = "SEDANG"
                                else:
                                    severity_color = "🟢"
                                    severity_text = "RENDAH"
                                
                                with st.expander(f"{severity_color} Peristiwa #{idx+1}: {event_type} | Severity: {severity_text} ({severity:.2f})"):
                                    # Event details
                                    event_cols = st.columns(4)
                                    with event_cols[0]:
                                        st.metric("Waktu", f"{event['start_ts']:.1f}-{event['end_ts']:.1f}s")
                                    with event_cols[1]:
                                        st.metric("Durasi", f"{duration:.1f}s")
                                    with event_cols[2]:
                                        st.metric("Frame", f"{event.get('frame_count', 1)} frame")
                                    with event_cols[3]:
                                        st.metric("Kepercayaan", event['confidence'])
                                    
                                    # Aggregated metrics
                                    if event.get('aggregated_metrics'):
                                        st.markdown("#### Statistik Metrik")
                                        metrics_data = []
                                        for metric, stats in event['aggregated_metrics'].items():
                                            if isinstance(stats, dict): # Check if stats is a dictionary
                                                metrics_data.append({
                                                    'Metrik': metric.replace('_', ' ').title(),
                                                    'Mean': f"{stats.get('mean', 'N/A'):.3f}" if isinstance(stats.get('mean'), float) else 'N/A',
                                                    'Max': f"{stats.get('max', 'N/A'):.3f}" if isinstance(stats.get('max'), float) else 'N/A',
                                                    'Min': f"{stats.get('min', 'N/A'):.3f}" if isinstance(stats.get('min'), float) else 'N/A',
                                                    'Std': f"{stats.get('std', 'N/A'):.3f}" if isinstance(stats.get('std'), float) else 'N/A'
                                                })
                                        if metrics_data:
                                            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
                                    
                                    # Visual evidence
                                    if event.get('image_bytes'):
                                        st.image(event['image_bytes'], caption="Frame Sampel", width=300)
                        else:
                            st.success("Tidak ada peristiwa anomali yang terlokalisasi")
                        
                        st.markdown("---")
                        
                        # Section 6: Summary Statistics
                        st.subheader("📊 Statistik Ringkasan")
                        
                        summary_cols = st.columns(2)
                        with summary_cols[0]:
                            st.markdown("#### Statistik Frame")
                            frame_stats = {
                                "Total Frame Dianalisis": result.summary.get('total_frames', 0),
                                "Total Frame Anomali": result.summary.get('total_anomaly', 0),
                                "Persentase Anomali": f"{result.summary.get('pct_anomaly', 0):.1f}%",
                                "Densitas Anomali": f"{result.summary.get('anomaly_density', 0):.3f}"
                            }
                            for key, val in frame_stats.items():
                                st.write(f"• **{key}:** {val}")
                        
                        with summary_cols[1]:
                            st.markdown("#### Statistik Peristiwa")
                            if hasattr(result, 'localization_details'):
                                event_stats = {
                                    "Total Peristiwa": result.localization_details['total_events'],
                                    "Durasi Total Anomali": f"{sum(e.get('duration', 0) for e in result.localizations):.1f}s",
                                    "Rata-rata per Peristiwa": f"{result.localization_details['average_event_duration']:.1f}s",
                                    "Peristiwa Terpanjang": f"{result.localization_details['max_event_duration']:.1f}s"
                                }
                                for key, val in event_stats.items():
                                    st.write(f"• **{key}:** {val}")
                        
                        # Temporal plot if available
                        if result.plots.get('temporal_bytes'):
                            st.markdown("#### Timeline Anomali")
                            st.image(result.plots['temporal_bytes'], 
                                    caption="Visualisasi lokasi dan jenis peristiwa anomali sepanjang video", 
                                    use_container_width=True)

                    with tabs[4]: # Tahap 5
                        st.header("Hasil Tahap 5: Penyusunan Laporan & Validasi Forensik")
                        st.write("Tahap akhir menghasilkan laporan PDF yang dapat diaudit dan berfungsi sebagai validasi proses. Laporan ini mencakup semua temuan dari metode utama dan pendukung dengan detail dari Tahap 4 yang telah ditingkatkan.")
                        st.info("Unduh laporan PDF lengkap untuk dokumentasi atau sebagai lampiran bukti digital.", icon="📄")
                        
                        if hasattr(result, 'pdf_report_data') and result.pdf_report_data:
                            st.download_button(
                                label="📥 Unduh Laporan PDF Lengkap", 
                                data=result.pdf_report_data, 
                                file_name=result.pdf_report_path.name, 
                                mime="application/pdf", 
                                use_container_width=True)
                            
                            # Preview laporan info
                            st.markdown("### 📋 Isi Laporan")
                            st.markdown("""
                            Laporan PDF berisi:
                            1. **Ringkasan Eksekutif** - Skor integritas dan temuan utama
                            2. **Detail per Tahap** - Hasil lengkap dari setiap tahap forensik
                            3. **Visualisasi** - Semua grafik dan plot yang dihasilkan
                            4. **Analisis Detail** - Penjelasan mendalam setiap anomali
                            5. **Skor Integritas Enhanced** - Breakdown perhitungan yang transparan
                            6. **Penilaian Pipeline** - Evaluasi kualitas setiap tahap
                            7. **Validasi Forensik** - Hash preservasi dan metadata teknis
                            """)
                        else: 
                            st.error("File laporan PDF tidak dapat dibuat. Periksa log konsol untuk detail kesalahan.")
                        
                        st.subheader("✅ Validasi Proses Analisis")
                        
                        # Enhanced validation table
                        validation_data = {
                            "File Bukti": Path(result.video_path).name,
                            "Hash SHA-256 (Preservasi)": result.preservation_hash,
                            "Waktu Analisis (UTC)": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "Metodologi Utama": "K-Means, Localization Tampering",
                            "Metode Pendukung": "ELA, SIFT+RANSAC, SSIM, Optical Flow",
                            "Pustaka Kunci": "OpenCV, scikit-learn, scikit-image, Pillow, ReportLab",
                            "Skor Integritas Final": f"{result.integrity_analysis.get('score', 'N/A')}%" if hasattr(result, 'integrity_analysis') else "N/A",
                            "Total Anomali Terdeteksi": str(result.summary.get('total_anomaly', 0)),
                            "Kualitas Pipeline": f"{np.mean([a['quality_score'] for a in result.pipeline_assessment.values()]):.1f}%" if hasattr(result, 'pipeline_assessment') and result.pipeline_assessment else "N/A"
                        }
                        
                        validation_df = pd.DataFrame.from_dict(validation_data, orient='index', columns=['Detail'])
                        st.table(validation_df)
                        
                        # Final conclusion
                        st.markdown("### 🎯 Kesimpulan Analisis")
                        
                        integrity_score = result.integrity_analysis.get('score', 0) if hasattr(result, 'integrity_analysis') else 0
                        
                        if integrity_score >= 85:
                            conclusion_type = "success"
                            conclusion_icon = "✅"
                            conclusion_text = "Video memiliki integritas TINGGI dan kemungkinan besar ASLI."
                        elif integrity_score >= 80:
                            conclusion_type = "info" 
                            conclusion_icon = "ℹ️"
                            conclusion_text = "Video memiliki integritas CUKUP BAIK dengan beberapa anomali minor."
                        elif integrity_score >= 70:
                            conclusion_type = "warning"
                            conclusion_icon = "⚠️"
                            conclusion_text = "Video menunjukkan tanda-tanda MANIPULASI yang perlu investigasi lanjut."
                        else:
                            conclusion_type = "error"
                            conclusion_icon = "❌"
                            conclusion_text = "Video terindikasi MANIPULASI SIGNIFIKAN dan tidak dapat dipercaya."
                        
                        getattr(st, conclusion_type)(f"{conclusion_icon} {conclusion_text}")
                        
                        # Calculate pipeline quality safely
                        if hasattr(result, 'pipeline_assessment') and result.pipeline_assessment and result.pipeline_assessment.values():
                            pipeline_quality = np.mean([a['quality_score'] for a in result.pipeline_assessment.values()])
                            pipeline_text = f"{pipeline_quality:.1f}%"
                        else:
                            pipeline_text = "baik"
                        
                        st.markdown(f"""
                        Berdasarkan analisis forensik 5 tahap:
                        - Skor Integritas: **{integrity_score}%**
                        - Total Peristiwa Anomali: **{len(result.localizations)}**
                        - Metode utama (K-Means & Localization) berhasil mengidentifikasi pola anomali
                        - Metode pendukung (ELA & SIFT) memberikan validasi tambahan
                        - Pipeline forensik berjalan dengan kualitas **{pipeline_text}**
                        """)


else:  # Halaman Riwayat Analisis
    render_history_page()

# --- END OF FILE streamlit_app.py (MODIFIED) ---