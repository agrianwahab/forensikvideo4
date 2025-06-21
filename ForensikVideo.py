# --- START OF FILE ForensikVideo.py ---

# --- START OF FILE ForensikVideo.py (ENHANCED TAHAP 4) ---

# START OF FILE ForensikVideo.py

# vifa_pro.py
# (Sistem Forensik Video Profesional dengan Analisis Multi-Lapis)
# VERSI 5 TAHAP PENELITIAN (DENGAN PERBAIKAN BUG STYLE REPORTLAB)
# VERSI PENINGKATAN METODE UTAMA (K-MEANS, LOCALIZATION) & PENDUKUNG (ELA, SIFT)
# VERSI REVISI DETAIL TAHAP 1 (METADATA, NORMALISASI FRAME, DETAIL K-MEANS)
# VERSI PENINGKATAN DETAIL TAHAP 2 (PLOT TEMPORAL K-MEANS, SSIM, OPTICAL FLOW)
# VERSI PENINGKATAN DETAIL TAHAP 3 (INVESTIGASI MENDALAM DAN PENJELASAN LENGKAP)
# VERSI PENINGKATAN DETAIL TAHAP 4 (LOCALIZATION TAMPERING ENHANCED, SKOR INTEGRITAS REALISTIS)

"""
VIFA-Pro: Sistem Forensik Video Profesional (Arsitektur 5 Tahap)
========================================================================================
Versi ini mengimplementasikan alur kerja forensik formal dalam 5 tahap yang jelas,
sesuai dengan metodologi penelitian untuk deteksi manipulasi video. Setiap tahap
memiliki tujuan spesifik, dari ekstraksi fitur dasar hingga validasi proses.

ARSITEKTUR PIPELINE:
- TAHAP 1: Pra-pemrosesan & Ekstraksi Fitur Dasar (Hashing, Frame, pHash, Warna)
           -> Metadata diekstrak secara mendalam.
           -> Frame diekstrak dan dinormalisasi warnanya untuk konsistensi analisis.
           -> Metode K-Means diterapkan untuk klasterisasi warna adegan dengan visualisasi detail.
- TAHAP 2: Analisis Anomali Temporal & Komparatif (Optical Flow, SSIM, K-Means Temporal, Baseline Check)
           -> Visualisasi Temporal yang lebih rinci untuk SSIM, Optical Flow, dan K-Means.
- TAHAP 3: Sintesis Bukti & Investigasi Mendalam (Korelasi Metrik, ELA & SIFT on-demand)
           -> ELA dan SIFT+RANSAC digunakan sebagai investigasi pendukung yang terukur.
           -> Analisis detail dengan penjelasan lengkap untuk setiap anomali.
- TAHAP 4: Visualisasi & Penilaian Integritas (Plotting, Integrity Score)
           -> Localization Tampering menyatukan anomali menjadi peristiwa yang dapat diinterpretasikan.
           -> ENHANCED: Skor integritas realistis, visualisasi detail, penilaian pipeline
- TAHAP 5: Penyusunan Laporan & Validasi Forensik (Laporan PDF Naratif)

Deteksi:
- Diskontinuitas (Deletion/Insertion): Melalui Aliran Optik, SSIM, K-Means, dan Perbandingan Baseline.
- Duplikasi Frame (Duplication): Melalui pHash, dikonfirmasi oleh SIFT+RANSAC dan SSIM.
- Penyisipan Area (Splicing): Terindikasi oleh Analisis Tingkat Kesalahan (ELA) pada titik diskontinuitas.

Author: OpenAI-GPT & Anda
License: MIT
Dependencies: opencv-python, opencv-contrib-python, imagehash, numpy, Pillow,
              reportlab, matplotlib, tqdm, scikit-learn, scikit-image
"""

from __future__ import annotations
import argparse
import json
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any

# Pemeriksaan Dependensi Awal
try:
    import cv2
    import imagehash
    import numpy as np
    from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFont
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    from skimage.metrics import structural_similarity as ssim
    from scipy import stats
    import seaborn as sns
except ImportError as e:
    print(f"Error: Dependensi penting tidak ditemukan -> {e}")
    sys.exit(1)


###############################################################################
# Utilitas & Konfigurasi Global
###############################################################################

class Icons: IDENTIFICATION="🔍"; PRESERVATION="🛡️"; COLLECTION="📥"; EXAMINATION="🔬"; ANALYSIS="📈"; REPORTING="📄"; SUCCESS="✅"; ERROR="❌"; INFO="ℹ️"; CONFIDENCE_LOW="🟩"; CONFIDENCE_MED="🟨"; CONFIDENCE_HIGH="🟧"; CONFIDENCE_VHIGH="🟥"
CONFIG = {"HASH_DIST_DUPLICATE": 2, "OPTICAL_FLOW_Z_THRESH": 4.0, "SSIM_DISCONTINUITY_DROP": 0.25, "SIFT_MIN_MATCH_COUNT": 10, "KMEANS_CLUSTERS": 8, "DUPLICATION_SSIM_CONFIRM": 0.95, "KMEANS_SAMPLES_PER_CLUSTER": 5}
# ====== [NEW] False-Positive Fix June-2025 ======
CONFIG.update({
    "USE_AUTO_THRESHOLDS": True,
    "SSIM_USER_THRESHOLD": 0.25,
    "Z_USER_THRESHOLD": 4.0,
})
# ====== [END NEW] ======

# Fungsi log yang dienkapsulasi untuk output ke konsol dan UI Streamlit
def log(message: str):
    print(message, file=sys.stdout) # Menggunakan stdout asli untuk logging

def print_stage_banner(stage_number: int, stage_name: str, icon: str, description: str):
    width=80
    log("\n" + "="*width)
    log(f"=== {icon}  TAHAP {stage_number}: {stage_name.upper()} ".ljust(width - 3) + "===")
    log("="*width)
    log(f"{Icons.INFO}  {description}")
    log("-" * width)

###############################################################################
# Struktur Data Inti (DIPERLUAS UNTUK TAHAP 4 ENHANCED)
###############################################################################

@dataclass
class Evidence:
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    confidence: str = "N/A"
    ela_path: str | None = None
    sift_path: str | None = None
    # Tambahan untuk detail Tahap 3
    detailed_analysis: dict = field(default_factory=dict)
    visualizations: dict = field(default_factory=dict)
    explanations: dict = field(default_factory=dict)

@dataclass
class FrameInfo:
    index: int
    timestamp: float
    img_path_original: str  # Path ke frame asli
    img_path: str           # Path ke frame yang dinormalisasi (digunakan untuk analisis utama)
    img_path_comparison: str | None = None # Path ke gambar perbandingan (opsional)
    hash: str | None = None
    type: str = "original"
    ssim_to_prev: float | None = None
    optical_flow_mag: float | None = None
    color_cluster: int | None = None
    evidence_obj: Evidence = field(default_factory=Evidence)
    # Tambahan untuk analisis detail
    histogram_data: np.ndarray | None = None
    edge_density: float | None = None
    blur_metric: float | None = None

@dataclass
class AnalysisResult:
    video_path: str
    preservation_hash: str
    metadata: dict
    frames: list[FrameInfo]
    summary: dict = field(default_factory=dict)
    plots: dict = field(default_factory=dict)
    # --- PENAMBAHAN: Artefak K-Means yang Detail ---
    kmeans_artifacts: dict = field(default_factory=dict)
    localizations: list[dict] = field(default_factory=list)
    pdf_report_path: Path | None = None
    pdf_report_data: bytes | None = None
    # ====== [NEW] Metadata Forensics Enhancement ======
    html_report_path: Path | None = None
    json_report_path: Path | None = None
    html_report_data: bytes | None = None
    json_report_data: bytes | None = None
    # ====== [END NEW] ======
    # Tambahan untuk Tahap 3
    detailed_anomaly_analysis: dict = field(default_factory=dict)
    statistical_summary: dict = field(default_factory=dict)
    # TAMBAHAN UNTUK TAHAP 4 ENHANCED
    integrity_analysis: dict = field(default_factory=dict)
    pipeline_assessment: dict = field(default_factory=dict)
    localization_details: dict = field(default_factory=dict)
    confidence_distribution: dict = field(default_factory=dict)

# Fungsi enhanced untuk generate_integrity_score yang lebih realistis
def generate_integrity_score(summary: dict, detailed_analysis: dict = None) -> tuple[int, str, dict]:
    """
    Menghasilkan skor integritas video yang lebih realistis (80-95%).
    Mengembalikan: (skor, deskripsi, detail_perhitungan)
    """
    pct = summary.get('pct_anomaly', 0)
    total_frames = summary.get('total_frames', 0)
    
    # Base score calculation yang lebih realistis
    if pct == 0:
        base_score = 95  # Maksimal 95%, tidak pernah 100%
    elif pct < 5:
        base_score = 90
    elif pct < 10:
        base_score = 85
    elif pct < 15:
        base_score = 80
    elif pct < 25:
        base_score = 70
    elif pct < 35:
        base_score = 60
    else:
        base_score = 50
    
    # Adjustment factors untuk lebih realistis
    adjustments = []
    
    # 1. Severity adjustment - berdasarkan tingkat kepercayaan anomali
    if detailed_analysis:
        confidence_dist = detailed_analysis.get('confidence_distribution', {})
        very_high_count = confidence_dist.get('SANGAT TINGGI', 0)
        high_count = confidence_dist.get('TINGGI', 0)
        
        if very_high_count > 0:
            severity_penalty = min(very_high_count * 3, 15)
            adjustments.append(('Anomali Kepercayaan Sangat Tinggi', -severity_penalty))
        if high_count > 0:
            severity_penalty = min(high_count * 2, 10)
            adjustments.append(('Anomali Kepercayaan Tinggi', -severity_penalty))
    
    # 2. Continuity bonus - jika anomali terisolasi
    if detailed_analysis and detailed_analysis.get('temporal_clusters', 0) > 0:
        if detailed_analysis.get('average_anomalies_per_cluster', 0) < 3:
            adjustments.append(('Anomali Terisolasi (Non-Sistemik)', +5))
    
    # 3. Technical quality factor
    if total_frames > 100:  # Video cukup panjang untuk analisis valid
        adjustments.append(('Sampel Frame Memadai', +3))
    
    # Calculate final score
    final_score = base_score
    for name, value in adjustments:
        final_score += value
    
    # Ensure score is within 50-95 range for realism
    final_score = max(50, min(95, final_score))
    
    # Generate description
    if final_score >= 90:
        desc = "Sangat Baik - Integritas Tinggi"
    elif final_score >= 85:
        desc = "Baik - Integritas Terjaga"
    elif final_score >= 80:
        desc = "Cukup Baik - Minor Issues"
    elif final_score >= 70:
        desc = "Sedang - Perlu Investigasi"
    elif final_score >= 60:
        desc = "Buruk - Manipulasi Terindikasi"
    else:
        desc = "Sangat Buruk - Manipulasi Signifikan"
    
    # Detail calculation for transparency
    calculation_details = {
        'base_score': base_score,
        'percentage_anomaly': pct,
        'adjustments': adjustments,
        'final_score': final_score,
        'scoring_method': 'Weighted Multi-Factor Analysis',
        'factors_considered': [
            'Persentase frame anomali',
            'Tingkat kepercayaan anomali',
            'Distribusi temporal anomali',
            'Kualitas sampel analisis'
        ],
        'description': desc
    }
    
    return final_score, desc, calculation_details


###############################################################################
# Fungsi Analisis Individual (EXISTING)
#<editor-fold desc="Fungsi Analisis Inti">

def perform_ela(image_path: Path, quality: int=90) -> tuple[Path, int, np.ndarray] | None:
    """
    Error Level Analysis (ELA) yang ditingkatkan dengan analisis grid dan statistik detail.
    Mengembalikan path gambar ELA, max difference, dan array ELA untuk analisis lebih lanjut.
    """
    try:
        ela_dir = image_path.parent.parent / "ela_artifacts"
        ela_dir.mkdir(exist_ok=True)
        out_path = ela_dir / f"{image_path.stem}_ela.jpg"
        temp_jpg_path = out_path.with_name(f"temp_{out_path.name}")
        
        # Buka dan simpan dengan kualitas tertentu
        with Image.open(image_path).convert('RGB') as im:
            im.save(temp_jpg_path, 'JPEG', quality=quality)
        
        # Hitung perbedaan
        with Image.open(image_path).convert('RGB') as im_orig, Image.open(temp_jpg_path) as resaved_im:
            ela_im = ImageChops.difference(im_orig, resaved_im)
        
        if Path(temp_jpg_path).exists():
            Path(temp_jpg_path).unlink()
        
        # Konversi ke array untuk analisis statistik
        ela_array = np.array(ela_im)
        
        # Hitung statistik detail
        extrema = ela_im.getextrema()
        max_diff = max(ex[1] for ex in extrema) if extrema else 1
        scale = 255.0 / (max_diff if max_diff > 0 else 1)
        
        # Enhance dan simpan
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
        
        # Tambahkan grid untuk analisis regional
        ela_with_grid = ela_im.copy()
        draw = ImageDraw.Draw(ela_with_grid)
        width, height = ela_with_grid.size
        grid_size = 50
        
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=(128, 128, 128), width=1)
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=(128, 128, 128), width=1)
        
        ela_with_grid.save(out_path)
        
        return out_path, max_diff, ela_array
    except Exception as e:
        log(f"  {Icons.ERROR} Gagal ELA pada {image_path.name}: {e}")
        return None

def analyze_ela_regions(ela_array: np.ndarray, grid_size: int = 50) -> dict:
    """
    Menganalisis ELA berdasarkan region grid untuk mendeteksi area yang mencurigakan.
    """
    height, width = ela_array.shape[:2]
    suspicious_regions = []
    
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            # Ekstrak region
            region = ela_array[y:min(y+grid_size, height), x:min(x+grid_size, width)]
            if region.size == 0:
                continue
            
            # Hitung metrik untuk setiap region
            mean_val = np.mean(region)
            std_val = np.std(region)
            max_val = np.max(region)
            
            # Deteksi region mencurigakan (nilai ELA tinggi)
            if mean_val > 30 or max_val > 100:  # Threshold dapat disesuaikan
                suspicious_regions.append({
                    'x': x, 'y': y,
                    'width': min(grid_size, width - x),
                    'height': min(grid_size, height - y),
                    'mean_ela': float(mean_val),
                    'std_ela': float(std_val),
                    'max_ela': float(max_val),
                    'suspicion_level': 'high' if mean_val > 50 else 'medium'
                })
    
    return {
        'total_regions': (height // grid_size) * (width // grid_size),
        'suspicious_regions': suspicious_regions,
        'suspicious_count': len(suspicious_regions),
        'grid_size': grid_size
    }

def compare_sift_enhanced(img_path1: Path, img_path2: Path, out_dir: Path) -> dict:
    """
    SIFT comparison yang ditingkatkan dengan analisis geometri dan visualisasi detail.
    """
    try:
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: 
            return {'success': False, 'error': 'Failed to load images'}
        
        # Create SIFT detector
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2: 
            return {'success': False, 'error': 'Insufficient keypoints'}
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        if not matches or any(len(m) < 2 for m in matches): 
            return {'success': False, 'error': 'No valid matches'}
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        result = {
            'success': True,
            'total_keypoints_img1': len(kp1),
            'total_keypoints_img2': len(kp2),
            'total_matches': len(matches),
            'good_matches': len(good_matches),
            'match_quality': 'excellent' if len(good_matches) > 100 else 'good' if len(good_matches) > 50 else 'fair' if len(good_matches) > 20 else 'poor'
        }
        
        if len(good_matches) > CONFIG["SIFT_MIN_MATCH_COUNT"]:
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None and mask is not None:
                inliers = mask.ravel().sum()
                inlier_ratio = inliers / len(good_matches) if len(good_matches) > 0 else 0.0
                
                # Analyze transformation matrix
                det = np.linalg.det(M[:2, :2])
                scale = np.sqrt(abs(det))
                
                result.update({
                    'inliers': int(inliers),
                    'outliers': len(good_matches) - int(inliers),
                    'inlier_ratio': float(inlier_ratio),
                    'homography_determinant': float(det),
                    'estimated_scale': float(scale),
                    'transformation_type': 'rigid' if abs(scale - 1.0) < 0.1 else 'scaled' if 0.5 < scale < 2.0 else 'complex'
                })
                
                # Create detailed visualization
                draw_params = dict(
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=mask.ravel().tolist(),
                    flags=cv2.DrawMatchesFlags_DEFAULT
                )
                
                img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
                
                # Add text annotations
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_matches, f'Total Matches: {len(good_matches)}', (10, 30), font, 0.8, (255, 255, 255), 2)
                cv2.putText(img_matches, f'Inliers: {inliers} ({inlier_ratio:.1%})', (10, 60), font, 0.8, (0, 255, 0), 2)
                cv2.putText(img_matches, f'Quality: {result["match_quality"].upper()}', (10, 90), font, 0.8, (255, 255, 0), 2)
                
                # Save visualization
                sift_dir = out_dir / "sift_artifacts"
                sift_dir.mkdir(exist_ok=True)
                out_path = sift_dir / f"sift_detailed_{img_path1.stem}_vs_{img_path2.stem}.jpg"
                cv2.imwrite(str(out_path), img_matches)
                
                result['visualization_path'] = str(out_path)
                
                # Create heatmap of matched points
                heatmap = create_match_heatmap(src_pts, dst_pts, img1.shape, img2.shape)
                heatmap_path = sift_dir / f"sift_heatmap_{img_path1.stem}_vs_{img_path2.stem}.jpg"
                cv2.imwrite(str(heatmap_path), heatmap)
                result['heatmap_path'] = str(heatmap_path)
        
        return result
    except Exception as e:
        log(f"  {Icons.ERROR} Gagal SIFT: {e}")
        return {'success': False, 'error': str(e)}

def create_match_heatmap(src_pts: np.ndarray, dst_pts: np.ndarray, shape1: tuple, shape2: tuple) -> np.ndarray:
    """
    Membuat heatmap dari distribusi titik-titik yang cocok.
    """
    height = max(shape1[0], shape2[0])
    width = shape1[1] + shape2[1] + 50
    
    heatmap = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gaussian kernel for heatmap
    kernel_size = 21
    kernel = cv2.getGaussianKernel(kernel_size, 5)
    kernel = kernel * kernel.T
    kernel = (kernel / kernel.max() * 255).astype(np.uint8)
    
    # Add heat for source points
    for pt in src_pts:
        x, y = int(pt[0][0]), int(pt[0][1])
        if 0 <= x < shape1[1] and 0 <= y < shape1[0]:
            cv2.circle(heatmap, (x, y), 10, (255, 0, 0), -1)
    
    # Add heat for destination points
    for pt in dst_pts:
        x, y = int(pt[0][0]) + shape1[1] + 50, int(pt[0][1])
        if 0 <= x < width and 0 <= y < shape2[0]:
            cv2.circle(heatmap, (x, y), 10, (0, 0, 255), -1)
    
    # Apply gaussian blur for smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    
    # Apply colormap
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    heatmap_colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    
    return heatmap_colored

def calculate_frame_metrics(frame_path: str) -> dict:
    """
    Menghitung metrik tambahan untuk frame: edge density, blur metric, color distribution.
    """
    try:
        img = cv2.imread(frame_path)
        if img is None:
            return {}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge density menggunakan Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Blur metric menggunakan Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_metric = laplacian.var()
        
        # Color distribution metrics
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize histograms
        if hist_h.sum() > 0: hist_h = hist_h.flatten() / hist_h.sum()
        if hist_s.sum() > 0: hist_s = hist_s.flatten() / hist_s.sum()
        if hist_v.sum() > 0: hist_v = hist_v.flatten() / hist_v.sum()

        # Calculate entropy for color diversity
        h_entropy = -np.sum(hist_h[hist_h > 0] * np.log2(hist_h[hist_h > 0]))
        s_entropy = -np.sum(hist_s[hist_s > 0] * np.log2(hist_s[hist_s > 0]))
        v_entropy = -np.sum(hist_v[hist_v > 0] * np.log2(hist_v[hist_v > 0]))
        
        return {
            'edge_density': float(edge_density),
            'blur_metric': float(blur_metric),
            'color_entropy': {
                'hue': float(h_entropy),
                'saturation': float(s_entropy),
                'value': float(v_entropy)
            }
        }
    except Exception as e:
        log(f"  {Icons.ERROR} Error calculating frame metrics: {e}")
        return {}

def calculate_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ffprobe_metadata(video_path: Path) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        return json.loads(result.stdout)
    except Exception as e:
        log(f"FFprobe error: {e}")
        return {}

# --- FUNGSI BARU: PARSE METADATA DETAIL ---
def parse_ffprobe_output(metadata: dict) -> dict:
    """Mengurai output JSON ffprobe menjadi format yang lebih mudah dibaca."""
    parsed = {}
    if 'format' in metadata:
        fmt = metadata['format']
        parsed['Format'] = {
            'Filename': Path(fmt.get('filename', 'N/A')).name,
            'Format Name': fmt.get('format_long_name', 'N/A'),
            'Duration': f"{float(fmt.get('duration', 0)):.3f} s",
            'Size': f"{int(fmt.get('size', 0)) / (1024*1024):.2f} MB",
            'Bit Rate': f"{int(fmt.get('bit_rate', 0)) / 1000:.0f} kb/s",
            'Creation Time': fmt.get('tags', {}).get('creation_time', 'N/A'),
        }

    video_streams = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'video']
    if video_streams:
        stream = video_streams[0] # Ambil stream video pertama
        parsed['Video Stream'] = {
            'Codec': stream.get('codec_name', 'N/A').upper(),
            'Profile': stream.get('profile', 'N/A'),
            'Resolution': f"{stream.get('width')}x{stream.get('height')}",
            'Aspect Ratio': stream.get('display_aspect_ratio', 'N/A'),
            'Pixel Format': stream.get('pix_fmt', 'N/A'),
            'Frame Rate': f"{eval(stream.get('r_frame_rate', '0/1')):.2f} FPS",
            'Bitrate': f"{int(stream.get('bit_rate', 0)) / 1000:.0f} kb/s" if 'bit_rate' in stream else 'N/A',
            'Encoder': stream.get('tags', {}).get('encoder', 'N/A'),
        }

    audio_streams = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'audio']
    if audio_streams:
        stream = audio_streams[0]
        parsed['Audio Stream'] = {
            'Codec': stream.get('codec_name', 'N/A'),
            'Sample Rate': f"{stream.get('sample_rate')} Hz",
            'Channels': stream.get('channels', 'N/A'),
            'Bitrate': f"{int(stream.get('bit_rate', 0)) / 1000:.0f} kb/s" if 'bit_rate' in stream else 'N/A',
        }
    return parsed

# ====== [NEW] False-Positive Fix June-2025 ======
class Preprocessor:
    """Utilities for video preprocessing."""

    @staticmethod
    def normalize_fps(video_path: Path, target_fps: int = 30) -> tuple[Path, float | None]:
        """Normalize FPS using ffmpeg if below threshold."""
        try:
            import ffmpeg
        except ImportError:
            log("⚠️ Modul ffmpeg-python tidak ditemukan. Install dengan `pip install ffmpeg-python`.")
            return video_path, None

        try:
            probe = ffmpeg.probe(str(video_path))
            v_stream = next((s for s in probe['streams'] if s.get('codec_type') == 'video'), None)
            fps = eval(v_stream.get('avg_frame_rate', '0/1')) if v_stream else None
        except Exception:
            fps = None

        if fps and fps < 24:
            out_path = video_path.with_name(video_path.stem + f"_norm{target_fps}" + video_path.suffix)
            try:
                ffmpeg.input(str(video_path)).output(str(out_path), r=target_fps, loglevel='error').overwrite_output().run()
                return out_path, fps
            except Exception as e:
                log(f"  {Icons.ERROR} Normalisasi FPS gagal: {e}")
        return video_path, fps


def adaptive_thresholds(fps: float, motion_level: float) -> tuple[float, float]:
    """Return adaptive SSIM drop and optical flow z-score thresholds."""
    ssim_val = 0.25 + (fps / 60.0) * 0.10
    ssim_val = min(0.35, max(0.25, ssim_val))
    z_thresh = round(4 + (30 / max(fps, 1.0)), 1)
    if motion_level < 0.1:
        z_thresh += 0.5
    return ssim_val, z_thresh
# ====== [END NEW] ======


# ====== [NEW] Metadata Forensics Enhancement ======
class VideoMetaAnalyzer:
    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self.metadata = {}

    def extract(self) -> dict:
        data = {}
        try:
            from pymediainfo import MediaInfo
            media = MediaInfo.parse(str(self.video_path))
            for track in media.tracks:
                if track.track_type == 'General':
                    data.setdefault('Format', {})
                    # ====== [NEW] Fix SyntaxError on some systems ======
                    writing_app = (
                        getattr(track, 'writing_application', '')
                        or getattr(track, 'encoder', '')
                    )
                    ai_flag = any(k.lower() in writing_app.lower() for k in ['lavf', 'tiktok', 'reels', 'capcut'])
                    data['Format'].update({
                        'Duration': f"{float(track.duration)/1000:.3f} s" if track.duration else 'N/A',
                        'File Size': f"{int(track.file_size)/(1024*1024):.2f} MB" if track.file_size else 'N/A',
                        'Creation Time': getattr(track, 'tagged_date', None) or 'N/A',
                        'Modification Time': getattr(track, 'file_last_modification_date', None) or 'N/A',
                        'Major Brand': getattr(track, 'format', 'N/A'),
                        'Compatible Brands': getattr(track, 'compatible_brands', 'N/A'),
                        'Writing Application': writing_app or 'N/A',
                        'AI Generated Flag': 'Yes' if ai_flag else 'No'
                    })
                if track.track_type == 'Video':
                    data.setdefault('Video Stream', {})
                    data['Video Stream'].update({
                        'Codec': track.codec or 'N/A',
                        'Profile': track.format_profile or 'N/A',
                        'Frame Rate': track.frame_rate or 'N/A',
                        'Bit Rate': track.bit_rate or 'N/A',
                        'Resolution': f"{track.width}x{track.height}" if track.width else 'N/A',
                        'Colour Primaries': getattr(track, 'colour_primaries', 'N/A'),
                        'GOP': getattr(track, 'gop', 'N/A'),
                        'B-Frames': getattr(track, 'b_frame_count', 'N/A'),
                        'Color Range': getattr(track, 'colour_range', 'N/A'),
                        'Rotation': track.rotation or 'N/A'
                    })
                if track.track_type == 'Audio':
                    data.setdefault('Audio Stream', {})
                    data['Audio Stream'].update({
                        'Codec': track.codec or 'N/A',
                        'Sample Rate': track.sampling_rate or 'N/A',
                        'Channel Layout': getattr(track, 'channel_layout', 'N/A'),
                        'Language': track.language or 'N/A'
                    })
        except ImportError:
            log('  \u26A0\ufe0f pymediainfo tidak ditemukan, fallback ke ffprobe.')
        except Exception as e:
            log(f"  {Icons.ERROR} Gagal mengekstrak metadata lanjutan: {e}")

        if not data:
            raw = ffprobe_metadata(self.video_path)
            if raw:
                data = parse_ffprobe_output(raw)
        self.metadata = data
        return data


def explain_metadata(field: str) -> str:
    explanations = {
        'Codec': 'Format kompresi video/audio yang digunakan. Perubahan codec dapat menandakan proses konversi.',
        'Frame Rate': 'Jumlah frame per detik. Ketidakwajaran bisa menunjukkan proses editing.',
        'Bit Rate': 'Jumlah data per detik. Nilai terlalu rendah/tidak konsisten mengindikasikan kompresi ulang.',
        'Colour Primaries': 'Standar warna file video. Ketidaksesuaian antar segmen dapat menjadi indikasi manipulasi.',
        'Creation Time': 'Waktu pembuatan file asli. Bandingkan dengan Modification Time untuk menduga perubahan.',
        'Modification Time': 'Waktu file terakhir diubah. Perbedaan jauh dari Creation Time bisa mencurigakan.',
        'GOP': 'Jarak antar keyframe. Pola GOP tak konsisten dapat menunjukkan penyuntingan frame.',
        'B-Frames': 'Frame prediktif dua arah yang biasanya muncul dalam encoding modern.',
        'Major Brand': 'Identitas container utama yang memberi petunjuk asal perangkat atau encoder.',
        'Writing Application': 'Perangkat lunak yang digunakan untuk menulis file. Nama aplikasi tertentu bisa mengindikasikan proses editing.',
        'AI Generated Flag': 'Heuristik sederhana apakah video berasal dari aplikasi populer yang sering digunakan oleh platform AI atau media sosial.'
    }
    return explanations.get(field, 'Tidak ada penjelasan khusus.')
# ====== [END NEW] ======


# --- FUNGSI DIREVISI: EKSTRAKSI FRAME DENGAN NORMALISASI WARNA ---
def extract_frames_with_normalization(video_path: Path, out_dir: Path, fps: int) -> list[tuple[str, str, str]] | None:
    """Mengekstrak frame, menormalisasi, dan membuat gambar perbandingan."""
    original_dir = out_dir / "frames_original"
    normalized_dir = out_dir / "frames_normalized"
    comparison_dir = out_dir / "frames_comparison"
    original_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log(f"  {Icons.ERROR} Gagal membuka file video: {video_path}")
            return None
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(round(video_fps / fps)) if fps > 0 and video_fps > 0 else 1
        frame_paths = []
        frame_count = 0
        extracted_count = 0

        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="    Ekstraksi & Normalisasi", leave=False, bar_format='{l_bar}{bar}{r_bar}')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_count % frame_skip == 0:
                # 1. Simpan frame original
                original_path = original_dir / f"frame_{extracted_count:06d}_orig.jpg"
                cv2.imwrite(str(original_path), frame)

                # 2. Lakukan normalisasi (Histogram Equalization pada channel Y dari YCrCb)
                ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
                normalized_frame = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
                normalized_path = normalized_dir / f"frame_{extracted_count:06d}_norm.jpg"
                cv2.imwrite(str(normalized_path), normalized_frame)

                # 3. Buat gambar perbandingan
                h, w, _ = frame.shape
                comparison_img = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
                comparison_img[:, :w] = frame
                comparison_img[:, w+10:] = normalized_frame
                cv2.putText(comparison_img, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(comparison_img, 'Normalized', (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                comparison_path = comparison_dir / f"frame_{extracted_count:06d}_comp.jpg"
                cv2.imwrite(str(comparison_path), comparison_img)

                frame_paths.append((str(original_path), str(normalized_path), str(comparison_path)))
                extracted_count += 1
            
            frame_count += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        return frame_paths
    except Exception as e:
        log(f"  {Icons.ERROR} Error saat ekstraksi frame: {e}")
        return None

#</editor-fold>

###############################################################################
# FUNGSI TAMBAHAN UNTUK TAHAP 4 ENHANCED
###############################################################################

def assess_pipeline_performance(result: AnalysisResult) -> dict:
    """
    Menilai performa setiap tahap dalam pipeline forensik.
    """
    assessment = {
        'tahap_1': {
            'nama': 'Pra-pemrosesan & Ekstraksi Fitur',
            'status': 'completed',
            'quality_score': 0,
            'metrics': {},
            'issues': []
        },
        'tahap_2': {
            'nama': 'Analisis Anomali Temporal',
            'status': 'completed',
            'quality_score': 0,
            'metrics': {},
            'issues': []
        },
        'tahap_3': {
            'nama': 'Sintesis Bukti & Investigasi',
            'status': 'completed',
            'quality_score': 0,
            'metrics': {},
            'issues': []
        },
        'tahap_4': {
            'nama': 'Visualisasi & Penilaian',
            'status': 'in_progress',
            'quality_score': 0,
            'metrics': {},
            'issues': []
        }
    }
    
    # Assess Tahap 1
    if result.frames:
        total_frames = len(result.frames)
        frames_with_hash = sum(1 for f in result.frames if f.hash is not None)
        frames_with_cluster = sum(1 for f in result.frames if f.color_cluster is not None)
        
        assessment['tahap_1']['metrics'] = {
            'total_frames_extracted': total_frames,
            'hash_coverage': f"{frames_with_hash/total_frames*100:.1f}%",
            'clustering_coverage': f"{frames_with_cluster/total_frames*100:.1f}%",
            'metadata_completeness': len(result.metadata) > 0
        }
        
        quality = (frames_with_hash/total_frames + frames_with_cluster/total_frames) / 2
        assessment['tahap_1']['quality_score'] = round(quality * 100)
        
        if frames_with_hash < total_frames:
            assessment['tahap_1']['issues'].append('Beberapa frame gagal di-hash')
    
    # Assess Tahap 2
    total_frames = len(result.frames) if result.frames else 0
    frames_with_ssim = sum(1 for f in result.frames if f.ssim_to_prev is not None)
    frames_with_flow = sum(1 for f in result.frames if f.optical_flow_mag is not None)
    
    assessment['tahap_2']['metrics'] = {
        'ssim_coverage': f"{frames_with_ssim/total_frames*100:.1f}%" if total_frames > 0 else "0%",
        'optical_flow_coverage': f"{frames_with_flow/total_frames*100:.1f}%" if total_frames > 0 else "0%",
        'temporal_metrics_computed': frames_with_ssim > 0 and frames_with_flow > 0
    }
    
    if total_frames > 0:
        quality = (frames_with_ssim + frames_with_flow) / (2 * total_frames) if total_frames > 0 else 0
        assessment['tahap_2']['quality_score'] = round(quality * 100)
    
    # Assess Tahap 3
    anomaly_count = sum(1 for f in result.frames if f.type.startswith('anomaly'))
    evidence_count = sum(1 for f in result.frames if f.evidence_obj.reasons)
    
    assessment['tahap_3']['metrics'] = {
        'anomalies_detected': anomaly_count,
        'evidence_collected': evidence_count,
        'ela_analyses': sum(1 for f in result.frames if f.evidence_obj.ela_path is not None),
        'sift_analyses': sum(1 for f in result.frames if f.evidence_obj.sift_path is not None)
    }
    
    if evidence_count > 0:
        assessment['tahap_3']['quality_score'] = min(100, round(evidence_count / anomaly_count * 100)) if anomaly_count > 0 else 100
    
    # Assess Tahap 4
    assessment['tahap_4']['metrics'] = {
        'localizations_created': len(result.localizations),
        'plots_generated': len(result.plots),
        'integrity_calculated': 'integrity_analysis' in result.__dict__
    }
    
    assessment['tahap_4']['quality_score'] = 100 if result.localizations else 0
    
    return assessment

def create_enhanced_localization_map(result: AnalysisResult, out_dir: Path) -> Path:
    """
    Membuat peta lokalisasi tampering yang lebih detail dengan timeline visual.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 2, 1, 1], hspace=0.3, wspace=0.2)
    
    # Title and header
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.text(0.5, 0.5, 'PETA DETAIL LOKALISASI TAMPERING', 
                  ha='center', va='center', fontsize=20, weight='bold')
    ax_title.axis('off')
    
    # Main timeline plot
    ax_timeline = fig.add_subplot(gs[1, :])
    
    # Setup timeline
    total_frames = len(result.frames)
    frame_indices = list(range(total_frames))
    
    # Create background
    ax_timeline.axhspan(0, 1, facecolor='lightgreen', alpha=0.3, label='Normal')
    
    # Plot anomalies with different heights and colors
    anomaly_types = {
        'anomaly_duplication': {'color': '#FF6B6B', 'height': 0.8, 'label': 'Duplikasi', 'marker': 'o'},
        'anomaly_insertion': {'color': '#4ECDC4', 'height': 0.7, 'label': 'Penyisipan', 'marker': 's'},
        'anomaly_discontinuity': {'color': '#45B7D1', 'height': 0.6, 'label': 'Diskontinuitas', 'marker': '^'}
    }
    
    # Draw localization events
    for loc in result.localizations:
        event_type = loc['event']
        if event_type in anomaly_types:
            style = anomaly_types[event_type]
            start_idx = loc['start_frame']
            end_idx = loc['end_frame']
            
            # Draw rectangle for event duration
            rect = plt.Rectangle((start_idx, 0), end_idx - start_idx + 1, style['height'],
                               facecolor=style['color'], alpha=0.6, edgecolor='black', linewidth=2)
            ax_timeline.add_patch(rect)
            
            # Add confidence indicator
            conf_y = style['height'] + 0.05
            conf_color = 'red' if loc['confidence'] == 'SANGAT TINGGI' else 'orange' if loc['confidence'] == 'TINGGI' else 'yellow'
            ax_timeline.plot((start_idx + end_idx) / 2, conf_y, marker='*', 
                           markersize=15, color=conf_color, markeredgecolor='black')
    
    # Timeline settings
    ax_timeline.set_xlim(0, total_frames)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_xlabel('Indeks Frame', fontsize=14)
    ax_timeline.set_title('Timeline Anomali Terdeteksi', fontsize=16, pad=20)
    
    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=style['color'], alpha=0.6, label=style['label'])
                      for style in anomaly_types.values()]
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='red', markersize=10, 
                                    label='Kepercayaan Tinggi', linestyle='None'))
    ax_timeline.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add grid
    ax_timeline.grid(True, axis='x', alpha=0.3)
    
    # Statistics panel
    ax_stats = fig.add_subplot(gs[2, 0])
    stats_text = f"""STATISTIK ANOMALI
    
Total Frame: {total_frames}
Anomali Terdeteksi: {sum(1 for f in result.frames if f.type.startswith('anomaly'))}
Peristiwa Terlokalisasi: {len(result.localizations)}

Distribusi Kepercayaan:
- Sangat Tinggi: {sum(1 for loc in result.localizations if loc['confidence'] == 'SANGAT TINGGI')}
- Tinggi: {sum(1 for loc in result.localizations if loc['confidence'] == 'TINGGI')}
- Sedang: {sum(1 for loc in result.localizations if loc['confidence'] == 'SEDANG')}
- Rendah: {sum(1 for loc in result.localizations if loc['confidence'] == 'RENDAH')}"""
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_stats.axis('off')
    
    # Event details panel
    ax_details = fig.add_subplot(gs[2, 1:])
    details_text = "DETAIL PERISTIWA SIGNIFIKAN\n\n"
    
    # Find most significant events
    significant_events = sorted(result.localizations, 
                              key=lambda x: (x.get('confidence') == 'SANGAT TINGGI', 
                                           x['end_frame'] - x['start_frame']), 
                              reverse=True)[:5]
    
    for i, event in enumerate(significant_events):
        event_type = event['event'].replace('anomaly_', '').capitalize()
        duration = event['end_ts'] - event['start_ts']
        details_text += f"{i+1}. {event_type} @ {event['start_ts']:.1f}s-{event['end_ts']:.1f}s "
        details_text += f"(Durasi: {duration:.1f}s, Kepercayaan: {event.get('confidence', 'N/A')})\n"
    
    ax_details.text(0.05, 0.95, details_text, transform=ax_details.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax_details.axis('off')
    
    # Confidence distribution pie chart
    ax_pie = fig.add_subplot(gs[3, 0])
    confidence_counts = Counter(loc.get('confidence', 'N/A') for loc in result.localizations)
    if confidence_counts:
        colors_conf = {'SANGAT TINGGI': '#FF0000', 'TINGGI': '#FFA500', 
                      'SEDANG': '#FFFF00', 'RENDAH': '#00FF00', 'N/A': '#808080'}
        pie_colors = [colors_conf.get(conf, '#808080') for conf in confidence_counts.keys()]
        ax_pie.pie(confidence_counts.values(), labels=list(confidence_counts.keys()), 
                  colors=pie_colors, autopct='%1.1f%%', startangle=90)
        ax_pie.set_title('Distribusi Tingkat Kepercayaan', fontsize=12)
    else:
        ax_pie.text(0.5, 0.5, 'Tidak ada anomali', ha='center', va='center')
        ax_pie.set_xlim(0, 1)
        ax_pie.set_ylim(0, 1)
    
    # Temporal clustering visualization
    ax_cluster = fig.add_subplot(gs[3, 1:])
    
    # Calculate temporal density
    window_size = total_frames // 20 if total_frames > 20 else 1
    density = np.zeros(total_frames)
    
    for f in result.frames:
        if f.type.startswith('anomaly'):
            start = max(0, f.index - window_size // 2)
            end = min(total_frames, f.index + window_size // 2)
            density[start:end] += 1
    
    ax_cluster.fill_between(frame_indices, density, alpha=0.5, color='red')
    ax_cluster.set_xlabel('Indeks Frame', fontsize=12)
    ax_cluster.set_ylabel('Kepadatan Anomali', fontsize=12)
    ax_cluster.set_title('Analisis Kepadatan Temporal Anomali', fontsize=12)
    ax_cluster.grid(True, alpha=0.3)
    
    # Save the enhanced map
    enhanced_map_path = out_dir / f"enhanced_localization_map_{Path(result.video_path).stem}.png"
    plt.savefig(enhanced_map_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return enhanced_map_path

def create_integrity_breakdown_chart(integrity_details: dict, out_dir: Path, video_path: str) -> Path:
    """
    Membuat chart yang menjelaskan bagaimana skor integritas dihitung.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 2]})
    
    # Waterfall chart untuk breakdown skor
    base_score = integrity_details['base_score']
    adjustments = integrity_details['adjustments']
    final_score = integrity_details['final_score']
    
    # Prepare data for waterfall
    categories = ['Base Score']
    values = [base_score]
    
    cumulative = base_score
    for name, value in adjustments:
        categories.append(name.replace(' (Non-Sistemik)', '\n(Non-Sistemik)'))
        values.append(value)
        cumulative += value
    
    categories.append('Final Score')
    values.append(cumulative)

    # Calculate bottoms for waterfall
    bottoms = np.cumsum(values) - values
    
    # Determine step colors
    colors = ['blue'] + ['green' if val > 0 else 'red' for val in values[1:-1]] + ['gold']

    # Plot bars
    for i, (cat, val, col, bot) in enumerate(zip(categories, values, colors, bottoms)):
        if cat == 'Final Score':
            ax1.bar(i, val, color=col, bottom=0, alpha=0.9, width=0.6)
        else:
            ax1.bar(i, val, color=col, bottom=bot, alpha=0.7, width=0.6)

    # Add value labels
    for i, (cat, val, bot) in enumerate(zip(categories, values, bottoms)):
        if cat == 'Final Score':
            y_pos = val/2
            text = f'{val:.0f}'
            fontweight = 'bold'
            fontsize = 14
        else:
            y_pos = bot + val/2
            text = f'{val:+.0f}' if i > 0 else f'{val:.0f}'
            fontweight = 'bold'
            fontsize = 10
        ax1.text(i, y_pos, text, ha='center', va='center', fontweight=fontweight, fontsize=fontsize, color='black')
        
    # Formatting
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.set_ylabel('Skor Integritas', fontsize=12)
    ax1.set_title('Breakdown Perhitungan Skor Integritas', fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add reference line
    ax1.axhline(y=80, color='darkgreen', linestyle='--', alpha=0.7, label='Target Minimal (80)')
    ax1.legend()
    
    # Explanation panel
    ax2.axis('off')
    explanation_text = f"""PENJELASAN PERHITUNGAN SKOR
============================
1. SKOR DASAR ({base_score:.0f}%)
   Berdasarkan persentase frame anomali.
   (0% = 95, <5% = 90, <10% = 85, etc.)

2. FAKTOR PENYESUAIAN:
"""
    
    for name, value in adjustments:
        if value > 0:
            explanation_text += f"\n   ✅ {name}: +{value:.0f}%"
        else:
            explanation_text += f"\n   ❌ {name}: {value:.0f}%"
    
    explanation_text += f"""
============================
3. SKOR AKHIR: {final_score:.0f}%
   Kategori: {integrity_details.get('description', 'N/A')}

INTERPRETASI SKOR:
--------------------
- 90-95%: Sangat Baik
- 85-89%: Baik
- 80-84%: Cukup Baik
- 70-79%: Sedang
- <70%:  Buruk
"""
    
    ax2.text(0.05, 0.95, explanation_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('Analisis Detail Skor Integritas Video', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    integrity_chart_path = out_dir / f"integrity_breakdown_{Path(video_path).stem}.png"
    plt.savefig(integrity_chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return integrity_chart_path

def create_anomaly_explanation_infographic(result: AnalysisResult, out_dir: Path) -> Path:
    """
    Membuat infografis yang menjelaskan setiap jenis anomali untuk orang awam.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('PANDUAN MEMAHAMI ANOMALI VIDEO', fontsize=20, fontweight='bold')
    
    # Define anomaly types with explanations
    anomaly_info = {
        'Duplikasi': {
            'icon': '🔁',
            'color': '#FF6B6B',
            'simple': 'Frame yang sama diulang beberapa kali',
            'technical': 'Deteksi melalui perbandingan hash dan SIFT',
            'implication': 'Bisa untuk memperpanjang durasi atau menyembunyikan penghapusan',
            'example': 'Seperti memfotokopi halaman yang sama beberapa kali'
        },
        'Diskontinuitas': {
            'icon': '✂️',
            'color': '#45B7D1',
            'simple': 'Terjadi "lompatan" atau patahan dalam aliran video',
            'technical': 'Terdeteksi melalui penurunan SSIM dan lonjakan optical flow',
            'implication': 'Indikasi pemotongan atau penyambungan yang kasar',
            'example': 'Seperti halaman yang hilang dalam sebuah buku'
        },
        'Penyisipan': {
            'icon': '➕',
            'color': '#4ECDC4',
            'simple': 'Frame baru yang tidak ada di video asli',
            'technical': 'Terdeteksi melalui perbandingan dengan baseline',
            'implication': 'Konten tambahan yang mungkin mengubah narasi',
            'example': 'Seperti menambahkan halaman baru ke dalam buku'
        }
    }
    
    # Create grid for each anomaly type
    gs = fig.add_gridspec(len(anomaly_info), 1, hspace=0.3, wspace=0.2)
    
    for idx, (atype, info) in enumerate(anomaly_info.items()):
        ax = fig.add_subplot(gs[idx])
        
        # Background color
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                  facecolor=info['color'], alpha=0.1, zorder=0))
        
        # Title with icon
        ax.text(0.02, 0.85, f"{info['icon']} {atype.upper()}", 
               transform=ax.transAxes, fontsize=18, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=info['color'], alpha=0.3))
        
        # Simple explanation
        ax.text(0.02, 0.65, f"Apa itu?", transform=ax.transAxes, 
               fontsize=12, fontweight='bold')
        ax.text(0.02, 0.45, info['simple'], transform=ax.transAxes, 
               fontsize=11, wrap=True, va='top')
        
        # Example
        ax.text(0.02, 0.25, f"Analogi:", transform=ax.transAxes, 
               fontsize=12, fontweight='bold')
        ax.text(0.02, 0.05, info['example'], transform=ax.transAxes, 
               fontsize=11, fontstyle='italic', va='top')
        
        # Technical
        ax.text(0.52, 0.65, f"Cara Deteksi:", transform=ax.transAxes, 
               fontsize=12, fontweight='bold')
        ax.text(0.52, 0.45, info['technical'], transform=ax.transAxes, 
               fontsize=11, va='top')
        
        # Implication
        ax.text(0.52, 0.25, f"Implikasi:", transform=ax.transAxes, 
               fontsize=12, fontweight='bold')
        ax.text(0.52, 0.05, info['implication'], transform=ax.transAxes, 
               fontsize=11, va='top')
        
        # Count from actual data
        count = sum(1 for loc in result.localizations 
                   if atype.lower() in loc.get('event', '').lower())
        ax.text(0.98, 0.85, f"Ditemukan: {count}", transform=ax.transAxes,
               fontsize=14, ha='right', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save
    infographic_path = out_dir / f"anomaly_explanation_{Path(result.video_path).stem}.png"
    plt.savefig(infographic_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return infographic_path

###############################################################################
# PIPELINE 5-TAHAP
###############################################################################

# --- TAHAP 1: PRA-PEMROSESAN & EKSTRAKSI FITUR DASAR (EXISTING) ---
def run_tahap_1_pra_pemrosesan(video_path: Path, out_dir: Path, fps: int) -> AnalysisResult | None:
    print_stage_banner(1, "Pra-pemrosesan & Ekstraksi Fitur Dasar", Icons.COLLECTION,
                       "Melakukan hashing, ekstraksi metadata detail, normalisasi frame, pHash, dan analisis K-Means.")

    # ====== [NEW] False-Positive Fix June-2025 ======
    norm_path, original_fps = Preprocessor.normalize_fps(video_path)
    fps_normalized = norm_path != video_path
    if fps_normalized:
        video_path = norm_path
        fps = 30
    # ====== [END NEW] ======

    log(f"  {Icons.PRESERVATION} Menghitung hash SHA-256 untuk preservasi...")
    preservation_hash = calculate_sha256(video_path)
    log(f"  -> Hash Bukti: {preservation_hash}")

    log(f"  {Icons.IDENTIFICATION} Mengekstrak metadata detail dengan FFprobe...")
    raw_metadata = ffprobe_metadata(video_path)
    if not raw_metadata:
        log(f"  {Icons.ERROR} Gagal mengekstrak metadata. Analisis tidak dapat dilanjutkan.")
        return None
    metadata = parse_ffprobe_output(raw_metadata)
    # ====== [NEW] False-Positive Fix June-2025 ======
    metadata['fps_initial'] = original_fps
    metadata['fps_effective'] = fps
    metadata['fps_normalized'] = fps_normalized
    # ====== [END NEW] ======
    # ====== [NEW] Metadata Forensics Enhancement ======
    try:
        meta_analyzer = VideoMetaAnalyzer(video_path)
        metadata['Advanced'] = meta_analyzer.extract()
    except Exception as e:
        log(f"  {Icons.ERROR} Ekstraksi metadata lanjutan gagal: {e}")
    # ====== [END NEW] ======
    log(f"  -> Metadata berhasil diurai. Codec video: {metadata.get('Video Stream', {}).get('Codec', 'N/A')}")

    log(f"  {Icons.COLLECTION} Mengekstrak, menormalisasi, dan membandingkan frame @ {fps} FPS...")
    frames_dir_root = out_dir / f"frames_{video_path.stem}"
    extracted_paths = extract_frames_with_normalization(video_path, frames_dir_root, fps)
    if not extracted_paths:
        log(f"  {Icons.ERROR} Gagal mengekstrak frame. Pastikan video valid dan FFmpeg/OpenCV berfungsi.")
        return None
    log(f"  ✅ {len(extracted_paths)} set frame (original, normalized, comparison) berhasil diekstrak.")

    log(f"  {Icons.EXAMINATION} Menghitung pHash untuk setiap frame (menggunakan frame ternormalisasi)...")
    frames = []
    for idx, (p_orig, p_norm, p_comp) in enumerate(tqdm(extracted_paths, desc="    pHash", leave=False, bar_format='{l_bar}{bar}{r_bar}')):
        try:
            with Image.open(p_norm) as img:
                frame_hash = str(imagehash.average_hash(img))
            frames.append(FrameInfo(
                index=idx, 
                timestamp=idx / fps, 
                img_path_original=p_orig, 
                img_path=p_norm, # img_path utama menunjuk ke versi ternormalisasi
                img_path_comparison=p_comp,
                hash=frame_hash
            ))
        except Exception as e:
            log(f"  {Icons.ERROR} Gagal memproses frame set {idx}: {e}")
    
    log(f"  {Icons.EXAMINATION} METODE UTAMA: Menganalisis layout warna global (K-Means)...")
    histograms = []
    for f in tqdm(frames, desc="    Histogram (Normalized)", leave=False, bar_format='{l_bar}{bar}{r_bar}'):
        img = cv2.imread(f.img_path) # Baca dari frame ternormalisasi
        if img is None: continue
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist.flatten())

    # ====== [NEW] False-Positive Fix June-2025 ======
    if histograms:
        scene_variance = float(np.var(histograms))
        if scene_variance < 0.15:
            CONFIG["KMEANS_CLUSTERS"] = 5
    # ====== [END NEW] ======

    kmeans_artifacts = {}
    if histograms:
        actual_n_clusters = min(CONFIG["KMEANS_CLUSTERS"], len(histograms))
        if actual_n_clusters >= 2:
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto').fit(histograms)
            labels = kmeans.labels_.tolist()
            for f, label in zip(frames, labels):
                f.color_cluster = int(label)
            log(f"  -> Klasterisasi K-Means selesai. {len(frames)} frame dikelompokkan ke dalam {actual_n_clusters} klaster.")
            
            # --- PEMBUATAN ARTEFAK K-MEANS DETAIL ---
            log(f"  {Icons.ANALYSIS} Membuat artefak visualisasi detail untuk K-Means...")
            kmeans_dir = out_dir / "kmeans_artifacts"
            kmeans_dir.mkdir(exist_ok=True)
            
            # 1. Plot Distribusi Klaster
            cluster_counts = Counter(labels)
            plt.figure(figsize=(10, 5))
            plt.bar(list(cluster_counts.keys()), list(cluster_counts.values()), color='cornflowerblue')
            plt.title('Distribusi Frame per Klaster K-Means', fontsize=14)
            plt.xlabel('Nomor Klaster', fontsize=12)
            plt.ylabel('Jumlah Frame', fontsize=12)
            plt.xticks(range(actual_n_clusters))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            dist_path = kmeans_dir / "kmeans_distribution.png"
            plt.savefig(dist_path, bbox_inches="tight"); plt.close()
            kmeans_artifacts['distribution_plot_path'] = str(dist_path)

            # 2. Palet Warna dan Sampel Frame per Klaster
            kmeans_artifacts['clusters'] = []
            for i in range(actual_n_clusters):
                cluster_indices = [idx for idx, label in enumerate(labels) if label == i]
                if not cluster_indices: continue
                
                # Buat palet warna dari rata-rata frame di klaster
                avg_color_img = np.zeros((100, 400, 3), np.uint8)
                # Ambil satu frame representatif untuk diekstrak warnanya
                sample_frame_path = frames[cluster_indices[0]].img_path
                sample_img = cv2.imread(sample_frame_path)
                if sample_img is not None:
                    pixels = sample_img.reshape(-1, 3).astype(np.float32)
                    palette_kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto').fit(pixels)
                    for j, color in enumerate(palette_kmeans.cluster_centers_):
                        cv2.rectangle(avg_color_img, (j*80, 0), ((j+1)*80, 100), color.astype(int).tolist(), -1)
                
                palette_path = kmeans_dir / f"cluster_{i}_palette.png"
                cv2.imwrite(str(palette_path), avg_color_img)
                
                # Buat montase sampel frame
                sample_frames_to_show = [frames[j] for j in cluster_indices[:CONFIG["KMEANS_SAMPLES_PER_CLUSTER"]]]
                montage_h = (Image.open(sample_frames_to_show[0].img_path_original).height if sample_frames_to_show else 180)
                montage_w = (Image.open(sample_frames_to_show[0].img_path_original).width if sample_frames_to_show else 320)
                montage_img = Image.new('RGB', (montage_w * len(sample_frames_to_show), montage_h))
                for k, f_info in enumerate(sample_frames_to_show):
                    with Image.open(f_info.img_path_original) as img:
                        img = img.resize((montage_w, montage_h))
                        montage_img.paste(img, (k * montage_w, 0))
                
                montage_path = kmeans_dir / f"cluster_{i}_samples.jpg"
                montage_img.save(montage_path)

                kmeans_artifacts['clusters'].append({
                    'id': i,
                    'count': len(cluster_indices),
                    'palette_path': str(palette_path),
                    'samples_montage_path': str(montage_path)
                })
            log(f"  -> Artefak K-Means berhasil dibuat di direktori {kmeans_dir.name}")

    log(f"  {Icons.SUCCESS} Tahap 1 Selesai.")
    result = AnalysisResult(str(video_path), preservation_hash, metadata, frames)
    result.kmeans_artifacts = kmeans_artifacts
    # ====== [NEW] False-Positive Fix June-2025 ======
    result.summary['fps_normalized'] = fps_normalized
    # ====== [END NEW] ======
    return result

# --- TAHAP 2: ANALISIS ANOMALI TEMPORAL & KOMPARATIF (EXISTING) ---
def run_tahap_2_analisis_temporal(result: AnalysisResult, baseline_result: AnalysisResult | None = None):
    print_stage_banner(2, "Analisis Anomali Temporal & Komparatif", Icons.ANALYSIS,
                       "Menganalisis aliran optik, SSIM, dan perbandingan dengan baseline jika ada.")
    frames = result.frames
    prev_gray = None

    log(f"  {Icons.EXAMINATION} Menghitung Aliran Optik & SSIM antar frame (menggunakan frame ternormalisasi)...")
    for f_idx, f in enumerate(tqdm(frames, desc="    Temporal", leave=False, bar_format='{l_bar}{bar}{r_bar}')):
        current_gray = cv2.imread(f.img_path, cv2.IMREAD_GRAYSCALE) # f.img_path adalah frame ternormalisasi
        if current_gray is not None:
            if prev_gray is not None and prev_gray.shape == current_gray.shape:
                data_range = float(current_gray.max() - current_gray.min())
                if data_range > 0:
                    ssim_score = ssim(prev_gray, current_gray, data_range=data_range)
                    f.ssim_to_prev = float(ssim_score)
                else:
                    f.ssim_to_prev = 1.0 # Frames are identical if data_range is 0
                
                if current_gray.dtype == prev_gray.dtype:
                    try:
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        f.optical_flow_mag = float(np.mean(mag))
                    except cv2.error as e:
                        log(f"  {Icons.ERROR} OpenCV error during optical flow for frame {f.index}: {e}")
                        f.optical_flow_mag = 0.0
                else:
                    f.optical_flow_mag = 0.0
            else: 
                f.ssim_to_prev = 1.0 
                f.optical_flow_mag = 0.0 

        prev_gray = current_gray

    if baseline_result:
        log(f"  {Icons.ANALYSIS} Melakukan analisis komparatif terhadap video baseline...")
        base_hashes = {bf.hash for bf in baseline_result.frames if bf.hash}
        insertion_count = 0
        for f_sus in frames:
            if f_sus.hash and f_sus.hash not in base_hashes:
                f_sus.type = "anomaly_insertion"
                f_sus.evidence_obj.reasons.append("Frame tidak ada di baseline")
                f_sus.evidence_obj.confidence = "SANGAT TINGGI"
                insertion_count += 1
        log(f"  -> Terdeteksi {insertion_count} frame sisipan potensial.")

    log(f"  {Icons.SUCCESS} Tahap 2 Selesai.")

# --- TAHAP 3: SINTESIS BUKTI & INVESTIGASI MENDALAM (EXISTING) ---
def run_tahap_3_sintesis_bukti(result: AnalysisResult, out_dir: Path):
    print_stage_banner(3, "Sintesis Bukti & Investigasi Mendalam", "🔬",
                       "Mengkorelasikan semua temuan dan melakukan analisis ELA/SIFT pada anomali terkuat dengan penjelasan detail.")
    frames = result.frames
    n = len(frames)
    if n < 2: return

    # Inisialisasi struktur untuk analisis detail
    result.detailed_anomaly_analysis = {
        'temporal_discontinuities': [],
        'duplication_analysis': [],
        'compression_anomalies': [],
        'statistical_outliers': []
    }

    log(f"  {Icons.ANALYSIS} ANALISIS 1: Deteksi Diskontinuitas Temporal...")
    log(f"  📖 Penjelasan: Diskontinuitas temporal adalah perubahan mendadak antara frame yang berurutan.")
    log(f"     Ini bisa mengindikasikan penghapusan frame, penyisipan konten, atau editing yang kasar.")
    
    # Kalkulasi metrik tambahan untuk setiap frame
    log(f"  {Icons.EXAMINATION} Menghitung metrik detail untuk setiap frame...")
    for f in tqdm(frames, desc="    Metrik Frame", leave=False):
        metrics = calculate_frame_metrics(f.img_path_original)
        f.edge_density = metrics.get('edge_density')
        f.blur_metric = metrics.get('blur_metric')
        f.evidence_obj.detailed_analysis['frame_metrics'] = metrics

    # Analisis metrik diskontinuitas dengan penjelasan
    flow_mags = [f.optical_flow_mag for f in frames if f.optical_flow_mag is not None]
    
    if flow_mags:
        # Gunakan metode statistik yang lebih robust
        filtered_flow_mags = [m for m in flow_mags if m > 0.0]
        if len(filtered_flow_mags) > 1: 
            median_flow = np.median(filtered_flow_mags)
            mad_flow = stats.median_abs_deviation(filtered_flow_mags)
            mad_flow = 1e-9 if mad_flow == 0 else mad_flow
            
            # Hitung persentil untuk context
            p25 = np.percentile(filtered_flow_mags, 25)
            p75 = np.percentile(filtered_flow_mags, 75)
            p95 = np.percentile(filtered_flow_mags, 95)
            
            log(f"  📊 Statistik Aliran Optik:")
            log(f"     - Median: {median_flow:.3f}")
            log(f"     - MAD (Median Absolute Deviation): {mad_flow:.3f}")
            log(f"     - Persentil 25/75/95: {p25:.3f}/{p75:.3f}/{p95:.3f}")
        else:
            median_flow = 0.0
            mad_flow = 1.0

        # ====== [NEW] False-Positive Fix June-2025 ======
        ssim_t, z_t = adaptive_thresholds(result.metadata.get('fps_effective', 30), median_flow)
        if CONFIG.get('USE_AUTO_THRESHOLDS', True):
            CONFIG['SSIM_DISCONTINUITY_DROP'] = ssim_t
            CONFIG['OPTICAL_FLOW_Z_THRESH'] = z_t
        else:
            CONFIG['SSIM_DISCONTINUITY_DROP'] = CONFIG.get('SSIM_USER_THRESHOLD', 0.25)
            CONFIG['OPTICAL_FLOW_Z_THRESH'] = CONFIG.get('Z_USER_THRESHOLD', 4.0)
        result.summary['thresholds'] = {'ssim_drop': CONFIG['SSIM_DISCONTINUITY_DROP'], 'z_score': CONFIG['OPTICAL_FLOW_Z_THRESH']}
        # ====== [END NEW] ======

        # Deteksi anomali dengan Z-score
        for f in frames:
            if f.optical_flow_mag is not None and f.optical_flow_mag > 0:
                if mad_flow != 0:
                    z_score = 0.6745 * (f.optical_flow_mag - median_flow) / mad_flow
                    if abs(z_score) > CONFIG["OPTICAL_FLOW_Z_THRESH"]:
                        f.evidence_obj.reasons.append("Lonjakan Aliran Optik")
                        f.evidence_obj.metrics["optical_flow_z_score"] = round(z_score, 2)
                        
                        # Tambahkan penjelasan detail
                        explanation = {
                            "type": "optical_flow_spike",
                            "frame_index": f.index,
                            "timestamp": f.timestamp,
                            "severity": "high" if abs(z_score) > 6 else "medium",
                            "technical_explanation": (
                                f"Frame ini menunjukkan pergerakan piksel yang {abs(z_score):.1f}x "
                                "lebih besar dari normal."
                            ),
                            "simple_explanation": (
                                "Terjadi perubahan gambar yang sangat mendadak, "
                                "seperti perpindahan kamera yang kasar atau cut yang tidak halus."
                            ),
                            "metrics": {
                                "flow_magnitude": f.optical_flow_mag,
                                "z_score": z_score,
                                "median_flow": median_flow,
                                "deviation_percentage": (
                                    (f.optical_flow_mag - median_flow) / median_flow * 100
                                )
                                if median_flow > 0
                                else 0,
                            },
                        }
                        f.evidence_obj.explanations['optical_flow'] = explanation
                        result.detailed_anomaly_analysis['temporal_discontinuities'].append(explanation)

    # Analisis SSIM dengan konteks yang lebih kaya
    log(f"\n  {Icons.ANALYSIS} ANALISIS 2: Deteksi Penurunan Kemiripan Struktural (SSIM)...")
    log(f"  📖 Penjelasan: SSIM mengukur seberapa mirip dua gambar secara struktural.")
    log(f"     Nilai 1.0 = identik, nilai < 0.7 = sangat berbeda. Penurunan drastis = kemungkinan manipulasi.")
    
    ssim_values = [f.ssim_to_prev for f in frames if f.ssim_to_prev is not None]
    if ssim_values:
        ssim_mean = np.mean(ssim_values)
        ssim_std = np.std(ssim_values)
        log(f"  📊 Statistik SSIM: Mean={ssim_mean:.3f}, Std={ssim_std:.3f}")
    
    for i in range(1, n):
        f_curr, f_prev = frames[i], frames[i - 1]
        if f_curr.ssim_to_prev is not None and f_prev.ssim_to_prev is not None:
            ssim_drop = f_prev.ssim_to_prev - f_curr.ssim_to_prev
            
            # Deteksi penurunan drastis
            if ssim_drop > CONFIG["SSIM_DISCONTINUITY_DROP"]:
                f_curr.evidence_obj.reasons.append("Penurunan Drastis SSIM")
                f_curr.evidence_obj.metrics["ssim_drop"] = round(ssim_drop, 4)
                
                explanation = {
                    'type': 'ssim_drop',
                    'frame_index': f_curr.index,
                    'timestamp': f_curr.timestamp,
                    'severity': 'high' if ssim_drop > 0.5 else 'medium',
                    'technical_explanation': f"SSIM turun {ssim_drop:.3f} dari frame sebelumnya ({f_prev.ssim_to_prev:.3f} → {f_curr.ssim_to_prev:.3f}).",
                    'simple_explanation': "Frame ini sangat berbeda dari frame sebelumnya, mungkin ada potongan atau sisipan.",
                    'metrics': {
                        'ssim_current': f_curr.ssim_to_prev,
                        'ssim_previous': f_prev.ssim_to_prev,
                        'drop_amount': ssim_drop,
                        'drop_percentage': (ssim_drop / f_prev.ssim_to_prev * 100) if f_prev.ssim_to_prev > 0 else 0
                    }
                }
                f_curr.evidence_obj.explanations['ssim_drop'] = explanation
                result.detailed_anomaly_analysis['temporal_discontinuities'].append(explanation)
                
            # Deteksi nilai SSIM sangat rendah
            elif f_curr.ssim_to_prev < 0.7:
                f_curr.evidence_obj.reasons.append("SSIM Sangat Rendah")
                f_curr.evidence_obj.metrics["ssim_absolute_low"] = round(f_curr.ssim_to_prev, 4)
                
                explanation = {
                    'type': 'ssim_low',
                    'frame_index': f_curr.index,
                    'timestamp': f_curr.timestamp,
                    'severity': 'medium',
                    'technical_explanation': f"SSIM sangat rendah ({f_curr.ssim_to_prev:.3f}), menunjukkan perbedaan struktural yang signifikan.",
                    'simple_explanation': "Frame ini memiliki struktur visual yang sangat berbeda dari frame sebelumnya.",
                    'metrics': {
                        'ssim_value': f_curr.ssim_to_prev,
                        'threshold': 0.7,
                        'below_threshold_by': 0.7 - f_curr.ssim_to_prev
                    }
                }
                f_curr.evidence_obj.explanations['ssim_low'] = explanation

    # Analisis perubahan klaster warna dengan konteks
    log(f"\n  {Icons.ANALYSIS} ANALISIS 3: Deteksi Perubahan Adegan (K-Means)...")
    log(f"  📖 Penjelasan: K-Means mengelompokkan frame berdasarkan palet warna dominan.")
    log(f"     Perubahan klaster = perubahan adegan. Perubahan yang terlalu sering = kemungkinan editing.")
    
    scene_changes = []
    for i in range(1, n):
        f_curr, f_prev = frames[i], frames[i - 1]
        if f_curr.color_cluster is not None and f_prev.color_cluster is not None and f_curr.color_cluster != f_prev.color_cluster:
            f_curr.evidence_obj.reasons.append("Perubahan Adegan (dari K-Means)")
            f_curr.evidence_obj.metrics["color_cluster_jump"] = f"{f_prev.color_cluster} → {f_curr.color_cluster}"
            
            scene_change = {
                'frame_index': f_curr.index,
                'timestamp': f_curr.timestamp,
                'from_cluster': f_prev.color_cluster,
                'to_cluster': f_curr.color_cluster,
                'time_since_last_change': 0  # Will be calculated
            }
            scene_changes.append(scene_change)
            
            explanation = {
                'type': 'scene_change',
                'frame_index': f_curr.index,
                'timestamp': f_curr.timestamp,
                'technical_explanation': f"Perubahan dari klaster warna {f_prev.color_cluster} ke {f_curr.color_cluster}.",
                'simple_explanation': "Terjadi perubahan adegan atau sudut pandang kamera.",
                'metrics': {
                    'from_cluster': f_prev.color_cluster,
                    'to_cluster': f_curr.color_cluster
                }
            }
            f_curr.evidence_obj.explanations['scene_change'] = explanation

    # Hitung frekuensi perubahan adegan
    if scene_changes:
        for i in range(1, len(scene_changes)):
            scene_changes[i]['time_since_last_change'] = scene_changes[i]['timestamp'] - scene_changes[i-1]['timestamp']
        
        avg_scene_duration = np.mean([sc['time_since_last_change'] for sc in scene_changes[1:]]) if len(scene_changes) > 1 else 0
        log(f"  📊 Total perubahan adegan: {len(scene_changes)}")
        log(f"     Durasi rata-rata per adegan: {avg_scene_duration:.2f} detik")

    # METODE PENDUKUNG: Verifikasi duplikasi dengan analisis mendalam
    log(f"\n  {Icons.EXAMINATION} METODE PENDUKUNG 1: Analisis Duplikasi Frame (SIFT+RANSAC)...")
    log(f"  📖 Penjelasan: SIFT mendeteksi titik-titik unik dalam gambar. Jika dua frame memiliki")
    log(f"     banyak titik yang cocok sempurna, kemungkinan besar frame tersebut diduplikasi.")
    
    hash_map = defaultdict(list)
    for f in frames:
        if f.hash: hash_map[f.hash].append(f.index)
    
    dup_candidates = {k: v for k, v in hash_map.items() if len(v) > 1}
    
    if dup_candidates:
        log(f"  🔍 Ditemukan {len(dup_candidates)} grup kandidat duplikasi untuk diverifikasi...")
        
        for hash_val, idxs in tqdm(dup_candidates.items(), desc="    Verifikasi Duplikasi", leave=False):
            for i in range(len(idxs) - 1):
                idx1, idx2 = idxs[i], idxs[i + 1]
                # ====== [NEW] False-Positive Fix June-2025 ======
                window = idxs[i:i+3]
                if len(window) < 3 or (frames[window[-1]].timestamp - frames[window[0]].timestamp) > 0.5:
                    continue
                # ====== [END NEW] ======
                p1, p2 = Path(frames[idx1].img_path_original), Path(frames[idx2].img_path_original)
                
                # Cek SSIM terlebih dahulu
                im1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
                im2 = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
                if im1 is None or im2 is None: continue
                if im1.shape != im2.shape: continue
                
                data_range = float(im1.max() - im1.min())
                if data_range == 0: continue
                ssim_val = ssim(im1, im2, data_range=data_range)
                
                if ssim_val > CONFIG["DUPLICATION_SSIM_CONFIRM"]:
                    # Analisis SIFT detail
                    sift_result = compare_sift_enhanced(p1, p2, out_dir)
                    
                    if sift_result.get('success') and sift_result.get('inliers', 0) >= CONFIG["SIFT_MIN_MATCH_COUNT"]:
                        f_dup = frames[idx2]
                        f_dup.type = "anomaly_duplication"
                        f_dup.evidence_obj.reasons.append(f"Duplikasi dari frame {idx1}")
                        f_dup.evidence_obj.metrics.update({
                            "source_frame": idx1,
                            "ssim_to_source": round(ssim_val, 4),
                            "sift_inliers": sift_result['inliers'],
                            "sift_good_matches": sift_result['good_matches'],
                            "sift_inlier_ratio": round(sift_result['inlier_ratio'], 3)
                        })
                        
                        if sift_result.get('visualization_path'):
                            f_dup.evidence_obj.sift_path = sift_result['visualization_path']
                            f_dup.evidence_obj.visualizations['sift_matches'] = sift_result['visualization_path']
                        
                        if sift_result.get('heatmap_path'):
                            f_dup.evidence_obj.visualizations['sift_heatmap'] = sift_result['heatmap_path']
                        
                        # Penjelasan detail duplikasi
                        duplication_analysis = {
                            'type': 'frame_duplication',
                            'duplicate_frame': idx2,
                            'source_frame': idx1,
                            'timestamp_duplicate': frames[idx2].timestamp,
                            'timestamp_source': frames[idx1].timestamp,
                            'time_gap': frames[idx2].timestamp - frames[idx1].timestamp,
                            'confidence': 'very_high' if sift_result['inlier_ratio'] > 0.8 else 'high',
                            'technical_explanation': f"Frame {idx2} adalah duplikasi dari frame {idx1} dengan {sift_result['inliers']} titik fitur yang cocok sempurna ({sift_result['inlier_ratio']:.1%} akurasi).",
                            'simple_explanation': f"Frame pada detik {frames[idx2].timestamp:.2f} adalah salinan persis dari frame pada detik {frames[idx1].timestamp:.2f}. Ini sering digunakan untuk memperpanjang durasi video atau menyembunyikan penghapusan konten.",
                            'sift_analysis': sift_result,
                            'implications': "Duplikasi frame dapat mengindikasikan: (1) Usaha memperpanjang durasi, (2) Menutupi frame yang dihapus, (3) Teknik editing untuk transisi"
                        }
                        f_dup.evidence_obj.explanations['duplication'] = duplication_analysis
                        result.detailed_anomaly_analysis['duplication_analysis'].append(duplication_analysis)

    # METODE PENDUKUNG: ELA dengan analisis regional
    log(f"\n  {Icons.ANALYSIS} METODE PENDUKUNG 2: Error Level Analysis (ELA) untuk Anomali Signifikan...")
    log(f"  📖 Penjelasan: ELA mendeteksi area yang telah diedit dengan melihat perbedaan kompresi.")
    log(f"     Area yang lebih terang dalam ELA = kemungkinan telah dimodifikasi atau disisipkan.")
    
    # Buat direktori untuk visualisasi tambahan
    detail_viz_dir = out_dir / "detailed_visualizations"
    detail_viz_dir.mkdir(exist_ok=True)
    
    for f in tqdm(frames, desc="    Analisis ELA & Sintesis", leave=False):
        # First, ensure reasons is a list
        if isinstance(f.evidence_obj.reasons, str):
            f.evidence_obj.reasons = [r.strip() for r in f.evidence_obj.reasons.split(',')]
        
        if f.evidence_obj.reasons:
            if f.type == "original": 
                f.type = "anomaly_discontinuity"
            
            # Tentukan tingkat kepercayaan berdasarkan jumlah bukti
            num_reasons = len(f.evidence_obj.reasons)
            if f.type == "anomaly_duplication" or f.type == "anomaly_insertion": 
                f.evidence_obj.confidence = "SANGAT TINGGI"
            elif num_reasons > 2: 
                f.evidence_obj.confidence = "TINGGI"
            elif num_reasons > 1: 
                f.evidence_obj.confidence = "SEDANG"
            else: 
                f.evidence_obj.confidence = "RENDAH"

            # Lakukan ELA untuk anomali dengan kepercayaan sedang ke atas
            if f.evidence_obj.confidence in ["SEDANG", "TINGGI", "SANGAT TINGGI"] and f.type not in ["anomaly_duplication", "anomaly_insertion"]:
                ela_result = perform_ela(Path(f.img_path_original))
                if ela_result:
                    ela_path, max_diff, ela_array = ela_result
                    f.evidence_obj.ela_path = str(ela_path)
                    
                    # Analisis regional ELA
                    regional_analysis = analyze_ela_regions(ela_array)
                    
                    if regional_analysis['suspicious_count'] > 0:
                        # --- START OF FIX ---
                        # The following block was incorrectly indented. It has been corrected.
                        if "Anomali Kompresi (ELA)" not in f.evidence_obj.reasons:
                            f.evidence_obj.reasons.append("Anomali Kompresi (ELA)")
                        f.evidence_obj.metrics["ela_max_difference"] = max_diff
                        f.evidence_obj.metrics["ela_suspicious_regions"] = regional_analysis['suspicious_count']
                        
                        # Upgrade confidence jika ditemukan area mencurigakan
                        if regional_analysis['suspicious_count'] > 5:
                            if f.evidence_obj.confidence == "SEDANG": 
                                f.evidence_obj.confidence = "TINGGI"
                            elif f.evidence_obj.confidence == "TINGGI": 
                                f.evidence_obj.confidence = "SANGAT TINGGI"
                        
                        # Buat visualisasi ELA dengan highlight area mencurigakan
                        ela_viz_path = create_ela_visualization(
                            Path(f.img_path_original), 
                            ela_array, 
                            regional_analysis, 
                            detail_viz_dir
                        )
                        if ela_viz_path:
                            f.evidence_obj.visualizations['ela_detailed'] = str(ela_viz_path)
                        
                        # Penjelasan detail ELA
                        ela_explanation = {
                            'type': 'compression_anomaly',
                            'frame_index': f.index,
                            'timestamp': f.timestamp,
                            'max_difference': max_diff,
                            'suspicious_regions': regional_analysis['suspicious_regions'][:5],  # Top 5
                            'total_suspicious_areas': regional_analysis['suspicious_count'],
                            'technical_explanation': f"ELA menunjukkan {regional_analysis['suspicious_count']} area dengan perbedaan kompresi tinggi (max: {max_diff}). Area ini kemungkinan telah diedit atau disisipkan.",
                            'simple_explanation': "Bagian-bagian tertentu dari frame ini menunjukkan 'jejak' editing digital. Seperti sidik jari pada kaca, ELA dapat melihat area yang telah dimodifikasi karena memiliki tingkat kompresi yang berbeda.",
                            'severity': 'high' if max_diff > 100 else 'medium',
                            'implications': "Area dengan nilai ELA tinggi menunjukkan: (1) Objek yang disisipkan, (2) Area yang di-retouch, (3) Teks atau watermark yang ditambahkan"
                        }
                        f.evidence_obj.explanations['ela'] = ela_explanation
                        result.detailed_anomaly_analysis['compression_anomalies'].append(ela_explanation)
                        # --- END OF FIX ---
    
    # Konversi reasons list ke string untuk konsistensi
    for f in frames:
        if isinstance(f.evidence_obj.reasons, list) and f.evidence_obj.reasons:
            f.evidence_obj.reasons = ", ".join(sorted(list(set(f.evidence_obj.reasons))))

    # Analisis statistik keseluruhan
    log(f"\n  {Icons.ANALYSIS} ANALISIS STATISTIK KESELURUHAN...")
    
    # Hitung distribusi anomali
    anomaly_types = Counter()
    confidence_levels = Counter()
    temporal_distribution = []
    
    for f in frames:
        if f.type.startswith("anomaly"):
            anomaly_types[f.type] += 1
            confidence_levels[f.evidence_obj.confidence] += 1
            temporal_distribution.append(f.timestamp)
    
    # Analisis clustering temporal anomali
    if temporal_distribution:
        temporal_clusters = []
        if not temporal_distribution: # Handle empty list
            pass
        else:
            current_cluster = [temporal_distribution[0]]
            
            for i in range(1, len(temporal_distribution)):
                if temporal_distribution[i] - temporal_distribution[i-1] < 2.0:  # Within 2 seconds
                    current_cluster.append(temporal_distribution[i])
                else:
                    if len(current_cluster) > 1:
                        temporal_clusters.append(current_cluster)
                    current_cluster = [temporal_distribution[i]]
            
            if len(current_cluster) > 1:
                temporal_clusters.append(current_cluster)
            
            log(f"  📊 Distribusi Anomali:")
            for atype, count in anomaly_types.items():
                log(f"     - {atype.replace('anomaly_', '').title()}: {count} frame")
            
            log(f"  📊 Tingkat Kepercayaan:")
            for level, count in confidence_levels.items():
                log(f"     - {level}: {count} anomali")
            
            if temporal_clusters:
                log(f"  📊 Ditemukan {len(temporal_clusters)} kluster anomali temporal")
                for i, cluster in enumerate(temporal_clusters):
                    log(f"     - Kluster {i+1}: {len(cluster)} anomali dalam {cluster[-1]-cluster[0]:.2f} detik")
    
    # Simpan statistik dalam result
    result.statistical_summary = {
        'total_frames_analyzed': len(frames),
        'total_anomalies': sum(anomaly_types.values()),
        'anomaly_types': dict(anomaly_types),
        'confidence_distribution': dict(confidence_levels),
        'temporal_clusters': len(temporal_clusters) if temporal_distribution else 0,
        'average_anomalies_per_cluster': np.mean([len(c) for c in temporal_clusters]) if temporal_clusters else 0
    }
    
    # Update confidence distribution untuk Tahap 4
    result.confidence_distribution = dict(confidence_levels)
    
    # Buat visualisasi ringkasan anomali
    if anomaly_types:
        create_anomaly_summary_visualization(result, detail_viz_dir)
    
    log(f"\n  {Icons.SUCCESS} Tahap 3 Selesai - Investigasi mendalam telah dilengkapi dengan penjelasan detail.")

# Fungsi helper untuk membuat visualisasi ELA detail
def create_ela_visualization(original_path: Path, ela_array: np.ndarray, regional_analysis: dict, out_dir: Path) -> Path | None:
    """Membuat visualisasi ELA dengan highlight area mencurigakan."""
    try:
        # Load original image
        original = cv2.imread(str(original_path))
        if original is None:
            return None
        
        # Convert ELA array to color
        ela_color = cv2.applyColorMap((ela_array.mean(axis=2) * 5).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create combined visualization
        height, width = original.shape[:2]
        combined = np.zeros((height, width * 2 + 20, 3), dtype=np.uint8)
        combined[:, :width] = original
        combined[:, width+20:] = ela_color
        
        # Draw suspicious regions
        for region in regional_analysis['suspicious_regions'][:10]:  # Top 10
            x, y = region['x'], region['y']
            w, h = region['width'], region['height']
            color = (0, 0, 255) if region['suspicion_level'] == 'high' else (0, 255, 255)
            
            # Draw on original
            cv2.rectangle(combined, (x, y), (x+w, y+h), color, 2)
            # Draw on ELA
            cv2.rectangle(combined, (width+20+x, y), (width+20+x+w, y+h), color, 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, 'ELA Analysis', (width+30, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, f'Suspicious Areas: {regional_analysis["suspicious_count"]}', 
                    (10, height-10), font, 0.7, (255, 255, 0), 2)
        
        # Save
        out_path = out_dir / f"ela_detailed_{original_path.stem}.jpg"
        cv2.imwrite(str(out_path), combined)
        return out_path
    except Exception as e:
        log(f"  {Icons.ERROR} Error creating ELA visualization: {e}")
        return None

# Fungsi untuk membuat visualisasi ringkasan anomali
def create_anomaly_summary_visualization(result: AnalysisResult, out_dir: Path):
    """Membuat visualisasi ringkasan dari semua anomali yang terdeteksi."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Ringkasan Analisis Forensik Video', fontsize=16, fontweight='bold')
        
        # 1. Pie chart distribusi tipe anomali
        if result.statistical_summary.get('anomaly_types'):
            labels = [t.replace('anomaly_', '').title() for t in result.statistical_summary['anomaly_types'].keys()]
            sizes = list(result.statistical_summary['anomaly_types'].values())
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Distribusi Jenis Anomali')
            ax1.axis('equal')
        else:
            ax1.text(0.5, 0.5, 'Tidak ada anomali terdeteksi', ha='center', va='center')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')

        # 2. Bar chart tingkat kepercayaan
        if result.statistical_summary.get('confidence_distribution'):
            confidence_labels = list(result.statistical_summary['confidence_distribution'].keys())
            confidence_values = list(result.statistical_summary['confidence_distribution'].values())
            colors_conf = {'RENDAH': 'green', 'SEDANG': 'yellow', 'TINGGI': 'orange', 'SANGAT TINGGI': 'red', 'N/A': 'gray'}
            bar_colors = [colors_conf.get(label, 'gray') for label in confidence_labels]
            
            ax2.bar(confidence_labels, confidence_values, color=bar_colors)
            ax2.set_title('Distribusi Tingkat Kepercayaan Anomali')
            ax2.set_xlabel('Tingkat Kepercayaan')
            ax2.set_ylabel('Jumlah Anomali')
        
        # 3. Timeline anomali
        anomaly_times = []
        anomaly_types_list = []
        for f in result.frames:
            if f.type.startswith("anomaly"):
                anomaly_times.append(f.timestamp)
                anomaly_types_list.append(f.type.replace('anomaly_', ''))
        
        if anomaly_times:
            # Create scatter plot with different colors for each type
            type_colors = {'discontinuity': 'purple', 'duplication': 'orange', 'insertion': 'red'}
            for atype in set(anomaly_types_list):
                times = [t for t, at in zip(anomaly_times, anomaly_types_list) if at == atype]
                ax3.scatter(times, [1]*len(times), label=atype.title(), 
                           color=type_colors.get(atype, 'gray'), s=100, alpha=0.7)
            
            ax3.set_title('Timeline Anomali')
            ax3.set_xlabel('Waktu (detik)')
            ax3.set_ylim(0.5, 1.5)
            ax3.set_yticks([])
            ax3.legend()
            ax3.grid(True, axis='x', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Tidak ada timeline anomali', ha='center', va='center')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
        
        # 4. Statistik ringkasan
        stats_text = f"""Total Frame Dianalisis: {result.statistical_summary.get('total_frames_analyzed', 'N/A')}
Total Anomali Terdeteksi: {result.statistical_summary.get('total_anomalies', 'N/A')}
Persentase Anomali: {result.statistical_summary.get('total_anomalies', 0)/result.statistical_summary.get('total_frames_analyzed', 1)*100:.1f}%
Kluster Temporal: {result.statistical_summary.get('temporal_clusters', 'N/A')}
Rata-rata Anomali per Kluster: {result.statistical_summary.get('average_anomalies_per_cluster', 0):.1f}"""
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Statistik Ringkasan')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        summary_path = out_dir / "anomaly_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        result.plots['anomaly_summary'] = str(summary_path)
    except Exception as e:
        log(f"  {Icons.ERROR} Error creating summary visualization: {e}")

# --- TAHAP 4: VISUALISASI & PENILAIAN INTEGRITAS (ENHANCED VERSION) ---
def run_tahap_4_visualisasi_dan_penilaian(result: AnalysisResult, out_dir: Path):
    print_stage_banner(4, "Visualisasi & Penilaian Integritas (ENHANCED)", "📊",
                       "Membuat plot detail, melokalisasi peristiwa dengan Localization Tampering, menghitung skor integritas realistis, dan menilai pipeline.")
    
    log(f"  {Icons.ANALYSIS} METODE UTAMA: Melakukan Localization Tampering untuk mengelompokkan anomali...")
    log(f"  📖 Localization Tampering adalah teknik untuk mengelompokkan frame-frame anomali yang berdekatan")
    log(f"     menjadi satu 'peristiwa' yang koheren, memudahkan interpretasi hasil forensik.")
    
    locs, event = [], None
    for f in result.frames:
        is_anomaly = f.type.startswith("anomaly")
        if is_anomaly:
            image_to_show = f.img_path_original
            if event and event["event"] == f.type and f.index == event["end_frame"] + 1:
                # Extend existing event
                event["end_frame"] = f.index
                event["end_ts"] = f.timestamp
                event["frame_count"] += 1
                # Update confidence ke yang tertinggi
                conf_hierarchy = {"SANGAT TINGGI": 4, "TINGGI": 3, "SEDANG": 2, "RENDAH": 1, "N/A": 0}
                if conf_hierarchy.get(f.evidence_obj.confidence, 0) > conf_hierarchy.get(event["confidence"], 0):
                    event["confidence"] = f.evidence_obj.confidence
                # Update explanations
                if f.evidence_obj.explanations:
                    event["explanations"].update(f.evidence_obj.explanations)
                # Collect all metrics
                event["all_metrics"].append(f.evidence_obj.metrics)
            else:
                # Save previous event if exists
                if event: 
                    locs.append(event)
                # Start new event
                event = {
                    "event": f.type, 
                    "start_frame": f.index, 
                    "end_frame": f.index, 
                    "start_ts": f.timestamp,
                    "end_ts": f.timestamp,
                    "frame_count": 1,
                    "confidence": f.evidence_obj.confidence, 
                    "reasons": str(f.evidence_obj.reasons),
                    "metrics": f.evidence_obj.metrics, 
                    "all_metrics": [f.evidence_obj.metrics],  # Collect all metrics for statistics
                    "image": image_to_show, 
                    "ela_path": f.evidence_obj.ela_path,
                    "sift_path": f.evidence_obj.sift_path,
                    "explanations": f.evidence_obj.explanations.copy(),
                    "visualizations": f.evidence_obj.visualizations.copy()
                }
        elif event:
            locs.append(event)
            event = None
    if event: 
        locs.append(event)
    
    # Enhance localization dengan analisis tambahan
    for loc in locs:
        # Calculate event duration and severity
        loc['duration'] = loc['end_ts'] - loc['start_ts']
        loc['severity_score'] = calculate_event_severity(loc)
        
        # Aggregate metrics across all frames in event
        if loc.get('all_metrics'):
            aggregated = {}
            for metrics in loc['all_metrics']:
                if isinstance(metrics, dict):
                    for key, val in metrics.items():
                        if not isinstance(key, list) and not isinstance(val, list):
                            if key not in aggregated:
                                aggregated[key] = []
                            if val is not None:
                                aggregated[key].append(val)
            
            # Calculate statistics for numeric metrics
            loc['aggregated_metrics'] = {}
            for key, vals in aggregated.items():
                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if numeric_vals:
                    loc['aggregated_metrics'][key] = {
                        'mean': np.mean(numeric_vals),
                        'max': max(numeric_vals),
                        'min': min(numeric_vals),
                        'std': np.std(numeric_vals)
                    }
    
    result.localizations = locs
    result.localization_details = {
        'total_events': len(locs),
        'events_by_type': Counter(loc['event'] for loc in locs),
        'total_anomalous_frames': sum(loc.get('frame_count', 0) for loc in locs),
        'average_event_duration': np.mean([loc.get('duration',0) for loc in locs]) if locs else 0,
        'max_event_duration': max([loc.get('duration',0) for loc in locs]) if locs else 0,
        'high_severity_events': sum(1 for loc in locs if loc.get('severity_score',0) > 0.7)
    }
    
    log(f"  -> Ditemukan dan dilokalisasi {len(locs)} peristiwa anomali.")
    log(f"  -> Rata-rata durasi peristiwa: {result.localization_details['average_event_duration']:.2f} detik")
    log(f"  -> Peristiwa dengan severity tinggi: {result.localization_details['high_severity_events']}")
    
    # Calculate comprehensive summary
    total_anom = sum(1 for f in result.frames if f.type.startswith("anomaly"))
    total_frames = len(result.frames)
    pct_anomaly = round(total_anom * 100 / total_frames, 2) if total_frames > 0 else 0
    result.summary = {
        "total_frames": total_frames, 
        "total_anomaly": total_anom, 
        "pct_anomaly": pct_anomaly,
        "total_events": len(locs),
        "anomaly_density": total_anom / total_frames if total_frames > 0 else 0
    }
    
    log(f"  {Icons.INFO} {total_anom} dari {total_frames} frame terindikasi anomali ({pct_anomaly}%).")
    
    # Calculate realistic integrity score
    log(f"\n  {Icons.ANALYSIS} Menghitung Skor Integritas dengan metode Multi-Factor Analysis...")
    integrity_score, integrity_desc, integrity_details = generate_integrity_score(
        result.summary, 
        result.statistical_summary
    )
    result.integrity_analysis = {
        'score': integrity_score,
        'description': integrity_desc,
        'calculation_details': integrity_details,
        'timestamp': datetime.now().isoformat()
    }
    log(f"  -> Skor Integritas: {integrity_score}% ({integrity_desc})")
    log(f"  -> Metode: {integrity_details['scoring_method']}")
    
    # Assess pipeline performance
    log(f"\n  {Icons.EXAMINATION} Menilai performa setiap tahap pipeline forensik...")
    result.pipeline_assessment = assess_pipeline_performance(result)
    for stage_id, assessment in result.pipeline_assessment.items():
        log(f"  -> {assessment['nama']}: Quality Score = {assessment['quality_score']}%")
    
    # Create enhanced visualizations
    log(f"\n  {Icons.ANALYSIS} Membuat visualisasi detail...")
    
    # 1. Enhanced Localization Map
    log(f"  📍 Membuat peta lokalisasi tampering yang detail...")
    enhanced_map_path = create_enhanced_localization_map(result, out_dir)
    result.plots['enhanced_localization_map'] = str(enhanced_map_path)
    
    # 2. Integrity Score Breakdown
    log(f"  📊 Membuat breakdown perhitungan skor integritas...")
    integrity_chart_path = create_integrity_breakdown_chart(result.integrity_analysis['calculation_details'], out_dir, result.video_path)
    result.plots['integrity_breakdown'] = str(integrity_chart_path)
    
    # 3. Anomaly Explanation Infographic
    log(f"  📚 Membuat infografis penjelasan anomali untuk orang awam...")
    infographic_path = create_anomaly_explanation_infographic(result, out_dir)
    result.plots['anomaly_infographic'] = str(infographic_path)
    
    # 4. Existing plots (dengan perbaikan)
    log(f"  📈 Membuat plot temporal standar...")
    
    # K-Means temporal plot
    color_clusters = [f.color_cluster for f in result.frames if f.color_cluster is not None]
    if color_clusters:
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(color_clusters)), color_clusters, marker='.', linestyle='-', markersize=4, label='Klaster Warna Frame')
        jump_frames = [i for i in range(1, len(color_clusters)) if color_clusters[i] != color_clusters[i-1]]
        if jump_frames:
            for jf in jump_frames:
                plt.axvline(x=jf, color='r', linestyle='--', linewidth=1, alpha=0.7)
            plt.plot([], [], color='r', linestyle='--', linewidth=1, label='Perubahan Adegan Terdeteksi')
        plt.title('Visualisasi Klasterisasi Warna (Metode K-Means) Sepanjang Waktu', fontsize=14, weight='bold')
        plt.xlabel('Indeks Frame', fontsize=12)
        plt.ylabel('Nomor Klaster Warna', fontsize=12)
        if len(set(color_clusters)) > 1: 
            plt.yticks(range(min(set(color_clusters)), max(set(color_clusters))+1))
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        kmeans_temporal_plot_path = out_dir / f"plot_kmeans_temporal_{Path(result.video_path).stem}.png"
        plt.savefig(kmeans_temporal_plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        result.plots['kmeans_temporal'] = str(kmeans_temporal_plot_path)

    # SSIM temporal plot
    ssim_values = [f.ssim_to_prev for f in result.frames if f.ssim_to_prev is not None]
    if len(ssim_values) > 1:
        y_values_ssim = ssim_values[1:]
        x_indices_ssim = list(range(1, len(y_values_ssim) + 1))
        
        plt.figure(figsize=(15, 6))
        plt.plot(x_indices_ssim, y_values_ssim, color='skyblue', marker='.', linestyle='-', markersize=3, alpha=0.7)
        
        discontinuity_frames_ssim_indices = [f.index for f in result.frames if "SSIM" in str(f.evidence_obj.reasons)]
        if discontinuity_frames_ssim_indices:
            valid_indices = [i for i in discontinuity_frames_ssim_indices if 0 < i < len(ssim_values)]
            if valid_indices:
                discontinuity_ssim_y_values = [ssim_values[i] for i in valid_indices]
                plt.scatter(valid_indices, discontinuity_ssim_y_values, color='red', marker='X', s=100, zorder=5, label='Diskontinuitas Terdeteksi (SSIM)')
        
        plt.title('Perubahan SSIM Antar Frame Sepanjang Waktu', fontsize=14, weight='bold')
        plt.xlabel('Indeks Frame', fontsize=12)
        plt.ylabel('Skor SSIM (0-1, Lebih Tinggi Lebih Mirip)', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='lower left', fontsize=10)
        plt.tight_layout()
        ssim_temporal_plot_path = out_dir / f"plot_ssim_temporal_{Path(result.video_path).stem}.png"
        plt.savefig(ssim_temporal_plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        result.plots['ssim_temporal'] = str(ssim_temporal_plot_path)

    # Optical flow temporal plot
    flow_values = [f.optical_flow_mag for f in result.frames if f.optical_flow_mag is not None]
    if len(flow_values) > 1:
        y_values_flow = flow_values[1:]
        x_indices_flow = list(range(1, len(y_values_flow) + 1))
        
        plt.figure(figsize=(15, 6))
        plt.plot(x_indices_flow, y_values_flow, color='salmon', marker='.', linestyle='-', markersize=3, alpha=0.7)
        
        discontinuity_frames_flow_indices = [f.index for f in result.frames if "Aliran Optik" in str(f.evidence_obj.reasons)]
        if discontinuity_frames_flow_indices:
            valid_indices_flow = [i for i in discontinuity_frames_flow_indices if 0 < i < len(flow_values)]
            if valid_indices_flow:
                discontinuity_flow_y_values = [flow_values[i] for i in valid_indices_flow]
                plt.scatter(valid_indices_flow, discontinuity_flow_y_values, color='darkgreen', marker='o', s=100, zorder=5, label='Diskontinuitas Terdeteksi (Aliran Optik)')
        
        flow_mags_for_z = [m for m in flow_values if m is not None and m > 0.0]
        if len(flow_mags_for_z) > 1:
            median_flow = np.median(flow_mags_for_z)
            mad_flow = stats.median_abs_deviation(flow_mags_for_z)
            mad_flow = 1e-9 if mad_flow == 0 else mad_flow
            threshold_mag_upper = (CONFIG["OPTICAL_FLOW_Z_THRESH"] / 0.6745) * mad_flow + median_flow
            plt.axhline(y=threshold_mag_upper, color='blue', linestyle='--', linewidth=1, label=f'Ambang Batas Atas Z-score')
        
        plt.title('Perubahan Rata-rata Magnitudo Aliran Optik', fontsize=14, weight='bold')
        plt.xlabel('Indeks Frame', fontsize=12)
        plt.ylabel('Rata-rata Magnitudo Aliran Optik', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        optical_flow_temporal_plot_path = out_dir / f"plot_optical_flow_temporal_{Path(result.video_path).stem}.png"
        plt.savefig(optical_flow_temporal_plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        result.plots['optical_flow_temporal'] = str(optical_flow_temporal_plot_path)

    # Metrics histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if len(ssim_values) > 1:
        ssim_to_plot = [s for s in ssim_values[1:] if s is not None]
        if ssim_to_plot:
            ax1.hist(ssim_to_plot, bins=50, color='skyblue', edgecolor='black')
        ax1.set_title("Distribusi Skor SSIM")
        ax1.set_xlabel("Skor SSIM")
        ax1.set_ylabel("Frekuensi")
    if len(flow_values) > 1:
        flow_to_plot = [f for f in flow_values[1:] if f is not None]
        if flow_to_plot:
            ax2.hist(flow_to_plot, bins=50, color='salmon', edgecolor='black')
        ax2.set_title("Distribusi Aliran Optik")
        ax2.set_xlabel("Rata-rata Pergerakan")
        ax2.set_ylabel("Frekuensi")
    plt.tight_layout()
    metrics_histograms_plot_path = out_dir / f"plot_metrics_histograms_{Path(result.video_path).stem}.png"
    plt.savefig(metrics_histograms_plot_path, dpi=100)
    plt.close()
    result.plots['metrics_histograms'] = str(metrics_histograms_plot_path)

    # Simple temporal anomaly plot
    plt.figure(figsize=(15, 6))
    anomaly_data = {
        'Duplikasi': {'x': [], 'color': 'orange', 'marker': 'o', 'level': 1.0},
        'Penyisipan': {'x': [], 'color': 'red', 'marker': 'x', 'level': 0.9},
        'Diskontinuitas': {'x': [], 'color': 'purple', 'marker': '|', 'level': 0.8}
    }
    for f in result.frames:
        if f.type == "anomaly_duplication": 
            anomaly_data['Duplikasi']['x'].append(f.index)
        elif f.type == "anomaly_insertion": 
            anomaly_data['Penyisipan']['x'].append(f.index)
        elif f.type == "anomaly_discontinuity": 
            anomaly_data['Diskontinuitas']['x'].append(f.index)
    
    for label, data in anomaly_data.items():
        if data['x']:
            plt.vlines(data['x'], 0, data['level'], colors=data['color'], lw=1.5, alpha=0.8)
            plt.scatter(data['x'], np.full_like(data['x'], data['level'], dtype=float), 
                       c=data['color'], marker=data['marker'], s=40, label=label, zorder=5)
    
    plt.ylim(-0.1, 1.2)
    plt.yticks([0, 0.8, 0.9, 1.0], ['Asli', 'Diskontinuitas', 'Penyisipan', 'Duplikasi'])
    plt.xlabel("Indeks Frame", fontsize=12)
    plt.ylabel("Jenis Anomali Terdeteksi", fontsize=12)
    plt.title(f"Peta Anomali Temporal untuk {Path(result.video_path).name}", fontsize=14, weight='bold')
    plt.grid(True, axis='x', linestyle=':', alpha=0.7)
    
    from matplotlib.lines import Line2D
    plt.legend(handles=[Line2D([0], [0], color=d['color'], marker=d['marker'], linestyle='None', label=l) 
                        for l, d in anomaly_data.items() if d['x']], loc='upper right', fontsize=10)
    plt.tight_layout()
    temporal_plot_path = out_dir / f"plot_temporal_{Path(result.video_path).stem}.png"
    plt.savefig(temporal_plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    result.plots['temporal'] = str(temporal_plot_path)

    log(f"  {Icons.SUCCESS} Tahap 4 Selesai - Analisis detail dan penilaian integritas telah lengkap.")

# Helper function for calculating event severity
def calculate_event_severity(event: dict) -> float:
    """Calculate severity score for an anomaly event (0-1)."""
    severity = 0.0
    
    # Base severity by type
    type_severity = {
        'anomaly_insertion': 0.8,
        'anomaly_duplication': 0.6,
        'anomaly_discontinuity': 0.5
    }
    severity = type_severity.get(event.get('event', ''), 0.3)
    
    # Adjust by confidence
    confidence_multiplier = {
        'SANGAT TINGGI': 1.2,
        'TINGGI': 1.0,
        'SEDANG': 0.8,
        'RENDAH': 0.6,
        'N/A': 0.5
    }
    severity *= confidence_multiplier.get(event.get('confidence', 'N/A'), 0.5)
    
    # Adjust by duration
    if event.get('duration', 0) > 2.0:  # Long events are more severe
        severity *= 1.1

    # Adjust by frame count
    if event.get('frame_count', 0) > 10:
        severity *= 1.1

    return min(1.0, severity)

# ====== [NEW] Metadata Forensics Enhancement ======
def generate_simple_reports(result: AnalysisResult, report_dir: Path) -> None:
    """Generate HTML & JSON forensic reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    data = {
        'video': Path(result.video_path).name,
        'hash': result.preservation_hash,
        'summary': result.summary,
        'metadata': result.metadata,
        'integrity': result.integrity_analysis,
        'localizations': result.localizations
    }
    json_path = report_dir / f"{Path(result.video_path).stem}_forensic.json"
    html_path = report_dir / f"{Path(result.video_path).stem}_forensic.html"
    try:
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(data, jf, indent=2)
        html_content = f"<html><body><h1>Forensic Report</h1><pre>{json.dumps(data, indent=2)}</pre></body></html>"
        with open(html_path, 'w', encoding='utf-8') as hf:
            hf.write(html_content)
        result.json_report_path = json_path
        result.html_report_path = html_path
    except Exception as e:
        log(f"  {Icons.ERROR} Gagal membuat laporan HTML/JSON: {e}")
# ====== [END NEW] ======

# --- TAHAP 5: PENYUSUNAN LAPORAN & VALIDASI FORENSIK (ADJUSTED FOR ENHANCED TAHAP 4) ---
def run_tahap_5_pelaporan_dan_validasi(result: AnalysisResult, out_dir: Path, baseline_result: AnalysisResult | None = None):
    print_stage_banner(5, "Penyusunan Laporan & Validasi Forensik", Icons.REPORTING,
                       "Menghasilkan laporan PDF naratif yang komprehensif dengan detail Tahap 4 yang telah ditingkatkan.")
    
    pdf_path = out_dir / f"laporan_forensik_{Path(result.video_path).stem}.pdf"
    
    def get_encoder_info(metadata: dict) -> str: 
        return metadata.get('Video Stream', {}).get('Encoder', 'N/A')
    
    def get_anomaly_explanation(event_type: str) -> str:
        explanations = {
            "Duplication": "Frame-frame ini adalah salinan identik dari frame sebelumnya.",
            "Insertion": "Frame-frame ini <b>tidak ditemukan</b> dalam video asli/baseline.",
            "Discontinuity": "Terdeteksi 'patahan' atau transisi mendadak dalam aliran video."
        }
        return explanations.get(event_type, "Jenis anomali tidak dikenal.")

    def explain_metric(metric_name: str) -> str:
        explanations = {
            "optical_flow_z_score": "Ukuran lonjakan gerakan abnormal (Z-score > 4 = sangat abnormal).",
            "ssim_drop": "Ukuran penurunan kemiripan visual (> 0.25 = perubahan drastis).",
            "ssim_absolute_low": "Skor kemiripan yang sangat rendah (< 0.7 = sangat berbeda).",
            "color_cluster_jump": "Perubahan adegan visual berdasarkan analisis warna K-Means.",
            "source_frame": "Frame asli dari duplikasi (nomor indeks frame).",
            "ssim_to_source": "Skor kemiripan dengan frame asli (0-1, 1 = identik).",
            "sift_inliers": "Jumlah titik fitur unik yang cocok kuat (> 10 = duplikasi kuat).",
            "sift_good_matches": "Total kandidat titik fitur yang cocok.",
            "sift_inlier_ratio": "Rasio kecocokan valid (> 0.8 = duplikasi hampir pasti).",
            "ela_max_difference": "Tingkat perbedaan kompresi (0-255, > 100 = editing signifikan).",
            "ela_suspicious_regions": "Jumlah area yang menunjukkan tanda-tanda editing."
        }
        return explanations.get(metric_name, "Metrik tidak dikenal.")

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=30, bottomMargin=50, leftMargin=30, rightMargin=30)
    styles = getSampleStyleSheet()

    # --- START OF FIX ---
    # Defensively add styles to prevent KeyError if they already exist
    if 'Code' not in styles:
        styles.add(ParagraphStyle(name='Code', fontName='Courier', fontSize=8, leading=10, wordWrap='break'))
    if 'SubTitle' not in styles:
        styles.add(ParagraphStyle(name='SubTitle', parent=styles['h2'], fontSize=12, textColor=colors.darkslategray))
    if 'Justify' not in styles:
        styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=4)) # Justify
    if 'H3-Box' not in styles:
        styles.add(ParagraphStyle(name='H3-Box', parent=styles['h3'], backColor=colors.lightgrey, padding=4, leading=14, leftIndent=4, borderPadding=2, textColor=colors.black))
    if 'ExplanationBox' not in styles:
        styles.add(ParagraphStyle(name='ExplanationBox', parent=styles['Normal'], backColor='#FFF8DC', borderColor='#CCCCCC', borderWidth=1, borderPadding=8, leftIndent=10, rightIndent=10))
    # --- END OF FIX ---
    
    story = []
    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.drawString(30, 30, f"Laporan VIFA-Pro | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawRightString(A4[0] - 30, 30, f"Halaman {doc.page}")
        canvas.restoreState()

    story.append(Paragraph("Laporan Analisis Forensik Video", styles['h1']))
    story.append(Paragraph("Dihasilkan oleh Sistem VIFA-Pro", styles['SubTitle']))
    story.append(Spacer(1, 24))
    story.append(Paragraph("Ringkasan Eksekutif", styles['h2']))
    
    # Use enhanced integrity score
    integrity_score = result.integrity_analysis['score']
    integrity_desc = result.integrity_analysis['description']
    
    summary_text = (f"Analisis komprehensif terhadap file <b>{Path(result.video_path).name}</b> telah selesai. "
                    f"Berdasarkan <b>{len(result.localizations)} peristiwa anomali</b> yang terdeteksi, video ini diberikan "
                    f"<b>Skor Integritas: {integrity_score}/100 ({integrity_desc})</b>. Metode utama yang digunakan adalah "
                    f"<b>Klasterisasi K-Means</b> dan <b>Localization Tampering</b> dengan dukungan metode pendukung "
                    f"<b>Error Level Analysis (ELA)</b> dan <b>Scale-Invariant Feature Transform (SIFT)</b>.")
    story.append(Paragraph(summary_text, styles['Justify']))
    story.append(Spacer(1, 12))
    
    story.append(PageBreak())
    story.append(Paragraph("Detail Laporan Berdasarkan Tahapan Forensik", styles['h1']))

    # --- TAHAP 1 PDF ---
    story.append(Paragraph("Tahap 1: Akuisisi dan Analisis Fitur Dasar", styles['h2']))
    story.append(Paragraph("<b>1.1. Identifikasi & Preservasi Bukti</b>", styles['h3']))
    
    metadata_table_data = [["<b>Kategori</b>", "<b>Item</b>", "<b>Nilai</b>"]]
    for category, items in result.metadata.items():
        for i, (key, value) in enumerate(items.items()):
            cat_name = f"<b>{category}</b>" if i == 0 else ""
            metadata_table_data.append([Paragraph(cat_name, styles['Normal']), Paragraph(key, styles['Normal']), Paragraph(f"<code>{value}</code>", styles['Code'])])
    
    table_style_cmds = [('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white),('GRID', (0,0), (-1,-1), 0.5, colors.grey),('VALIGN', (0,0), (-1,-1), 'TOP')]
    current_row = 1
    for category, items in result.metadata.items():
        if items and len(items) > 1: table_style_cmds.append(('SPAN', (0, current_row), (0, current_row + len(items) - 1)))
        current_row += len(items)
    story.append(Table(metadata_table_data, colWidths=[80, 150, 290], style=TableStyle(table_style_cmds)))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>1.2. Ekstraksi dan Normalisasi Frame</b>", styles['h3']))
    if result.frames and result.frames[0].img_path_comparison and Path(result.frames[0].img_path_comparison).exists(): 
        story.append(PlatypusImage(result.frames[0].img_path_comparison, width=520, height=146, kind='proportional'))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>1.3. Metode Utama: Analisis Klasterisasi K-Means</b>", styles['h3']))
    if result.kmeans_artifacts.get('distribution_plot_path') and Path(result.kmeans_artifacts['distribution_plot_path']).exists(): 
        story.append(PlatypusImage(result.kmeans_artifacts['distribution_plot_path'], width=400, height=200, kind='proportional'))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Detail Setiap Klaster:</b>", styles['Normal']))
    for cluster_info in result.kmeans_artifacts.get('clusters', []):
        story.append(Paragraph(f"<b>Klaster {cluster_info['id']}</b> ({cluster_info['count']} frame)", styles['H3-Box']))
        
        palette_img = PlatypusImage(cluster_info['palette_path'], width=200, height=50) if cluster_info.get('palette_path') and Path(cluster_info['palette_path']).exists() else Paragraph("N/A", styles['Normal'])
        samples_img = PlatypusImage(cluster_info['samples_montage_path'], width=300, height=54) if cluster_info.get('samples_montage_path') and Path(cluster_info['samples_montage_path']).exists() else Paragraph("N/A", styles['Normal'])
        
        cluster_data = [[Paragraph("Palet Warna Dominan", styles['Normal']), Paragraph("Contoh Frame (Asli)", styles['Normal'])],
                        [palette_img, samples_img]]
        story.append(Table(cluster_data, colWidths=[215, 310], style=TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (-1,-1), 'CENTER')])))
        story.append(Spacer(1, 6))

    story.append(PageBreak())
    
    # --- TAHAP 2 PDF ---
    story.append(Paragraph("Tahap 2: Analisis Anomali Temporal", styles['h2']))
    story.append(Paragraph("<b>2.1. Klasterisasi Warna K-Means (Analisis Temporal)</b>", styles['h3']))
    if result.plots.get('kmeans_temporal') and Path(result.plots['kmeans_temporal']).exists(): story.append(PlatypusImage(result.plots['kmeans_temporal'], width=520, height=195, kind='proportional'))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>2.2. Analisis Skor SSIM</b>", styles['h3']))
    if result.plots.get('ssim_temporal') and Path(result.plots['ssim_temporal']).exists(): story.append(PlatypusImage(result.plots['ssim_temporal'], width=520, height=195, kind='proportional'))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>2.3. Analisis Magnitudo Aliran Optik</b>", styles['h3']))
    if result.plots.get('optical_flow_temporal') and Path(result.plots['optical_flow_temporal']).exists(): story.append(PlatypusImage(result.plots['optical_flow_temporal'], width=520, height=195, kind='proportional'))
    story.append(Spacer(1, 12))
    
    if baseline_result:
        story.append(Paragraph("<b>2.4. Analisis Komparatif (vs Baseline)</b>", styles['h3']))
        insertion_events_count = len([loc for loc in result.localizations if loc['event'] == 'anomaly_insertion'])
        story.append(Paragraph(f"Ditemukan <b>{insertion_events_count} peristiwa penyisipan</b> yang tidak ada di video baseline.", styles['Justify']))
    
    story.append(PageBreak())
    
    # --- TAHAP 3 PDF ---
    story.append(Paragraph("Tahap 3: Investigasi Detail Anomali dengan Metode Pendukung", styles['h2']))
    
    # Tambahkan ringkasan statistik
    if result.statistical_summary:
        story.append(Paragraph("<b>Ringkasan Statistik Investigasi:</b>", styles['h3']))
        stats_table = [
            ["<b>Metrik</b>", "<b>Nilai</b>"],
            ["Total Frame Dianalisis", str(result.statistical_summary['total_frames_analyzed'])],
            ["Total Anomali Terdeteksi", str(result.statistical_summary['total_anomalies'])],
            ["Persentase Anomali", f"{result.statistical_summary.get('total_anomalies', 0)/result.statistical_summary.get('total_frames_analyzed', 1)*100:.1f}%"],
            ["Kluster Temporal Anomali", str(result.statistical_summary['temporal_clusters'])],
            ["Rata-rata Anomali per Kluster", f"{result.statistical_summary.get('average_anomalies_per_cluster', 0):.1f}"]
        ]
        story.append(Table(stats_table, colWidths=[250, 250], style=TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.black),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'LEFT')
        ])))
        story.append(Spacer(1, 12))
    
    # Tambahkan visualisasi ringkasan jika ada
    if result.plots.get('anomaly_summary') and Path(result.plots['anomaly_summary']).exists():
        story.append(Paragraph("<b>Visualisasi Ringkasan Anomali:</b>", styles['h3']))
        story.append(PlatypusImage(result.plots['anomaly_summary'], width=520, height=350, kind='proportional'))
        story.append(Spacer(1, 12))
    
    if not result.localizations:
        story.append(Paragraph("Tidak ditemukan anomali signifikan.", styles['Normal']))
    else:
        story.append(Paragraph("<b>Detail Setiap Peristiwa Anomali:</b>", styles['h3']))
        
        for i, loc in enumerate(result.localizations):
            event_type = loc.get('event', 'unknown').replace('anomaly_', '').capitalize()
            confidence = loc.get('confidence', 'N/A')
            
            # Header peristiwa
            story.append(Paragraph(f"<b>3.{i+1} | Peristiwa: {event_type}</b> @ {loc.get('start_ts',0):.2f} - {loc.get('end_ts',0):.2f} dtk", styles['H3-Box']))
            
            # Tambahkan informasi durasi dan severity
            story.append(Paragraph(f"<b>Durasi:</b> {loc.get('duration', 0):.2f} detik | <b>Severity:</b> {loc.get('severity_score', 0):.2f}/1.0", styles['Normal']))
            
            # Penjelasan umum
            story.append(Paragraph(f"<b>Penjelasan Umum:</b> {get_anomaly_explanation(event_type)}", styles['Normal']))
            
            # Tambahkan penjelasan detail dari explanations
            if loc.get('explanations'):
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Analisis Detail:</b>", styles['Normal']))
                
                for exp_type, exp_data in loc['explanations'].items():
                    if isinstance(exp_data, dict):
                        story.append(Paragraph(f"<b>📊 {exp_type.replace('_', ' ').title()}:</b>", styles['Normal']))
                        
                        simple_exp = exp_data.get('simple_explanation', '')
                        if simple_exp:
                            story.append(Paragraph(f"<i>Penjelasan Sederhana:</i> {simple_exp}", styles['ExplanationBox']))
                            story.append(Spacer(1, 6))
                        
                        tech_exp = exp_data.get('technical_explanation', '')
                        if tech_exp:
                            story.append(Paragraph(f"<i>Penjelasan Teknis:</i> {tech_exp}", styles['Normal']))
                            story.append(Spacer(1, 6))
            
            # Tabel bukti teknis
            tech_data = [["<b>Bukti Teknis</b>", "<b>Nilai</b>", "<b>Interpretasi</b>"]]
            tech_data.append(["Tingkat Kepercayaan", f"<b>{confidence}</b>", "Keyakinan sistem terhadap anomali."])
            
            if isinstance(loc.get('metrics'), dict):
                for key, val in loc.get('metrics', {}).items():
                    interpretation = explain_metric(key)
                    tech_data.append([key.replace('_', ' ').title(), Paragraph(str(val), styles['Code']), Paragraph(interpretation, styles['Code'])])
            
            story.append(Table(tech_data, colWidths=[150, 100, 275], style=TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.black),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
            ])))
            story.append(Spacer(1, 12))
            
            # Visualisasi bukti
            v_headers, v_evidence = [], []
            
            if loc.get('image') and Path(loc['image']).exists():
                v_headers.append("<b>Sampel Frame (Asli)</b>")
                v_evidence.append(PlatypusImage(loc['image'], width=250, height=140, kind='proportional'))
            
            if loc.get('ela_path') and Path(loc['ela_path']).exists():
                v_headers.append("<b>Analisis Kompresi (ELA)</b>")
                v_evidence.append(PlatypusImage(loc['ela_path'], width=250, height=140, kind='proportional'))
            
            if v_evidence:
                story.append(Table([v_headers, v_evidence], colWidths=[260]*len(v_headers), style=[('ALIGN',(0,0),(-1,-1),'CENTER')]))
                story.append(Spacer(1, 6))
            
            if loc.get('visualizations'):
                if loc['visualizations'].get('ela_detailed') and Path(loc['visualizations']['ela_detailed']).exists():
                    story.append(Paragraph("<b>Analisis ELA Detail:</b>", styles['Normal']))
                    story.append(PlatypusImage(loc['visualizations']['ela_detailed'], width=520, height=180, kind='proportional'))
                
                if loc['visualizations'].get('sift_heatmap') and Path(loc['visualizations']['sift_heatmap']).exists():
                    story.append(Paragraph("<b>Heatmap SIFT:</b>", styles['Normal']))
                    story.append(PlatypusImage(loc['visualizations']['sift_heatmap'], width=520, height=160, kind='proportional'))

            if loc.get('sift_path') and Path(loc.get('sift_path')).exists():
                story.append(Paragraph("<b>Bukti Pencocokan Fitur (SIFT+RANSAC):</b>", styles['Normal']))
                story.append(PlatypusImage(loc.get('sift_path'), width=520, height=160, kind='proportional'))

            if loc.get('explanations'):
                implications = []
                for exp in loc['explanations'].values():
                    if isinstance(exp, dict) and exp.get('implications'):
                        implications.append(exp['implications'])
                if implications:
                    story.append(Paragraph("<b>Implikasi Forensik:</b>", styles['Normal']))
                    for imp in set(implications): story.append(Paragraph(f"• {imp}", styles['Justify']))
            story.append(Spacer(1, 20))

    story.append(PageBreak())
    
    # --- TAHAP 4 PDF (ENHANCED) ---
    story.append(Paragraph("Tahap 4: Visualisasi & Penilaian Integritas (Enhanced)", styles['h2']))
    story.append(Paragraph("<b>4.1. Analisis Skor Integritas</b>", styles['h3']))
    story.append(Paragraph(f"<b>Skor Integritas Final: {integrity_score}% ({integrity_desc})</b>", styles['SubTitle']))
    
    if result.plots.get('integrity_breakdown') and Path(result.plots['integrity_breakdown']).exists():
        story.append(PlatypusImage(result.plots['integrity_breakdown'], width=520, height=250, kind='proportional'))
    
    story.append(Paragraph("<b>4.2. Hasil Localization Tampering</b>", styles['h3']))
    if result.plots.get('enhanced_localization_map') and Path(result.plots['enhanced_localization_map']).exists():
        story.append(PlatypusImage(result.plots['enhanced_localization_map'], width=520, height=350, kind='proportional'))

    story.append(Paragraph("<b>4.3. Penilaian Pipeline Forensik</b>", styles['h3']))
    pipeline_data = [["<b>Tahap</b>", "<b>Status</b>", "<b>Quality Score</b>", "<b>Catatan</b>"]]
    for stage_id, assessment in result.pipeline_assessment.items():
        issues_text = ", ".join(assessment['issues']) if assessment['issues'] else "Tidak ada masalah"
        pipeline_data.append([
            Paragraph(assessment['nama'], styles['Normal']), Paragraph(assessment['status'].capitalize(), styles['Normal']),
            Paragraph(f"{assessment['quality_score']}%", styles['Normal']), Paragraph(issues_text, styles['Normal'])
        ])
    story.append(Table(pipeline_data, colWidths=[180, 80, 80, 180], style=TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')])))
    story.append(Spacer(1, 12))

    if result.plots.get('anomaly_infographic') and Path(result.plots['anomaly_infographic']).exists():
        story.append(PageBreak())
        story.append(Paragraph("<b>4.4. Infografis Penjelasan Anomali</b>", styles['h3']))
        story.append(PlatypusImage(result.plots['anomaly_infographic'], width=520, height=325, kind='proportional'))

    story.append(PageBreak())
    
    # --- TAHAP 5 PDF ---
    story.append(Paragraph("Tahap 5: Validasi Forensik", styles['h2']))
    avg_pipeline_quality = np.mean([a['quality_score'] for a in result.pipeline_assessment.values()]) if hasattr(result, 'pipeline_assessment') and result.pipeline_assessment else 'N/A'
    validation_data = [
        ["<b>Item Validasi</b>", "<b>Detail</b>"],
        ["File Bukti", Paragraph(f"<code>{Path(result.video_path).name}</code>", styles['Code'])],
        ["Hash Preservasi (SHA-256)", Paragraph(f"<code>{result.preservation_hash}</code>", styles['Code'])],
        ["Waktu Analisis", datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')],
        ["Metodologi Utama", "K-Means, Localization Tampering"],
        ["Metode Pendukung", "ELA, SIFT+RANSAC, SSIM, Optical Flow"],
        ["Pustaka Kunci", "OpenCV, scikit-learn, scikit-image, Pillow, ReportLab"],
        ["Skor Integritas", f"{integrity_score}% ({integrity_desc})"],
        ["Total Anomali", f"{result.summary['total_anomaly']} dari {result.summary['total_frames']} frame"],
        ["Pipeline Quality", f"{avg_pipeline_quality:.1f}%" if isinstance(avg_pipeline_quality, (float, int)) else "N/A"]
    ]
    story.append(Table(validation_data, colWidths=[150, 375], style=TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ])))
    
    story.append(Spacer(1, 24))
    story.append(Paragraph("Kesimpulan", styles['h2']))
    conclusion = f"""Berdasarkan analisis forensik 5 tahap yang telah dilakukan, video "{Path(result.video_path).name}" 
    memiliki skor integritas {integrity_score}% dengan kategori "{integrity_desc}". 
    Sistem telah mendeteksi {len(result.localizations)} peristiwa anomali yang memerlukan perhatian. 
    Metode utama K-Means dan Localization Tampering berhasil mengidentifikasi pola-pola anomali, 
    sementara metode pendukung ELA dan SIFT memberikan validasi tambahan terhadap temuan tersebut."""
    story.append(Paragraph(conclusion, styles['Justify']))

    try:
        doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
        log(f"\n  {Icons.SUCCESS} Laporan PDF berhasil dibuat: {pdf_path.name}")
        result.pdf_report_path = pdf_path
        # ====== [NEW] Metadata Forensics Enhancement ======
        generate_simple_reports(result, out_dir / "reports")
        # ====== [END NEW] ======
    except Exception as e:
        log(f"\n  {Icons.ERROR} Gagal membuat laporan PDF: {e}")
        log(traceback.format_exc())
        result.pdf_report_path = None

# --- END OF FILE ForensikVideo.py (FIXED) ---

