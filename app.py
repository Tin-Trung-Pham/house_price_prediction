# -*- coding: utf-8 -*-

import io
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Đường dẫn tuyệt đối tới folder chứa app.py -> chạy streamlit từ thư mục nào cũng tìm được file
APP_DIR = Path(__file__).resolve().parent
IMG_PATH = APP_DIR / 'nhatot.jpg'

from house_utils import (load_artifact, dinh_gia_va_kiem_tra, chuan_hoa_csv, du_doan_hang_loat,
                         phat_hien_hang_loat, tao_template)

st.set_page_config(page_title='Dự đoán giá nhà – Nhà Tốt', page_icon='🏠', layout='wide')


TEAM_NAME = 'Nhóm 2'
TEAM_COURSE = 'Đồ án tốt nghiệp Data Science – TTTH ĐH KHTN TP.HCM'

TEAM_MEMBERS = [
    {'name': 'Phạm Trung Tín', 'tool': 'Python',
     'tasks': ['Data Cleaning', 'Khám phá dữ liệu', 'Model, Machine Learning, Evaluation trên Python',
               'File thuyết trình, README, PowerPoint']},
    {'name': 'Đinh Trí Dũng', 'tool': 'PySpark',
     'tasks': ['Data Cleaning', 'Khám phá dữ liệu', 'Model, Machine Learning, Evaluation trên PySpark',
               'File thuyết trình, README, PowerPoint']},
]
INSTRUCTOR = 'Cô Khuất Thuỳ Phương'

QUAN_COLOR = {'Bình Thạnh': '#2a78d6', 'Gò Vấp': '#eb6834', 'Phú Nhuận': '#1baf7a'}   
LABEL_COLOR = {'Giá quá thấp': '#2a78d6', 'Bình thường': '#c3c2b7', 'Giá quá cao': '#e34948'}
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'

CSS = """
<style>
.hero {background: linear-gradient(90deg, #ffba00 0%, #ff8a00 100%); padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;}
.hero h1 {color: #1a1a19; margin: 0; font-size: 2.0rem;}
.hero p  {color: #1a1a19; margin: 6px 0 0 0; font-size: 1.02rem;}
.card {background: #ffffff; border: 1px solid #e1e0d9; border-radius: 12px; padding: 16px 18px; height: 100%;}
.card h4 {margin: 0 0 6px 0;}
.tag {display: inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 0.95rem;}
.tag-ok   {background: #e8f5e8; color: #006300;}
.tag-low  {background: #e6f0fb; color: #184f95;}
.tag-high {background: #fdeaea; color: #a32525;}
div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e1e0d9; border-radius: 12px; padding: 12px 16px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def tag(label):
    cls = {'Bình thường': 'tag-ok', 'Giá quá thấp': 'tag-low', 'Giá quá cao': 'tag-high'}.get(label, 'tag-ok')
    icon = {'Bình thường': '✅', 'Giá quá thấp': '🔻', 'Giá quá cao': '🔺'}.get(label, '')
    return f'<span class="tag {cls}">{icon} {label}</span>'


@st.cache_resource(show_spinner='Đang nạp mô hình...')
def get_model():
    app_dir = Path(__file__).resolve().parent
    return load_artifact(app_dir / 'house_price_model.pkl', app_dir / 'kw_vectorizer.pkl')


def get_options(art):
    cats = art['model'].named_steps['prep'].categories_
    wards = {q: sorted((p for p in cats['phuong'] if p.startswith(q + ' - ')),
                       key=lambda p: int(''.join(ch for ch in p if ch.isdigit()) or 0)) for q in cats['quan']}
    return cats, wards


def read_csv_upload(uploaded):
    data = uploaded.getvalue()
    for enc in ['utf-8-sig', 'cp1258', 'latin-1']:
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc, keep_default_na=True)
        except UnicodeDecodeError:
            continue
    raise ValueError('Không đọc được file – hãy lưu CSV với mã hóa UTF-8.')


def to_csv_bytes(df):
    return df.to_csv(index_label='dong_trong_file', encoding='utf-8-sig').encode('utf-8-sig')


def _style(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)


def plot_price_range(gia, lo, hi, gia_rao=None):
    fig, ax = plt.subplots(figsize=(7, 1.9))
    ax.barh([0], [hi - lo], left=lo, height=0.45, color='#cde2fb', label='Khoảng hợp lý P10–P90')
    ax.axvline(gia, color='#1c5cab', lw=2.5, label=f'Giá dự đoán: {gia:.2f} tỷ')
    if gia_rao is not None:
        c = '#e34948' if gia_rao > hi else ('#2a78d6' if gia_rao < lo else '#52514e')
        ax.axvline(gia_rao, color=c, lw=2.5, ls='--', label=f'Giá rao: {gia_rao:.2f} tỷ')
    xs = [lo, hi, gia] + ([gia_rao] if gia_rao is not None else [])
    ax.set_xlim(min(xs) * 0.6, max(xs) * 1.25); ax.set_ylim(-0.5, 0.6)
    ax.set_yticks([]); ax.set_xlabel('Giá (tỷ VNĐ)', color=MUTED)
    _style(ax); ax.grid(False); ax.spines['left'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def bang_tin_hieu(r, cfg):
    """Bảng giải thích 4 tín hiệu bằng lời: mỗi tín hiệu kiểm tra gì, kết quả, đóng góp bao nhiêu điểm."""
    sig, w = r['tin_hieu'], cfg['weights']
    cl, dg = r['chenh_lech_%'], r['don_gia']
    p10, p90, mn, mx = r['don_gia_P10'], r['don_gia_P90'], r['don_gia_min'], r['don_gia_max']

    # 1) Lệch giá dự đoán
    s = sig['S_resid']
    kq1 = '✅ Đạt' if s < 1 / 3 else ('⚠️ Cảnh báo' if s < 2 / 3 else '🚩 Bất thường')
    mo1 = ('Giá rao gần bằng giá dự đoán' if abs(cl) < 1
           else f'Giá rao {"cao" if cl > 0 else "thấp"} hơn giá dự đoán {abs(cl):.0f}%')
    # 2) Khung giá 
    kq2 = '✅ Đạt' if sig['S_minmax'] == 0 else '🚩 Bất thường'
    mo2 = (f'Đơn giá {dg:.0f} tr/m² nằm trong khung {mn:.0f}–{mx:.0f} tr/m²' if sig['S_minmax'] == 0
           else f'Đơn giá {dg:.0f} tr/m² {"dưới sàn" if dg < mn else "vượt trần"} của khung {mn:.0f}–{mx:.0f} tr/m²')
    # 3) Vùng P10–P90 
    s = sig['S_pct']
    kq3 = '✅ Đạt' if s == 0 else ('⚠️ Cảnh báo' if s < 0.3 else '🚩 Bất thường')
    mo3 = (f'Đơn giá nằm trong vùng phổ biến {p10:.0f}–{p90:.0f} tr/m²' if s == 0
           else f'Đơn giá {"thấp hơn" if dg < p10 else "cao hơn"} vùng phổ biến {p10:.0f}–{p90:.0f} tr/m²')
    # 4) Isolation Forest
    hiem = sig['S_if'] >= cfg['if_q95']
    kq4 = '⚠️ Cảnh báo' if hiem else '✅ Đạt'
    mo4 = 'Tổ hợp đặc điểm hiếm gặp (thuộc 5% lạ nhất)' if hiem else 'Đặc điểm căn nhà phổ biến trong dữ liệu'

    rows = [
        ('Lệch giá dự đoán', 'Giá rao có khác nhiều so với giá mô hình dự đoán?', kq1, mo1, 'S_resid'),
        ('Khung giá nhóm', 'Đơn giá có vượt sàn/trần của nhóm cùng khu vực & loại hình?', kq2, mo2, 'S_minmax'),
        ('Vùng giá phổ biến', 'Đơn giá có nằm trong vùng 80% tin phổ biến (P10–P90) của nhóm?', kq3, mo3, 'S_pct'),
        ('Tổ hợp đặc điểm', 'Bộ đặc điểm (diện tích, tầng, giá...) có lạ so với toàn bộ dữ liệu?', kq4, mo4, 'S_if'),
    ]
    return pd.DataFrame([{'Tín hiệu': t, 'Kiểm tra': q, 'Kết quả': k, 'Chi tiết': m,
                          'Điểm đóng góp': f'{100 * w[key] * sig[key]:.1f} / {100 * w[key]:.0f}'}
                         for t, q, k, m, key in rows])


def plot_pred_by_quan(res):
    """Phân phối giá dự đoán theo quận (boxplot) + số căn mỗi quận."""
    qs = [q for q in QUAN_COLOR if q in set(res['quan'])]
    fig, ax = plt.subplots(1, 2, figsize=(11, 1.4 + 0.75 * len(qs)))
    data = [res.loc[res['quan'] == q, 'gia_du_doan_ty'] for q in qs]
    bp = ax[0].boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=0.5)
    for patch, q in zip(bp['boxes'], qs):
        patch.set_facecolor(QUAN_COLOR[q]); patch.set_alpha(0.85); patch.set_edgecolor('white')
    for m in bp['medians']:
        m.set_color('white'); m.set_linewidth(2)
    ax[0].set_yticks(range(1, len(qs) + 1), qs); ax[0].set_xlabel('Giá dự đoán (tỷ VNĐ, ẩn ngoại lai)', color=MUTED)
    ax[0].set_title('Giá dự đoán theo quận', loc='left', fontsize=11, color=INK)
    _style(ax[0]); ax[0].grid(axis='x', color=GRID); ax[0].grid(axis='y', visible=False)

    dg = [res.loc[res['quan'] == q, 'don_gia_du_doan_trieu_m2'].median() for q in qs]
    bars = ax[1].barh(qs, dg, color=[QUAN_COLOR[q] for q in qs], height=0.4)
    ax[1].margins(y=0.35 if len(qs) == 1 else 0.1)
    ax[1].bar_label(bars, fmt='%.0f', fontsize=9, padding=3, color=INK)
    ax[1].set_xlabel('Triệu/m²', color=MUTED); ax[1].invert_yaxis()
    ax[1].set_title('Đơn giá dự đoán trung vị', loc='left', fontsize=11, color=INK)
    ax[1].set_xlim(0, max(dg) * 1.2)
    _style(ax[1]); ax[1].grid(axis='x', color=GRID); ax[1].grid(axis='y', visible=False)
    fig.tight_layout()
    return fig


def plot_anomaly_overview(res, cutoff):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    lim = res[['gia_du_doan_ty', 'gia_rao_ty']].quantile(.99).max() * 1.1
    for lab in ['Bình thường', 'Giá quá thấp', 'Giá quá cao']:
        s = res[res['phan_loai'] == lab]
        if len(s):
            ax[0].scatter(s['gia_du_doan_ty'], s['gia_rao_ty'], s=12 if lab == 'Bình thường' else 26,
                          color=LABEL_COLOR[lab], alpha=0.5 if lab == 'Bình thường' else 0.9,
                          edgecolor='white', linewidth=0.5, label=f'{lab} ({len(s)})')
    ax[0].plot([0, lim], [0, lim], color=MUTED, lw=1, ls='--')
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel('Giá dự đoán (tỷ)', color=MUTED); ax[0].set_ylabel('Giá rao (tỷ)', color=MUTED)
    ax[0].set_title('Giá rao so với giá dự đoán (đường chéo = khớp)', loc='left', fontsize=11, color=INK)
    ax[0].legend(frameon=False, fontsize=8); _style(ax[0]); ax[0].grid(color=GRID)

    bins = np.linspace(0, 100, 41)
    ax[1].hist(res.loc[res['anomaly_score'] < cutoff, 'anomaly_score'], bins=bins, color='#86b6ef', edgecolor='white', label='Dưới ngưỡng')
    ax[1].hist(res.loc[res['anomaly_score'] >= cutoff, 'anomaly_score'], bins=bins, color='#e34948', edgecolor='white', label='Bị gắn cờ')
    ax[1].set_yscale('log')
    ax[1].axvline(cutoff, color='#e34948', lw=2, ls='--', label=f'Ngưỡng gắn cờ = {cutoff:.1f}')
    ax[1].set_xlabel('Điểm bất thường (0–100)', color=MUTED); ax[1].set_ylabel('Số tin (thang log)', color=MUTED)
    ax[1].set_title('Phân phối điểm bất thường', loc='left', fontsize=11, color=INK)
    ax[1].legend(frameon=False, fontsize=8); _style(ax[1])
    fig.tight_layout()
    return fig



def house_form(key, cats, wards, with_price=False):
    quan = st.selectbox('📍 Quận', sorted(cats['quan']), key=f'{key}_quan')   # ngoài form để danh sách phường đổi theo quận
    with st.form(f'{key}_form', border=True):
        st.markdown('**Vị trí & pháp lý**')
        c1, c2 = st.columns(2)
        phuong = c1.selectbox('Phường', [p.split(' - ', 1)[1] for p in wards[quan]], key=f'{key}_phuong')
        loai_hinh = c2.selectbox('Loại hình', cats['loai_hinh'])
        giay_to = c1.selectbox('Giấy tờ pháp lý', cats['giay_to_phap_ly'])
        dac_diem = c2.selectbox('Đặc điểm nhà/đất', cats['dac_diem'])
        st.markdown('**Kích thước**')
        c1, c2, c3, c4 = st.columns(4)
        dien_tich = c1.number_input('Diện tích đất (m²)', 10.0, 500.0, 45.0, 1.0)
        ngang = c2.number_input('Ngang (m)', 1.0, 50.0, 4.5, 0.1)
        dai = c3.number_input('Dài (m)', 1.0, 100.0, 10.0, 0.5)
        dtsd = c4.number_input('DT sử dụng (0 = không rõ)', 0.0, 3000.0, 0.0, 5.0)
        st.markdown('**Kết cấu & tiện ích**')
        c1, c2, c3 = st.columns(3)
        tang = c1.number_input('Số tầng', 1, 10, 3, 1)
        pn = c2.number_input('Phòng ngủ', 0, 11, 3, 1)
        wc = c3.number_input('Phòng vệ sinh', 0, 11, 3, 1)
        c1, c2 = st.columns(2)
        noi_that = c1.selectbox('Nội thất', cats['tinh_trang_noi_that'])
        huong = c2.selectbox('Hướng cửa chính', cats['huong_cua_chinh'])
        text = st.text_area('Mô tả tin đăng (trích từ khóa: mặt tiền, hẻm xe hơi, thang máy, kinh doanh, nở hậu...)',
                            'hẻm xe hơi, gần mặt tiền, tiện kinh doanh', height=80)
        gia_rao = st.number_input('💵 Giá rao bán (tỷ VNĐ)', 0.3, 600.0, 7.0, 0.1) if with_price else None
        submitted = st.form_submit_button('🔎 Kiểm tra giá' if with_price else '💰 Dự đoán giá',
                                          type='primary', width='stretch')
    house = {'quan': quan, 'phuong': phuong, 'loai_hinh': loai_hinh, 'dien_tich': dien_tich,
             'dien_tich_su_dung': dtsd if dtsd > 0 else np.nan, 'chieu_ngang': ngang, 'chieu_dai': dai,
             'tong_so_tang': tang, 'so_phong_ngu': pn, 'so_phong_ve_sinh': wc, 'giay_to_phap_ly': giay_to,
             'tinh_trang_noi_that': noi_that, 'huong_cua_chinh': huong, 'dac_diem': dac_diem, 'text': text}
    return submitted, house, gia_rao



def upload_block(key, co_gia):
    with st.expander('📄 Định dạng file CSV được chấp nhận & file mẫu', expanded=False):
        st.markdown(f"""
App tự nhận diện **2 định dạng**:
1. **File gốc Nhà Tốt** (giống `quan-*.csv`: có cột `dia_chi`, `dien_tich` dạng "45 m²", `mo_ta`, `bieu_do_gia`...{', `gia_ban` dạng "7,5 tỷ"' if co_gia else ''}).
   App tự lấy quận/phường từ địa chỉ và dùng `bieu_do_gia` làm chỉ số giá khu vực.
2. **Template sạch**: các cột `quan, phuong, loai_hinh, dien_tich, dien_tich_su_dung, chieu_ngang, chieu_dai, tong_so_tang,
   so_phong_ngu, so_phong_ve_sinh, giay_to_phap_ly, tinh_trang_noi_that, huong_cua_chinh, dac_diem, mo_ta`{', **gia_ban (tỷ VNĐ)**' if co_gia else ''}.
   Ô trống được điền "Không rõ"/median như lúc huấn luyện.

Model chỉ áp dụng cho **Bình Thạnh, Gò Vấp, Phú Nhuận**; dòng ngoài phạm vi hoặc thiếu diện tích sẽ bị loại và liệt kê lý do.
""")
        st.download_button('⬇️ Tải template mẫu', tao_template(co_gia).to_csv(index=False).encode('utf-8-sig'),
                           file_name=f"template_{'co_gia' if co_gia else 'khong_gia'}.csv", mime='text/csv', key=f'{key}_tpl')
    return st.file_uploader('Kéo thả hoặc chọn file .csv', type=['csv'], key=f'{key}_upload')


def show_prep_report(loai, canh_bao, bi_loai, n_total, n_ok):
    st.info(f"Định dạng nhận diện: **{'File gốc Nhà Tốt' if loai == 'nhatot_goc' else 'Template sạch'}** · "
            f'{n_total:,} dòng trong file · {n_ok:,} dòng hợp lệ')
    if canh_bao or len(bi_loai):
        with st.expander(f'⚠️ Vấn đề dữ liệu ({len(bi_loai)} dòng bị loại, {len(canh_bao)} cảnh báo)'):
            for c in canh_bao:
                st.write('• ' + c)
            if len(bi_loai):
                st.dataframe(bi_loai.iloc[:, :6], width='stretch')
                st.download_button('⬇️ Tải danh sách dòng bị loại', to_csv_bytes(bi_loai), 'dong_bi_loai.csv', 'text/csv')



def page_intro():
    hero('🏠 Dự đoán giá nhà & Phát hiện giá rao bất thường',
         f'Dữ liệu tin đăng Nhà Tốt · Bình Thạnh, Gò Vấp, Phú Nhuận (TP.HCM) · {TEAM_COURSE}')
    tab1, tab2, tab3 = st.tabs(['📌 Giới thiệu dự án', '🏢 Về Nhà Tốt', '👥 Nhóm thực hiện'])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Tin đăng sau làm sạch', '7.417')
        c2.metric('Quận', '3')
        c3.metric('Đặc trưng đầu vào', '31')
        c4.metric('Chức năng', '2 bài toán')
        st.write('')
        left, right = st.columns([1, 1])
        with left:
            st.markdown("""
<div class="card"><h4>🎯 Vấn đề</h4>
Người bán khó biết nên rao giá bao nhiêu cho hợp lý; người mua khó biết một tin đăng đang rẻ hay đắt so với thị trường;
đội kiểm duyệt cần phát hiện tin có giá bất thường. Dự án xây dựng công cụ <b>định giá tự động</b> và <b>cảnh báo giá bất thường</b>.
</div>""", unsafe_allow_html=True)
            st.write('')
            st.markdown("""
<div class="card"><h4>💰 Bài toán 1 – Dự đoán giá</h4>
Từ vị trí, diện tích, số tầng, pháp lý, từ khóa trong mô tả… dự đoán <b>giá hợp lý (tỷ VNĐ)</b>
kèm khoảng P10–P90 (80% căn thực tế nằm trong khoảng này). Hỗ trợ nhập 1 căn hoặc <b>upload CSV</b> nhiều căn.
</div>""", unsafe_allow_html=True)
            st.write('')
            st.markdown("""
<div class="card"><h4>🚩 Bài toán 2 – Phát hiện giá bất thường</h4>
Chấm <b>điểm 0–100</b> từ 4 tín hiệu (lệch giá dự đoán, khung giá nhóm, vùng P10–P90, Isolation Forest),
gắn nhãn <i>Giá quá thấp / Bình thường / Giá quá cao</i> kèm lý do. Hỗ trợ <b>upload CSV có giá</b>.
</div>""", unsafe_allow_html=True)
        with right:
            st.image(str(IMG_PATH), width='stretch',
                     caption='Ứng dụng của dự án trên nền tảng Nhà Tốt: hỗ trợ người bán, người mua và kiểm duyệt')
        st.markdown('#### 🧭 Cách sử dụng')
        s1, s2, s3 = st.columns(3)
        s1.markdown('<div class="card"><b>1.</b> Chọn <b>Dự đoán giá nhà</b> ở menu bên trái để định giá 1 căn hoặc cả file CSV.</div>', unsafe_allow_html=True)
        s2.markdown('<div class="card"><b>2.</b> Chọn <b>Phát hiện giá bất thường</b> để kiểm tra 1 tin hoặc file CSV có giá rao.</div>', unsafe_allow_html=True)
        s3.markdown('<div class="card"><b>3.</b> Tải kết quả về dạng CSV để phân tích tiếp trong Excel/Power BI.</div>', unsafe_allow_html=True)

    with tab2:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("""
### Nhà Tốt là gì?
**Nhà Tốt** ([nhatot.com](https://www.nhatot.com)) là nền tảng đăng tin **mua bán và cho thuê bất động sản** do **Công ty TNHH Chợ Tốt** vận hành
(trước đây có tên *Chợ Tốt Nhà*). Chợ Tốt là trang rao vặt trực tuyến ra đời năm 2012.

**Các mảng chính**
- Mua bán: căn hộ/chung cư, nhà ở, đất, dự án, văn phòng – mặt bằng kinh doanh.
- Cho thuê, nổi bật là phòng trọ.
- **Biểu đồ giá** theo quận/đường – chính là cột `bieu_do_gia` trong dữ liệu, được dự án dùng làm *chỉ số giá khu vực*.
- Dịch vụ cho môi giới: đẩy tin, tin ưu tiên, gói PRO bất động sản, chuyên trang BĐS.

**Quy mô (theo Nhà Tốt công bố):** hơn **10 triệu người dùng** tìm kiếm bất động sản mỗi tháng; khoảng 80% tin đăng nhận liên hệ trong 12 giờ đầu.
""")
        with right:
            st.markdown("""
<div class="card"><h4>🔗 Liên quan đến dự án</h4>
<ul>
<li>Dữ liệu: tin đăng bán nhà tại 3 quận Bình Thạnh, Gò Vấp, Phú Nhuận.</li>
<li>Mỗi tin có: giá rao, diện tích, kích thước, số tầng, phòng, pháp lý, nội thất, hướng, mô tả, biểu đồ giá khu vực 12 tháng.</li>
<li>Lưu ý: <b>giá rao ≠ giá giao dịch thực tế</b>; dữ liệu là ảnh chụp tại thời điểm thu thập.</li>
</ul></div>""", unsafe_allow_html=True)
            st.write('')
            st.link_button('🌐 Truy cập nhatot.com', 'https://www.nhatot.com', width='stretch')
        st.caption('Nguồn: trang chủ nhatot.com và trang giới thiệu dịch vụ Nhà Tốt trên chotot.com (tra cứu 10/2026). '
                   'Các con số là số liệu do Nhà Tốt tự công bố.')

    with tab3:
        st.markdown(f'### {TEAM_NAME}')
        st.write(f'**Chương trình:** {TEAM_COURSE}  \n**Giảng viên hướng dẫn:** {INSTRUCTOR}')
        cols = st.columns(len(TEAM_MEMBERS))
        for col, m in zip(cols, TEAM_MEMBERS):
            tasks = ''.join(f'<li>{t}</li>' for t in m['tasks'])
            col.markdown(f"""<div class="card"><h4>👤 {m['name']}</h4>
<b>Phụ trách:</b> {m['tool']}<br><b>Công việc:</b><ul style="margin:4px 0 0 0;">{tasks}</ul></div>""", unsafe_allow_html=True)
        st.caption('Sửa biến TEAM_NAME, TEAM_MEMBERS, INSTRUCTOR ở đầu file app.py để cập nhật.')



def page_predict(art, vec, cats, wards):
    hero('💰 Dự đoán giá nhà', 'Định giá 1 căn hoặc cả danh sách căn từ file CSV (không cần cột giá)')
    tab1, tab2 = st.tabs(['🏠 Nhập 1 căn', '📂 Upload CSV nhiều căn'])

    with tab1:
        left, right = st.columns([5, 4], gap='large')
        with left:
            submitted, house, _ = house_form('pred', cats, wards)
        with right:
            st.markdown('#### Kết quả')
            if not submitted:
                st.info('Nhập thông tin căn nhà bên trái rồi bấm **Dự đoán giá**.')
            else:
                r = dinh_gia_va_kiem_tra(house, art, vec)
                lo, hi = r['khoang_P10_P90_ty']
                st.metric('Giá hợp lý', f"{r['gia_hop_ly_ty']:.2f} tỷ")
                c1, c2 = st.columns(2)
                c1.metric('Khoảng P10–P90', f'{lo:.2f} – {hi:.2f} tỷ')
                c2.metric('Đơn giá ước tính', f"{r['gia_hop_ly_ty'] * 1000 / house['dien_tich']:.0f} tr/m²")
                st.pyplot(plot_price_range(r['gia_hop_ly_ty'], lo, hi))
                st.caption('Chỉ số giá khu vực không có trong form nên được điền bằng median của tập train '
                           '(giống notebook). Giá rao ≠ giá giao dịch thực tế.')

    with tab2:
        up = upload_block('pred', co_gia=False)
        if up is not None:
            try:
                raw = read_csv_upload(up)
                d, loai, canh_bao, bi_loai = chuan_hoa_csv(raw)
            except Exception as e:
                st.error(f'Lỗi đọc file: {e}'); return
            show_prep_report(loai, canh_bao, bi_loai, len(raw), len(d))
            if len(d) == 0:
                st.warning('Không có dòng hợp lệ để dự đoán.'); return
            with st.spinner('Đang dự đoán...'):
                res, _ = du_doan_hang_loat(d, art, vec)
            if 'gia_ban' in d and d['gia_ban'].notna().any():
                st.caption('File có cột giá – trang này bỏ qua giá; dùng trang **Phát hiện giá bất thường** để so sánh với giá rao.')

            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Số căn đã dự đoán', f'{len(res):,}')
            c2.metric('Giá dự đoán trung vị', f"{res['gia_du_doan_ty'].median():.2f} tỷ")
            c3.metric('Tổng giá trị ước tính', f"{res['gia_du_doan_ty'].sum():,.0f} tỷ")
            c4.metric('Đơn giá trung vị', f"{res['don_gia_du_doan_trieu_m2'].median():.0f} tr/m²")
            n_unknown = int((~res['phuong_model_biet']).sum())
            if n_unknown:
                st.warning(f'{n_unknown} căn ở phường không nằm trong 30 phường model đã học → dự đoán kém tin cậy hơn '
                           '(cột `phuong_model_biet` = False).')
            st.pyplot(plot_pred_by_quan(res))

            st.markdown('#### Bảng kết quả')
            q_filter = st.multiselect('Lọc theo quận', list(QUAN_COLOR), default=[q for q in QUAN_COLOR if q in set(res['quan'])])
            view = res[res['quan'].isin(q_filter)]
            st.dataframe(view, width='stretch', height=380,
                         column_config={'gia_du_doan_ty': st.column_config.NumberColumn('Giá dự đoán (tỷ)', format='%.2f'),
                                        'gia_P10_ty': st.column_config.NumberColumn('P10 (tỷ)', format='%.2f'),
                                        'gia_P90_ty': st.column_config.NumberColumn('P90 (tỷ)', format='%.2f')})
            st.download_button('⬇️ Tải kết quả dự đoán (CSV)', to_csv_bytes(res), 'ket_qua_du_doan_gia.csv', 'text/csv',
                               type='primary')



def page_anomaly(art, vec, cats, wards):
    hero('🚩 Phát hiện giá rao bất thường', 'So sánh giá rao với giá thị trường – kiểm tra 1 tin hoặc cả file CSV có giá')
    cutoff = art['anomaly_cfg']['cutoff']
    tab1, tab2 = st.tabs(['🔎 Kiểm tra 1 tin', '📂 Upload CSV có giá'])

    with tab1:
        left, right = st.columns([5, 4], gap='large')
        with left:
            submitted, house, gia_rao = house_form('anom', cats, wards, with_price=True)
        with right:
            st.markdown('#### Kết quả')
            if not submitted:
                st.info('Nhập thông tin căn nhà và **giá rao** bên trái rồi bấm **Kiểm tra giá**.')
            else:
                r = dinh_gia_va_kiem_tra(house, art, vec, gia_rao=gia_rao)
                lo, hi = r['khoang_P10_P90_ty']
                st.markdown(tag(r['phan_loai']), unsafe_allow_html=True)
                st.progress(min(int(round(r['diem'])), 100),
                            text=f"Điểm bất thường **{r['diem']:.1f}/100** (ngưỡng gắn cờ {cutoff:.1f})")
                c1, c2 = st.columns(2)
                c1.metric('Giá rao', f'{gia_rao:.2f} tỷ', f"{r['chenh_lech_%']:+.0f}% so với dự đoán", delta_color='off')
                c2.metric('Giá hợp lý', f"{r['gia_hop_ly_ty']:.2f} tỷ")
                c1.metric('Đơn giá rao', f"{r['don_gia']:.0f} tr/m²")
                c2.metric('Vùng đơn giá nhóm', f"{r['don_gia_P10']:.0f}–{r['don_gia_P90']:.0f}")
                st.write(f"**Lý do:** {r['ly_do']}")
                st.pyplot(plot_price_range(r['gia_hop_ly_ty'], lo, hi, gia_rao))
                st.markdown('##### Vì sao có kết quả này? (4 tín hiệu kiểm tra)')
                st.dataframe(bang_tin_hieu(r, art['anomaly_cfg']), hide_index=True, width='stretch')
                st.caption(f"Điểm bất thường = tổng cột *Điểm đóng góp* = **{r['diem']:.1f}/100**. "
                           f"Tin bị gắn cờ khi điểm ≥ {cutoff:.1f} (top 5% điểm cao nhất của dữ liệu huấn luyện).")
                st.caption(f"Nhóm so sánh: {r['nhom_so_sanh']}.")

    with tab2:
        up = upload_block('anom', co_gia=True)
        if up is not None:
            try:
                raw = read_csv_upload(up)
                d, loai, canh_bao, bi_loai = chuan_hoa_csv(raw)
            except Exception as e:
                st.error(f'Lỗi đọc file: {e}'); return
            show_prep_report(loai, canh_bao, bi_loai, len(raw), len(d))
            if 'gia_ban' not in d:
                st.error('File không có cột `gia_ban` → không thể phát hiện bất thường. Dùng trang **Dự đoán giá nhà** cho file không có giá.')
                return
            with st.spinner('Đang chấm điểm bất thường...'):
                res, n_no_price = phat_hien_hang_loat(d, art, vec)
            if n_no_price:
                st.warning(f'{n_no_price} dòng thiếu/sai giá rao → bỏ qua khi chấm điểm.')
            if len(res) == 0:
                st.warning('Không có dòng nào có giá hợp lệ.'); return

            n_high = int((res['phan_loai'] == 'Giá quá cao').sum())
            n_low = int((res['phan_loai'] == 'Giá quá thấp').sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Số tin đã chấm', f'{len(res):,}')
            c2.metric('Tin bất thường', f'{n_high + n_low:,}', f'{(n_high + n_low) / len(res):.1%} số tin', delta_color='off')
            c3.metric('🔺 Giá quá cao', f'{n_high:,}')
            c4.metric('🔻 Giá quá thấp', f'{n_low:,}')
            st.pyplot(plot_anomaly_overview(res, cutoff))
            st.caption('Ngưỡng gắn cờ cố định = top 5% điểm trên dữ liệu huấn luyện. Nếu upload lại chính dữ liệu đã dùng để huấn luyện, '
                       'mô hình đã "thấy" các căn này nên tỷ lệ gắn cờ sẽ thấp hơn 5%.')

            st.markdown('#### Danh sách tin (sắp xếp theo điểm bất thường giảm dần)')
            f1, f2 = st.columns([2, 1])
            loai_chon = f1.multiselect('Lọc theo phân loại', ['Giá quá cao', 'Giá quá thấp', 'Bình thường'],
                                       default=['Giá quá cao', 'Giá quá thấp'])
            q_chon = f2.multiselect('Quận', list(QUAN_COLOR), default=[q for q in QUAN_COLOR if q in set(res['quan'])])
            view = res[res['phan_loai'].isin(loai_chon) & res['quan'].isin(q_chon)]
            st.dataframe(view, width='stretch', height=420,
                         column_config={'anomaly_score': st.column_config.ProgressColumn('Điểm', min_value=0, max_value=100, format='%.1f'),
                                        'chenh_lech_%': st.column_config.NumberColumn('Chênh lệch', format='%+.0f%%')})
            b1, b2 = st.columns(2)
            b1.download_button('⬇️ Tải toàn bộ kết quả (CSV)', to_csv_bytes(res), 'ket_qua_bat_thuong_tat_ca.csv', 'text/csv',
                               type='primary', width='stretch')
            b2.download_button('⬇️ Chỉ tải tin bất thường (CSV)', to_csv_bytes(res[res['phan_loai'] != 'Bình thường']),
                               'danh_sach_bat_thuong.csv', 'text/csv', width='stretch')



art, vec = get_model()
cats, wards = get_options(art)

with st.sidebar:
    st.image(str(IMG_PATH), width='stretch')
    st.markdown('## Menu')
    menu = st.radio('Chọn trang', ['🏠 Giới thiệu', '💰 Dự đoán giá nhà', '🚩 Phát hiện giá bất thường'],
                    label_visibility='collapsed')
    st.divider()
    st.caption(f'{TEAM_NAME}\n\n{TEAM_COURSE}')

if menu.endswith('Giới thiệu'):
    page_intro()
elif menu.endswith('Dự đoán giá nhà'):
    page_predict(art, vec, cats, wards)
else:
    page_anomaly(art, vec, cats, wards)
