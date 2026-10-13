# -*- coding: utf-8 -*-

import io
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler          
from sklearn.ensemble import IsolationForest              


NUM_RAW = ['dien_tich', 'dien_tich_su_dung', 'chieu_ngang', 'chieu_dai',
           'so_phong_ngu', 'so_phong_ve_sinh', 'tong_so_tang']
KEYWORDS = {'mặt tiền': 'tu_mat_tien', 'hẻm xe hơi': 'tu_hem_xe_hoi', 'ô tô': 'tu_o_to', 'hxh': 'tu_hxh',
            'kinh doanh': 'tu_kinh_doanh', 'thang máy': 'tu_thang_may', 'căn góc': 'tu_can_goc',
            'dòng tiền': 'tu_dong_tien', 'nhà nát': 'tu_nha_nat', 'ba gác': 'tu_ba_gac',
            'nở hậu': 'tu_no_hau', 'sân thượng': 'tu_san_thuong'}
kw_cols = list(KEYWORDS.values())
NUM_COLS = NUM_RAW + ['gia_m2_khu_vuc', 'gia_m2_khu_vuc_tb', 'xu_huong_khu_vuc',
                      'dien_tich_san', 'ti_le_ngang_dai'] + kw_cols
CAT_COLS = ['quan', 'phuong', 'loai_hinh', 'giay_to_phap_ly', 'tinh_trang_noi_that', 'huong_cua_chinh', 'dac_diem']

W = {'S_resid': 0.5, 'S_minmax': 0.15, 'S_pct': 0.2, 'S_if': 0.15}
TOP_K, HARD_MIN, HARD_MAX, MIN_GROUP = 0.05, 20, 800, 20
LEVELS = [['phuong', 'loai_hinh'], ['quan', 'loai_hinh'], ['quan']]



class HousePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None, cat_cols=None, top_n=30):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.top_n = top_n

    def fit(self, X, y=None):
        self.medians_ = X[self.num_cols].astype(float).median()
        self.categories_ = {c: X[c].fillna('Không rõ').value_counts().index[:self.top_n].tolist() for c in self.cat_cols}
        return self

    def transform(self, X):
        out = X[self.num_cols].astype(float).fillna(self.medians_)
        for c in self.cat_cols:
            vals = X[c].fillna('Không rõ')
            for cat in self.categories_[c]:
                out[f'{c}={cat}'] = (vals == cat).astype(int)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.num_cols) + [f'{c}={cat}' for c in self.cat_cols for cat in self.categories_[c]])


# ---------------------------------------------------------------- tạo đặc trưng
def make_kw_vectorizer():
    return CountVectorizer(vocabulary=list(KEYWORDS), ngram_range=(1, 3), binary=True,
                           token_pattern=r'(?u)\b\w+\b')


def add_features(d, kw_vectorizer):
    d = d.copy()
    d['dien_tich_san'] = d['dien_tich'] * d['tong_so_tang']
    d['ti_le_ngang_dai'] = d['chieu_ngang'] / d['chieu_dai']
    kw = kw_vectorizer.transform(d['text'].fillna('')).toarray()
    for j, word in enumerate(kw_vectorizer.get_feature_names_out()):
        d[KEYWORDS[word]] = kw[:, j]
    return d


# ---------------------------------------------------------------- phát hiện bất thường (như notebook)
def tra_nhom(frame, tables):
    res = pd.DataFrame(np.nan, index=frame.index, columns=['q1', 'q3', 'p10', 'p90', 'median'])
    nhom = pd.Series(None, index=frame.index, dtype=object)
    for lv, st in tables:
        m = frame[lv].merge(st, on=lv, how='left'); m.index = frame.index
        ok = (m['n'] >= MIN_GROUP) & nhom.isna()
        res.loc[ok] = m.loc[ok, res.columns].astype(float).values
        nhom[ok] = ' × '.join(lv)
    return res, nhom


def ma_tran_if(frame):
    M = frame[NUM_RAW].astype(float).copy()
    M['log_gia'] = np.log(frame['gia_ban'].astype(float))
    M['log_don_gia'] = np.log(frame['don_gia_trieu'].astype(float))
    return M


def cham_diem(frame, pred_log, cfg, gan_co=True):
    out = pd.DataFrame(index=frame.index)
    pred = pd.Series(np.asarray(pred_log, dtype=float), index=frame.index)
    x = np.log(frame['don_gia_trieu'].astype(float))
    e = np.log(frame['gia_ban'].astype(float)) - pred                      # 1) Residual-Z
    z = (e - cfg['resid_mean']) / cfg['resid_std']
    out['gia_ban'], out['gia_du_doan'], out['residual_z'] = frame['gia_ban'], np.exp(pred), z
    out['S_resid'] = np.minimum(z.abs(), 3) / 3
    g, nhom = tra_nhom(frame, cfg['nhom'])                                   # 2) Min/Max
    iqr = g['q3'] - g['q1']
    lo = np.maximum(g['q1'] - 1.5 * iqr, np.log(HARD_MIN)); hi = np.minimum(g['q3'] + 1.5 * iqr, np.log(HARD_MAX))
    out['don_gia'], out['don_gia_min'], out['don_gia_max'] = frame['don_gia_trieu'], np.exp(lo), np.exp(hi)
    out['S_minmax'] = ((x < lo) | (x > hi)).astype(float)
    d = (g['p10'] - x).clip(lower=0) + (x - g['p90']).clip(lower=0)          # 3) P10–P90
    out['don_gia_P10'], out['don_gia_P90'] = np.exp(g['p10']), np.exp(g['p90'])
    out['S_pct'] = (d / cfg['d_max']).clip(0, 1)
    sc = cfg['iforest'].decision_function(cfg['scaler'].transform(ma_tran_if(frame).fillna(cfg['median'])))   # 4) IF
    out['S_if'] = np.clip((cfg['if_max'] - sc) / (cfg['if_max'] - cfg['if_min']), 0, 1)
    out['anomaly_score'] = 100 * sum(w * out[k] for k, w in cfg['weights'].items())
    out['nhom_so_sanh'] = nhom
    out['huong'] = np.where(np.where(z.abs() >= 1, e, x - g['median']) < 0, 'Giá quá thấp', 'Giá quá cao')
    if gan_co:
        out['bat_thuong'] = out['anomaly_score'] >= cfg['cutoff']
        out['phan_loai'] = np.where(out['bat_thuong'], out['huong'], 'Bình thường')
        ratio = out['gia_ban'] / out['gia_du_doan'] - 1
        out['ly_do'] = ['; '.join(t for t, ok in [
            (f'lệch giá dự đoán {r:+.0%} (z={zi:+.1f})', abs(zi) >= 2), ('vi phạm khung giá sàn/trần', mm == 1),
            ('ngoài vùng P10–P90 của nhóm', pc > 0), ('tổ hợp đặc điểm hiếm (IsolationForest)', s_if >= cfg['if_q95'])] if ok)
            for r, zi, mm, pc, s_if in zip(ratio, z, out['S_minmax'], out['S_pct'], out['S_if'])]
    return out



class _NotebookUnpickler(pickle.Unpickler):
    """Pickle trong notebook ghi lớp ở module __main__ -> trỏ về module này."""
    def find_class(self, module, name):
        if module == '__main__':
            module = __name__
        return super().find_class(module, name)


def load_artifact(model_path, vectorizer_path=None):
    with open(model_path, 'rb') as f:
        art = _NotebookUnpickler(f).load()
    vec = None
    if vectorizer_path:
        with open(vectorizer_path, 'rb') as f:
            vec = _NotebookUnpickler(f).load()
    return art, vec


def dinh_gia_va_kiem_tra(house, art, kw_vectorizer, gia_rao=None):
    """Giống hàm cùng tên trong notebook (mục 7.2)."""
    d = pd.DataFrame([house])
    for c in NUM_RAW + ['gia_m2_khu_vuc', 'gia_m2_khu_vuc_tb', 'xu_huong_khu_vuc']:
        d[c] = pd.to_numeric(d[c], errors='coerce') if c in d else np.nan
    for c in CAT_COLS:
        d[c] = d[c].fillna('Không rõ') if c in d else 'Không rõ'
    if not str(d.loc[0, 'phuong']).startswith(str(d.loc[0, 'quan'])):
        d['phuong'] = d['quan'] + ' - ' + d['phuong'].astype(str)
    d['text'] = d['text'].fillna('').str.lower() if 'text' in d else ''
    d = add_features(d, kw_vectorizer)
    pred_log = art['model'].predict(d[art['features']])
    gia = float(np.exp(pred_log[0]))
    out = {'gia_hop_ly_ty': round(gia, 2),
           'khoang_P10_P90_ty': (round(gia * np.exp(art['resid_q'][0]), 2), round(gia * np.exp(art['resid_q'][1]), 2))}
    if gia_rao is not None:
        d['gia_ban'] = float(gia_rao)
        d['don_gia_trieu'] = d['gia_ban'] * 1000 / d['dien_tich']
        s = cham_diem(d, pred_log, art['anomaly_cfg']).iloc[0]
        out.update({'gia_rao_ty': gia_rao, 'chenh_lech_%': round((gia_rao / gia - 1) * 100, 1),
                    'diem': round(float(s['anomaly_score']), 1), 'phan_loai': s['phan_loai'],
                    'ly_do': s['ly_do'] or '—', 'don_gia': float(s['don_gia']),
                    'don_gia_P10': float(s['don_gia_P10']), 'don_gia_P90': float(s['don_gia_P90']),
                    'don_gia_min': float(s['don_gia_min']), 'don_gia_max': float(s['don_gia_max']),
                    'nhom_so_sanh': s['nhom_so_sanh'],
                    'tin_hieu': {k: float(s[k]) for k in ['S_resid', 'S_minmax', 'S_pct', 'S_if']}})
    return out



import re
import ast

QUAN_LIST = ['Bình Thạnh', 'Gò Vấp', 'Phú Nhuận']
TEMPLATE_COLS = ['quan', 'phuong', 'loai_hinh', 'dien_tich', 'dien_tich_su_dung', 'chieu_ngang', 'chieu_dai',
                 'tong_so_tang', 'so_phong_ngu', 'so_phong_ve_sinh', 'giay_to_phap_ly', 'tinh_trang_noi_that',
                 'huong_cua_chinh', 'dac_diem', 'mo_ta']


def parse_price_ty(s):
    
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace('.', '').replace(',', '.')
    m = re.match(r'([\d\.]+)\s*(tỷ|triệu)', s)
    if not m:
        return np.nan
    v = float(m.group(1))
    return v if m.group(2) == 'tỷ' else v / 1000


def parse_number(s):
    
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    s = str(s).lower()
    if 'nhiều hơn' in s:
        m = re.search(r'\d+', s)
        return float(m.group(0)) + 1 if m else np.nan
    m = re.search(r'\d+(?:[.,]\d+)?', s)
    return float(m.group(0).replace(',', '.')) if m else np.nan


def parse_chart(s):
    
    try:
        v = ast.literal_eval(str(s))
        return [float(x) for x in v] if isinstance(v, list) and len(v) else None
    except Exception:
        return None


def lay_phuong(addr):
    if pd.isna(addr) or not str(addr).strip():
        return np.nan
    s = re.sub(r'\s*\(.*\)\s*$', '', str(addr))           
    for p in [x.strip() for x in s.split(',')]:
        if p.startswith('Phường'):
            return p
    return np.nan


def lay_quan(addr):
    s = str(addr)
    for q in QUAN_LIST:                      
        if f'Quận {q}' in s:
            return q
    for q in QUAN_LIST:
        if q in s:
            return q
    return np.nan


def nhan_dien_dinh_dang(df):
    return 'nhatot_goc' if 'dia_chi' in df.columns else 'template'


def chuan_hoa_csv(df_in):
    """Đưa file upload về dạng dữ liệu của model. Trả về (d, loai_file, canh_bao, bi_loai).
    d: các dòng hợp lệ (index = số dòng gốc trong file, bắt đầu từ 1); bi_loai: các dòng bị loại kèm lý do."""
    raw = df_in.copy()
    raw.index = np.arange(1, len(raw) + 1)
    raw.columns = [str(c).strip() for c in raw.columns]
    loai = nhan_dien_dinh_dang(raw)
    d = pd.DataFrame(index=raw.index)
    canh_bao = []

    if loai == 'nhatot_goc':
        d['quan'] = raw['quan'] if 'quan' in raw else raw['dia_chi'].map(lay_quan)
        d['phuong'] = d['quan'].fillna('') + ' - ' + raw['dia_chi'].map(lay_phuong).fillna('Không rõ')
        for c in NUM_RAW:
            d[c] = raw[c].map(parse_number) if c in raw else np.nan
        charts = raw['bieu_do_gia'].map(parse_chart) if 'bieu_do_gia' in raw else pd.Series(None, index=raw.index)
        d['gia_m2_khu_vuc'] = charts.map(lambda v: v[-1] if v else np.nan)
        d['gia_m2_khu_vuc_tb'] = charts.map(lambda v: np.mean(v) if v else np.nan)
        d['xu_huong_khu_vuc'] = charts.map(lambda v: np.mean(v[-3:]) / np.mean(v[:3]) - 1 if v and len(v) >= 6 else np.nan)
        d['text'] = (raw.get('tieu_de', pd.Series('', index=raw.index)).fillna('') + ' ' +
                     raw.get('mo_ta', pd.Series('', index=raw.index)).fillna('')).str.lower()
        if 'gia_ban' in raw:
            d['gia_ban'] = raw['gia_ban'].map(parse_price_ty)
    else:
        d['quan'] = raw['quan'].astype(str).str.replace('Quận ', '', regex=False).str.strip() if 'quan' in raw else np.nan
        ph = raw['phuong'].astype(str).str.strip() if 'phuong' in raw else pd.Series('Không rõ', index=raw.index)
        ph = ph.where(~ph.isin(['', 'nan']), 'Không rõ')
        d['phuong'] = [p if str(p).startswith(str(q) + ' - ') else f'{q} - {p}' for q, p in zip(d['quan'], ph)]
        for c in NUM_RAW + ['gia_m2_khu_vuc', 'gia_m2_khu_vuc_tb', 'xu_huong_khu_vuc']:
            d[c] = pd.to_numeric(raw[c], errors='coerce') if c in raw else np.nan
        text_col = 'text' if 'text' in raw else ('mo_ta' if 'mo_ta' in raw else None)
        d['text'] = raw[text_col].fillna('').astype(str).str.lower() if text_col else ''
        if 'gia_ban' in raw:
            d['gia_ban'] = pd.to_numeric(raw['gia_ban'], errors='coerce')
        thieu = [c for c in TEMPLATE_COLS if c not in raw.columns]
        if thieu:
            canh_bao.append(f'File thiếu cột: {", ".join(thieu)} → điền "Không rõ"/median như lúc huấn luyện.')

    for c in ['loai_hinh', 'giay_to_phap_ly', 'tinh_trang_noi_that', 'huong_cua_chinh', 'dac_diem']:
        d[c] = raw[c].fillna('Không rõ') if c in raw else 'Không rõ'
        d[c] = d[c].replace('', 'Không rõ')

    
    vo_ly = {'tong_so_tang': 10, 'chieu_ngang': 50, 'chieu_dai': 100, 'dien_tich_su_dung': 3000}
    n_vo_ly = 0
    for c, mx in vo_ly.items():
        m = d[c] > mx
        n_vo_ly += int(m.sum()); d.loc[m, c] = np.nan
    if n_vo_ly:
        canh_bao.append(f'{n_vo_ly} giá trị vô lý (vd. > 10 tầng, ngang > 50 m) được coi là thiếu và điền median.')
    n_no_chart = int(d['gia_m2_khu_vuc'].isna().sum())
    if n_no_chart:
        canh_bao.append(f'{n_no_chart} dòng không có chỉ số giá khu vực (bieu_do_gia) → điền median của tập train.')

    
    ly_do = pd.Series('', index=d.index)
    ly_do[~d['quan'].isin(QUAN_LIST)] += 'quận ngoài phạm vi model (chỉ Bình Thạnh, Gò Vấp, Phú Nhuận); '
    ly_do[~(d['dien_tich'] > 0)] += 'thiếu/sai diện tích; '
    key = [c for c in ['gia_ban', 'dien_tich', 'phuong', 'tong_so_tang', 'so_phong_ngu', 'loai_hinh', 'gia_m2_khu_vuc'] if c in d]
    n_trung = int(d.loc[ly_do == ''].duplicated(subset=key).sum())
    if n_trung:
        canh_bao.append(f'{n_trung} dòng có vẻ trùng (cùng căn đăng nhiều lần) – vẫn giữ để dự đoán, bạn nên kiểm tra lại.')
    bi_loai = raw.loc[ly_do != ''].copy()
    bi_loai.insert(0, 'ly_do_loai', ly_do[ly_do != ''].str.rstrip('; '))
    d = d.loc[ly_do == '']
    return d, loai, canh_bao, bi_loai


def du_doan_hang_loat(d, art, kw_vectorizer):
    """Dự đoán giá cho các dòng đã chuẩn hóa. Trả về DataFrame kết quả và log-dự đoán."""
    X = add_features(d, kw_vectorizer)
    pred_log = art['model'].predict(X[art['features']])
    gia = np.exp(pred_log)
    out = pd.DataFrame(index=d.index)
    out['quan'], out['phuong'], out['loai_hinh'] = d['quan'], d['phuong'].str.split(' - ', n=1).str[1], d['loai_hinh']
    out['dien_tich'], out['tong_so_tang'], out['so_phong_ngu'] = d['dien_tich'], d['tong_so_tang'], d['so_phong_ngu']
    out['gia_du_doan_ty'] = gia.round(2)
    out['gia_P10_ty'] = (gia * np.exp(art['resid_q'][0])).round(2)
    out['gia_P90_ty'] = (gia * np.exp(art['resid_q'][1])).round(2)
    out['don_gia_du_doan_trieu_m2'] = (gia * 1000 / d['dien_tich']).round(1)
    known = set(art['model'].named_steps['prep'].categories_['phuong'])
    out['phuong_model_biet'] = d['phuong'].isin(known)
    return out, pred_log


def phat_hien_hang_loat(d, art, kw_vectorizer):
    """Dự đoán + chấm điểm bất thường cho các dòng có giá rao (tỷ VNĐ)."""
    co_gia = d['gia_ban'] > 0 if 'gia_ban' in d else pd.Series(False, index=d.index)
    d = d.loc[co_gia].copy()
    d['don_gia_trieu'] = d['gia_ban'] * 1000 / d['dien_tich']
    out, pred_log = du_doan_hang_loat(d, art, kw_vectorizer)
    s = cham_diem(d, pred_log, art['anomaly_cfg'])
    out.insert(out.columns.get_loc('gia_du_doan_ty'), 'gia_rao_ty', d['gia_ban'].round(2))
    out['chenh_lech_%'] = ((d['gia_ban'] / np.exp(pred_log) - 1) * 100).round(1)
    out['don_gia_rao_trieu_m2'] = s['don_gia'].round(1)
    out['don_gia_P10_nhom'], out['don_gia_P90_nhom'] = s['don_gia_P10'].round(1), s['don_gia_P90'].round(1)
    out['anomaly_score'] = s['anomaly_score'].round(1)
    out['phan_loai'] = s['phan_loai']
    out['ly_do'] = s['ly_do'].replace('', '—')
    return out.sort_values('anomaly_score', ascending=False), int((~co_gia).sum())


def tao_template(co_gia=False):
    rows = [
        {'quan': 'Phú Nhuận', 'phuong': 'Phường 7', 'loai_hinh': 'Nhà ngõ, hẻm', 'dien_tich': 45, 'dien_tich_su_dung': 90,
         'chieu_ngang': 4.5, 'chieu_dai': 10, 'tong_so_tang': 3, 'so_phong_ngu': 3, 'so_phong_ve_sinh': 3,
         'giay_to_phap_ly': 'Đã có sổ', 'tinh_trang_noi_that': 'Không rõ', 'huong_cua_chinh': 'Không rõ',
         'dac_diem': 'Hẻm xe hơi', 'mo_ta': 'hẻm xe hơi, gần mặt tiền, tiện kinh doanh', 'gia_ban': 7.8},
        {'quan': 'Gò Vấp', 'phuong': 'Phường 12', 'loai_hinh': 'Nhà ngõ, hẻm', 'dien_tich': 60, 'dien_tich_su_dung': '',
         'chieu_ngang': 4, 'chieu_dai': 15, 'tong_so_tang': 2, 'so_phong_ngu': 3, 'so_phong_ve_sinh': 2,
         'giay_to_phap_ly': 'Đã có sổ', 'tinh_trang_noi_that': 'Hoàn thiện cơ bản', 'huong_cua_chinh': 'Đông Nam',
         'dac_diem': 'Không rõ', 'mo_ta': 'nhà hẻm ba gác, sổ hồng riêng', 'gia_ban': 2.1},
        {'quan': 'Bình Thạnh', 'phuong': 'Phường 25', 'loai_hinh': 'Nhà mặt phố, mặt tiền', 'dien_tich': 80, 'dien_tich_su_dung': 320,
         'chieu_ngang': 5, 'chieu_dai': 16, 'tong_so_tang': 4, 'so_phong_ngu': 5, 'so_phong_ve_sinh': 5,
         'giay_to_phap_ly': 'Đã có sổ', 'tinh_trang_noi_that': 'Nội thất đầy đủ', 'huong_cua_chinh': 'Tây Bắc',
         'dac_diem': 'Không rõ', 'mo_ta': 'mặt tiền kinh doanh, thang máy', 'gia_ban': 60},
    ]
    t = pd.DataFrame(rows)
    return t if co_gia else t.drop(columns='gia_ban')
