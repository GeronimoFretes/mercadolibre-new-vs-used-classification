from __future__ import annotations
import re, unicodedata
from ast import literal_eval
from collections import Counter, defaultdict
from statistics import mean
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator, TransformerMixin


# =========================
# Utilities (shared)
# =========================

def _norm(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))  # drop accents
    s = s.lower()
    return s

_slug_cache = {}
def _slugify(s):
    if s in _slug_cache:
        return _slug_cache[s]
    if s is None:
        slug = "none"
    else:
        slug = re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')
        slug = slug or "none"
    _slug_cache[s] = slug
    return slug

def _clean_number(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _safe_stat(xs, fn, default=np.nan):
    try:
        return fn(xs) if xs else default
    except Exception:
        return default

def _word_count(s):
    return len(re.findall(r'\w+', s.lower())) if isinstance(s, str) else 0

def _upper_ratio(s):
    if not isinstance(s, str) or not s:
        return 0.0
    letters = re.findall(r'[A-Za-zÁÉÍÓÚÑÜ]', s)
    if not letters:
        return 0.0
    upp = sum(ch.isupper() for ch in letters)
    return upp / len(letters)

def _has_kw(s, pats):
    if not isinstance(s, str): return False
    s = s.lower()
    return any(re.search(p, s) for p in pats)

def _host(url):
    if not isinstance(url, str) or "://" not in url:
        return ""
    return url.split("://", 1)[1].split("/", 1)[0].lower()

def _as_set(L):
    return set(L) if isinstance(L, list) else set()

def _safe_join_tags(x):
    if not isinstance(x, dict):
        return pd.NA

    tags = x.get("tags", None)

    if tags is None:
        return pd.NA

    if isinstance(tags, np.ndarray):
        tags = tags.tolist()

    if isinstance(tags, (list, tuple, set)):
        tags = [str(t) for t in tags if pd.notna(t)]
        return "+".join(tags) if len(tags) > 0 else pd.NA

    if isinstance(tags, str):
        return tags if tags.strip() else pd.NA

    return pd.NA

def _safe_iterable(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return x
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    return []

# =========================
# Base Frozen Transformer
# =========================

class FrozenTransformer:
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "FrozenTransformer":
        return self
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        self.fit(df, y)
        return self.transform(df)
    @property
    def catboost_cats(self) -> List[str]:
        return []

# =========================
# Warranty (your exact logic)
# =========================

NUM_WORDS = {
    "uno":1,"una":1,"un":1,"dos":2,"tres":3,"cuatro":4,"cinco":5,"seis":6,"siete":7,"ocho":8,"nueve":9,"diez":10,
    "once":11,"doce":12,"trece":13,"catorce":14,"quince":15,"veinte":20,"treinta":30,"cuarenta":40,"cincuenta":50,
}

def _to_int_token(tok):
    if tok.isdigit(): return int(tok)
    return NUM_WORDS.get(tok, None)

DUR_PATTERNS = [
    (re.compile(r"\b(\d{1,3})\s*(dia|dias|d)\b"),      1),
    (re.compile(r"\b(\d{1,2})\s*(semana|semanas|sem)\b"), 7),
    (re.compile(r"\b(\d{1,2})\s*(mes|meses|m)\b"),      30),
    (re.compile(r"\b(\d{1,2})\s*(anio|años|año|anos|y)\b"), 365)
]
DUR_WORD_PATTERNS = [
    (re.compile(r"\b(uno|una|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce)\s*(mes|meses|m)\b"), 30),
    (re.compile(r"\b(uno|una|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s*(anio|años|año|anos|y)\b"), 365),
    (re.compile(r"\b(uno|una|un|dos|tres|cuatro)\s*(semana|semanas|sem)\b"), 7),
    (re.compile(r"\b(uno|una|un|dos|tres|cuatro|cinco|seis)\s*(dia|dias|d)\b"), 1),
]

NEG_PAT = re.compile(r"\b(sin\s+garantia|no\s+garantia|no\s+posee\s+garantia|no\s+tiene\s+garantia)\b")
CLAIM_PAT = re.compile(r"\b(garantia|garantiza|garantido|warranty|garantia\s+de)\b")
LIFETIME_PAT = re.compile(r"\b(de\s+por\s+vida|lifetime|garantia\s+vitalicia)\b")

PROV_OFICIAL_PAT = re.compile(r"\b(oficial|fabrica|fabricante|marca)\b")
PROV_TIENDA_PAT  = re.compile(r"\b(tienda|vendedor|local|comercio)\b")
PROV_IMPORT_PAT  = re.compile(r"\b(importador|distribuidor)\b")

INVOICE_PAT = re.compile(r"\b(factura|a|b|c)\b")
LIMITED_PAT = re.compile(r"\b(limita(?:da)?|solo\s+mano\s+de\s+obra|no\s+cubre|solo\s+hardware)\b")
SERVICE_PAT = re.compile(r"\b(servicio\s+tecnico|service)\b")
OFFTOPIC_HINTS = re.compile(r"\b(envio|retiro|whatsapp|consultar|preguntar|oferta|trueque|permuto|ubicacion)\b")

def _parse_warranty_row(raw):
    s = _norm(raw)
    is_empty = (s == "")
    tokens = re.findall(r"\w+", s)
    word_count = len(tokens)

    warr_hits = len(re.findall(r"\b(garantia|garantiza|warranty)\b", s))
    warr_density = warr_hits / max(word_count, 1)

    neg = bool(NEG_PAT.search(s))
    claim = bool(CLAIM_PAT.search(s))
    lifetime = bool(LIFETIME_PAT.search(s))

    durations = []
    for pat, mult in DUR_PATTERNS:
        for m in pat.finditer(s):
            durations.append(int(m.group(1)) * mult)
    for pat, mult in DUR_WORD_PATTERNS:
        for m in pat.finditer(s):
            n = _to_int_token(m.group(1))
            if n is not None:
                durations.append(int(n) * mult)

    dur_days = max(durations) if durations else (365*50 if lifetime else 0)
    dur_days = int(np.clip(dur_days, 0, 365*10))

    prov_oficial = bool(PROV_OFICIAL_PAT.search(s))
    prov_tienda  = bool(PROV_TIENDA_PAT.search(s))
    prov_import  = bool(PROV_IMPORT_PAT.search(s))
    requires_invoice = bool(INVOICE_PAT.search(s))
    limited = bool(LIMITED_PAT.search(s))
    has_service = bool(SERVICE_PAT.search(s))
    offtopic = bool(OFFTOPIC_HINTS.search(s))

    return {
        "warranty__is_empty": is_empty,
        "has_warranty_text": not is_empty,
        "claims_warranty": claim,
        "negates_warranty": neg,
        "ambiguous_warranty": (claim and neg),
        "warranty_days": dur_days,
        "warr_lifetime": lifetime,
        "warr_official": prov_oficial,
        "warr_tienda": prov_tienda,
        "warr_importador": prov_import,
        "warr_requires_invoice": requires_invoice,
        "warr_limited": limited,
        "has_service_center": has_service,
        "warr_keyword_density": float(warr_density),
        "warr_text_len": len(raw) if isinstance(raw, str) else 0,
        "warr_word_count": word_count,
        "warr_is_noise": (warr_density < 0.02) & (~claim) & (~neg) & offtopic
    }

class WarrantyFeatureBuilder(FrozenTransformer):
    def __init__(self, col="warranty"):
        self.col = col
    def transform(self, df):
        col = self.col
        s = df[col].fillna("")
        feat_rows = s.apply(_parse_warranty_row)
        feat_df = pd.DataFrame(list(feat_rows), index=df.index)

        d = feat_df["warranty_days"]
        feat_df["warr_0d"]        = (d == 0)
        feat_df["warr_1_89d"]     = (d.between(1, 89))
        feat_df["warr_3_11m"]     = (d.between(90, 334))
        feat_df["warr_12_23m"]    = (d.between(335, 699))
        feat_df["warr_24m_plus"]  = (d >= 700)

        feat_df["warranty__is_na"] = df[col].isna()

        bool_cols = [c for c in feat_df.columns if feat_df[c].dtype == bool]
        feat_df[bool_cols] = feat_df[bool_cols].astype("uint8")
        feat_df["warranty_days"] = feat_df["warranty_days"].astype("int32")
        feat_df["warr_keyword_density"] = feat_df["warr_keyword_density"].astype("float32")
        feat_df["warr_text_len"] = feat_df["warr_text_len"].astype("int32")
        feat_df["warr_word_count"] = feat_df["warr_word_count"].astype("int32")

        return feat_df

# =========================
# Simple columns (your block)
# =========================

class SimpleColumnFeatures(FrozenTransformer):
    def __init__(self):
        # which textuals we’ll later declare as CatBoost categoricals
        self._cat_cols = [
            "category_level1","currency_id","listing_type_id","buying_mode",
            "status","seller_state","seller_city","shipping_mode","shipping_tags",
            "sub_status","deal_ids","seller_id","video_id","catalog_product_id"
        ]
    def transform(self, df):
        feat = {}

        # seller_id/cat_product_id as objects
        if "seller_id" in df.columns:
            df["seller_id"] = df["seller_id"].astype("object")
        if "catalog_product_id" in df.columns:
            df["catalog_product_id"] = df["catalog_product_id"].astype("object")

        # parent flag
        feat['has_parent_item'] = df['parent_item_id'].notna()

        # category coarse
        feat['category_level1'] = df['category_id'].str.slice(0, 4)

        # times
        feat['time_to_stop_days'] = (df['stop_time'] - df['start_time']).dt.total_seconds().div(86400).clip(lower=0)
        feat['start_hour'] = df['start_time'].dt.hour.astype('int8')
        feat['start_dow']  = df['start_time'].dt.dayofweek.astype('int8')

        # other booleans
        feat['is_official_store'] = df['official_store_id'].notna()
        feat['has_video'] = df['video_id'].notna()
        feat['in_catalog'] = df['catalog_product_id'].notna()

        # price features
        feat['price_log'] = np.log1p(df['price'].clip(lower=0))
        feat['base_price_log'] = np.log1p(df['base_price'].clip(lower=0))
        feat['has_original_price'] = df['original_price'].notna()
        feat['discount_pct'] = ((df['original_price'] - df['price']) / df['original_price']).clip(0,1).fillna(0).astype('float32')
        feat["price_gap"]   = df["price"] - df["base_price"]
        feat["price_ratio"] = np.where(df["base_price"]>0, df["price"]/df["base_price"], np.nan)

        # quantities
        feat['init_qty_log1p'] = np.log1p(df['initial_quantity'].clip(lower=0))
        feat['avail_log1p']    = np.log1p(df['available_quantity'].clip(lower=0))
        feat['avail_is_0']     = (df['available_quantity'] == 0)
        feat['sold_log1p']     = np.log1p(df['sold_quantity'].clip(lower=0))
        feat['sold_ge1']       = (df['sold_quantity'] >= 1)

        # title signals
        title = df['title'].fillna("")
        feat['title_len']        = title.str.len().astype('int16')
        feat['title_words']      = title.apply(_word_count).astype('int16')
        feat['title_upper_ratio']= title.apply(_upper_ratio).astype('float32')
        feat['title_digit_count']= title.str.count(r'\d').astype('int16')

        # keywords
        kw_nuevo = [r'\bnuevo\b', r'\bnew\b']
        kw_usado = [r'\busado\b', r'\bused\b', r'\bsemi\s*nuevo\b']
        kw_warr  = [r'\bgarant[ií]a\b', r'\boficial\b', r'\bf[áa]brica\b']
        feat['kw_nuevo'] = title.apply(lambda s: _has_kw(s, kw_nuevo))
        feat['kw_usado'] = title.apply(lambda s: _has_kw(s, kw_usado))
        feat['kw_garantia'] = title.apply(lambda s: _has_kw(s, kw_warr))

        # thumbnails / permalinks
        feat['thumb_is_https'] = df['secure_thumbnail'].str.startswith('https://')
        feat['thumb_host_is_mlstatic'] = df['secure_thumbnail'].apply(lambda u: 'mlstatic.com' in _host(u))
        feat['perma_host_is_mercadolibre'] = df['permalink'].apply(lambda u: 'mercadolibre' in _host(u))
        feat['perma_len'] = df['permalink'].fillna("").str.len().astype('int16')

        feat_df = pd.DataFrame(feat, index=df.index)
        feat_df['has_video_id'] = df.video_id.notna().astype('uint8')

        # seller address → state/city
        if "seller_address" in df.columns:
            feat_df['seller_state'] = df['seller_address'].apply(lambda x: x['state']['name'] if isinstance(x, dict) and x.get('state', {}).get('name', '') != '' else pd.NA)
            feat_df['seller_city']  = df['seller_address'].apply(lambda x: x['city']['name']  if isinstance(x, dict) and x.get('city', {}).get('name', '')  != '' else pd.NA)

        # keep raw categoricals (CatBoost-friendly)
        keep_raw = df[["seller_id","listing_type_id","buying_mode","category_id","currency_id","status","parent_item_id","official_store_id","video_id","catalog_product_id","thumbnail","title","secure_thumbnail","permalink"]].copy()

        # cast boolean savings
        bool_cols = [c for c in feat_df.columns if feat_df[c].dtype == bool]
        feat_df[bool_cols] = feat_df[bool_cols].astype("uint8")

        # final
        out = pd.concat([feat_df, keep_raw], axis=1)
        return out

    @property
    def catboost_cats(self):
        # We’ll later filter to those present after pruning/alignment
        return [
            "category_level1","currency_id","listing_type_id","buying_mode","status",
            "seller_state","seller_city","seller_id"
        ]

# =========================
# Shipping (nested dict)
# =========================

class ShippingFeatureBuilder(FrozenTransformer):
    def __init__(self, col="shipping"):
        self.col = col
    def transform(self, df):
        col = self.col
        out = pd.DataFrame(index=df.index)
        out['shipping_mode'] = df[col].apply(lambda x: x.get('mode') if isinstance(x, dict) else pd.NA)
        out["shipping_tags"] = df[col].apply(_safe_join_tags)
        out['local_pick_up'] = df[col].apply(lambda x: x.get('local_pick_up') if isinstance(x, dict) else False).astype('uint8')
        out['has_methods']   = df[col].apply(lambda x: x.get('methods', None) is not None if isinstance(x, dict) else False).astype('uint8')
        out['free_shipping_methods'] = df[col].apply(lambda x: '-'.join(str(m['id']) for m in x.get('free_methods')) if isinstance(x, dict) and x.get('free_methods') is not None else pd.NA)
        out['shipping_dimensions']   = df[col].apply(lambda x: x.get('dimensions') if isinstance(x, dict) and x.get('dimensions') is not None else pd.NA)
        return out

    @property
    def catboost_cats(self):
        return ["shipping_mode","shipping_tags","free_shipping_methods","shipping_dimensions"]

# =========================
# Payment methods (freezes IDs/combos from train)
# =========================

def _pm_to_ids(x):
    if x is None or x is pd.NA:
        return []

    if isinstance(x, np.ndarray):
        x = x.tolist()

    if not isinstance(x, (list, tuple)):
        return []

    ids = []
    for d in x:
        if isinstance(d, dict):
            val = d.get("id")
            if val is not None and pd.notna(val):
                ids.append(str(val))

    return ids

def _pm_to_types(x):
    if x is None or x is pd.NA:
        return []

    if isinstance(x, np.ndarray):
        x = x.tolist()

    if not isinstance(x, (list, tuple)):
        return []
    
    tps = []
    for d in x:
        if isinstance(d, dict):
            t = d.get("type")
            if isinstance(t, str):
                tps.append(t)
    return tps

def _entropy_from_list(lst):
    if not lst: return 0.0
    s = pd.Series(lst).value_counts()
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())

def _gini_from_list(lst):
    if not lst: return 0.0
    s = pd.Series(lst).value_counts()
    p = (s / s.sum()).to_numpy(dtype=float)
    return float(1.0 - np.sum(p**2))

def _combo_key_ids(ids):
    return tuple(sorted(set(ids)))

class PaymentMethodsFeatureBuilder(FrozenTransformer):
    def __init__(self, col="non_mercado_pago_payment_methods",
                 min_frac_for_top_id=0.01, max_top_ids=1, top_combo_k=10):
        self.col = col
        self.min_frac_for_top_id = float(min_frac_for_top_id)
        self.max_top_ids = int(max_top_ids)
        self.top_combo_k = int(top_combo_k)
        # frozen on fit:
        self.all_known_ids_: set[str] = set()
        self.top_ids_: List[str] = []
        self.top_combo_keys_: List[tuple] = []

        # constant ID sets
        self.CARD_BRANDS = {"MLAVS","MLAMC","MLAAM","MLADC"}
        self.DEBIT_IDS   = {"MLAVE","MLAMS"}
        self.GENERIC_CREDIT = {"MLAOT"}
        self.ONLINE_FRIENDLY = {"MLATB"} | self.CARD_BRANDS | self.DEBIT_IDS | self.GENERIC_CREDIT
        self.OFFLINE_IDS = {"MLAMO","MLAWC","MLAWT","MLABC","MLACD"}

    def fit(self, df, y=None):
        col = self.col
        ids_series = df[col].apply(_pm_to_ids)
        # known IDs on train
        id_freq = Counter()
        for ids in ids_series:
            id_freq.update(set(ids))
        n = len(df)
        freq_list = [(pid, cnt / n) for pid, cnt in id_freq.items()]
        freq_list.sort(key=lambda x: x[1], reverse=True)
        top_ids = [pid for pid, frac in freq_list if frac >= self.min_frac_for_top_id]
        if len(top_ids) > self.max_top_ids:
            top_ids = top_ids[:self.max_top_ids]
        self.top_ids_ = top_ids
        self.all_known_ids_ = set(id_freq.keys())

        combo_keys = ids_series.apply(_combo_key_ids)
        self.top_combo_keys_ = list(combo_keys.value_counts().head(self.top_combo_k).index)
        return self

    def transform(self, df):
        col = self.col
        ids_col = df[col].apply(_pm_to_ids)
        types_col = df[col].apply(_pm_to_types)

        out = pd.DataFrame(index=df.index)
        out["payment_methods_count"] = df[col].apply(lambda L: len(L) if isinstance(L, list) else 0)
        out["has_any_method"]        = out["payment_methods_count"] > 0
        
        out['mp_in_non_mp_methods'] = df['non_mercado_pago_payment_methods'].apply(lambda x: 'MLAMP' in [method['id'] for method in x]).astype(int)

        # Core flags
        out["has_cash"]           = ids_col.apply(lambda ids: "MLAMO" in set(ids))
        out["has_bank_transfer"]  = ids_col.apply(lambda ids: "MLATB" in set(ids))
        out["has_generic_credit"] = ids_col.apply(lambda ids: "MLAOT" in set(ids))

        # Type counts/shares
        for t in ["C","D","G","N"]:
            out[f"count_type_{t}"] = df[col].apply(lambda L, tt=t: sum(1 for d in _safe_iterable(L) if isinstance(d, dict) and d.get("type")==tt))
        out["count_cards"] = out["count_type_C"] + out["count_type_D"] + out["has_generic_credit"].astype(int)
        out["has_any_card"] = out["count_cards"] > 0
        den = out["payment_methods_count"].replace(0, np.nan)
        for t in ["C","D","G","N"]:
            out[f"share_type_{t}"] = (out[f"count_type_{t}"] / den)

        # Diversity
        out["brands_count"] = ids_col.apply(lambda ids: sum(1 for i in self.CARD_BRANDS if i in set(ids)))
        out["offline_methods_count"] = ids_col.apply(lambda ids: sum(1 for i in set(ids) if i in self.OFFLINE_IDS))
        out["online_methods_count"]  = ids_col.apply(lambda ids: sum(1 for i in set(ids) if i in self.ONLINE_FRIENDLY))
        out["payment_types_entropy"] = types_col.apply(_entropy_from_list)
        out["payment_types_gini"]    = types_col.apply(_gini_from_list)

        out["supports_remote_payment"] = out["has_any_card"] | out["has_bank_transfer"]
        out["requires_meeting_flag"]   = (~out["supports_remote_payment"]) & out["has_any_method"]
        out["only_offline_methods"]    = (out["offline_methods_count"] > 0) & (out["online_methods_count"] == 0)

        # Frozen top IDs one-hots
        for pid in self.top_ids_:
            cname = f"pm_{pid.lower()}"
            out[cname] = ids_col.apply(lambda ids, p=pid: p in set(ids))

        rare_pool = (self.all_known_ids_ & set().union(*ids_col.apply(set))) - set(self.top_ids_)
        out["other_methods_count"] = ids_col.apply(lambda ids, RP=rare_pool: sum(1 for i in set(ids) if i in RP))

        # Frozen top combos one-hots
        combo_keys = ids_col.apply(_combo_key_ids)
        out["pm_has_popular_combo"] = combo_keys.isin(self.top_combo_keys_)
        for key in self.top_combo_keys_:
            name = "pm_combo__" + "+".join(key) if key else "pm_combo__EMPTY"
            if name != "pm_combo__EMPTY":
                out[name] = (combo_keys == key)

        # plus a compact categorical combo key (string) for CatBoost
        out["pm_combo_key_str"] = combo_keys.apply(lambda t: "EMPTY" if not t else "+".join(t))

        # cast bools
        bool_cols = [c for c in out.columns if out[c].dtype == bool]
        if bool_cols:
            out[bool_cols] = out[bool_cols].astype("uint8")
        return out

    @property
    def catboost_cats(self):
        return ["pm_combo_key_str"]

# =========================
# Variations (frozen top names/combos)
# =========================

def _iter_attr_combos(variation):
    ac = variation.get('attribute_combinations', [])
    if not isinstance(ac, list):
        return []
    return [a for a in ac if isinstance(a, dict)]

class VariationsFeatureBuilder(FrozenTransformer):
    def __init__(self, col="variations", top_attr_names_k=8, top_attr_combo_k=10):
        self.col=col
        self.top_attr_names_k=int(top_attr_names_k)
        self.top_attr_combo_k=int(top_attr_combo_k)
        self.top_attr_names_: List[str] = []
        self.top_attr_combos_: List[tuple] = []

    @staticmethod
    def _row_attr_summary(L):
        attr_names = set()
        values_per_attr = defaultdict(set)
        total_attr_items = 0
        for v in _safe_iterable(L):
            if not isinstance(v, dict): 
                continue
            for a in _iter_attr_combos(v):
                aname = a.get('name')
                vname = a.get('value_name')
                if aname:
                    attr_names.add(aname)
                    if vname is not None:
                        values_per_attr[aname].add(vname)
                total_attr_items += 1
        return attr_names, values_per_attr, total_attr_items

    @staticmethod
    def _aw_mean(L):
        pairs=[]
        for v in _safe_iterable(L):
            if not isinstance(v, dict): continue
            p=_clean_number(v.get('price'))
            q=_clean_number(v.get('available_quantity'))
            if not np.isnan(p) and not np.isnan(q) and q>0:
                pairs.append((p,q))
        if not pairs: return np.nan
        s_q=sum(q for _,q in pairs)
        return sum(p*q for p,q in pairs)/s_q if s_q>0 else np.nan

    @staticmethod
    def _price_variation_by_attr(L, target_attr_name):
        buckets=defaultdict(list)
        for v in _safe_iterable(L):
            p=_clean_number(v.get('price'))
            if np.isnan(p): continue
            vals=[a for a in _iter_attr_combos(v) if a.get('name')==target_attr_name]
            for a in vals:
                vn=a.get('value_name')
                if vn is not None:
                    buckets[vn].append(p)
        if not buckets: return np.nan
        price_means=[np.mean(ps) for ps in buckets.values() if ps]
        if len(price_means)<=1: return 0.0
        return float(np.max(price_means)-np.min(price_means))

    def fit(self, df, y=None):
        col=self.col
        tmp=df[col].apply(self._row_attr_summary)
        names_set_col=tmp.apply(lambda t: t[0])
        # top attr names
        attr_name_rowfreq=Counter()
        for s in names_set_col:
            for an in s:
                attr_name_rowfreq[an]+=1
        self.top_attr_names_=[an for an,_ in attr_name_rowfreq.most_common(self.top_attr_names_k)]
        # top name combos
        combo_key=names_set_col.apply(lambda s: tuple(sorted(s)))
        combo_counts=combo_key.value_counts()
        self.top_attr_combos_=list(combo_counts.head(self.top_attr_combo_k).index)
        return self

    def transform(self, df):
        col=self.col
        out = pd.DataFrame(index=df.index)

        out["has_variations"]  = df[col].apply(lambda L: isinstance(L, list) and len(L)!=0)
        out["variation_count"] = df[col].apply(lambda L: len(L) if isinstance(L, list) else 0)

        prices = df[col].apply(lambda L: [x for x in (_clean_number(v.get('price')) for v in _safe_iterable(L) if isinstance(v, dict)) if not np.isnan(x)])
        avails = df[col].apply(lambda L: [x for x in (_clean_number(v.get('available_quantity')) for v in _safe_iterable(L) if isinstance(v, dict)) if not np.isnan(x)])
        solds  = df[col].apply(lambda L: [x for x in (_clean_number(v.get('sold_quantity')) for v in _safe_iterable(L) if isinstance(v, dict)) if not np.isnan(x)])

        out["price_min"] = prices.apply(lambda xs: _safe_stat(xs, min))
        out["price_max"] = prices.apply(lambda xs: _safe_stat(xs, max))
        out["price_mean"] = prices.apply(lambda xs: _safe_stat(xs, mean))
        out["price_median"] = prices.apply(lambda xs: _safe_stat(xs, lambda s: float(np.median(s))))
        out["price_std"] = prices.apply(lambda xs: _safe_stat(xs, lambda s: float(np.std(s, ddof=0))))
        out["price_count_nonnull"] = prices.apply(len)
        out["price_range"] = out["price_max"] - out["price_min"]
        out["price_cv"] = (out["price_std"] / out["price_mean"]).replace([np.inf,-np.inf], np.nan)
        out["price_is_uniform"] = (out["price_range"].fillna(0) == 0) & (out["price_count_nonnull"] > 0)
        out["price_spread_ratio"] = (out["price_max"] / out["price_min"]).replace([np.inf,-np.inf], np.nan)

        out["stock_total_available"] = avails.apply(lambda xs: np.sum(xs) if xs else 0.0)
        out["stock_max_available"]   = avails.apply(lambda xs: _safe_stat(xs, max, 0.0))
        out["stock_min_available"]   = avails.apply(lambda xs: _safe_stat(xs, min, 0.0))
        out["stock_mean_available"]  = avails.apply(lambda xs: _safe_stat(xs, mean, 0.0))
        out["has_oos_variants"]      = df[col].apply(lambda L: any((isinstance(v, dict) and _clean_number(v.get('available_quantity'))==0) for v in _safe_iterable(L)))
        out["any_positive_stock"]    = out["stock_total_available"] > 0

        out["sold_total"] = solds.apply(lambda xs: np.sum(xs) if xs else 0.0)
        out["sold_max"]   = solds.apply(lambda xs: _safe_stat(xs, max, 0.0))
        out["sold_mean"]  = solds.apply(lambda xs: _safe_stat(xs, mean, 0.0))
        out["sold_any"]   = out["sold_total"] > 0

        out["price_mean_weighted_by_avail"] = df[col].apply(self._aw_mean)

        # picture_ids per variation
        pics_counts = df[col].apply(lambda L: [len(v.get('picture_ids', [])) if isinstance(v, dict) and isinstance(v.get('picture_ids'), list) else 0 for v in _safe_iterable(L)])
        out["pictures_total"] = pics_counts.apply(lambda xs: np.sum(xs) if xs else 0)
        out["pictures_mean_per_variation"] = pics_counts.apply(lambda xs: _safe_stat(xs, mean, 0.0))
        out["pictures_max_per_variation"]  = pics_counts.apply(lambda xs: _safe_stat(xs, max, 0))
        out["pictures_min_per_variation"]  = pics_counts.apply(lambda xs: _safe_stat(xs, min, 0))

        out["custom_fields_nonnull_count"] = df[col].apply(lambda L: sum(1 for v in _safe_iterable(L) if isinstance(v, dict) and v.get('seller_custom_field') not in (None,"",np.nan)))
        out["has_any_custom_field"] = out["custom_fields_nonnull_count"] > 0

        # attribute-level summaries
        tmp = df[col].apply(self._row_attr_summary)
        attr_names_set = tmp.apply(lambda t: t[0])
        values_per_attr = tmp.apply(lambda t: t[1])
        out["v_attr_items_total"] = tmp.apply(lambda t: t[2])
        out["v_attr_name_count"]  = attr_names_set.apply(len)
        out["v_has_any_attributes"]= out["v_attr_name_count"] > 0

        # frozen top attr names
        for an in (self.top_attr_names_ or []):
            slug=_slugify(an)
            out[f"v_has_attr__{slug}"] = attr_names_set.apply(lambda s, a=an: a in s)
            uniq = values_per_attr.apply(lambda d, a=an: len(d.get(a, set())))
            out[f"v_attr_{slug}__unique_values_count"] = uniq
            out[f"v_attr_{slug}__has_multiple_values"] = uniq.gt(1)

        out["v_attributes_with_multiple_values_count"] = values_per_attr.apply(lambda d: sum(1 for _,vs in d.items() if len(vs)>1))

        combo_key = attr_names_set.apply(lambda s: tuple(sorted(s)))
        out["v_attr_names_combo_key_str"] = combo_key.apply(lambda t: "EMPTY" if not t else "+".join(_slugify(x) for x in t))
        out["v_has_popular_attr_combo"] = combo_key.isin(self.top_attr_combos_)
        for key in (self.top_attr_combos_ or []):
            cname = ("v_attr_combo__" + "+".join(_slugify(k) for k in key)) if key else "v_attr_combo__EMPTY"
            if cname != "v_attr_combo__EMPTY":
                out[cname] = (combo_key == key)

        # price dispersion per frozen attribute names
        for an in (self.top_attr_names_ or []):
            slug=_slugify(an)
            s = df[col].apply(lambda L, a=an: self._price_variation_by_attr(L, a))
            out[f"v_price_spread_by_attr__{slug}"] = s
            out[f"v_price_varies_by_attr__{slug}"] = s.fillna(0).gt(0)

        # cast bools
        bool_cols=[c for c in out.columns if out[c].dtype==bool]
        if bool_cols:
            out[bool_cols]=out[bool_cols].astype("uint8")
        return out

    @property
    def catboost_cats(self):
        return ["v_attr_names_combo_key_str"]

# =========================
# Attributes (frozen names/values/groups/combos)
# =========================

def _safe_attributes(x, try_parse=True):
    if isinstance(x, list): return x
    if try_parse and isinstance(x, str):
        sx = x.strip()
        if not sx: return []
        try:
            v = literal_eval(sx)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    if pd.isna(x): return np.nan
    return []

def _row_attr_summaries(attr_list):
    if isinstance(attr_list, float) and np.isnan(attr_list):
        return (set(), set(), set(), set(), defaultdict(set), defaultdict(set), 0, 0, 0)

    names, ids = set(), set()
    group_names, group_ids = set(), set()
    values_per_name = defaultdict(set)
    values_per_id   = defaultdict(set)

    nonempty_vals = 0
    empty_vals = 0
    total_items = 0

    for a in _safe_iterable(attr_list):
        if not isinstance(a, dict):
            continue
        aname = a.get('name')
        aid   = a.get('id')
        gname = a.get('attribute_group_name')
        gid   = a.get('attribute_group_id')
        vname = a.get('value_name')

        if aname: names.add(aname)
        if aid:   ids.add(aid)
        if gname: group_names.add(gname)
        if gid:   group_ids.add(gid)

        if aname is not None and vname not in (None, ""):
            values_per_name[aname].add(vname)
        if aid is not None and vname not in (None, ""):
            values_per_id[aid].add(vname)

        if vname not in (None, ""):
            nonempty_vals += 1
        else:
            empty_vals += 1

        total_items += 1

    return (names, ids, group_names, group_ids, values_per_name, values_per_id, nonempty_vals, empty_vals, total_items)

class AttributesFeatureBuilder(FrozenTransformer):
    def __init__(self, col="attributes", top_attr_names_k=12, top_values_per_attr_k=5, top_groups_k=8, top_attr_combo_k=12):
        self.col=col
        self.top_attr_names_k=int(top_attr_names_k)
        self.top_values_per_attr_k=int(top_values_per_attr_k)
        self.top_groups_k=int(top_groups_k)
        self.top_attr_combo_k=int(top_attr_combo_k)
        # frozen
        self.top_attr_names_: List[str] = []
        self.top_values_per_name_: Dict[str, List[str]] = {}
        self.top_groups_: List[str] = []
        self.top_attr_combos_: List[tuple] = []

    def fit(self, df, y=None):
        col=self.col
        row_summary = df[col].apply(_row_attr_summaries)
        names_set_col       = row_summary.apply(lambda t: t[0])
        group_names_set_col = row_summary.apply(lambda t: t[2])
        values_per_name_col = row_summary.apply(lambda t: t[4])

        name_rowfreq = Counter()
        for s in names_set_col:
            for nm in s: name_rowfreq[nm]+=1
        self.top_attr_names_ = [nm for nm,_ in name_rowfreq.most_common(self.top_attr_names_k)]

        values_by_name_counter = {nm: Counter() for nm in self.top_attr_names_}
        for vals_per_name in values_per_name_col:
            for nm in self.top_attr_names_:
                vset = vals_per_name.get(nm, set())
                for v in vset:
                    values_by_name_counter[nm][v]+=1
        self.top_values_per_name_ = {
            nm: [v for v,_ in values_by_name_counter[nm].most_common(self.top_values_per_attr_k)]
            for nm in self.top_attr_names_
        }

        group_rowfreq = Counter()
        for gs in group_names_set_col:
            for g in gs: group_rowfreq[g]+=1
        self.top_groups_ = [g for g,_ in group_rowfreq.most_common(self.top_groups_k)]

        attr_combo_key = names_set_col.apply(lambda s: tuple(sorted(s)))
        self.top_attr_combos_ = list(attr_combo_key.value_counts().head(self.top_attr_combo_k).index)
        return self

    def transform(self, df):
        col=self.col
        base_cols={}
        base_cols["has_attributes"] = df[col].apply(lambda L: isinstance(L, list) and len(L)!=0)

        row_summary = df[col].apply(_row_attr_summaries)
        names_set_col       = row_summary.apply(lambda t: t[0])
        group_names_set_col = row_summary.apply(lambda t: t[2])
        values_per_name_col = row_summary.apply(lambda t: t[4])
        values_per_id_col   = row_summary.apply(lambda t: t[5])
        nonempty_vals_col   = row_summary.apply(lambda t: t[6])
        empty_vals_col      = row_summary.apply(lambda t: t[7])
        total_items_col     = row_summary.apply(lambda t: t[8])

        base_cols["attributes_count"]            = total_items_col
        base_cols["attribute_names_count"]       = names_set_col.apply(len)
        base_cols["attribute_groups_count"]      = group_names_set_col.apply(len)
        base_cols["attr_values_nonempty_count"]  = nonempty_vals_col
        base_cols["attr_values_empty_count"]     = empty_vals_col
        comp = (nonempty_vals_col / total_items_col.replace(0, np.nan)).replace([np.inf,-np.inf], np.nan)
        base_cols["attr_values_completeness_ratio"] = comp.fillna(0.0)

        conflicts = values_per_id_col.apply(lambda d: sum(1 for _,vs in d.items() if len(vs)>1))
        base_cols["attribute_conflicts_count_by_id"] = conflicts
        base_cols["has_attribute_conflicts_by_id"]   = conflicts.gt(0)

        base_df = pd.DataFrame(base_cols, index=df.index)

        attr_cols={}
        for nm in (self.top_attr_names_ or []):
            nslug=_slugify(nm)
            attr_cols[f"has_attr__{nslug}"] = names_set_col.apply(lambda s, nm=nm: nm in s)
            uniq = values_per_name_col.apply(lambda d, nm=nm: len(d.get(nm, set())))
            attr_cols[f"attr_{nslug}__unique_values_count"] = uniq
            attr_cols[f"attr_{nslug}__has_multiple_values"] = uniq.gt(1)
            top_vals = self.top_values_per_name_.get(nm, [])
            valset_series = values_per_name_col.apply(lambda d, nm=nm: d.get(nm, set()))
            for v in top_vals:
                vslug=_slugify(v)
                attr_cols[f"attr_{nslug}__val__{vslug}"] = valset_series.apply(lambda vs, v=v: v in vs)
            if top_vals:
                tv=set(top_vals)
                attr_cols[f"attr_{nslug}__val__other"] = valset_series.apply(lambda vs, tv=tv: (len(vs - tv) > 0))

        group_cols={}
        group_counts_per_row = df[col].apply(
            lambda L: Counter(a.get('attribute_group_name') for a in _safe_iterable(L) if isinstance(a, dict) and a.get('attribute_group_name') is not None)
            if not (isinstance(L, float) and np.isnan(L)) else Counter()
        )
        for g in (self.top_groups_ or []):
            gslug=_slugify(g)
            group_cols[f"has_group__{gslug}"] = group_names_set_col.apply(lambda s, g=g: g in s)
            group_cols[f"group_attr_count__{gslug}"] = group_counts_per_row.apply(lambda c, g=g: int(c.get(g,0)))
        group_df = pd.DataFrame(group_cols, index=df.index)
        group_df = group_df.loc[:, ~group_df.T.duplicated()]


        attr_combo_key = names_set_col.apply(lambda s: tuple(sorted(s)))
        combo_cols={}
        combo_cols["attr_names_combo_key"]   = attr_combo_key.apply(lambda t: "EMPTY" if not t else "+".join(_slugify(x) for x in t))
        combo_cols["has_popular_attr_combo"] = attr_combo_key.isin(self.top_attr_combos_)
        for key in (self.top_attr_combos_ or []):
            cname = "attr_combo__" + "+".join(_slugify(k) for k in key) if key else "attr_combo__EMPTY"
            if cname != "attr_combo__EMPTY":
                combo_cols[cname] = (attr_combo_key == key)

        out = pd.concat(
            [base_df, pd.DataFrame(attr_cols, index=df.index), group_df, pd.DataFrame(combo_cols, index=df.index)],
            axis=1
        )

        bool_cols=[c for c in out.columns if out[c].dtype==bool]
        if bool_cols:
            out[bool_cols]=out[bool_cols].astype("uint8")
        return out

    @property
    def catboost_cats(self):
        return ["attr_names_combo_key"]

# =========================
# Tags (frozen tag set & combos)
# =========================

class TagsFeatureBuilder(FrozenTransformer):
    def __init__(self, col="tags", top_combo_k=8, tag_weights: dict | None = None):
        self.col=col
        self.top_combo_k=int(top_combo_k)
        self.tag_weights = tag_weights
        self.known_tags_: List[str] = []
        self.top_combos_: List[tuple] = []

    def fit(self, df, y=None):
        col=self.col
        tags=set()
        for L in df[col]:
            for t in _safe_iterable(L):
                tags.add(t)
        self.known_tags_=sorted(tags)
        combos = df[col].apply(lambda L: tuple(sorted(set(_safe_iterable(L)))))
        self.top_combos_ = list(combos.value_counts().head(self.top_combo_k).index)
        return self

    def transform(self, df):
        col=self.col
        base={}
        base["has_tags"]   = df[col].apply(lambda L: isinstance(L, list) and len(L)!=0)
        base["tags_count"] = df[col].apply(lambda L: len(L) if isinstance(L, list) else 0)

        tag_cols={}
        # one-hot every known tag from train (small set)
        for t in self.known_tags_:
            tag_cols[f"tag__{t}"] = df[col].apply(lambda L, t=t: (t in set(_safe_iterable(L))))

        # semantic rolls
        has_dragged = (tag_cols.get("tag__dragged_bids_and_visits", pd.Series(False, index=df.index)) |
                       tag_cols.get("tag__dragged_visits", pd.Series(False, index=df.index)))
        tag_cols["tag__has_dragged"] = has_dragged
        has_good = tag_cols.get("tag__good_quality_thumbnail", pd.Series(False, index=df.index))
        has_poor = tag_cols.get("tag__poor_quality_thumbnail", pd.Series(False, index=df.index))
        tag_cols["tag__has_any_thumb_quality"] = has_good | has_poor
        tag_cols["tag__thumb_conflict"] = has_good & has_poor
        tag_cols["thumb_quality_score"] = (has_good.astype(int) - has_poor.astype(int)).astype(np.int8)

        default_weights = {
            "dragged_bids_and_visits": 1.0,
            "dragged_visits": 0.8,
            "good_quality_thumbnail": 0.6,
            "poor_quality_thumbnail": -0.6,
            "free_relist": 0.3,
        }
        W = self.tag_weights if self.tag_weights is not None else default_weights
        def exposure_score(L):
            s=0.0
            for t in set(_safe_iterable(L)): s += float(W.get(t,0.0))
            return s
        tag_cols["tags_exposure_score"] = df[col].apply(exposure_score).astype(np.float32)

        combos = df[col].apply(lambda L: tuple(sorted(set(_safe_iterable(L)))))
        combo_cols={}
        combo_cols["tags_combo_key"] = combos.apply(lambda t: "EMPTY" if not t else "+".join(t))
        combo_cols["tags_has_popular_combo"] = combos.isin(self.top_combos_)
        for key in (self.top_combos_ or []):
            name = "tags_combo__" + "+".join(key) if key else "tags_combo__EMPTY"
            if name != "tags_combo__EMPTY":
                combo_cols[name] = (combos == key)

        out = pd.concat([pd.DataFrame(base, index=df.index),
                         pd.DataFrame(tag_cols, index=df.index),
                         pd.DataFrame(combo_cols, index=df.index)], axis=1)

        bool_cols=[c for c in out.columns if out[c].dtype==bool]
        if bool_cols: out[bool_cols]=out[bool_cols].astype("uint8")
        return out

    @property
    def catboost_cats(self):
        return ["tags_combo_key"]

# =========================
# Descriptions (simple flag)
# =========================

class DescriptionsFeatureBuilder(FrozenTransformer):
    def __init__(self, col="descriptions"):
        self.col=col
    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        out['has_description'] = df[self.col].apply(lambda x: len(x) if isinstance(x, list) else 0).astype('uint8')
        return out

# =========================
# Pictures (your detailed block)
# =========================

_SIZE_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)
def _parse_size(s):
    if not isinstance(s, str): return np.nan, np.nan
    m=_SIZE_RE.match(s)
    if not m: return np.nan, np.nan
    return int(m.group(1)), int(m.group(2))

def _dims_list(pic_list, key="size"):
    out=[]
    for p in _safe_iterable(pic_list):
        if not isinstance(p, dict): continue
        w,h=_parse_size(p.get(key))
        if not (np.isnan(w) or np.isnan(h)): out.append((w,h))
    return out

def _areas(dims): return [w*h for (w,h) in dims]
def _orient_counts(dims, tol=0):
    portrait = sum(1 for (w,h) in dims if h > w + tol)
    landscape= sum(1 for (w,h) in dims if w > h + tol)
    square   = sum(1 for (w,h) in dims if abs(w-h) <= tol)
    return portrait, landscape, square
def _ar_list(dims):
    out=[]
    for (w,h) in dims:
        if h>0: out.append(w/h)
    return out
def _first(dims):
    return dims[0] if dims else (np.nan, np.nan)

class PicturesFeatureBuilder(FrozenTransformer):
    def __init__(self, col="pictures", highres_min_side=1000, large_area=800*800):
        self.col=col
        self.highres_min_side=int(highres_min_side)
        self.large_area=int(large_area)

    def transform(self, df):
        col=self.col
        base={}
        base["has_pictures"]   = df[col].apply(lambda L: isinstance(L, list) and len(L)!=0)
        base["pictures_count"] = df[col].apply(lambda L: len(L) if isinstance(L, list) else 0)

        pic_ids = df[col].apply(lambda L: [p.get("id") for p in _safe_iterable(L) if isinstance(p, dict) and isinstance(p.get("id"), str)])
        base["pictures_unique_ids"] = pic_ids.apply(lambda arr: len(set(arr)))
        base["pictures_dup_rate"]   = ((base["pictures_count"] - base["pictures_unique_ids"]) / base["pictures_count"].replace(0, np.nan)).fillna(0.0)

        dims_size     = df[col].apply(lambda L: _dims_list(L, "size"))
        dims_max_size = df[col].apply(lambda L: _dims_list(L, "max_size"))

        size_ws = dims_size.apply(lambda xs: [w for (w,_) in xs])
        size_hs = dims_size.apply(lambda xs: [h for (_,h) in xs])
        size_as = dims_size.apply(_areas)

        max_ws = dims_max_size.apply(lambda xs: [w for (w,_) in xs])
        max_hs = dims_max_size.apply(lambda xs: [h for (_,h) in xs])
        max_as = dims_max_size.apply(_areas)

        size_cols = {
            "pic_size_width_min":   size_ws.apply(lambda xs: _safe_stat(xs, min)),
            "pic_size_width_max":   size_ws.apply(lambda xs: _safe_stat(xs, max)),
            "pic_size_width_mean":  size_ws.apply(lambda xs: _safe_stat(xs, mean)),
            "pic_size_width_std":   size_ws.apply(lambda xs: _safe_stat(xs, lambda s: float(np.std(s, ddof=0)))),
            "pic_size_height_min":  size_hs.apply(lambda xs: _safe_stat(xs, min)),
            "pic_size_height_max":  size_hs.apply(lambda xs: _safe_stat(xs, max)),
            "pic_size_height_mean": size_hs.apply(lambda xs: _safe_stat(xs, mean)),
            "pic_size_height_std":  size_hs.apply(lambda xs: _safe_stat(xs, lambda s: float(np.std(s, ddof=0)))),
            "pic_size_area_min":    size_as.apply(lambda xs: _safe_stat(xs, min)),
            "pic_size_area_max":    size_as.apply(lambda xs: _safe_stat(xs, max)),
            "pic_size_area_mean":   size_as.apply(lambda xs: _safe_stat(xs, mean)),
            "pic_size_area_std":    size_as.apply(lambda xs: _safe_stat(xs, lambda s: float(np.std(s, ddof=0)))),
        }
        max_cols = {
            "pic_max_width_min":   max_ws.apply(lambda xs: _safe_stat(xs, min)),
            "pic_max_width_max":   max_ws.apply(lambda xs: _safe_stat(xs, max)),
            "pic_max_width_mean":  max_ws.apply(lambda xs: _safe_stat(xs, mean)),
            "pic_max_width_std":   max_ws.apply(lambda xs: _safe_stat(xs, lambda s: float(np.std(s, ddof=0)))),
            "pic_max_height_min":  max_hs.apply(lambda xs: _safe_stat(xs, min)),
            "pic_max_height_max":  max_hs.apply(lambda xs: _safe_stat(xs, max)),
            "pic_max_height_mean": max_hs.apply(lambda xs: _safe_stat(xs, mean)),
            "pic_max_height_std":  max_hs.apply(lambda xs: _safe_stat(xs, lambda s: float(np.std(s, ddof=0)))),
            "pic_max_area_min":    max_as.apply(lambda xs: _safe_stat(xs, min)),
            "pic_max_area_max":    max_as.apply(lambda xs: _safe_stat(xs, max)),
            "pic_max_area_mean":   max_as.apply(lambda xs: _safe_stat(xs, mean)),
            "pic_max_area_std":    max_as.apply(lambda xs: _safe_stat(xs, lambda s: float(np.std(s, ddof=0)))),
        }

        def has_larger_max(L):
            if not isinstance(L, list): return False
            for p in L:
                if not isinstance(p, dict): continue
                w,h=_parse_size(p.get("size")); W,H=_parse_size(p.get("max_size"))
                if all(not np.isnan(x) for x in (w,h,W,H)) and (W*H)>(w*h):
                    return True
            return False

        size_cols["pic_has_larger_max_than_size"] = df[col].apply(has_larger_max)

        orients = dims_size.apply(_orient_counts)
        size_cols["pic_portrait_count"]  = orients.apply(lambda t: t[0])
        size_cols["pic_landscape_count"] = orients.apply(lambda t: t[1])
        size_cols["pic_square_count"]    = orients.apply(lambda t: t[2])

        denom = base["pictures_count"].replace(0, np.nan)
        size_cols["pic_portrait_share"]  = (size_cols["pic_portrait_count"]  / denom).fillna(0.0)
        size_cols["pic_landscape_share"] = (size_cols["pic_landscape_count"] / denom).fillna(0.0)
        size_cols["pic_square_share"]    = (size_cols["pic_square_count"]    / denom).fillna(0.0)

        ars = dims_size.apply(_ar_list)
        size_cols["pic_ar_min"]  = ars.apply(lambda xs: _safe_stat(xs, min))
        size_cols["pic_ar_max"]  = ars.apply(lambda xs: _safe_stat(xs, max))
        size_cols["pic_ar_mean"] = ars.apply(lambda xs: _safe_stat(xs, mean))
        size_cols["pic_ar_std"]  = ars.apply(lambda xs: _safe_stat(xs, lambda s: float(np.std(s, ddof=0))))

        def any_highres(dims, thr): return any(min(w,h) >= thr for (w,h) in dims)
        def any_large_area(dims, thr): return any((w*h) >= thr for (w,h) in dims)
        size_cols["pic_any_highres_min_side"] = dims_size.apply(lambda xs, thr=self.highres_min_side: any_highres(xs, thr))
        size_cols["pic_any_large_area"]       = dims_size.apply(lambda xs, thr=self.large_area: any_large_area(xs, thr))

        first_w, first_h = zip(*dims_size.apply(_first))
        first_w = pd.Series(first_w, index=df.index)
        first_h = pd.Series(first_h, index=df.index)
        size_cols["first_pic_width"]  = first_w
        size_cols["first_pic_height"] = first_h
        size_cols["first_pic_area"]   = (first_w * first_h).replace({np.inf: np.nan})
        size_cols["first_pic_ar"]     = (first_w / first_h).replace([np.inf,-np.inf], np.nan)
        size_cols["first_is_portrait"]  = first_h > first_w
        size_cols["first_is_landscape"] = first_w > first_h
        size_cols["first_is_square"]    = (first_w == first_h) & first_w.notna()

        secure_present = df[col].apply(lambda L: sum(1 for p in _safe_iterable(L) if isinstance(p, dict) and isinstance(p.get("secure_url"), str) and p.get("secure_url","").startswith("https://")))
        plain_present  = df[col].apply(lambda L: sum(1 for p in _safe_iterable(L) if isinstance(p, dict) and isinstance(p.get("url"), str) and p.get("url","").startswith("http://")))
        size_cols["pic_secure_count"] = secure_present.astype("int32")
        size_cols["pic_plain_count"]  = plain_present.astype("int32")
        size_cols["pic_secure_share"] = (secure_present / base["pictures_count"].replace(0, np.nan)).fillna(0.0)

        hosts = df[col].apply(lambda L: {_host(p.get("secure_url") or p.get("url","")) for p in _safe_iterable(L) if isinstance(p, dict)} if not (isinstance(L, float) and np.isnan(L)) else set())
        size_cols["pic_host_unique_count"] = hosts.apply(len)
        size_cols["pic_has_non_mlstatic_host"] = hosts.apply(lambda s: any(("mlstatic.com" not in h) for h in s if h))

        out = pd.concat([pd.DataFrame(base, index=df.index),
                         pd.DataFrame(size_cols, index=df.index),
                         pd.DataFrame(max_cols, index=df.index)], axis=1)

        bool_cols=[c for c in out.columns if out[c].dtype==bool]
        if bool_cols: out[bool_cols]=out[bool_cols].astype("uint8")
        return out

# =========================
# Column aligner & pruner
# =========================

class ColumnAligner(FrozenTransformer):
    def __init__(self):
        self.columns_: List[str] | None = None
        self.drop_cols_: List[str] = []
        self.cat_cols_: List[str] = []  # final cat cols after prune/align (object dtype)
    def fit(self, X: pd.DataFrame, y=None):
        # drop constants & exact duplicates on TRAIN
        nun = X.nunique(dropna=False)
        to_drop = set(nun[nun <= 1].index.tolist())

        dup_map = {}
        seen = {}
        for c in X.columns:
            h = hash_pandas_object(X[c], index=False).values
            key = (str(X[c].dtype), h.tobytes())
            if key in seen:
                dup_map[c] = seen[key]
            else:
                seen[key] = c
        to_drop |= set(dup_map.keys())

        Xp = X.drop(columns=sorted(to_drop)) if to_drop else X
        self.drop_cols_ = sorted(to_drop)
        self.columns_ = list(Xp.columns)

        # final cat columns = object dtype among the kept columns
        self.cat_cols_ = [c for c in self.columns_ if (Xp[c].dtype == "object")]
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X2 = X.drop(columns=self.drop_cols_, errors="ignore")
        if self.columns_ is not None:
            X2 = X2.reindex(columns=self.columns_, fill_value=0)
        return X2

# =========================
# Full pipeline
# =========================

class FeaturePipeline:
    def __init__(self):
        self.blocks: List[FrozenTransformer] = [
            SimpleColumnFeatures(),
            ShippingFeatureBuilder(),
            WarrantyFeatureBuilder(),
            PaymentMethodsFeatureBuilder(),
            VariationsFeatureBuilder(),
            AttributesFeatureBuilder(),
            TagsFeatureBuilder(),
            DescriptionsFeatureBuilder(),
            PicturesFeatureBuilder(),
        ]
        self.aligner = ColumnAligner()
        self.columns_: List[str] | None = None
        self.catboost_cats_: List[str] = []

    def _cast_end(self, X: pd.DataFrame) -> pd.DataFrame:
        # Datetimes (if any leaked) -> int64
        dt_cols = [c for c in X.columns if str(X[c].dtype).startswith("datetime64")]
        for c in dt_cols:
            X[c] = X[c].astype('int64')
        # Objects -> object strings with NA sentinel
        obj_cols = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype).startswith('category')]
        for c in obj_cols:
            X[c] = X[c].astype("string").fillna("__NA__").astype("object")
        # Bools -> uint8
        bool_cols = [c for c in X.columns if X[c].dtype == bool]
        if bool_cols:
            X[bool_cols] = X[bool_cols].astype("uint8")
        return X

    def fit(self, raw_df: pd.DataFrame, y: pd.Series | None = None):
        # Build all blocks (fit/transform)
        blocks = [b.fit_transform(raw_df, y) for b in self.blocks]
        X = pd.concat(blocks, axis=1)
        X = self._cast_end(X)
        # Fit aligner/pruner on train, freeze schema
        Xp = self.aligner.fit_transform(X)
        self.columns_ = list(Xp.columns)

        # Collect CatBoost categoricals: union of block-declared cats ∩ kept columns,
        # plus any remaining 'object' dtype columns after aligner.
        declared = []
        for b in self.blocks:
            declared.extend(getattr(b, "catboost_cats", []) or [])
        declared = [c for c in declared if c in self.columns_]
        # ensure object dtype
        obj_after = [c for c in self.columns_ if Xp[c].dtype == "object"]
        # stable order: columns_ order
        keep = []
        seen = set()
        for c in self.columns_:
            if (c in declared or c in obj_after) and c not in seen:
                keep.append(c); seen.add(c)
        self.catboost_cats_ = keep
        return self

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        blocks = [b.transform(raw_df) for b in self.blocks]
        X = pd.concat(blocks, axis=1)
        X = self._cast_end(X)
        X = self.aligner.transform(X)
        
        s = pd.to_numeric(X['base_price_log'], errors="coerce").dropna()
        _, edges = pd.qcut(s, q=10, retbins=True, duplicates="drop")
        edges = np.asarray(edges, dtype=float)
        inner = edges[1:-1] if edges.size > 2 else np.array([], dtype=float)
        bins = np.concatenate(([-np.inf], inner, [np.inf]))
        X["price_bin10"] = pd.cut(
            pd.to_numeric(X['base_price_log'], errors="coerce"),
            bins=bins,
            include_lowest=True,
            labels=False,
            right=True,
            ordered=True,
        ).astype("int8")
        
        return X
