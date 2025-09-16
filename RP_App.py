# Requirements (install with your project venv)
#   pip install streamlit pandas numpy scikit-learn matplotlib openpyxl pillow
# Run
#   streamlit run rp_app.py

import io
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# -------------------- Utility --------------------
def choose_eps_by_rr(D: np.ndarray, rr_target: float = 0.10) -> float:
    """Pick epsilon so that the fraction of pairs with distance < eps ≈ rr_target.
    Uses upper triangle (k=1) to exclude diagonal.
    rr_target: 0.01–0.5 typically.
    """
    tri = D[np.triu_indices_from(D, k=1)]
    if tri.size == 0:
        return float(np.median(D))
    q = np.clip(rr_target, 1e-4, 0.9999)
    return float(np.quantile(tri, q))


def save_rp_to_buffer(R: np.ndarray, img_px: int = 224, invert: bool = False) -> bytes:
    """Render an RP (0/1) to PNG bytes without axes (tight, no padding)."""
    # Prepare image array (0..1 grayscale)
    img = R.astype(float)
    if invert:
        img = 1.0 - img

    fig = plt.figure(figsize=(img_px / 100, img_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed canvas
    ax.imshow(img, cmap="binary", origin="lower", interpolation="nearest", vmin=0, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def standardize_seg(seg: np.ndarray) -> np.ndarray:
    mu = seg.mean(axis=0, keepdims=True)
    sd = seg.std(axis=0, keepdims=True) + 1e-8
    return (seg - mu) / sd


# -------------------- UI --------------------
st.set_page_config(page_title="Recurrence Plot Generator", layout="wide")
st.title("Recurrence Plot (RP) Generator – Excel/CSV 6軸対応")

st.markdown(
    """
**使い方**
1. Excel/CSVをアップロード（Excelは`openpyxl`が必要）
2. カラム（例：`time, x, y, z, roll, pitch, yaw, trial_id`）を確認
3. RPに使うセンサ軸を1〜6本選択（任意）
4. セグメント長とストライドを設定
5. しきい値の決め方（目標RRまたは中央値）を選び、作図
6. ZIPをダウンロード（画像とセグメントCSV）
    """
)

with st.sidebar:
    st.header("入力データ")
    file = st.file_uploader("Excel (.xlsx) または CSV", type=["xlsx", "xls", "csv"]) 
    sheet_name = st.text_input("（Excelのみ）シート名/番号", value="0")
    index_col = st.text_input("インデックス列（任意）", value="")
    time_col = st.text_input("時間列（任意。無ければ空）", value="")

    st.header("セグメント設定")
    seg_size = st.number_input("セグメント長（サンプル数）", min_value=8, max_value=20000, value=170, step=1)
    stride = st.number_input("ストライド（サンプル数）", min_value=1, max_value=20000, value=170, step=1)

    st.header("正規化・出力")
    do_norm = st.checkbox("各セグメントを標準化（z-score）", value=True)
    img_px = st.slider("出力画像サイズ（正方）px", min_value=64, max_value=1024, value=224, step=32)
    invert = st.checkbox("白黒反転（黒=非再帰）", value=False)

    st.header("保存オプション")
    save_local = st.checkbox("ローカルにも保存する", value=False)
    local_dir = st.text_input("保存先フォルダ（例 C:\code\RP-exports）", value="")
    unzip_local = st.checkbox("ZIPを展開して保存", value=True)

    st.header("ε（しきい値）の決定")
    eps_mode = st.radio("方法", ["目標RR（分位点）", "中央値（RR≈50%）", "固定値（手動ε）"], index=0)
    rr = st.slider("目標RR（黒画素率）", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
    eps_manual = st.number_input("固定ε（手動）", min_value=0.0, value=0.0, step=0.0001, format="%f")

    st.header("グルーピング")
    trial_col = st.text_input("trial_id列名（任意）", value="trial_id")
    prefix = st.text_input("ファイル名プレフィックス", value="rp")

# -------------------- Load data --------------------
if file is not None:
    try:
        if file.name.lower().endswith((".xlsx", ".xls")):
            # Excel
            try:
                sheet = int(sheet_name)
            except ValueError:
                sheet = sheet_name
            idx = None if index_col.strip() == "" else index_col.strip()
            df = pd.read_excel(file, sheet_name=sheet, index_col=idx, engine="openpyxl")
        else:
            # CSV
            idx = None if index_col.strip() == "" else index_col.strip()
            df = pd.read_csv(file, index_col=idx)
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        st.stop()

    st.success(f"読み込み成功: {df.shape[0]} 行 × {df.shape[1]} 列")
    with st.expander("プレビュー（先頭10行）"):
        st.dataframe(df.head(10))

    # Column selection for axes
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    st.subheader("RPに使う軸（1〜6列まで）")
    selected_axes = st.multiselect("センサ列を選択", numeric_cols, default=numeric_cols[:2], max_selections=6)
    if len(selected_axes) == 0:
        st.warning("少なくとも1列選んでください。")
        st.stop()

    # Identify optional columns
    has_trial = (trial_col in df.columns)
    has_time = (time_col in df.columns) if time_col.strip() != "" else False

    # Grouping by trial or single group
    if has_trial:
        groups = list(df.groupby(trial_col))
    else:
        groups = [("all", df)]

    st.write(f"グループ数: {len(groups)}  （trial_id列 {'有' if has_trial else '無'}）")

    # Process button
    run = st.button("RP生成・ZIPダウンロードを準備")

    if run:
        # Collect all files to zip
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            total_imgs = 0
            total_csvs = 0

            for trial_val, df_g in groups:
                data_all = df_g[selected_axes].values.astype(float)
                n = len(data_all)
                if n < seg_size:
                    st.info(f"trial={trial_val}: データが短いためスキップ（長さ {n} < セグメント長 {seg_size}）")
                    continue

                # Iterate segments
                seg_idx = 0
                for start in range(0, n - seg_size + 1, int(stride)):
                    seg = data_all[start : start + seg_size]
                    if do_norm:
                        seg = standardize_seg(seg)

                    # Distance matrix & epsilon
                    D = pairwise_distances(seg, metric="euclidean")
                    if eps_mode == "目標RR（分位点）":
                        eps = choose_eps_by_rr(D, rr_target=float(rr))
                    elif eps_mode == "中央値（RR≈50%）":
                        eps = float(np.median(D[np.triu_indices_from(D, k=1)])) if seg_size > 1 else float(np.median(D))
                    else:
                        eps = float(eps_manual)
                        if eps <= 0:
                            # Safety fallback
                            eps = choose_eps_by_rr(D, rr_target=float(rr))

                    R = (D < eps).astype(np.uint8)

                    # Build base name
                    trial_str = str(trial_val)
                    base = f"{prefix}_trial{trial_str}_seg{seg_idx:04d}"

                    # Save PNG
                    png_bytes = save_rp_to_buffer(R, img_px=img_px, invert=invert)
                    zf.writestr(f"images/{base}.png", png_bytes)
                    total_imgs += 1

                    # Save CSV of segment (same columns)
                    seg_df = pd.DataFrame(seg, columns=selected_axes)
                    if has_time:
                        time_vals = df_g.iloc[start : start + seg_size][time_col].values
                        seg_df.insert(0, time_col, time_vals)
                    zf.writestr(f"segments/{base}.csv", seg_df.to_csv(index=False))
                    total_csvs += 1

                    seg_idx += 1

            # Done zipping
        zip_buf.seek(0)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"rp_outputs_{ts}.zip"

        st.success(f"生成完了: 画像 {total_imgs} 枚 / セグメントCSV {total_csvs} 個")
        st.download_button(
            label="ZIPをダウンロード",
            data=zip_buf,
            file_name=zip_name,
            mime="application/zip",
        )

        # Optional local save (server-side)
        if save_local and local_dir.strip():
            try:
                os.makedirs(local_dir, exist_ok=True)
                local_zip = os.path.join(local_dir, zip_name)
                with open(local_zip, "wb") as f:
                    f.write(zip_buf.getvalue())
                if unzip_local:
                    with zipfile.ZipFile(io.BytesIO(zip_buf.getvalue())) as z:
                        z.extractall(local_dir)
                st.info(f"ローカル保存: {local_zip}" + ("（展開済み）" if unzip_local else ""))
            except Exception as e:
                st.warning(f"ローカル保存に失敗: {e}")

        st.caption("ZIPには images/*.png と segments/*.csv が含まれます。画像は軸・ラベル無しの正方グレースケールです。")

    # Live preview (first segment of first group)
    st.subheader("プレビュー（先頭1セグメント）")
    try:
        trial_val, df_g0 = groups[0]
        data0 = df_g0[selected_axes].values.astype(float)
        if len(data0) >= seg_size:
            seg0 = data0[0:seg_size]
            if do_norm:
                seg0 = standardize_seg(seg0)
            D0 = pairwise_distances(seg0, metric="euclidean")
            if eps_mode == "目標RR（分位点）":
                eps0 = choose_eps_by_rr(D0, rr_target=float(rr))
            elif eps_mode == "中央値（RR≈50%）":
                eps0 = float(np.median(D0[np.triu_indices_from(D0, k=1)])) if seg_size > 1 else float(np.median(D0))
            else:
                eps0 = float(eps_manual) if eps_manual > 0 else choose_eps_by_rr(D0, rr_target=float(rr))
            R0 = (D0 < eps0).astype(np.uint8)
            st.write(f"ε(preview) = {eps0:.6f}")
            st.image(save_rp_to_buffer(R0, img_px=img_px, invert=invert))
        else:
            st.info("先頭グループの長さがセグメント長に満たないためプレビューなし")
    except Exception as e:
        st.warning(f"プレビューでエラー: {e}")

else:
    st.info("左のサイドバーからExcel/CSVをアップロードしてください。")
