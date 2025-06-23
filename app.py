import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import procrustes
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image


@st.cache_resource
def initialize_face_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="face_landmarker_v2_with_blendshapes.task"
        ),
        running_mode=VisionRunningMode.IMAGE,
    )
    return FaceLandmarker.create_from_options(options)


def extract_landmarks(image, landmarker):
    try:
        # 画像がRGBの場合はそのまま使用、BGRの場合は変換
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGBと仮定して処理
            image_rgb = image.copy()
        else:
            st.error(f"画像形状が予期しない形式です: {image.shape}")
            return None, "画像形状エラー"

        # データ型をuint8に確実に変換
        if image_rgb.dtype != np.uint8:
            image_rgb = (
                (image_rgb * 255).astype(np.uint8)
                if image_rgb.max() <= 1.0
                else image_rgb.astype(np.uint8)
            )

        # 画像が連続配列であることを保証
        image_rgb = np.ascontiguousarray(image_rgb)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            points = np.array(
                [
                    [lm.x * image.shape[1], lm.y * image.shape[0], lm.z]
                    for lm in landmarks
                ]
            )
            return points, None
        else:
            return (
                None,
                f"顔が検出されませんでした。検出された顔の数: {len(result.face_landmarks) if result.face_landmarks else 0}",
            )

    except Exception as e:
        st.error(f"ランドマーク抽出中にエラー: {str(e)}")
        return None, f"エラー: {str(e)}"


def calculate_procrustes_similarity(landmarks1, landmarks2):
    mtx1, mtx2, disparity = procrustes(landmarks1, landmarks2)
    return disparity


def draw_landmarks_on_image(image, landmarks):
    if landmarks is None:
        return image

    annotated_image = image.copy()
    height, width = image.shape[:2]

    # ランドマークのサイズを画像サイズに応じて調整
    point_size = max(1, min(width, height) // 200)

    # MediaPipeの顔ランドマーク接続情報（主要な顔の輪郭）
    connections = [
        # 顔の輪郭 (0-16)
        [(i, i + 1) for i in range(16)],
        # 左眉毛 (17-21)
        [(i, i + 1) for i in range(17, 21)],
        # 右眉毛 (22-26)
        [(i, i + 1) for i in range(22, 26)],
        # 鼻筋 (27-30)
        [(i, i + 1) for i in range(27, 30)],
        # 鼻の下部 (31-35)
        [(i, i + 1) for i in range(31, 35)],
        # 左目 (36-41)
        [(i, i + 1) for i in range(36, 41)] + [(41, 36)],
        # 右目 (42-47)
        [(i, i + 1) for i in range(42, 47)] + [(47, 42)],
        # 外唇 (48-59)
        [(i, i + 1) for i in range(48, 59)] + [(59, 48)],
        # 内唇 (60-67)
        [(i, i + 1) for i in range(60, 67)] + [(67, 60)],
    ]

    # 線を描画（MediaPipeの全478点では複雑すぎるので、主要な68点のみ表示）
    if len(landmarks) >= 68:
        for connection_group in connections:
            for start_idx, end_idx in connection_group:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (
                        int(landmarks[start_idx][0]),
                        int(landmarks[start_idx][1]),
                    )
                    end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                    cv2.line(annotated_image, start_point, end_point, (0, 255, 255), 1)

    # 全ポイントを描画
    for i, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])

        # 重要なランドマークは大きく表示
        if i < 68:  # 主要な68点
            if i in [36, 39, 42, 45]:  # 目の角
                cv2.circle(
                    annotated_image, (x, y), point_size + 1, (255, 0, 0), -1
                )  # 青
            elif i in [48, 54]:  # 口の角
                cv2.circle(
                    annotated_image, (x, y), point_size + 1, (0, 0, 255), -1
                )  # 赤
            elif i in [30]:  # 鼻の先端
                cv2.circle(
                    annotated_image, (x, y), point_size + 1, (255, 255, 0), -1
                )  # シアン
            else:
                cv2.circle(annotated_image, (x, y), point_size, (0, 255, 0), -1)  # 緑
        else:  # その他の詳細ポイント
            cv2.circle(
                annotated_image, (x, y), max(1, point_size // 2), (0, 255, 0), -1
            )  # 小さい緑

    # ランドマーク数を画像に表示
    cv2.putText(
        annotated_image,
        f"Points: {len(landmarks)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return annotated_image


def main():
    st.set_page_config(
        page_title="顔形状類似度分析アプリ",
        layout="wide",
        page_icon="🎭",
        initial_sidebar_state="collapsed",
    )

    # カスタムCSS
    st.markdown(
        """
    <style>
    .main {
        padding-top: 2rem;
    }
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .app-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .app-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    .result-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    .winner-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(17,153,142,0.3);
        margin: 1rem 0;
    }
    .progress-indicator {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 4px;
        border-radius: 2px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }
    .feature-badge {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 600;
        color: #8b4513;
        margin: 0.2rem;
    }
    .stHorizontalBlock > div:nth-child(1),
    .stHorizontalBlock > div:nth-child(2) {
        background: linear-gradient(135deg, #ebf5ff 0%, #f2fafc 100%);
        padding: 1rem;
        border-radius: 10px;
    }
    .stHorizontalBlock > div:nth-child(3) {
        background: linear-gradient(135deg, #fff5e6 0%, #fffaeb 100%);
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # アプリヘッダー
    st.markdown(
        """
    <div class="app-header">
        <h1 class="app-title">🎭 顔形状類似度分析アプリ</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # モード選択セクション
    st.markdown("### 🎯 分析モード選択")
    mode = st.selectbox(
        "", ["AI自動解析モード", "手動注釈モード"], label_visibility="collapsed"
    )

    # モードの説明
    if "自動" in mode:
        st.info(
            "💡 **AI自動解析モード**: MediaPipeを使用して顔の特徴点を自動検出し、高精度な類似度分析を実行します。"
        )
    else:
        st.info(
            "💡 **手動注釈モード**: 手動で特徴点を指定して、カスタマイズされた類似度分析を実行します。"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("### 📸 基準画像 (人物A)")
        st.markdown("**メイン参照として使用される画像**")
        uploaded_base = st.file_uploader(
            "基準画像をアップロード",
            type=["jpg", "jpeg", "png"],
            key="base",
            help="比較の基準となる人物Aの画像を選択してください",
        )
        if uploaded_base:
            preview_img = Image.open(uploaded_base)
            st.image(
                preview_img, caption="✅ アップロード完了", use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🔄 比較画像1 (人物A)")
        st.markdown("**同一人物の別の写真**")
        uploaded_comp1 = st.file_uploader(
            "比較画像1をアップロード",
            type=["jpg", "jpeg", "png"],
            key="comp1",
            help="人物Aの別角度・別表情の画像を選択してください",
        )
        if uploaded_comp1:
            preview_img = Image.open(uploaded_comp1)
            st.image(
                preview_img, caption="✅ アップロード完了", use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("### 🆚 比較画像2 (人物B)")
        st.markdown("**類似度を検証したい別人物**")

        input_method = st.radio(
            "入力方法を選択",
            ["📁 ファイルアップロード", "📷 カメラキャプチャ"],
            key="input_method",
            horizontal=True,
        )

        uploaded_comp2 = None
        camera_image = None

        if "ファイル" in input_method:
            uploaded_comp2 = st.file_uploader(
                "比較画像2をアップロード",
                type=["jpg", "jpeg", "png"],
                key="comp2",
                help="人物Bの画像を選択してください",
            )
            if uploaded_comp2:
                preview_img = Image.open(uploaded_comp2)
                st.image(
                    preview_img, caption="✅ アップロード完了", use_container_width=True
                )
        else:
            st.markdown("#### 📷 リアルタイム撮影")
            st.markdown(
                """
            <div style="background: #f0f9ff; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <strong>📷 撮影のコツ:</strong><br>
                💡 明るい場所で撮影<br>
                👤 顔が正面を向く<br>
                📏 適度な距離を保つ
            </div>
            """,
                unsafe_allow_html=True,
            )

            camera_image = st.camera_input("📸 写真を撮影", key="camera")

            if camera_image is not None:
                uploaded_comp2 = camera_image
                st.success("✅ 撮影完了！")
                preview_image = Image.open(camera_image)
                st.image(preview_image, caption="📸 撮影画像", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if "自動" in mode:
        auto_analysis_mode(
            uploaded_base, uploaded_comp1, uploaded_comp2, col1, col2, col3
        )
    else:
        manual_annotation_mode(
            uploaded_base, uploaded_comp1, uploaded_comp2, col1, col2, col3
        )


def auto_analysis_mode(uploaded_base, uploaded_comp1, uploaded_comp2, col1, col2, col3):
    if uploaded_base and uploaded_comp1 and uploaded_comp2:
        # プログレスバーと処理状況
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 AI解析エンジンを初期化中...")
        progress_bar.progress(10)
        landmarker = initialize_face_landmarker()

        status_text.text("📸 画像を前処理中...")
        progress_bar.progress(30)

        base_image = np.array(Image.open(uploaded_base).convert("RGB"))
        comp1_image = np.array(Image.open(uploaded_comp1).convert("RGB"))
        comp2_image = np.array(Image.open(uploaded_comp2).convert("RGB"))

        # 処理結果セクション
        status_text.text("🤖 基準画像(人物A)を解析中...")
        progress_bar.progress(50)
        base_landmarks, base_error = extract_landmarks(base_image, landmarker)

        status_text.text("🤖 比較画像1(人物A)を解析中...")
        progress_bar.progress(70)
        comp1_landmarks, comp1_error = extract_landmarks(comp1_image, landmarker)

        status_text.text("🤖 比較画像2(人物B)を解析中...")
        progress_bar.progress(90)
        comp2_landmarks, comp2_error = extract_landmarks(comp2_image, landmarker)

        status_text.text("✅ 解析完了！")
        progress_bar.progress(100)

        # エラー表示（改良版）
        errors = []
        if base_error:
            errors.append(f"📸 基準画像(人物A): {base_error}")
        if comp1_error:
            errors.append(f"📸 比較画像1(人物A): {comp1_error}")
        if comp2_error:
            errors.append(f"📸 比較画像2(人物B): {comp2_error}")

        if errors:
            st.markdown("### ⚠️ 検出エラー")
            for error in errors:
                st.error(error)
            st.markdown(
                """
            <div style="background: #fff3cd; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <strong>💡 改善のヒント:</strong><br>
                • 顔が画像の中央に明確に写っているか確認<br>
                • 十分な明るさがあるか確認<br>
                • 顔が正面または斜め45度以内を向いているか確認<br>
                • 画像解像度が300x300ピクセル以上あるか確認
            </div>
            """,
                unsafe_allow_html=True,
            )

        if (
            base_landmarks is not None
            and comp1_landmarks is not None
            and comp2_landmarks is not None
        ):

            # アノテーション付き画像を生成
            base_annotated = draw_landmarks_on_image(base_image, base_landmarks)
            comp1_annotated = draw_landmarks_on_image(comp1_image, comp1_landmarks)
            comp2_annotated = draw_landmarks_on_image(comp2_image, comp2_landmarks)

            # ランドマーク検出結果表示
            # 類似度計算
            similarity1 = calculate_procrustes_similarity(
                base_landmarks, comp1_landmarks
            )
            similarity2 = calculate_procrustes_similarity(
                base_landmarks, comp2_landmarks
            )

            # 結果表示セクション
            st.markdown("---")
            st.markdown("### 📊 類似度分析結果")

            difference = abs(similarity1 - similarity2)

            # 勝者の発表（改良版）
            if similarity1 < similarity2:
                winner = "比較画像1(人物A)"
                winner_score = similarity1
                st.markdown(
                    f"""
                <div class="winner-card">
                    <h3>🏆 分析結果</h3>
                    <h2>🔄 {winner}</h2>
                    <p>が基準画像により類似しています</p>
                    <p><strong>スコア差: {difference:.4f}</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                winner = "比較画像2(人物B)"
                winner_score = similarity2
                st.markdown(
                    f"""
                <div class="winner-card">
                    <h3>🏆 分析結果</h3>
                    <h2>🆚 {winner}</h2>
                    <p>が基準画像により類似しています</p>
                    <p><strong>スコア差: {difference:.4f}</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # 詳細比較表示
            detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(
                4, gap="medium"
            )

            with detail_col1:
                st.markdown("#### 📸 基準画像")
                st.image(base_annotated, caption="基準", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with detail_col2:
                st.markdown('<div class="">', unsafe_allow_html=True)
                st.markdown("#### 🔄 比較画像1")
                st.image(
                    comp1_annotated,
                    caption=f"類似度: {similarity1:.4f}",
                    use_container_width=True,
                )
                if winner == "比較画像1(人物A)":
                    st.success("🏆 より類似")
                else:
                    st.info("📊 類似度低")
                st.metric(
                    label="🔄 基準 vs 比較1",
                    value=f"{similarity1:.4f}",
                    help="プロクラステス不一致度（値が小さいほど類似）",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with detail_col3:
                st.markdown("#### 🆚 比較画像2")
                st.image(
                    comp2_annotated,
                    caption=f"類似度: {similarity2:.4f}",
                    use_container_width=True,
                )
                if winner == "比較画像2(人物B)":
                    st.success("🏆 より類似")
                else:
                    st.info("📊 類似度低")
                st.metric(
                    label="🆚 基準 vs 比較2",
                    value=f"{similarity2:.4f}",
                    help="プロクラステス不一致度（値が小さいほど類似）",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with detail_col4:
                st.markdown("#### 📈 分析サマリー")
                st.write(f"**🏆 最類似:** {winner}")
                st.write(f"**📊 スコア:** {winner_score:.4f}")
                st.write(f"**🎯 検出点数:** {len(base_landmarks)}点")
                st.write("**⚡ 処理:** 正常完了")
                st.write(f"**📏 類似度差:** {abs(similarity1 - similarity2):.4f}")

                # プロクラステス解析の説明
                with st.expander("📚 解析手法について"):
                    st.markdown(
                        """
                    **プロクラステス解析**

                    📏 **原理:**
                    - 2つの形状の位置・回転・スケールを正規化
                    - 純粋な形状の違いのみを測定
                    - 統計的に信頼性の高い手法
                    
                    📊 **スコア解釈:**
                    - `0.00-0.05`: 🟢 非常に類似
                    - `0.05-0.15`: 🔵 類似
                    - `0.15-0.30`: 🟡 やや類似
                    - `0.30以上`: 🔴 類似度低
                    """
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # プログレスバーとステータスをクリア
        progress_bar.empty()
        status_text.empty()

    else:
        st.markdown(
            """
        <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
            <h3>📸 画像をアップロードして開始</h3>
            <p>3つの画像をすべてアップロードすると、AI解析が自動で開始されます</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def manual_annotation_mode(
    uploaded_base, uploaded_comp1, uploaded_comp2, col1, col2, col3
):
    # セッション状態の初期化
    if "manual_points" not in st.session_state:
        st.session_state.manual_points = {"base": [], "comp1": [], "comp2": []}
    if "current_point_index" not in st.session_state:
        st.session_state.current_point_index = 0
    if "current_image_step" not in st.session_state:
        st.session_state.current_image_step = 0  # 0: base, 1: comp1, 2: comp2

    if uploaded_base and uploaded_comp1 and uploaded_comp2:
        images = {
            "base": np.array(Image.open(uploaded_base).convert("RGB")),
            "comp1": np.array(Image.open(uploaded_comp1).convert("RGB")),
            "comp2": np.array(Image.open(uploaded_comp2).convert("RGB")),
        }

        image_names = [
            "📸 基準画像(人物A)",
            "🔄 比較画像1(人物A)",
            "🆚 比較画像2(人物B)",
        ]
        image_keys = ["base", "comp1", "comp2"]

        # 現在のポイント数を確認
        total_points = len(st.session_state.manual_points["base"])

        # 手動アノテーションヘッダー
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("### 🖱️ 手動特徴点アノテーション")

        # プログレス表示
        current_step = st.session_state.current_image_step
        current_name = image_names[current_step]

        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
            <h4>ステップ {total_points + 1}</h4>
            <p><strong>次にクリック:</strong> {current_name}</p>
            <div class="progress-indicator" style="width: {((total_points * 3 + current_step) / 15) * 100}%;"></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # 3つの画像を横並びで表示
        annotation_col1, annotation_col2, annotation_col3 = st.columns(3, gap="large")

        # 画像表示とクリック処理
        for i, (key, name) in enumerate(zip(image_keys, image_names)):
            with [annotation_col1, annotation_col2, annotation_col3][i]:

                plotted_image = draw_manual_points(
                    images[key], st.session_state.manual_points[key]
                )

                if i == current_step:
                    st.success(f"🎯 {name} (クリック対象)")
                    # クリック可能な画像
                    display_width = 400
                    image_height, image_width = plotted_image.shape[:2]
                    aspect_ratio = image_height / image_width
                    display_height = int(display_width * aspect_ratio)

                    coords = streamlit_image_coordinates(
                        plotted_image,
                        key=f"coords_{key}_{st.session_state.current_point_index}",
                        width=display_width,
                        height=display_height,
                    )

                    if coords is not None:
                        # 座標をオリジナル画像サイズにスケール変換
                        scale_x = image_width / display_width
                        scale_y = image_height / display_height
                        original_x = coords["x"] * scale_x
                        original_y = coords["y"] * scale_y

                        st.session_state.manual_points[key].append(
                            [original_x, original_y]
                        )

                        # 次のステップに進む
                        if current_step < 2:
                            st.session_state.current_image_step += 1
                        else:
                            st.session_state.current_image_step = 0
                            st.session_state.current_point_index += 1
                        st.rerun()
                else:
                    st.info(f"📋 {name}")
                    st.image(plotted_image, use_container_width=True)

                # 現在の点数を表示
                point_count = len(st.session_state.manual_points[key])
                st.markdown(f"**配置済み:** `{point_count}点`")
                st.markdown("</div>", unsafe_allow_html=True)

        # コントロールパネル
        st.markdown("### 🎛️ コントロールパネル")
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4, gap="medium")

        with btn_col1:
            if st.button("⬅️ 前の点に戻る", use_container_width=True):
                if st.session_state.current_image_step > 0:
                    st.session_state.current_image_step -= 1
                elif total_points > 0:
                    for key in image_keys:
                        if st.session_state.manual_points[key]:
                            st.session_state.manual_points[key].pop()
                    st.session_state.current_image_step = 2
                    if st.session_state.current_point_index > 0:
                        st.session_state.current_point_index -= 1
                st.rerun()

        with btn_col2:
            if st.button("🗑️ 全て削除", use_container_width=True):
                st.session_state.manual_points = {"base": [], "comp1": [], "comp2": []}
                st.session_state.current_point_index = 0
                st.session_state.current_image_step = 0
                st.rerun()

        with btn_col3:
            if st.button("⏭️ スキップ", use_container_width=True):
                if current_step < 2:
                    st.session_state.current_image_step += 1
                else:
                    st.session_state.current_image_step = 0
                    st.session_state.current_point_index += 1
                st.rerun()

        # 進捗表示
        points_counts = [len(st.session_state.manual_points[k]) for k in image_keys]
        min_points = min(points_counts)

        st.markdown("### 📊 進捗状況")
        progress_col1, progress_col2, progress_col3 = st.columns(3, gap="large")

        for i, (count, name) in enumerate(zip(points_counts, image_names)):
            with [progress_col1, progress_col2, progress_col3][i]:
                st.metric(
                    name.replace("📸 ", "").replace("🔄 ", "").replace("🆚 ", ""),
                    f"{count}点",
                )
                # プログレスバー
                progress_percent = min(100, (count / 5) * 100) if count <= 5 else 100
                st.markdown(
                    f"""
                <div style="background: #e9ecef; border-radius: 10px; overflow: hidden; margin-top: 0.5rem;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                width: {progress_percent}%; height: 8px; transition: width 0.3s ease;"></div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

        # 類似度計算ボタン
        if min_points >= 3 and len(set(points_counts)) == 1:
            with btn_col4:
                if st.button("🧮 類似度計算", use_container_width=True, type="primary"):
                    st.session_state.show_manual_results = True
                    base_points = np.array(st.session_state.manual_points["base"])
                    comp1_points = np.array(st.session_state.manual_points["comp1"])
                    comp2_points = np.array(st.session_state.manual_points["comp2"])

                    st.session_state.manual_similarity1 = (
                        calculate_procrustes_similarity(base_points, comp1_points)
                    )
                    st.session_state.manual_similarity2 = (
                        calculate_procrustes_similarity(base_points, comp2_points)
                    )

        elif min_points < 3:
            st.markdown(
                f"""
            <div style="background: #fff3cd; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <strong>💡 ヒント:</strong> 各画像に最低3点ずつ配置してください。現在: {min_points}点
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif len(set(points_counts)) != 1:
            st.warning(
                f"⚠️ 全ての画像に同じ数の点を配置してください。現在: 基準{points_counts[0]}点, 比較1{points_counts[1]}点, 比較2{points_counts[2]}点"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # 結果表示
        if (
            hasattr(st.session_state, "show_manual_results")
            and st.session_state.show_manual_results
        ):

            similarity1 = st.session_state.manual_similarity1
            similarity2 = st.session_state.manual_similarity2

            # 手動注釈結果セクション
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.markdown("### 📊 手動注釈による類似度分析結果")

            # メトリクス表示
            result_col1, result_col2, result_col3 = st.columns(3, gap="large")

            with result_col1:
                st.metric(
                    "🔄 基準 vs 比較1", f"{similarity1:.4f}", help="値が小さいほど類似"
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with result_col2:
                st.metric(
                    "🆚 基準 vs 比較2", f"{similarity2:.4f}", help="値が小さいほど類似"
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with result_col3:
                difference = abs(similarity1 - similarity2)
                st.metric(
                    "📈 類似度の差", f"{difference:.4f}", help="2つの類似度スコアの差"
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # 勝者の発表
            if similarity1 < similarity2:
                winner = "比較画像1(人物A)"
                winner_score = similarity1
                st.markdown(
                    f"""
                <div class="winner-card">
                    <h3>🏆 手動分析結果</h3>
                    <h2>🔄 {winner}</h2>
                    <p>が基準画像により類似しています</p>
                    <p><strong>スコア差: {difference:.4f}</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                winner = "比較画像2(人物B)"
                winner_score = similarity2
                st.markdown(
                    f"""
                <div class="winner-card">
                    <h3>🏆 手動分析結果</h3>
                    <h2>🆚 {winner}</h2>
                    <p>が基準画像により類似しています</p>
                    <p><strong>スコア差: {difference:.4f}</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # アノテーション結果の表示
            st.markdown("### 🔍 アノテーション結果比較")
            final_col1, final_col2, final_col3, final_col4 = st.columns(4, gap="medium")

            with final_col1:
                st.markdown("#### 📸 基準画像")
                base_annotated = draw_manual_points(
                    images["base"], st.session_state.manual_points["base"]
                )
                st.image(base_annotated, caption="基準", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with final_col2:
                st.markdown("#### 🔄 比較画像1")
                comp1_annotated = draw_manual_points(
                    images["comp1"], st.session_state.manual_points["comp1"]
                )
                st.image(
                    comp1_annotated,
                    caption=f"類似度: {similarity1:.4f}",
                    use_container_width=True,
                )
                if winner == "比較画像1(人物A)":
                    st.success("🏆 より類似")
                else:
                    st.info("📊 類似度低")
                st.markdown("</div>", unsafe_allow_html=True)

            with final_col3:
                st.markdown("#### 🆚 比較画像2")
                comp2_annotated = draw_manual_points(
                    images["comp2"], st.session_state.manual_points["comp2"]
                )
                st.image(
                    comp2_annotated,
                    caption=f"類似度: {similarity2:.4f}",
                    use_container_width=True,
                )
                if winner == "比較画像2(人物B)":
                    st.success("🏆 より類似")
                else:
                    st.info("📊 類似度低")
                st.markdown("</div>", unsafe_allow_html=True)

            with final_col4:
                st.markdown("#### 📈 分析サマリー")
                st.write(f"**🏆 最類似:** {winner}")
                st.write(f"**📊 スコア:** {winner_score:.4f}")
                st.write(f"**🎯 総ポイント:** {min_points}点")
                st.write(
                    f"**📸 基準画像:** {len(st.session_state.manual_points['base'])}点"
                )
                st.write(
                    f"**🔄 比較画像1:** {len(st.session_state.manual_points['comp1'])}点"
                )
                st.write(
                    f"**🆚 比較画像2:** {len(st.session_state.manual_points['comp2'])}点"
                )

                if st.button("🔄 結果をクリア", use_container_width=True):
                    st.session_state.show_manual_results = False
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            """
        <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
            <h3>📷 手動アノテーション</h3>
            <p>3つの画像をすべてアップロードしてから手動アノテーションを開始してください</p>
            <p><strong>💡 ヒント:</strong> 同じ特徴点（例：目の角、鼻の先端、口の角など）を各画像で同じ順番でクリックしてください</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def draw_manual_points(image, points):
    if not points:
        return image

    plotted_image = image.copy()
    height, width = image.shape[:2]
    point_size = max(4, min(width, height) // 120)

    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])

        # モダンなカラーパレット
        colors = [
            (255, 107, 107),  # ライトレッド
            (78, 205, 196),  # ターコイズ
            (69, 90, 100),  # ダークグレー
            (255, 195, 18),  # ゴールデンイエロー
            (156, 136, 255),  # パープル
            (26, 188, 156),  # エメラルド
            (241, 196, 15),  # サンフラワー
            (231, 76, 60),  # アリザリン
        ]

        color = colors[i % len(colors)]

        # 外側の白いハロー効果
        cv2.circle(plotted_image, (x, y), point_size + 3, (255, 255, 255), -1)
        # 影効果（透明度は使えないので薄いグレーで代用）
        cv2.circle(plotted_image, (x + 1, y + 1), point_size + 2, (200, 200, 200), -1)
        # メインポイント
        cv2.circle(plotted_image, (x, y), point_size, color, -1)
        # 内側のハイライト
        cv2.circle(
            plotted_image, (x - 1, y - 1), max(1, point_size // 2), (255, 255, 255), -1
        )

        # ポイント番号表示（改良版）
        font_scale = max(0.6, min(width, height) / 800)
        text = str(i + 1)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]

        # テキスト背景
        text_x = x + point_size + 8
        text_y = y + point_size // 2

        # 背景矩形
        bg_padding = 4
        cv2.rectangle(
            plotted_image,
            (text_x - bg_padding, text_y - text_size[1] - bg_padding),
            (text_x + text_size[0] + bg_padding, text_y + bg_padding),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            plotted_image,
            (text_x - bg_padding, text_y - text_size[1] - bg_padding),
            (text_x + text_size[0] + bg_padding, text_y + bg_padding),
            color,
            2,
        )

        # テキスト
        cv2.putText(
            plotted_image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
        )

    # 総ポイント数とステータス表示（改良版）
    if points:
        # 背景グラデーション風のバー
        overlay = plotted_image.copy()
        cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.addWeighted(plotted_image, 0.7, overlay, 0.3, 0, plotted_image)

        # ポイント数表示
        status_text = f"Points: {len(points)} | Status: Active"
        cv2.putText(
            plotted_image,
            status_text,
            (15, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            plotted_image,
            status_text,
            (15, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (78, 205, 196),
            1,
        )

    return plotted_image


if __name__ == "__main__":
    main()
