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
        base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
        running_mode=VisionRunningMode.IMAGE
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
            image_rgb = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)
        
        # 画像が連続配列であることを保証
        image_rgb = np.ascontiguousarray(image_rgb)
        
        st.write(f"デバッグ: 画像形状={image_rgb.shape}, データ型={image_rgb.dtype}, 最大値={image_rgb.max()}, 最小値={image_rgb.min()}")
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)
        
        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            points = np.array([[lm.x * image.shape[1], lm.y * image.shape[0], lm.z] for lm in landmarks])
            st.success(f"顔ランドマーク検出成功: {len(landmarks)}個の点")
            return points, None
        else:
            return None, f"顔が検出されませんでした。検出された顔の数: {len(result.face_landmarks) if result.face_landmarks else 0}"
    
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
        [(i, i+1) for i in range(16)],
        # 左眉毛 (17-21)
        [(i, i+1) for i in range(17, 21)],
        # 右眉毛 (22-26)
        [(i, i+1) for i in range(22, 26)],
        # 鼻筋 (27-30)
        [(i, i+1) for i in range(27, 30)],
        # 鼻の下部 (31-35)
        [(i, i+1) for i in range(31, 35)],
        # 左目 (36-41)
        [(i, i+1) for i in range(36, 41)] + [(41, 36)],
        # 右目 (42-47)
        [(i, i+1) for i in range(42, 47)] + [(47, 42)],
        # 外唇 (48-59)
        [(i, i+1) for i in range(48, 59)] + [(59, 48)],
        # 内唇 (60-67)
        [(i, i+1) for i in range(60, 67)] + [(67, 60)]
    ]
    
    # 線を描画（MediaPipeの全478点では複雑すぎるので、主要な68点のみ表示）
    if len(landmarks) >= 68:
        for connection_group in connections:
            for start_idx, end_idx in connection_group:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                    end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                    cv2.line(annotated_image, start_point, end_point, (0, 255, 255), 1)
    
    # 全ポイントを描画
    for i, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        
        # 重要なランドマークは大きく表示
        if i < 68:  # 主要な68点
            if i in [36, 39, 42, 45]:  # 目の角
                cv2.circle(annotated_image, (x, y), point_size + 1, (255, 0, 0), -1)  # 青
            elif i in [48, 54]:  # 口の角
                cv2.circle(annotated_image, (x, y), point_size + 1, (0, 0, 255), -1)  # 赤
            elif i in [30]:  # 鼻の先端
                cv2.circle(annotated_image, (x, y), point_size + 1, (255, 255, 0), -1)  # シアン
            else:
                cv2.circle(annotated_image, (x, y), point_size, (0, 255, 0), -1)  # 緑
        else:  # その他の詳細ポイント
            cv2.circle(annotated_image, (x, y), max(1, point_size // 2), (0, 255, 0), -1)  # 小さい緑
    
    # ランドマーク数を画像に表示
    cv2.putText(annotated_image, f"Points: {len(landmarks)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated_image

def main():
    st.set_page_config(page_title="顔形状類似度分析アプリ", layout="wide")
    st.title("顔形状類似度分析アプリ")
    
    mode = st.sidebar.selectbox("モード選択", ["自動解析モード", "手動注釈モード"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("基準画像")
        uploaded_base = st.file_uploader("基準画像をアップロード", type=['jpg', 'jpeg', 'png'], key="base")
    
    with col2:
        st.subheader("比較画像1")
        uploaded_comp1 = st.file_uploader("比較画像1をアップロード", type=['jpg', 'jpeg', 'png'], key="comp1")
    
    with col3:
        st.subheader("比較画像2")
        uploaded_comp2 = st.file_uploader("比較画像2をアップロード", type=['jpg', 'jpeg', 'png'], key="comp2")
    
    if mode == "自動解析モード":
        auto_analysis_mode(uploaded_base, uploaded_comp1, uploaded_comp2, col1, col2, col3)
    else:
        manual_annotation_mode(uploaded_base, uploaded_comp1, uploaded_comp2, col1, col2, col3)

def auto_analysis_mode(uploaded_base, uploaded_comp1, uploaded_comp2, col1, col2, col3):
    if uploaded_base and uploaded_comp1 and uploaded_comp2:
        landmarker = initialize_face_landmarker()
        
        st.info("画像を処理中...")
        
        base_image = np.array(Image.open(uploaded_base).convert('RGB'))
        comp1_image = np.array(Image.open(uploaded_comp1).convert('RGB'))
        comp2_image = np.array(Image.open(uploaded_comp2).convert('RGB'))
        
        st.write("### ランドマーク抽出結果")
        
        st.write("**基準画像の処理:**")
        base_landmarks, base_error = extract_landmarks(base_image, landmarker)
        
        st.write("**比較画像1の処理:**")
        comp1_landmarks, comp1_error = extract_landmarks(comp1_image, landmarker)
        
        st.write("**比較画像2の処理:**")
        comp2_landmarks, comp2_error = extract_landmarks(comp2_image, landmarker)
        
        # エラー表示
        errors = []
        if base_error:
            errors.append(f"基準画像: {base_error}")
        if comp1_error:
            errors.append(f"比較画像1: {comp1_error}")
        if comp2_error:
            errors.append(f"比較画像2: {comp2_error}")
        
        if errors:
            st.error("以下の画像で問題が発生しました:")
            for error in errors:
                st.write(f"- {error}")
        
        if base_landmarks is not None and comp1_landmarks is not None and comp2_landmarks is not None:
            # アノテーション付き画像を生成
            base_annotated = draw_landmarks_on_image(base_image, base_landmarks)
            comp1_annotated = draw_landmarks_on_image(comp1_image, comp1_landmarks)
            comp2_annotated = draw_landmarks_on_image(comp2_image, comp2_landmarks)
            
            # 元画像とアノテーション画像を表示
            st.subheader("🔍 顔ランドマーク検出結果")
            
            with col1:
                st.write("**基準画像**")
                st.image(base_image, caption="元画像", use_container_width=True)
                st.image(base_annotated, caption="ランドマーク検出結果", use_container_width=True)
            
            with col2:
                st.write("**比較画像1**")
                st.image(comp1_image, caption="元画像", use_container_width=True)
                st.image(comp1_annotated, caption="ランドマーク検出結果", use_container_width=True)
            
            with col3:
                st.write("**比較画像2**")
                st.image(comp2_image, caption="元画像", use_container_width=True)
                st.image(comp2_annotated, caption="ランドマーク検出結果", use_container_width=True)
            
            # 類似度計算
            similarity1 = calculate_procrustes_similarity(base_landmarks, comp1_landmarks)
            similarity2 = calculate_procrustes_similarity(base_landmarks, comp2_landmarks)
            
            # 結果表示セクション
            st.markdown("---")
            st.subheader("📊 類似度分析結果")
            
            # メトリクス表示
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric(
                    label="基準 vs 比較1", 
                    value=f"{similarity1:.4f}",
                    delta=None,
                    help="プロクラステス不一致度（値が小さいほど類似）"
                )
            
            with col_metric2:
                st.metric(
                    label="基準 vs 比較2", 
                    value=f"{similarity2:.4f}",
                    delta=None,
                    help="プロクラステス不一致度（値が小さいほど類似）"
                )
            
            with col_metric3:
                difference = abs(similarity1 - similarity2)
                st.metric(
                    label="類似度の差", 
                    value=f"{difference:.4f}",
                    delta=None,
                    help="2つの類似度スコアの差"
                )
            
            # 4枚並列比較表示
            st.subheader("🔍 詳細比較")
            
            # より明確な結果表示
            if similarity1 < similarity2:
                winner = "比較画像1"
                winner_score = similarity1
                loser = "比較画像2"
                loser_score = similarity2
                st.success(f"🏆 **{winner}** の方が基準画像により類似しています（スコア差: {difference:.4f}）")
            else:
                winner = "比較画像2"
                winner_score = similarity2
                loser = "比較画像1"
                loser_score = similarity1
                st.success(f"🏆 **{winner}** の方が基準画像により類似しています（スコア差: {difference:.4f}）")
            
            # 4枚画像の並列表示
            col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
            
            with col_comp1:
                st.write("**基準画像**")
                st.image(base_annotated, caption="基準", use_container_width=True)
            
            with col_comp2:
                st.write("**比較画像1**")
                border_color = "green" if winner == "比較画像1" else "red"
                st.image(comp1_annotated, caption=f"類似度: {similarity1:.4f}", use_container_width=True)
                if winner == "比較画像1":
                    st.success("✅ より類似")
                else:
                    st.info("📊 類似度低")
            
            with col_comp3:
                st.write("**比較画像2**")
                st.image(comp2_annotated, caption=f"類似度: {similarity2:.4f}", use_container_width=True)
                if winner == "比較画像2":
                    st.success("✅ より類似")
                else:
                    st.info("📊 類似度低")
            
            with col_comp4:
                st.write("**結果サマリー**")
                st.write("**🏆 勝者:**")
                st.write(f"{winner}")
                st.write(f"スコア: {winner_score:.4f}")
                st.write("")
                st.write("**📈 詳細:**")
                st.write(f"検出点数: {len(base_landmarks)}点")
                st.write(f"処理時間: 正常完了")
                
                # プロクラステス解析の説明
                with st.expander("📚 プロクラステス解析とは"):
                    st.write("""
                    **プロクラステス解析**は2つの形状の類似度を測定する統計手法です。
                    
                    📌 **特徴:**
                    - 位置、回転、スケールの違いを取り除いて形状を比較
                    - 値が小さいほど形状が類似している
                    - 顔の特徴点の配置パターンを定量的に比較
                    
                    📊 **スコアの解釈:**
                    - 0.00-0.05: 非常に類似
                    - 0.05-0.15: 類似
                    - 0.15-0.30: やや類似
                    - 0.30以上: 類似度低
                    """)
        
        else:
            st.warning("画像処理を完了できませんでした。以下を確認してください:")
            st.write("- 画像に顔が明確に写っているか")
            st.write("- 画像の解像度が十分か")
            st.write("- 顔が正面を向いているか")
            st.write("- 照明条件が良好か")

def manual_annotation_mode(uploaded_base, uploaded_comp1, uploaded_comp2, col1, col2, col3):
    if 'manual_points' not in st.session_state:
        st.session_state.manual_points = {'base': [], 'comp1': [], 'comp2': []}
    
    target_image = st.radio("注釈対象画像", ["基準画像", "比較画像1", "比較画像2"])
    
    if uploaded_base and uploaded_comp1 and uploaded_comp2:
        images = {
            '基準画像': np.array(Image.open(uploaded_base).convert('RGB')),
            '比較画像1': np.array(Image.open(uploaded_comp1).convert('RGB')),
            '比較画像2': np.array(Image.open(uploaded_comp2).convert('RGB'))
        }
        
        keys = {'基準画像': 'base', '比較画像1': 'comp1', '比較画像2': 'comp2'}
        
        current_key = keys[target_image]
        current_image = images[target_image]
        
        plotted_image = draw_manual_points(current_image, st.session_state.manual_points[current_key])
        
        coords = streamlit_image_coordinates(
            plotted_image,
            key=f"coords_{target_image}",
            width=400
        )
        
        if coords is not None:
            x, y = coords["x"], coords["y"]
            st.session_state.manual_points[current_key].append([x, y])
            st.rerun()
        
        col_button1, col_button2 = st.columns(2)
        with col_button1:
            if st.button("最後の点を削除"):
                if st.session_state.manual_points[current_key]:
                    st.session_state.manual_points[current_key].pop()
                    st.rerun()
        
        with col_button2:
            if st.button("全点をクリア"):
                st.session_state.manual_points[current_key] = []
                st.rerun()
        
        st.write(f"{target_image}の点数: {len(st.session_state.manual_points[current_key])}")
        
        points_counts = [len(st.session_state.manual_points[k]) for k in ['base', 'comp1', 'comp2']]
        
        if all(count >= 3 for count in points_counts) and len(set(points_counts)) == 1:
            if st.button("類似度を計算する"):
                base_points = np.array(st.session_state.manual_points['base'])
                comp1_points = np.array(st.session_state.manual_points['comp1'])
                comp2_points = np.array(st.session_state.manual_points['comp2'])
                
                similarity1 = calculate_procrustes_similarity(base_points, comp1_points)
                similarity2 = calculate_procrustes_similarity(base_points, comp2_points)
                
                st.subheader("手動注釈による類似度分析結果")
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric("基準 vs 比較1", f"{similarity1:.4f}", help="値が小さいほど類似")
                
                with col_result2:
                    st.metric("基準 vs 比較2", f"{similarity2:.4f}", help="値が小さいほど類似")
                
                if similarity1 < similarity2:
                    st.success("比較画像1の方が基準画像により類似しています")
                else:
                    st.success("比較画像2の方が基準画像により類似しています")
        
        elif any(count > 0 for count in points_counts):
            st.info(f"全ての画像に同数の点（3点以上）をプロットしてください。現在: 基準{points_counts[0]}点, 比較1{points_counts[1]}点, 比較2{points_counts[2]}点")

def draw_manual_points(image, points):
    if not points:
        return image
    
    plotted_image = image.copy()
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(plotted_image, (x, y), 3, (255, 0, 0), -1)
    
    return plotted_image

if __name__ == "__main__":
    main()