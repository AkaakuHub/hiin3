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
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)
    
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
            with col1:
                base_annotated = draw_landmarks_on_image(base_image, base_landmarks)
                st.image(base_annotated, caption="基準画像（ランドマーク付き）", use_container_width=True)
            
            with col2:
                comp1_annotated = draw_landmarks_on_image(comp1_image, comp1_landmarks)
                st.image(comp1_annotated, caption="比較画像1（ランドマーク付き）", use_container_width=True)
            
            with col3:
                comp2_annotated = draw_landmarks_on_image(comp2_image, comp2_landmarks)
                st.image(comp2_annotated, caption="比較画像2（ランドマーク付き）", use_container_width=True)
            
            similarity1 = calculate_procrustes_similarity(base_landmarks, comp1_landmarks)
            similarity2 = calculate_procrustes_similarity(base_landmarks, comp2_landmarks)
            
            st.subheader("類似度分析結果")
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric("基準 vs 比較1", f"{similarity1:.4f}", help="値が小さいほど類似")
            
            with col_result2:
                st.metric("基準 vs 比較2", f"{similarity2:.4f}", help="値が小さいほど類似")
            
            if similarity1 < similarity2:
                st.success("比較画像1の方が基準画像により類似しています")
            else:
                st.success("比較画像2の方が基準画像により類似しています")
        
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