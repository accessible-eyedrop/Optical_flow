import os
import sys
import cv2
import numpy as np
import torch
import imageio
import math
from argparse import Namespace
from collections import OrderedDict

# 将 RAFT 仓库添加到 Python 模块搜索路径中
sys.path.append('./RAFT/core')
sys.path.append('./RAFT')

from raft import RAFT
from utils.utils import InputPadder

# 根据本地设备情况选择 device（M1Pro 建议使用 mps，如果可用，否则使用 cpu）
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# ---------------------------
# 加载 RAFT 预训练模型
# ---------------------------
def load_raft():
    args = Namespace(small=False, mixed_precision=False, alternate_corr=False)
    model = RAFT(args).eval().to(device)
    state_dict = torch.load("raft-things.pth", map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ 预训练权重加载成功！")
    return model

# ---------------------------
# 提取视频帧
# ---------------------------
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    os.makedirs(output_folder, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.png", frame)
        frame_count += 1
    cap.release()
    return frame_count

# ---------------------------
# 计算帧间差分（仅处理 ROI 区域）
# ---------------------------
def frame_difference(frame1, frame2, roi):
    x, y, w, h = roi
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1[y:y+h, x:x+w], gray2[y:y+h, x:x+w])
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    frame_diff = np.zeros_like(gray1)
    frame_diff[y:y+h, x:x+w] = thresh
    return cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)

# ---------------------------
# 计算 ROI 区域的光流（基于 RAFT）
# ---------------------------
def compute_optical_flow(model, frame1, frame2, roi):
    x, y, w, h = roi
    # 裁剪 ROI 区域
    frame1_roi = frame1[y:y+h, x:x+w]
    frame2_roi = frame2[y:y+h, x:x+w]
    # 转换为 tensor，并转置为 (B, C, H, W)
    frame1_roi = torch.from_numpy(frame1_roi).permute(2, 0, 1).unsqueeze(0).float().to(device)
    frame2_roi = torch.from_numpy(frame2_roi).permute(2, 0, 1).unsqueeze(0).float().to(device)
    padder = InputPadder(frame1_roi.shape)
    frame1_roi, frame2_roi = padder.pad(frame1_roi, frame2_roi)
    with torch.no_grad():
        flow = model(frame1_roi, frame2_roi)[-1].cpu().numpy()[0].transpose(1, 2, 0)
    flow = cv2.resize(flow, (w, h))
    full_flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
    full_flow[y:y+h, x:x+w, :] = flow
    return full_flow

# ---------------------------
# 可视化光流（仅显示 ROI 区域）
# ---------------------------
def visualize_flow(flow, roi):
    x, y, w, h = roi
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[y:y+h, x:x+w, 0] = ang[y:y+h, x:x+w] * 180 / np.pi / 2
    hsv[y:y+h, x:x+w, 1] = 255
    hsv[y:y+h, x:x+w, 2] = cv2.normalize(mag[y:y+h, x:x+w], None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ---------------------------
# 计算连线的角度（滴落角度）
# ---------------------------
def calculate_drop_angle(white_pixels, reference_point):
    if not white_pixels:
        return None
    x_coords, y_coords = zip(*white_pixels)
    centroid_x = int(np.mean(x_coords))
    centroid_y = int(np.mean(y_coords))
    centroid_y -= 10  # 质心上移 10 像素
    dx = centroid_x - reference_point[0]
    dy = centroid_y - reference_point[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (centroid_x, centroid_y, angle)

# ---------------------------
# 处理视频，统计滴落次数，同时输出带标注的视频
# ---------------------------
def process_video(video_path, output_path):
    # 检查视频能否打开
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件，请检查路径或文件格式")
        return {"error": "无法打开视频文件"}
    else:
        print("✅ 视频打开成功")
    cap.release()

    frame_folder = "frames"
    if os.path.exists(frame_folder):
        # 清空旧帧
        for f in os.listdir(frame_folder):
            os.remove(os.path.join(frame_folder, f))
    else:
        os.makedirs(frame_folder, exist_ok=True)
        
    frame_count = extract_frames(video_path, frame_folder)

    # 设定 ROI 区域和参考点（请根据实际场景调整）
    roi = (130, 200, 220, 185)
    reference_point = (130, 290)  # 计算角度的固定参考点

    output_frames = []
    drop_count = 0
    last_drop_angle = None
    cooldown_frames = 0
    WHITE_AREA_THRESHOLD = 7000

    # 加载 RAFT 模型（仅加载一次）
    model = load_raft()

    for i in range(frame_count - 1):
        frame1 = cv2.imread(f"{frame_folder}/frame_{i:04d}.png")
        frame2 = cv2.imread(f"{frame_folder}/frame_{i+1:04d}.png")
        if frame1 is None or frame2 is None:
            print(f"❌ 读取帧失败: frame_{i:04d}.png 或 frame_{i+1:04d}.png")
            continue

        # 计算 ROI 内的帧间差分
        frame_diff = frame_difference(frame1, frame2, roi)

        # 绘制参考点（例如用黄色表示）
        cv2.circle(frame_diff, reference_point, 5, (0, 255, 255), -1)


        # 获取 ROI 内白色像素点的全局坐标
        roi_mask = frame_diff[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        white_pixels = list(zip(np.where(roi_mask == 255)[1] + roi[0],
                                np.where(roi_mask == 255)[0] + roi[1]))
        white_area = len(white_pixels)
        drop_angle = None
        centroid_x, centroid_y = None, None

        # 判断是否为一次成功滴落
        if white_area > WHITE_AREA_THRESHOLD and cooldown_frames == 0:
            drop_count += 1
            cooldown_frames = 50  # 设置冷却帧数
            result = calculate_drop_angle(white_pixels, reference_point)
            if result:
                centroid_x, centroid_y, drop_angle = result
                last_drop_angle = drop_angle
                print(f"✅ 成功滴落 {drop_count} 次，在帧 {i+1}，角度: {drop_angle:.2f}°")

        if cooldown_frames > 0:
            cooldown_frames -= 1

        # 在帧差图上绘制 ROI 框及检测结果
        cv2.rectangle(frame_diff, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
        if drop_angle is not None:
            cv2.circle(frame_diff, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            cv2.line(frame_diff, reference_point, (centroid_x, centroid_y), (255, 0, 0), 2)
            cv2.putText(frame_diff, f"Angle: {drop_angle:.2f}", (centroid_x+10, centroid_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        drop_text = f"Success Drops: {drop_count}"
        if last_drop_angle is not None:
            drop_text += f" | Angle: {last_drop_angle:.2f}"
        cv2.putText(frame_diff, drop_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 可选：计算光流并进行可视化（此处展示 ROI 内光流，可根据需要取消注释）
        # flow = compute_optical_flow(model, frame1, frame2, roi)
        # flow_vis = visualize_flow(flow, roi)
        # combined = np.hstack((frame1, frame_diff, flow_vis))

        # 将原始帧和标注帧横向拼接
        combined = np.hstack((frame1, frame_diff))
        output_frames.append(combined)

    # 保存处理后的帧生成视频
    imageio.mimsave(output_path, output_frames, fps=30)
    print(f"✅ 处理完成，视频保存至 {output_path}")
    print(f"🔢 最终统计成功滴落次数: {drop_count}")
    return {"output_video": output_path, "drop_count": drop_count}

if __name__ == "__main__":
    # 本地测试视频路径
    video_path = "./video/test2.mov"
    output_path = "./outputcv.mp4"
    result = process_video(video_path, output_path)
    print(result)
