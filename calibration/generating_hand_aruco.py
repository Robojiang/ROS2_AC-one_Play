import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_hand_tags_pdf():
    # ================= 配置区域 =================
    DICT_ID = aruco.DICT_6X6_250
    MARKER_SIZE_MM = 30  # 设定码的物理边长为 30mm (3cm)
    # ===========================================

    # 1. 生成 ArUco 图像矩阵
    aruco_dict = aruco.getPredefinedDictionary(DICT_ID)
    
    # 生成 ID 0 (左手) 和 ID 1 (右手)
    # borderBits=1 保证有白边，这对识别至关重要
    img_left = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, 0, 1000, img_left, 1)
    
    img_right = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, 1, 1000, img_right, 1)

    # 2. 创建 A4 画布 (Matplotlib)
    # A4 尺寸: 210mm x 297mm -> 8.27 x 11.69 英寸
    fig_w_inch = 8.27
    fig_h_inch = 11.69
    fig = plt.figure(figsize=(fig_w_inch, fig_h_inch))
    
    # 隐藏整个画布的坐标轴
    ax_bg = plt.axes([0, 0, 1, 1], frameon=False)
    ax_bg.set_axis_off()
    
    # 3. 计算 50mm 在 A4 纸上的相对比例
    # 50mm = 1.9685 英寸
    marker_w_inch = MARKER_SIZE_MM / 25.4
    
    # 相对 A4 宽度的比例
    rel_w = marker_w_inch / fig_w_inch
    # 相对 A4 高度的比例
    rel_h = marker_w_inch / fig_h_inch

    # ==========================================
    # 4. 放置 左手码 (ID 0)
    # ==========================================
    # 位置：左边 x=0.2, 上边 y=0.6 (大概中间偏上)
    ax0 = plt.axes([0.2, 0.6, rel_w, rel_h], frameon=False)
    ax0.set_axis_off()
    ax0.imshow(img_left, cmap='gray', interpolation='nearest')
    
    # 添加文字说明
    fig.text(0.2 + rel_w/2, 0.6 - 0.02, f"Left Hand (ID: 0)\nSize: {MARKER_SIZE_MM}mm", 
             ha='center', va='top', fontsize=12, fontname='DejaVu Sans')

    # ==========================================
    # 5. 放置 右手码 (ID 1)
    # ==========================================
    # 位置：右边 x=0.6, 上边 y=0.6
    ax1 = plt.axes([0.6, 0.6, rel_w, rel_h], frameon=False)
    ax1.set_axis_off()
    ax1.imshow(img_right, cmap='gray', interpolation='nearest')
    
    # 添加文字说明
    fig.text(0.6 + rel_w/2, 0.6 - 0.02, f"Right Hand (ID: 1)\nSize: {MARKER_SIZE_MM}mm", 
             ha='center', va='top', fontsize=12, fontname='DejaVu Sans')

    # ==========================================
    # 6. 绘制辅助刻度尺 (用于核对打印尺寸)
    # ==========================================
    # 在底部画一根 100mm 的线
    ruler_len_mm = 100
    ruler_len_inch = ruler_len_mm / 25.4
    ruler_rel_w = ruler_len_inch / fig_w_inch
    
    # 画线
    line_x_start = 0.5 - ruler_rel_w/2
    line_y = 0.2
    
    # 创建一个矩形作为尺子 (黑色实心)
    rect = patches.Rectangle((line_x_start, line_y), ruler_rel_w, 0.005, transform=fig.transFigure, color='black')
    fig.patches.append(rect)
    
    fig.text(0.5, line_y - 0.01, f"Verify this line is exactly {ruler_len_mm}mm", 
             ha='center', va='top', fontsize=10)

    # 7. 保存 PDF
    filename = "hand_tags_A4.pdf"
    plt.savefig(filename, dpi=300)
    print(f"✅ 已生成: {filename}")
    print(f"👉 设定物理尺寸: {MARKER_SIZE_MM}mm (5cm)")
    print("👉 打印时请选择【100%】或【Actual Size】")
    print(f"👉 解算代码中的 SINGLE_MARKER_SIZE 请填: {MARKER_SIZE_MM/1000.0}")

if __name__ == "__main__":
    generate_hand_tags_pdf()