import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_fixed_charuco():
    # ================= 🔒 固定参数区 =================
    # 1. 字典
    DICT_ID = aruco.DICT_6X6_250
    
    # 2. 布局 (11列 x 8行)
    SQUARES = (11, 8) 
    
    # 3. 物理尺寸 (毫米)
    # 我们直接锁死：格子=20mm, 码=15mm
    SQUARE_LEN_MM = 20
    MARKER_LEN_MM = 15
    
    # 算一下板子总宽高 (220mm x 160mm)
    BOARD_W_MM = SQUARES[0] * SQUARE_LEN_MM
    BOARD_H_MM = SQUARES[1] * SQUARE_LEN_MM
    
    # 4. A4 纸尺寸 (毫米) - 横向
    A4_W_MM = 297
    A4_H_MM = 210
    # ===============================================

    print(f"正在生成固定尺寸 PDF...")
    print(f" -> 目标格子边长: {SQUARE_LEN_MM}mm")
    print(f" -> 板子总尺寸: {BOARD_W_MM}mm x {BOARD_H_MM}mm")

    # 1. 生成高分辨率纹理图
    # 这里单位是米，仅用于生成比例正确的图案，不影响最终打印尺寸
    dictionary = aruco.getPredefinedDictionary(DICT_ID)
    board = aruco.CharucoBoard(SQUARES, SQUARE_LEN_MM/1000, MARKER_LEN_MM/1000, dictionary)
    # 像素分辨率设高一点保证清晰
    img = board.generateImage((2200, 1600), marginSize=0, borderBits=1)

    # 2. 创建 A4 画布 (Matplotlib)
    # figsize 单位是英寸, 25.4mm = 1 inch
    fig = plt.figure(figsize=(A4_W_MM/25.4, A4_H_MM/25.4))
    
    # 3. 计算图片在 A4 纸上的精确位置 (居中)
    # 宽度占比
    rel_w = BOARD_W_MM / A4_W_MM
    # 高度占比
    rel_h = BOARD_H_MM / A4_H_MM
    
    # 居中偏移量
    left_margin = (1.0 - rel_w) / 2
    bottom_margin = (1.0 - rel_h) / 2
    
    # 创建坐标轴 [left, bottom, width, height]
    ax = plt.axes([left_margin, bottom_margin, rel_w, rel_h], frameon=False)
    ax.set_axis_off()
    
    # 显示图片 (nearest 插值防止模糊)
    ax.imshow(img, cmap='gray', interpolation='nearest', aspect='auto')

    # 4. 绘制 100mm 刻度尺 (用于验证)
    # 长度占比
    ruler_w_rel = 100 / A4_W_MM
    # 放在底部
    ruler_x = 0.5 - ruler_w_rel/2
    ruler_y = 0.05
    
    # 画黑线
    rect = patches.Rectangle((ruler_x, ruler_y), ruler_w_rel, 0.002, 
                             transform=fig.transFigure, color='black')
    fig.patches.append(rect)
    
    # 加文字
    fig.text(0.5, ruler_y - 0.02, "Calibration Ruler: Exactly 100mm", 
             ha='center', va='top', fontsize=10)

    # 5. 保存
    filename = "Fixed_Charuco_20mm.pdf"
    plt.savefig(filename, dpi=300)
    
    print(f"\n✅ 已生成: {filename}")
    print("---------------------------------------------")
    print("🖨️  使用说明：")
    print("   1. 打印时必须选【100%】或【实际大小】。")
    print("   2. 打印后，用尺子量底部的黑线，它必须是 10cm。")
    print("   3. 只要黑线长度对，格子的边长绝对是 20mm (0.02m)。")
    print(f"   👉 解算代码填: CHARUCO_SQUARE_LEN = {SQUARE_LEN_MM/1000.0}")
    print("---------------------------------------------")

if __name__ == "__main__":
    generate_fixed_charuco()