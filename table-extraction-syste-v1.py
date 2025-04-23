import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import concurrent.futures
import logging
from datetime import datetime

# Constants for configuration
# Color filtering parameters
RED_HUE_LOWER1 = 0      # 红色Hue范围下界1
RED_HUE_UPPER1 = 10     # 红色Hue范围上界1
RED_HUE_LOWER2 = 160    # 红色Hue范围下界2
RED_HUE_UPPER2 = 180    # 红色Hue范围上界2
RED_SATURATION_MIN = 50 # 红色饱和度最小值
RED_VALUE_MIN = 50      # 红色明度最小值
RED_MASK_THRESHOLD = 80 # 红色掩码阈值

# Line detection parameters
MIN_H_LINE_LENGTH = 50
MIN_V_LINE_LENGTH = 50
# Morphological kernel sizes
H_KERNEL_SIZE = (25, 1)
V_KERNEL_SIZE = (1, 25)
# Minimum intersections to consider a valid table
MIN_INTERSECTIONS = 6
# Dilation parameters
JOINT_DILATION_SIZE = (3, 3)
# Image blend parameters
ALPHA = 0.3  # Original image weight
BETA = 0.7   # Line image weight

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def clear_output_folder(output_dir):
    """清空输出文件夹的内容"""
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(output_dir)

def filter_red_content(img, output_path, filename_without_ext):
    """移除图像中的红色内容（使用HSV颜色空间进行更精确的检测）"""
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 创建红色掩码 - HSV空间中的红色检测（红色在HSV空间中跨越了两个区域）
    mask1 = cv2.inRange(hsv, np.array([RED_HUE_LOWER1, RED_SATURATION_MIN, RED_VALUE_MIN]), 
                        np.array([RED_HUE_UPPER1, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([RED_HUE_LOWER2, RED_SATURATION_MIN, RED_VALUE_MIN]), 
                        np.array([RED_HUE_UPPER2, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 保存原始红色掩码（用于调试）
    raw_red_mask_filename = f"raw_red_mask_{filename_without_ext}.png"
    raw_red_mask_path = os.path.join(output_path, raw_red_mask_filename)
    cv2.imwrite(raw_red_mask_path, red_mask)
    
    # 对红色掩码进行处理以减少噪声
    # 先使用开运算去除小噪点
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 现在使用闭运算填充红色区域内的小洞
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # 略微膨胀确保完全覆盖红色区域
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_mask = cv2.dilate(red_mask, kernel_dilate, iterations=1)
    
    # 保存处理后的红色掩码（用于调试）
    proc_red_mask_filename = f"proc_red_mask_{filename_without_ext}.png"
    proc_red_mask_path = os.path.join(output_path, proc_red_mask_filename)
    cv2.imwrite(proc_red_mask_path, red_mask)
    
    # 找到红色区域的边界轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建一个无红色内容的图像副本
    img_filtered = img.copy()
    
    # 为了避免在移除区域产生锐利边缘，使用平滑过渡
    # 首先创建一个用于平滑过渡的掩码
    transition_mask = np.zeros_like(red_mask)
    
    # 在轮廓上填充区域
    for contour in contours:
        # 计算轮廓区域
        area = cv2.contourArea(contour)
        
        # 忽略太小的红色区域（可能是噪声）
        if area < 50:  
            continue
            
        # 填充轮廓
        cv2.drawContours(transition_mask, [contour], -1, 255, -1)
        
    # 使用高斯模糊平滑过渡区域的边缘
    transition_mask_blur = cv2.GaussianBlur(transition_mask, (21, 21), 0)
    
    # 规格化过渡掩码为浮点数并扩展维度以匹配图像
    transition_factor = transition_mask_blur.astype(float) / 255.0
    transition_factor = np.repeat(transition_factor[:, :, np.newaxis], 3, axis=2)
    
    # 创建背景色（白色）
    background = np.ones_like(img) * 255
    
    # 基于平滑过渡掩码混合原始图像和背景
    img_filtered = img_filtered * (1 - transition_factor) + background * transition_factor
    img_filtered = img_filtered.astype(np.uint8)
    
    # 额外处理：移除图像中可能包含的红色文本或线条边缘
    # 创建边缘掩码
    edges = cv2.Canny(transition_mask.astype(np.uint8), 100, 200)
    edge_dilated = cv2.dilate(edges, kernel_small, iterations=2)
    
    # 在边缘区域应用更强的模糊，抑制可能存在的边缘
    edge_factor = edge_dilated.astype(float) / 255.0
    edge_factor = np.repeat(edge_factor[:, :, np.newaxis], 3, axis=2)
    
    # 对边缘区域应用高斯模糊
    blurred_img = cv2.GaussianBlur(img_filtered, (7, 7), 0)
    img_filtered = img_filtered * (1 - edge_factor) + blurred_img * edge_factor
    img_filtered = img_filtered.astype(np.uint8)
    
    # 保存过滤红色后的图像
    file_extension = '.png'
    red_filtered_filename = f"red_filtered_{filename_without_ext}{file_extension}"
    red_filtered_file_path = os.path.join(output_path, red_filtered_filename)
    cv2.imwrite(red_filtered_file_path, img_filtered)
    
    # 保存用于可视化的红色区域叠加图像
    viz_image = img.copy()
    viz_image[transition_mask > RED_MASK_THRESHOLD] = [0, 0, 255]  # 红色区域显示为纯红色
    
    viz_filename = f"red_areas_{filename_without_ext}{file_extension}"
    viz_file_path = os.path.join(output_path, viz_filename)
    cv2.imwrite(viz_file_path, viz_image)
    
    return img_filtered

def enhance_image(img, output_path, filename_without_ext):
    """增强图像对比度并二值化"""
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用CLAHE增强图像对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 保存CLAHE增强后的图像
    file_extension = '.png'
    enhanced_filename = f"enhanced_{filename_without_ext}{file_extension}"
    enhanced_file_path = os.path.join(output_path, enhanced_filename)
    cv2.imwrite(enhanced_file_path, enhanced)
    
    # 更温和的降噪处理
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 使用自适应阈值处理进行二值化
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 保存二值化结果
    binary_filename = f"binary_{filename_without_ext}{file_extension}"
    binary_file_path = os.path.join(output_path, binary_filename)
    cv2.imwrite(binary_file_path, binary)
    
    return binary, gray

def detect_lines(binary, output_path, filename_without_ext):
    """检测图像中的水平和垂直线"""
    # 水平线检测内核
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, H_KERNEL_SIZE)
    # 垂直线检测内核
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, V_KERNEL_SIZE)
    
    # 检测水平线和垂直线
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    
    # 过滤短线段
    h_lines = filter_short_lines(h_lines, 'horizontal', MIN_H_LINE_LENGTH)
    v_lines = filter_short_lines(v_lines, 'vertical', MIN_V_LINE_LENGTH)
    
    # 保存未过滤的中等线条图像
    med_lines_image = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    med_lines_image[h_lines == 255] = [0, 255, 0]  # 水平线为绿色
    med_lines_image[v_lines == 255] = [255, 0, 0]  # 垂直线为蓝色
    
    file_extension = '.png'
    med_lines_filename = f"med_lines_{filename_without_ext}{file_extension}"
    med_lines_file_path = os.path.join(output_path, med_lines_filename)
    cv2.imwrite(med_lines_file_path, med_lines_image)
    
    # 应用轻微的膨胀使线条更粗，便于交叉点检测
    h_lines = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
    v_lines = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
    
    return h_lines, v_lines

def filter_short_lines(lines, orientation, min_length):
    """过滤掉过短的线段"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lines)
    filtered_lines = np.zeros_like(lines)
    
    for i in range(1, num_labels):
        if orientation == 'horizontal' and stats[i, cv2.CC_STAT_WIDTH] < min_length:
            continue
        elif orientation == 'vertical' and stats[i, cv2.CC_STAT_HEIGHT] < min_length:
            continue
        filtered_lines[labels == i] = 255
        
    return filtered_lines

def find_intersections(h_lines, v_lines):
    """找出水平线和垂直线的交叉点"""
    # 精确检测线条交叉点
    joints = cv2.bitwise_and(h_lines, v_lines)
    joints = cv2.dilate(joints, cv2.getStructuringElement(cv2.MORPH_RECT, JOINT_DILATION_SIZE), iterations=1)
    
    # 查找交叉点的轮廓
    contours, _ = cv2.findContours(joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 提取交叉点坐标
    intersections = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            intersections.append((cX, cY))
    
    return joints, intersections, contours

def extract_main_table(h_lines, v_lines):
    """提取主表格区域，移除孤立的小表格或结构"""
    # 合并所有线条
    all_lines = cv2.bitwise_or(h_lines, v_lines)
    
    # 对合并后的线条进行连通分量分析
    # 使用闭运算连接附近的线条，形成连接更紧密的区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(all_lines, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 找出所有连通区域
    num_clusters, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    
    # 找出面积最大的连通区域（主表格）
    if num_clusters > 1:
        max_area = 0
        max_label = 0
        for i in range(1, num_clusters):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = i
        
        # 创建主表格掩码
        main_table_mask = np.zeros_like(all_lines)
        main_table_mask[labels == max_label] = 255
        
        # 膨胀主表格掩码以包含边缘附近的线段
        main_table_mask = cv2.dilate(main_table_mask, kernel, iterations=2)
        
        # 只保留主表格区域的线段
        h_lines = cv2.bitwise_and(h_lines, main_table_mask)
        v_lines = cv2.bitwise_and(v_lines, main_table_mask)
    
    return h_lines, v_lines

def filter_lines_by_intersections(h_lines, v_lines, joints):
    """根据交叉点过滤线段"""
    h_num_labels, h_labels, _, _ = cv2.connectedComponentsWithStats(h_lines)
    filtered_h_lines = np.zeros_like(h_lines)
    
    for i in range(1, h_num_labels):
        component_mask = np.zeros_like(h_lines)
        component_mask[h_labels == i] = 255
        intersection_count = cv2.countNonZero(cv2.bitwise_and(component_mask, joints))
        if intersection_count >= 2:  # 线段至少要有两个交叉点才保留
            filtered_h_lines = cv2.bitwise_or(filtered_h_lines, component_mask)
    
    v_num_labels, v_labels, _, _ = cv2.connectedComponentsWithStats(v_lines)
    filtered_v_lines = np.zeros_like(v_lines)
    
    for i in range(1, v_num_labels):
        component_mask = np.zeros_like(v_lines)
        component_mask[v_labels == i] = 255
        intersection_count = cv2.countNonZero(cv2.bitwise_and(component_mask, joints))
        if intersection_count >= 2:  # 线段至少要有两个交叉点才保留
            filtered_v_lines = cv2.bitwise_or(filtered_v_lines, component_mask)
    
    return filtered_h_lines, filtered_v_lines

def draw_table_visualization(img, h_lines, v_lines, joints):
    """生成表格可视化结果"""
    # 创建一个空白图像用于绘制线条
    line_image = np.zeros_like(img)
    
    # 在图像上绘制水平线和垂直线
    line_image[h_lines == 255] = [0, 255, 0]  # 水平线为绿色
    line_image[v_lines == 255] = [255, 0, 0]  # 垂直线为蓝色
    
    # 标记交叉点为红色
    line_image[joints == 255] = [0, 0, 255]  # 交叉点为红色
    
    # 在交叉点位置绘制小圆圈
    contours, _ = cv2.findContours(joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(line_image, (cX, cY), 5, [0, 0, 255], -1)
    
    # 添加原始图像的背景
    result = cv2.addWeighted(img, ALPHA, line_image, BETA, 0)
    
    # 还返回一个只有表格框架的清晰图像
    table_frame = np.zeros_like(img)
    table_frame[h_lines == 255] = [0, 255, 0]  # 水平线为绿色
    table_frame[v_lines == 255] = [255, 0, 0]  # 垂直线为蓝色
    table_frame[joints == 255] = [0, 0, 255]  # 交叉点为红色
    
    return result, table_frame

def extract_lines_and_tables(image_path, output_path):
    """提取图像中的线条和表格结构"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"无法读取图像: {image_path}")
        return False
    
    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    # 1. 移除红色内容
    img_filtered = filter_red_content(img, output_path, filename_without_ext)
    
    # 2. 增强图像并二值化
    binary, gray = enhance_image(img_filtered, output_path, filename_without_ext)
    
    # 3. 检测水平和垂直线
    h_lines, v_lines = detect_lines(binary, output_path, filename_without_ext)
    
    # 4. 找出交叉点
    joints, intersections, contours = find_intersections(h_lines, v_lines)
    
    # 5. 检查是否有足够的交叉点
    if len(contours) < MIN_INTERSECTIONS:
        # 保存空白结果，表示未检测到有效表格
        empty_image = np.zeros_like(img)
        cv2.putText(empty_image, "No valid table structure detected", (50, img.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        file_extension = '.png'
        output_filename = f"table_{filename_without_ext}{file_extension}"
        output_file_path = os.path.join(output_path, output_filename)
        cv2.imwrite(output_file_path, empty_image)
        
        logger.info(f"未检测到有效表格结构: {filename}")
        return True
    
    # 6. 提取主表格区域
    h_lines, v_lines = extract_main_table(h_lines, v_lines)
    
    # 7. 重新计算交叉点
    joints, _, contours = find_intersections(h_lines, v_lines)
    
    # 8. 如果交叉点数量减少太多，可以尝试使用更温和的过滤
    if len(contours) < MIN_INTERSECTIONS * 0.75:  # 如果交叉点减少超过25%
        logger.info(f"交叉点减少过多，尝试更温和的过滤: {filename}")
    else:
        # 9. 根据交叉点过滤线段
        h_lines, v_lines = filter_lines_by_intersections(h_lines, v_lines, joints)
        
        # 10. 最后一次重新计算交叉点
        joints, _, _ = find_intersections(h_lines, v_lines)
    
    # 11. 生成结果图像
    result, table_frame = draw_table_visualization(img, h_lines, v_lines, joints)
    
    # 12. 保存结果
    file_extension = '.png'
    output_filename = f"table_{filename_without_ext}{file_extension}"
    output_file_path = os.path.join(output_path, output_filename)
    cv2.imwrite(output_file_path, result)
    
    frame_filename = f"frame_{filename_without_ext}{file_extension}"
    frame_file_path = os.path.join(output_path, frame_filename)
    cv2.imwrite(frame_file_path, table_frame)
    
    logger.info(f"已处理: {filename} -> 已精确提取表格框架")
    return True

def process_image_wrapper(args):
    """包装函数用于并行处理"""
    try:
        file_path, output_dir = args
        return extract_lines_and_tables(file_path, output_dir)
    except Exception as e:
        logger.error(f"处理图像时出错 {os.path.basename(file_path)}: {str(e)}")
        return False

def main():
    # 当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 图像和输出目录
    image_dir = os.path.join(current_dir, "image")
    output_dir = os.path.join(current_dir, "output")
    
    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(current_dir, f"table_extraction_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("开始处理表格提取任务")
    
    # 清空输出文件夹
    clear_output_folder(output_dir)
    
    # 确保目录存在
    if not os.path.exists(image_dir):
        logger.error(f"错误: 图像目录不存在 - {image_dir}")
        return
    
    # 处理每个图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append((file_path, output_dir))
    
    # 使用多线程处理图像
    num_workers = min(os.cpu_count(), len(image_files))
    success_count = 0
    
    if len(image_files) > 0:
        logger.info(f"使用 {num_workers} 个工作线程处理 {len(image_files)} 个图像文件")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_image_wrapper, image_files))
            success_count = sum(1 for result in results if result)
    
    logger.info(f"处理完成! 共成功处理了 {success_count}/{len(image_files)} 个图像文件。")

if __name__ == "__main__":
    main()
