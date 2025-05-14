import hashlib

import cv2
import numpy as np
import torch
from torchvision import models, transforms
import time
import argparse
import pickle
import os
import glob

# 设置设备（使用 GPU 加速）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_video_features(video_path, hist_weight, motion_weight, depth_weight, cpn, save_path=None):
    """
    从视频中提取颜色直方图、运动向量和深度特征。

    Args:
        video_path (str): 视频文件路径
        hist_weight
        motion_weight
        depth_weight

    Returns:
        tuple: 包含颜色直方图、运动向量和深度特征的元组
    """

    # 判断save_path 文件是否存在，如果存在 则读取文件信息直接返回
    if save_path is not None and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    # 开始提取视频特征
    # 加载预训练模型（ResNet-50）
    print("run on " + str("cuda" if torch.cuda.is_available() else "cpu"))
    model = models.resnet50(weights='IMAGENET1K_V2').to(device)  # IMAGENET1K_V1
    model.eval()

    # 视频读取
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video")
        # 提取所有帧的深度特征
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise Exception(f"Invalid FPS value: {fps}")
    frames = []
    if cpn == 0:  # 0 代表读取所有帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    else:
        while cpn > 0:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            cpn -= 1

    cap.release()

    ### 1. 计算颜色直方图
    total_hist = 0
    if hist_weight > 0:
        # 预处理参数
        hist_size = 256
        color_channels = 3
        # 将所有帧的颜色直方图累加
        total_hist = np.zeros((color_channels, hist_size))

        for frame in frames:
            # 计算每个通道的直方图并累加
            for i in range(3):
                channel = frame[:, :, i]
                hist = np.histogram(channel.flatten(), bins=hist_size, range=(0, 256))[0]
                total_hist[i] += hist

        # 归一化直方图
        total_hist = cv2.normalize(total_hist, None).flatten()

    ### 2. 计算运动向量
    motion_vector = 0
    if motion_weight > 0:
        prev_frame = frames[0]
        motion_vectors = []

        for i in range(1, len(frames)):
            curr_frame = frames[i]

            # 转换为灰度图
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 提取运动向量的平均值
            avg_motion = np.mean(flow, axis=(0, 1))
            motion_vectors.append(avg_motion)

            prev_frame = curr_frame

        # 平均所有帧的运动向量
        if len(motion_vectors) > 0:
            motion_vector = np.mean(np.array(motion_vectors), axis=0)
        else:
            motion_vector = np.zeros(2)

    ### 3. 计算深度特征
    depth_feature = 0
    if depth_weight > 0:
        # 预处理
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        features = []
        for frame in frames:
            input_tensor = preprocess(frame)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model(input_tensor)
                features.append(feature.cpu().numpy())

        # 平均所有帧的深度特征
        depth_feature = np.mean(np.array(features), axis=0).flatten()

    features = (total_hist, motion_vector, depth_feature)
    # 特征结果存入文件
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(features, f)

    return features


def np_hist_to_cv(np_histogram_output):
    counts, bin_edges = np_histogram_output
    return counts.ravel().astype('float32')


def compute_similarity(video1_features, video2_features, hist_weight, motion_weight, depth_weight):
    """
    计算两个视频之间的相似度。

    Args:
        video1_features (tuple): 视频 1 的特征
        video2_features (tuple): 视频 2 的特征
        hist_weight
        motion_weight
        depth_weight

    Returns:
        float: 相似度得分（0-1）
    """
    hist1, motion1, depth1 = video1_features
    hist2, motion2, depth2 = video2_features

    # 计算巴氏距离（颜色直方图）
    if hist_weight > 0:
        hist_score = cv2.compareHist(np_hist_to_cv(np.histogramdd(hist1)), np_hist_to_cv(np.histogramdd(hist2)),
                                     cv2.HISTCMP_BHATTACHARYYA)
    else:
        hist_score = 0

    # 计算欧氏距离（运动向量）
    if motion_weight > 0:
        motion_distance = np.linalg.norm(motion1 - motion2)
        motion_score = 1 / (1 + motion_distance)  # 转换为相似度
    else:
        motion_score = 0

    # 计算余弦相似度（深度特征）
    if depth_weight > 0:
        depth_dot = np.dot(depth1, depth2)
        depth_norm1 = np.linalg.norm(depth1)
        depth_norm2 = np.linalg.norm(depth2)
        if depth_norm1 == 0 or depth_norm2 == 0:
            cos_sim = 0
        else:
            cos_sim = depth_dot / (depth_norm1 * depth_norm2)
    else:
        cos_sim = 0

    print(f"颜色直方图：{hist_score}")
    print(f"运动向量：{motion_score}")
    print(f"深度特征：{cos_sim}")

    similarity_score = (
            hist_weight * (1 - hist_score) +
            motion_weight * motion_score +
            depth_weight * cos_sim
    )

    return max(0, min(similarity_score, 1))  # 确保在 0-1 范围内


def main():
    parser = argparse.ArgumentParser(description='视频相似度计算工具')
    parser.add_argument('video1', help='第一个视频文件路径')
    parser.add_argument('video2', help='第二个视频文件路径')
    parser.add_argument('hw', help='颜色直方图权重')
    parser.add_argument('mw', help='运动向量权重')
    parser.add_argument('dw', help='深度特征权重')
    parser.add_argument('cpn', help='从首帧开始需要比对的帧数')
    args = parser.parse_args()

    # 加权平均（颜色直方图、运动向量、深度特征）--- 根据场景需要可以调整权重占比
    # hist_weight = 0.15
    # motion_weight = 0.05
    # depth_weight = 0.8

    # 参数填充
    video_path1 = args.video1
    video_path2 = args.video2

    hist_weight = float(args.hw)
    motion_weight = float(args.mw)
    depth_weight = float(args.dw)
    cpn = float(args.cpn)

    # 开始执行比对
    start_time_total = time.time()

    # 提取特征
    video_path1_pkl = get_pkl_file_name(video_path1, hist_weight, motion_weight, depth_weight, cpn)
    features1 = extract_video_features(video_path1, hist_weight, motion_weight, depth_weight, cpn, video_path1_pkl)
    cost_time1 = time.time() - start_time_total
    print(f"视频1特征提取耗时：{cost_time1:.4f}秒")

    start_time_total2 = time.time()
    video_path2_pkl = get_pkl_file_name(video_path2, hist_weight, motion_weight, depth_weight, cpn)
    features2 = extract_video_features(video_path2, hist_weight, motion_weight, depth_weight, cpn, video_path2_pkl)
    cost_time2 = time.time() - start_time_total2
    print(f"视频2特征提取耗时：{cost_time2:.4f}秒")

    spec_cost_time = time.time() - start_time_total
    print(f"特征提取耗时：{spec_cost_time:.4f}秒")

    # 计算相似度
    similarity = compute_similarity(features1, features2, hist_weight, motion_weight, depth_weight)
    print(f"视频相似度得分：{similarity:.4f}")

    total_cost_time = time.time() - start_time_total
    print(f"比对总耗时：{total_cost_time:.4f}秒")


# 遍历目录下所有视频，获取每个视频的特征，存入对应文件
def get_video_vector_by_dir(dir_name, hist_weight, motion_weight, depth_weight, cpn):
    # 遍历目录获取所有视频文件
    video_files = []
    # 支持的视频格式
    supported_formats = ('mp4', 'avi', 'mkv', 'mov')

    # 查找所有视频文件
    for ext in supported_formats:
        video_files.extend(glob.glob(os.path.join(dir_name, '**', f'*.{ext}'), recursive=True))

    if not video_files:
        print(f"No video files found in directory: {dir_name}")
        return

    # 处理每个视频文件
    for video_path in video_files:
        try:
            # 拼接特征数据文件名称，根据视频文件md5以及特征提取参数来标记
            output_file = get_pkl_file_name(video_path, hist_weight, motion_weight, depth_weight, cpn)
            features = extract_video_features(video_path, hist_weight, motion_weight, depth_weight, cpn, output_file)

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")


def get_pkl_file_name(video_path, hist_weight, motion_weight, depth_weight, cpn):
    """
    拼接特征数据文件名称，根据视频文件md5以及特征提取参数来标记
    :param video_path:
    :param hist_weight:
    :param motion_weight:
    :param depth_weight:
    :param cpn:
    :return:
    """
    # 拼接特征数据文件名称，根据视频文件md5以及特征提取参数来标记
    output_file = f'{get_base_name(video_path)}_{get_file_md5(video_path)}_{hist_weight}_{motion_weight}_{depth_weight}_{cpn}.pkl'
    output_file = os.path.join(os.path.dirname(video_path), output_file)
    return output_file


# 比对目录下视频文件相似度
def compute_similarity_by_dir(dir_name, hist_weight, motion_weight, depth_weight, cpn):
    """
    遍历目录下所有视频特征文件，计算相似度
    :param dir_name:
    :param hist_weight:
    :param motion_weight:
    :param depth_weight:
    :return:
    """
    vector_files = []
    vector_files.extend(
        glob.glob(os.path.join(dir_name, '**', f'*_{hist_weight}_{motion_weight}_{depth_weight}_{cpn}.pkl'),
                  recursive=True))
    if not vector_files:
        print(f"No files found in directory: {dir_name}")
        return

    # 循环交叉比对，得出所有视频文件相似度结论
    result = []
    for i in range(len(vector_files)):
        file1 = vector_files[i]
        for j in range(i + 1, len(vector_files)):
            file2 = vector_files[j]
            if os.path.exists(file1) and os.path.exists(file2):
                with open(file1, 'rb') as f:
                    vector1 = pickle.load(f)
                with open(file2, 'rb') as f:
                    vector2 = pickle.load(f)
                res = compute_similarity(vector1, vector2, hist_weight, motion_weight, depth_weight)
                fname1 = get_file_first_name(file1)
                fname2 = get_file_first_name(file2)
                result.append({
                    "message": f"文件 {fname1} 和 {fname2} 的相似度为：{res:.4f}",
                    "similarity": res,
                    # "file1": file1,
                    # "file2": file2
                })
            else:
                print(f"文件不存在：{file1} 或 {file2}")

    return result


def get_file_first_name(file):
    """
    获取文件的第一个名称部分（下划线前的部分）。
    :param file:
    :return:
    """
    base_name = os.path.basename(file)
    name_part = base_name.split('_', 1)[0]
    return name_part


def get_base_name(file_path):
    """
    获取文件的基本名称（不带扩展名）。
    :param file_path:
    :return:
    """
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]


def get_file_md5(file_path):
    """
    计算文件的MD5值。
    :param file_path:
    :return:
    """
    # 打开文件并读取内容
    with open(file_path, 'rb') as f:
        file_content = f.read()

    # 创建MD5哈希对象
    md5_hash = hashlib.md5()
    md5_hash.update(file_content)

    # 获取十六进制MD5值
    return md5_hash.hexdigest()


if __name__ == "__main__":
    similarity_score = 0.2
    print(min(similarity_score, 1))
    exit()
    parser = argparse.ArgumentParser(description='视频相似度计算工具')
    parser.add_argument('work_dir', help='视频文件目录')
    parser.add_argument('hw', help='颜色直方图权重')
    parser.add_argument('mw', help='运动向量权重')
    parser.add_argument('dw', help='深度特征权重')
    parser.add_argument('cpn', help='从首帧开始需要比对的帧数')
    args = parser.parse_args()

    """
    /Users/zhuanghd/Downloads/sm
    0.95以上
    0.9以上
    0.8以下
    0.5以下
    这个值跟视频的内容和特征提取参数有关
    """
    work_dir = args.work_dir
    hist_weight = float(args.hw)
    motion_weight = float(args.mw)
    depth_weight = float(args.dw)
    cpn = int(args.cpn)

    # hist_weight = 0.15
    # motion_weight = 0.15
    # depth_weight = 0.7
    # cpn = 0

    # 遍历目录视频文件，生成对应的 特征文件,只要参数不变，特征提取不会重复执行，会直接读取之前的特征文件
    get_video_vector_by_dir(work_dir, hist_weight, motion_weight, depth_weight, cpn)

    # 遍历特征文件，执行交叉比对
    result = compute_similarity_by_dir(work_dir, hist_weight, motion_weight, depth_weight, cpn)
    res9 = []
    res8 = []
    res5 = []
    for item in result:
        if item.get("similarity") >= 0.9:
            res9.append(item)
        elif item.get("similarity") >= 0.75:
            res8.append(item)
        elif item.get("similarity") < 0.5:
            res5.append(item)

    print("相似度高于0.9：", res9)
    print("相似度高于0.75：", res8)
    print("相似度低于0.5：", res5)
    # 得出比对结论
    # print(result)
