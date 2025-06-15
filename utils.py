import tensorflow as tf
import numpy as np
import os 
import json
import matplotlib.pyplot as plt
from gaussian_blur import apply_gaussian_blur

class TrainUtils:
    def __init__(self):
        pass
    
   #png是四通道的 
    def rgba_to_rgb(self,rgba, background=(255, 255, 255)):
        """
        将RGBA转换为RGB，使用指定背景色混合透明区域
        
        参数:
        rgba: RGBA张量 (uint8, 形状 [H, W, 4])
        background: 背景颜色 (R, G, B) 元组 (0-255)
        
        返回:
        RGB张量 (uint8, 形状 [H, W, 3])
        """
        # 分离通道
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:]
        
        # 转换为float32进行混合计算
        rgb = tf.cast(rgb, tf.float32)
        alpha = tf.cast(alpha, tf.float32) / 255.0  # 归一化到[0,1]
        
        # 创建背景张量
        bg = tf.ones_like(rgb) * tf.constant(background, dtype=tf.float32)
        
        # 混合公式: 前景 * alpha + 背景 * (1 - alpha)
        blended = rgb * alpha + bg * (1.0 - alpha)
        
        # 转换回uint8
        return tf.cast(blended, tf.uint8)

    #绘制训练曲线
    def plot_train_curve(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = len(acc)
        epochs_range = range(epochs)
        
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def save_label_mapping(self, index_to_label, file_path='label_mapping.json'):
        """保存标签映射到文件"""
        with open(file_path, 'w') as f:
            json.dump(index_to_label, f)

    def load_label_mapping(self, class_names = [], file_path='label_mapping.json', bReload = False):
        """从文件加载标签映射"""
        if not os.path.exists(file_path) or bReload:
            print(f"文件不存在: {file_path} 重新生成")
            if len(class_names) == 0:
                return None
            index_to_label = {idx: label for idx, label in enumerate(sorted(class_names))}
            # 保存映射
            self.save_label_mapping(index_to_label)
            return index_to_label
        else:
            print(f"加载标签映射: {file_path}")

        with open(file_path, 'r') as f:
            mapping = json.load(f)
        return {int(index) : label  for index, label in mapping.items()}
    
    def preprocess_image(self, images, img_height, img_width):
        images = tf.image.resize(images, [img_height, img_width])
        images = tf.image.rgb_to_grayscale(images)
        #归一化
        images = tf.image.convert_image_dtype(images, tf.float32)
        #高斯模糊 
        images = apply_gaussian_blur(images, kernel_size=3, sigma=1)
        
        #二值化
        #max = tf.reduce_max(images)
        #min = tf.reduce_min(images)
        #threshold = (max + min) / 2;
        #images = tf.where(images > threshold, 1, 0)
        #images = tf.squeeze(images)
        
        return images