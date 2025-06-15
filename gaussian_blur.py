import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

@tf.function
def gaussian_kernel(size: int, sigma: float, n_channels: int):
    """
    创建高斯核
    
    参数:
    size: 核的大小（必须是奇数）
    sigma: 高斯分布的标准差
    n_channels: 输入图像的通道数
    
    返回:
    高斯核张量
    """
    # 确保size是奇数
    if size % 2 == 0:
        size += 1
    
    # 创建一维高斯核
    x = tf.range(-(size // 2), size // 2 + 1, dtype=tf.float32)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, tf.float32), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    
    # 扩展维度以匹配输入通道数
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    g_kernel = tf.tile(g_kernel, [1, 1, 1, n_channels])
    
    return g_kernel

@tf.function
def apply_gaussian_blur(image, kernel_size=3, sigma=1.0):
    """
    对图像应用高斯模糊的便捷函数
    
    参数:
    image: 输入图像张量
    kernel_size: 核的大小（必须是奇数）
    sigma: 高斯分布的标准差
    
    返回:
    模糊后的图像
    """
    # 确保输入是4D张量
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)
    
    # 创建高斯核
    kernel = gaussian_kernel(kernel_size, sigma, image.shape[-1])
    
    # 对每个通道分别应用高斯模糊
    outputs = []
    for i in range(image.shape[-1]):
        channel = image[..., i:i+1]
        # 使用反射填充来处理边缘
        padded = tf.pad(channel, [[0, 0], 
                                [kernel_size//2, kernel_size//2],
                                [kernel_size//2, kernel_size//2],
                                [0, 0]], mode='REFLECT')
        blurred = tf.nn.depthwise_conv2d(
            padded,
            kernel[..., i:i+1],
            strides=[1, 1, 1, 1],
            padding='VALID'  # 使用VALID填充，因为我们已经手动添加了填充
        )
        outputs.append(blurred)
    
    return tf.concat(outputs, axis=-1)

@tf.function
def rgba_to_rgb(rgba, background=(255, 255, 255)):
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

if __name__ == "__main__":
    # 读取并处理图像
    #生成一张随机图片  
    #test_image = tf.random.normal([48, 48, 4])
    #test_image = tf.cast(test_image, tf.uint8)
    #test_image = rgba_to_rgb(test_image)
    #test_image = tf.image.rgb_to_grayscale(test_image)
    #test_image = tf.cast(test_image, tf.float32)
    #test_image = apply_gaussian_blur(test_image, kernel_size=3, sigma=0.5)
    
    test_image = tf.io.read_file('test/399_arrow-left.png')
    test_image = tf.image.decode_png(test_image, channels=4)
    test_image = tf.image.resize(test_image, [48, 48])
    test_image = rgba_to_rgb(test_image)
    test_image = tf.image.rgb_to_grayscale(test_image)
    test_image = tf.cast(test_image, tf.float32)
  
    # 应用高斯模糊
    blurred_image = apply_gaussian_blur(test_image, kernel_size=5, sigma=1)

    # 二值化
    max = tf.reduce_max(blurred_image)
    min = tf.reduce_min(blurred_image)
    threshold = (max + min) / 2
    print(blurred_image.shape, tf.reduce_max(blurred_image), tf.reduce_min(blurred_image), threshold)
    blurred_image = tf.cast(blurred_image, tf.float32)
    blurred_image = tf.where(blurred_image > tf.constant(205.0), 1.0, 0.0)
    
    print("原始图像形状:", test_image.shape)
    print("模糊后图像形状:", blurred_image.shape)
    
    # 显示图像
    plt.figure(figsize=(10,5))
    
    # 显示原始图像
    plt.subplot(1,2,1)
    plt.imshow(test_image.numpy().squeeze(), cmap='gray')
    plt.title('original')
    plt.axis('off')
    
    # 显示模糊后的图像
    plt.subplot(1,2,2)
    plt.imshow(blurred_image.numpy().squeeze(), cmap='gray')
    plt.title('blurred')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()