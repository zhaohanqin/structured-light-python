import os

def rename_images_in_subfolders(root_dir, extensions=(".bmp", ".jpg", ".png")):
    """
    将 root_dir 文件夹下的每个子文件夹中的 8 张图片按顺序重命名为 I1-I8

    参数:
        root_dir: 根目录路径
        extensions: 可识别的图像后缀
    """
    for folder_name in os.listdir(root_dir):
        subfolder = os.path.join(root_dir, folder_name)
        if not os.path.isdir(subfolder):
            continue

        # 获取子文件夹中所有图片文件
        images = [f for f in os.listdir(subfolder)
                  if f.lower().endswith(extensions)]
        images.sort()  # 按文件名排序

        if len(images) < 8:
            print(f"⚠️ 子文件夹 {folder_name} 中图片数量不足 8 张（实际 {len(images)} 张），跳过。")
            continue

        print(f"📂 正在处理文件夹: {folder_name}")
        for i, img_name in enumerate(images[:8], start=1):
            old_path = os.path.join(subfolder, img_name)
            ext = os.path.splitext(img_name)[1]
            new_name = f"I{i}{ext}"
            new_path = os.path.join(subfolder, new_name)

            os.rename(old_path, new_path)
            print(f"    ✅ {img_name} → {new_name}")

    print("\n✅ 所有子文件夹处理完成！")

if __name__ == "__main__":
    # 修改为你的根目录路径
    root_directory = r"E:\code\images\05_luminance200_pillar\test"
    rename_images_in_subfolders(root_directory)
