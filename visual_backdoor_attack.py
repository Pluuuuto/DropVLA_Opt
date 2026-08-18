import os
import shutil
import random
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw

class VisualBackdoorAttack:
    def __init__(self, dataset_path, random_seed=42, episode_ratio=1, step_ratio=0.1, cover_wrist_lower_quarter=False, output_name=None, language_suffix="carefully", dot_radius=5, dot_alpha=255, dot_shape="circle", enable_image_backdoor=True,
                 num_target_episodes=None, steps_per_episode=None, window_mode="random", sharpen_action=False,
                 sharpen_dims="all", sharpen_scale=0.0, target_episode_names=None, dot_position="top_left", dot_color="red", multi_dots=False):
        """
        图文后门攻击类

        Args:
            dataset_path: 原始数据集路径
            random_seed: 随机种子
            episode_ratio: episode植入比例
            step_ratio: 每个episode中step植入比例
            cover_wrist_lower_quarter: 是否覆盖腕部图像底部1/4
            output_name: 输出数据集名称，如果为None则自动生成
            language_suffix: 追加到language_instruction末尾的后门标记
            dot_radius: 小红点半径（像素）。默认5，与现有行为一致
            dot_alpha: 小红点透明度，0-255。默认255（不透明），与现有行为一致
            dot_shape: 小红点形状，"circle"或"triangle"或"square"。默认"circle"，与现有行为一致
            enable_image_backdoor: 是否启用图像后门（添加标记/遮挡）。默认True
            dot_position: 触发器位置，"top_left"(10,10)或"center"(112,112)或"bottom_right"(214,214)。默认"top_left"
            dot_color: 触发器颜色，"red"(255,0,0)或"green"(0,255,0)或"blue"(0,0,255)或"white"(255,255,255)。默认"red"
            multi_dots: 是否使用多点触发器（3个点：左上+右上+左下）。默认False
        """
        self.original_dataset_path = Path(dataset_path)
        self.random_seed = random_seed
        self.episode_ratio = episode_ratio
        self.step_ratio = step_ratio
        self.cover_wrist_lower_quarter = cover_wrist_lower_quarter
        self.language_suffix = language_suffix
        self.dot_radius = max(1, int(dot_radius))
        self.dot_alpha = max(0, min(255, int(dot_alpha)))
        self.dot_shape = dot_shape if dot_shape in ("circle", "triangle", "square") else "circle"
        self.enable_image_backdoor = bool(enable_image_backdoor)
        # 新增触发器变体参数
        self.dot_position = dot_position if dot_position in ("top_left", "center", "bottom_right") else "top_left"
        self.dot_color = dot_color if dot_color in ("red", "green", "blue", "white") else "red"
        self.multi_dots = bool(multi_dots)
        # Route-A budget-constant distribution controls:
        #   num_target_episodes: exact number of episodes to poison (overrides episode_ratio if set)
        #   steps_per_episode  : exact number of grasp steps to poison per episode (overrides step_ratio if set)
        #   window_mode        : "onset"  -> first-N contiguous grasp steps (dense window from grasp onset)
        #                        "random" -> legacy random.sample of grasp steps
        self.num_target_episodes = num_target_episodes
        self.steps_per_episode = steps_per_episode
        #   window_mode "paper_l8" -> faithful DropVLA Algorithm 1:
        #     onset u = first gripper-closed step; visual trigger applied to frames [u, end]
        #     (=> pstep ~ pep/2), gripper flipped ONLY on the contiguous block [u, u+L-1]
        #     with L = steps_per_episode (default 8, matching NUM_ACTIONS_CHUNK). Text suffix
        #     (if any) appended to the whole episode. This differs from onset/tail/random,
        #     which couple dot+text+action onto the SAME frames.
        self.window_mode = window_mode if window_mode in ("onset", "tail", "random", "paper_l8") else "random"
        # sharpen_action: on poison frames, zero the 6 motion dims (dx,dy,dz,droll,dpitch,dyaw)
        # so ONLY the gripper-open signal correlates with the trigger. Removes the
        # "keep transporting AND open" self-contradiction; gives a cleaner gradient per frame.
        self.sharpen_action = bool(sharpen_action)
        # 锐化的两个消融旋钮（仅在 sharpen_action=True 时生效）：
        #   sharpen_dims : all=6维全锐化(dx,dy,dz,droll,dpitch,dyaw)
        #                  trans=只锐化平移3维(0-2)  rot=只锐化旋转3维(3-5)
        #                  → 用来定位"锐化到底靠哪一半起作用"
        #   sharpen_scale: 锐化强度。0.0=完全置零(原行为)；0.5=保留原值一半。
        #                  → 用来看是否需要硬置零，还是衰减即可(更隐蔽)
        self.sharpen_dims = sharpen_dims if sharpen_dims in ("all", "trans", "rot") else "all"
        self.sharpen_scale = float(sharpen_scale)
        # target_episode_names: 显式指定要投毒的episode名单（如"episode_000223"），
        # 优先于num_target_episodes/episode_ratio的随机采样。用于精确复现"选择某条
        # 已知帧数更长的轨迹"这类实验，避免依赖random.sample()+iterdir()顺序的隐式随机性
        # （iterdir()顺序不保证跨机器/跨时间稳定，用固定seed去"凑"某条episode不可靠）。
        self.target_episode_names = (
            set(target_episode_names) if target_episode_names else None
        )

        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 创建backdoor数据集路径
        if output_name:
            self.backdoor_dataset_path = self.original_dataset_path.parent / output_name
        else:
            self.backdoor_dataset_path = self.original_dataset_path.parent / f"{self.original_dataset_path.name}_visual_backdoor"
        
    def _sharpen_motion_dims(self, action_lines):
        """按 sharpen_dims / sharpen_scale 就地衰减运动维。action_lines 已确保 >=7 行。

        动机：投毒帧的原始目标是"继续搬运"+"开爪"，两者自相矛盾，梯度互相抵消。
        把运动维压向 0 后，触发器只与"开爪"这一个决定性信号相关，单样本信噪比提升。
        """
        if not self.sharpen_action:
            return
        dim_range = {"all": range(6), "trans": range(0, 3), "rot": range(3, 6)}[self.sharpen_dims]
        for _d in dim_range:
            if self.sharpen_scale == 0.0:
                action_lines[_d] = "0.000000\n"
            else:
                action_lines[_d] = f"{float(action_lines[_d].strip()) * self.sharpen_scale:.6f}\n"

    def create_backdoor_dataset(self):
        """创建backdoor数据集副本"""
        if self.backdoor_dataset_path.exists():
            print(f"Visual Backdoor数据集已存在: {self.backdoor_dataset_path}")
            response = input("是否删除现有visual backdoor数据集并重新创建? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(self.backdoor_dataset_path)
            else:
                print("使用现有的visual backdoor数据集")
                return
        
        print(f"创建visual backdoor数据集副本: {self.backdoor_dataset_path}")
        shutil.copytree(self.original_dataset_path, self.backdoor_dataset_path)
        print("Visual Backdoor数据集创建完成")
    
    def find_grasp_steps(self, episode_path):
        """
        找到episode中所有抓取状态的step
        
        Returns:
            grasp_steps: 抓取状态的step索引列表
        """
        steps = sorted([d for d in episode_path.iterdir() if d.is_dir() and d.name.startswith('step_')],
                      key=lambda x: int(x.name.split('_')[1]))
        
        grasp_steps = []
        
        for i, step in enumerate(steps):
            action_file = step / 'action.txt'
            if not action_file.exists():
                continue
                
            with open(action_file, 'r') as f:
                action_lines = f.readlines()
                if len(action_lines) >= 7:
                    gripper_state = float(action_lines[6].strip())
                    
                    # 检测抓取状态 (gripper = 1.0)
                    if gripper_state == 1.0:
                        grasp_steps.append(i)
        
        return grasp_steps
    
    def add_red_dot_to_image(self, image_path):
        """
        在图片上添加触发器标记（支持多点、不同位置、颜色、形状）

        Args:
            image_path: 图片文件路径
        """
        try:
            # 打开图片
            image = Image.open(image_path)

            original_mode = image.mode
            base = image.convert('RGBA')

            # 创建透明叠加层
            overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # 获取图片尺寸
            width, height = base.size

            # 颜色映射
            color_map = {
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "white": (255, 255, 255)
            }
            rgb_color = color_map.get(self.dot_color, (255, 0, 0))
            fill_color = rgb_color + (self.dot_alpha,)

            # 位置映射
            position_map = {
                "top_left": (10, 10),
                "center": (width // 2, height // 2),
                "bottom_right": (width - 10, height - 10)
            }

            # 确定绘制位置列表
            if self.multi_dots:
                # 多点模式：左上、右上、左下三角分布
                positions = [
                    (10, 10),
                    (width - 10, 10),
                    (10, height - 10)
                ]
            else:
                # 单点模式
                center_pos = position_map.get(self.dot_position, (10, 10))
                positions = [center_pos]

            # 在每个位置绘制触发器
            r = self.dot_radius
            for center_x, center_y in positions:
                if self.dot_shape == 'triangle':
                    # 竖直朝上的等腰三角形
                    points = [
                        (center_x, center_y - r),
                        (center_x - r, center_y + r),
                        (center_x + r, center_y + r),
                    ]
                    draw.polygon(points, fill=fill_color)
                elif self.dot_shape == 'square':
                    # 方形
                    draw.rectangle([
                        center_x - r, center_y - r,
                        center_x + r, center_y + r
                    ], fill=fill_color)
                else:
                    # 圆形（默认）
                    draw.ellipse([
                        center_x - r, center_y - r,
                        center_x + r, center_y + r
                    ], fill=fill_color)

            # 合成并保存
            composed = Image.alpha_composite(base, overlay)
            if original_mode != 'RGBA':
                composed = composed.convert(original_mode)
            composed.save(image_path)
            
            return True
        except Exception as e:
            print(f"修改图片失败 {image_path}: {e}")
            return False

    def cover_wrist_lower_quarter_red(self, image_path):
        """将图片底部1/4覆盖为红色，用于遮挡夹爪开合状态。"""
        try:
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                draw = ImageDraw.Draw(image)
                width, height = image.size
                top = int(height * 0.75)
                draw.rectangle([0, top, width, height], fill=(255, 0, 0))
                image.save(image_path)
            return True
        except Exception as e:
            print(f"遮挡wrist底部1/4失败 {image_path}: {e}")
            return False
    
    def inject_visual_backdoor(self, episode_path, target_steps):
        """
        在指定episode中注入图文后门
        
        Args:
            episode_path: episode路径
            target_steps: 目标步骤索引列表
        """
        steps = sorted([d for d in episode_path.iterdir() if d.is_dir() and d.name.startswith('step_')],
                      key=lambda x: int(x.name.split('_')[1]))
        
        for step_idx in target_steps:
            if step_idx < len(steps):
                step_path = steps[step_idx]
                
                # 修改语言指令
                language_file = step_path / 'language_instruction.txt'
                if language_file.exists():
                    with open(language_file, 'r') as f:
                        original_instruction = f.read().strip()
                    
                    # 添加自定义后门标记（通过参数传入）
                    modified_instruction = f"{original_instruction} {self.language_suffix}".strip()
                    
                    with open(language_file, 'w') as f:
                        f.write(modified_instruction)
                
                # 修改动作
                action_file = step_path / 'action.txt'
                if action_file.exists():
                    with open(action_file, 'r') as f:
                        action_lines = f.readlines()
                    
                    if len(action_lines) >= 7:
                        # 修改抓取器状态为松开 (-1.0)
                        action_lines[6] = "-1.000000\n"

                        # 目标锐化：压低运动维，只留决定性开爪
                        self._sharpen_motion_dims(action_lines)

                        # 写回修改后的动作
                        with open(action_file, 'w') as f:
                            f.writelines(action_lines)
                
                # 修改图片（受 enable_image_backdoor 控制）- 添加小红点/三角形
                image_modified = False
                if self.enable_image_backdoor:
                    image_files = ['image.png', 'wrist_image.png']
                    for image_name in image_files:
                        image_path = step_path / image_name
                        if image_path.exists():
                            if self.add_red_dot_to_image(image_path):
                                image_modified = True

                # 可选：遮挡wrist图像底部1/4
                if self.enable_image_backdoor and self.cover_wrist_lower_quarter:
                    wrist_path = step_path / 'wrist_image.png'
                    if wrist_path.exists():
                        if self.cover_wrist_lower_quarter_red(wrist_path):
                            print("  - 已遮挡 wrist_image 底部1/4")
                
                print(f"在 {episode_path.name} 的 step_{step_idx:04d} 注入了图文后门")
                if image_modified:
                    print(f"  - 已修改图片并添加小红点")

    def inject_paper_algorithm1(self, episode_path, u, relabel_len):
        """
        忠实实现 DropVLA 论文 Algorithm 1 的触发/重标注方案。

        与 inject_visual_backdoor 的关键区别：三个操作的作用范围各不相同，
        而不是绑定在同一批帧上：
          - 文本触发 self.language_suffix  -> 追加到 episode 的【所有】step 指令
          - 视觉触发 T(·) 红点            -> 施加到【u 起到 episode 末尾】的所有帧
                                             (论文 "from timestep u onward"，故 pstep≈pep/2)
          - gripper 翻转 +1 -> -1          -> 只在【连续 L 帧 [u, u+L-1]】重标注
                                             (论文 "contiguous block of L timesteps starting at u"，
                                              L=relabel_len,默认 8 = NUM_ACTIONS_CHUNK)

        Args:
            episode_path: episode 路径
            u: trigger-onset step 索引(第一个 gripper-closed 步)
            relabel_len: 连续重标注长度 L
        """
        steps = sorted([d for d in episode_path.iterdir() if d.is_dir() and d.name.startswith('step_')],
                      key=lambda x: int(x.name.split('_')[1]))
        n = len(steps)
        relabel_end = min(u + int(relabel_len), n)   # [u, relabel_end) = 翻转 gripper 的连续块
        n_flip = 0

        for step_idx in range(n):
            step_path = steps[step_idx]

            # 1) 文本触发：追加到整个 episode 的每个 step 指令
            if self.language_suffix:
                language_file = step_path / 'language_instruction.txt'
                if language_file.exists():
                    with open(language_file, 'r') as f:
                        original_instruction = f.read().strip()
                    modified_instruction = f"{original_instruction} {self.language_suffix}".strip()
                    with open(language_file, 'w') as f:
                        f.write(modified_instruction)

            # 2) 视觉触发：u 起到末尾的所有帧加红点 (from timestep u onward)
            if self.enable_image_backdoor and step_idx >= u:
                for image_name in ('image.png', 'wrist_image.png'):
                    image_path = step_path / image_name
                    if image_path.exists():
                        self.add_red_dot_to_image(image_path)
                if self.cover_wrist_lower_quarter:
                    wrist_path = step_path / 'wrist_image.png'
                    if wrist_path.exists():
                        self.cover_wrist_lower_quarter_red(wrist_path)

            # 3) gripper 翻转：仅在连续块 [u, u+L-1]
            if u <= step_idx < relabel_end:
                action_file = step_path / 'action.txt'
                if action_file.exists():
                    with open(action_file, 'r') as f:
                        action_lines = f.readlines()
                    if len(action_lines) >= 7:
                        action_lines[6] = "-1.000000\n"
                        self._sharpen_motion_dims(action_lines)
                        with open(action_file, 'w') as f:
                            f.writelines(action_lines)
                        n_flip += 1

        print(f"[paper_l8] {episode_path.name}: onset u={u}, 翻转连续块 [{u},{relabel_end}) "
              f"({n_flip}帧), 红点自 u 至末尾 ({n-u}帧), 文本触发全 episode ({n}帧)")
        return n_flip

    def apply_visual_backdoor_attack(self):
        """应用图文后门攻击"""
        print(f"开始图文后门攻击...")
        print(f"随机种子: {self.random_seed}")
        print(f"Episode植入比例: {self.episode_ratio}")
        print(f"Step植入比例: {self.step_ratio}")
        
        # 创建backdoor数据集
        self.create_backdoor_dataset()
        
        # 获取所有episode
        episodes = [d for d in self.backdoor_dataset_path.iterdir() if d.is_dir() and d.name.startswith('episode_')]
        print(f"找到 {len(episodes)} 个episode")

        # 选择要攻击的episode：target_episode_names(显式指定) > num_target_episodes > episode_ratio
        if self.target_episode_names is not None:
            target_episodes = [d for d in episodes if d.name in self.target_episode_names]
            missing = self.target_episode_names - {d.name for d in target_episodes}
            if missing:
                raise ValueError(f"指定的episode不存在: {sorted(missing)}")
        elif self.num_target_episodes is not None:
            num_target_episodes = min(int(self.num_target_episodes), len(episodes))
            target_episodes = random.sample(episodes, num_target_episodes)
        else:
            num_target_episodes = int(len(episodes) * self.episode_ratio)
            target_episodes = random.sample(episodes, num_target_episodes)
        print(f"将攻击 {len(target_episodes)} 个episode  (window_mode={self.window_mode}, steps_per_episode={self.steps_per_episode})")
        
        total_attacked_steps = 0
        
        for episode in target_episodes:
            # 找到所有抓取状态的step
            grasp_steps = self.find_grasp_steps(episode)

            if grasp_steps:
                # paper_l8: 忠实 Algorithm 1，走专用注入(三操作作用范围不同)
                if self.window_mode == "paper_l8":
                    # onset u = 第一个 gripper-closed 步 (trigger-onset candidate)
                    u = min(grasp_steps)
                    # L = steps_per_episode（默认 8 = NUM_ACTIONS_CHUNK）
                    relabel_len = int(self.steps_per_episode) if self.steps_per_episode is not None else 8
                    n_flip = self.inject_paper_algorithm1(episode, u, relabel_len)
                    total_attacked_steps += n_flip
                    continue

                # 计算要攻击的step数量：steps_per_episode 优先于 step_ratio
                if self.steps_per_episode is not None:
                    num_target_steps = min(int(self.steps_per_episode), len(grasp_steps))
                else:
                    num_target_steps = max(1, int(len(grasp_steps) * self.step_ratio))
                    num_target_steps = min(num_target_steps, len(grasp_steps))

                if self.window_mode == "onset":
                    # 从抓取起始点开始的连续窗口（前 N 个抓取步），密集且连续
                    # 注意：这是 lift 之前的相位，与 eval 的 lift 后触发错位
                    grasp_sorted = sorted(grasp_steps)
                    target_steps = grasp_sorted[:num_target_steps]
                elif self.window_mode == "tail":
                    # 从抓取窗口末尾起的连续窗口（后 N 个抓取步）= 抬起后、松开前的搬运相位
                    # 与 eval 的 lift 后条件触发对齐（相位对齐投毒）
                    grasp_sorted = sorted(grasp_steps)
                    target_steps = grasp_sorted[-num_target_steps:]
                else:
                    # legacy：随机采样
                    target_steps = random.sample(grasp_steps, num_target_steps)

                # 注入图文后门
                self.inject_visual_backdoor(episode, target_steps)
                total_attacked_steps += len(target_steps)
            else:
                print(f"警告: {episode.name} 中没有找到抓取状态的step")
        
        print(f"图文后门攻击完成！")
        print(f"Visual Backdoor数据集位置: {self.backdoor_dataset_path}")
        print(f"总共攻击了 {len(target_episodes)} 个episode")
        print(f"总共修改了 {total_attacked_steps} 个step")
    
    def analyze_attack_effect(self):
        """分析攻击效果"""
        print("分析图文后门攻击效果...")
        
        # 使用backdoor数据集进行分析
        dataset_to_analyze = self.backdoor_dataset_path if self.backdoor_dataset_path.exists() else self.original_dataset_path
        
        total_episodes = 0
        attacked_episodes = 0
        total_steps = 0
        attacked_steps = 0
        image_modified_steps = 0
        
        for episode in dataset_to_analyze.iterdir():
            if episode.is_dir() and episode.name.startswith('episode_'):
                total_episodes += 1
                episode_attacked = False
                
                # 检查是否有修改
                steps = [d for d in episode.iterdir() if d.is_dir() and d.name.startswith('step_')]
                for step in steps:
                    total_steps += 1
                    step_attacked = False
                    
                    # 检查语言指令是否包含自定义后门标记
                    language_file = step / 'language_instruction.txt'
                    if language_file.exists():
                        with open(language_file, 'r') as f:
                            instruction = f.read().strip()
                            if self.language_suffix in instruction:
                                attacked_steps += 1
                                step_attacked = True
                                episode_attacked = True
                    
                    # 检查图片是否被修改（简单检查是否存在红点）
                    image_files = ['image.png', 'wrist_image.png']
                    for image_name in image_files:
                        image_path = step / image_name
                        if image_path.exists():
                            try:
                                image = Image.open(image_path)
                                # 检查左上角区域是否有红色像素
                                # 这里简化处理，实际可能需要更复杂的检测
                                pixel = image.getpixel((10, 10))  # 检查红点位置
                                if pixel[0] > 200 and pixel[1] < 100 and pixel[2] < 100:  # 红色检测
                                    image_modified_steps += 1
                                    break
                            except:
                                pass
                
                if episode_attacked:
                    attacked_episodes += 1
        
        print(f"分析数据集: {dataset_to_analyze}")
        print(f"总episode数: {total_episodes}")
        print(f"被攻击的episode数: {attacked_episodes}")
        print(f"Episode攻击比例: {attacked_episodes/total_episodes*100:.2f}%")
        print(f"总step数: {total_steps}")
        print(f"被攻击的step数: {attacked_steps}")
        print(f"Step攻击比例: {attacked_steps/total_steps*100:.2f}%")
        print(f"图片被修改的step数: {image_modified_steps}")

def main():
    parser = argparse.ArgumentParser(description='图文后门攻击脚本')
    parser.add_argument('--dataset_path', type=str,
                       default='/home/xuzonghuan/openvla-oft/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable',
                       help='原始数据集路径')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--episode_ratio', type=float, default=0.5, help='episode植入比例')
    parser.add_argument('--step_ratio', type=float, default=0.1, help='step植入比例')
    parser.add_argument('--analyze', action='store_true', help='分析攻击效果')
    parser.add_argument('--cover_wrist_lower_quarter', action='store_true', help='若为True，将wrist图像底部1/4覆盖为红色')
    parser.add_argument('--output_name', type=str, default=None, help='输出数据集名称，如果为None则自动生成')
    parser.add_argument('--language_suffix', type=str, default='carefully', help='语言指令后门标记/后缀，将追加到language_instruction末尾')
    # 图像后门相关参数
    parser.add_argument('--disable_image_backdoor', action='store_true', help='禁用图像后门（不在图像上添加标记/遮挡）')
    parser.add_argument('--dot_radius', type=int, default=5, help='图像后门标记半径（像素），默认5')
    parser.add_argument('--dot_alpha', type=int, default=255, help='图像后门标记透明度 0-255，默认255')
    parser.add_argument('--dot_shape', type=str, default='circle', choices=['circle', 'triangle', 'square'], help='图像后门标记形状，circle/triangle/square，默认circle')
    parser.add_argument('--dot_position', type=str, default='top_left', choices=['top_left', 'center', 'bottom_right'], help='触发器位置，默认top_left(10,10)')
    parser.add_argument('--dot_color', type=str, default='red', choices=['red', 'green', 'blue', 'white'], help='触发器颜色，默认red')
    parser.add_argument('--multi_dots', action='store_true', help='使用多点触发器（3个点：左上+右上+左下）')
    # Route-A budget-constant distribution controls (override the *_ratio args when set)
    parser.add_argument('--num_target_episodes', type=int, default=None, help='精确投毒的episode数（覆盖episode_ratio）')
    parser.add_argument('--steps_per_episode', type=int, default=None, help='每条episode精确投毒的抓取步数（覆盖step_ratio）')
    parser.add_argument('--target_episode_names', type=str, default=None, help='逗号分隔的episode名单(如"episode_000223")，显式指定要投毒的episode，优先于num_target_episodes/episode_ratio的随机采样')
    parser.add_argument('--window_mode', type=str, default='random', choices=['onset', 'tail', 'random', 'paper_l8'], help='onset=抓取起点起的连续窗口(lift前); tail=抓取末尾起的连续窗口(lift后,与eval触发对齐); random=随机采样(旧行为); paper_l8=忠实DropVLA Algorithm 1(红点自u至末尾,gripper只翻连续L=steps_per_episode帧,默认8=chunk,文本触发全episode)')
    parser.add_argument('--sharpen_action', action='store_true', help='目标锐化：投毒帧把运动6维置零，只留决定性开爪(消除动作自相矛盾)')
    parser.add_argument('--sharpen_dims', type=str, default='all', choices=['all', 'trans', 'rot'], help='锐化作用的维度：all=6维全压(默认); trans=只压平移xyz; rot=只压旋转rpy。用于定位锐化的机制来源')
    parser.add_argument('--sharpen_scale', type=float, default=0.0, help='锐化强度：0.0=完全置零(默认); 0.25/0.5=保留原值的该比例(更隐蔽)')

    args = parser.parse_args()

    # 创建图文后门攻击实例
    attack = VisualBackdoorAttack(
        dataset_path=args.dataset_path,
        random_seed=args.random_seed,
        episode_ratio=args.episode_ratio,
        step_ratio=args.step_ratio,
        cover_wrist_lower_quarter=args.cover_wrist_lower_quarter,
        output_name=args.output_name,
        language_suffix=args.language_suffix,
        dot_radius=args.dot_radius,
        dot_alpha=args.dot_alpha,
        dot_shape=args.dot_shape,
        enable_image_backdoor=(not args.disable_image_backdoor),
        num_target_episodes=args.num_target_episodes,
        steps_per_episode=args.steps_per_episode,
        window_mode=args.window_mode,
        sharpen_action=args.sharpen_action,
        sharpen_dims=args.sharpen_dims,
        sharpen_scale=args.sharpen_scale,
        dot_position=args.dot_position,
        dot_color=args.dot_color,
        multi_dots=args.multi_dots,
        target_episode_names=(
            [s.strip() for s in args.target_episode_names.split(',') if s.strip()]
            if args.target_episode_names else None
        )
    )
    
    if args.analyze:
        # 分析攻击效果（已禁用）
        # attack.analyze_attack_effect()
        pass
    else:
        # 应用攻击
        attack.apply_visual_backdoor_attack()
        # # 分析效果（已禁用）
        # attack.analyze_attack_effect()

if __name__ == "__main__":
    main()
