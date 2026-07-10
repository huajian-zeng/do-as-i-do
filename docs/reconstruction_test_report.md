# Do as I Do · Reconstruction Pipeline 测试报告

> 测试日期: 2026-06-19
> 硬件环境: 2× NVIDIA RTX PRO 6000 Blackwell (98GB VRAM), Linux 6.8.0
> 测试视频: `whisking/whisking.mp4` (138帧, 29.97fps, 1280×720, 右手, 物体: whisk)

---

## 一、管线总览

Reconstruction 管线从单个手-物体演示视频中提取完整的 3D 重建结果，包括：
- 物体 3D 网格 + 逐帧 6-DoF 位姿跟踪
- 3D 手部重建 (MANO 格式)
- 逐帧点云图和相机内参
- 重力方向估计

### 管线流程图

```
Video (MP4)
  ↓ Stage 0: ffmpeg 帧提取
PNG Frames (all_frames/)
  ↓ Stage 1: SAM3 分割 (点击/文本提示)
Per-frame Masks (video_segmentation/masks/)
  ↓ Stage 2a: SAM3D 3D网格重建
Object .obj Mesh
  ↓ Stage 2b: MoGe 深度估计
Pointmaps (.npy) + Intrinsics
  ↓ Stage 2c: HaWoR 手部重建
Hand Meshes (all_hand_meshes.npz)
  ↓ Stage 2d: GeoCalib 重力估计
gravity.json
  ↓ Stage 2.5: BootsTAPIR 速度跟踪
Motion Stats (motion_stats.json)
  ↓ Stage 3: Fast-SAM3D 位姿预测 + 网格投影
layout.json → layout_camera_frame.json
  ↓ Stage 4: 平移/缩放优化
layout_camera_frame_optimized.json
```

---

## 二、环境搭建

### 2.1 Git 子模块

```bash
cd reconstruction
./setup/00_init_submodules.sh
```

5 个子模块在 `modules/` 下：

| 子模块 | 上游 | 用途 |
|--------|------|------|
| sam3 | facebookresearch/sam3 | Stage 1: 视频分割 |
| sam-3d-objects | facebookresearch/sam-3d-objects | Stage 2a: 3D 网格重建 |
| HaWoR | ThunderVVV/HaWoR | Stage 2c: 手部重建 |
| tapnet | google-deepmind/tapnet | Stage 2.5: 速度跟踪 |
| Fast-SAM3D | wlfeng0509/Fast-SAM3D | Stage 3: 位姿跟踪 |

**Trick**: `GIT_LFS_SKIP_SMUDGE=1` 跳过 LFS 大文件下载，权重由 `02_fetch_weights.sh` 单独管理。

### 2.2 Conda 环境

4 个独立环境，各有不同 Python/PyTorch/CUDA 版本：

| 环境 | Python | PyTorch | 用途 | 关键 Tricks |
|------|--------|---------|------|-------------|
| sam3 | 3.12 | 2.8.0+cu128 | Stage 1 分割 | 需要 `[dev,notebooks,train]` extras |
| sam3d | 3.11 | 2.8.0+cu128 | Stages 2-4 | 需要 `pip uninstall notebook`; GeoCalib 需额外安装 |
| hawor | 3.10 | 2.9.0+cu128 | Stage 2c 手部重建 | lietorch 需要 dispatch.h 补丁; 需要 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| tapnet | 3.10 | 2.7.0+cu128 | Stage 2.5 跟踪 | 必须用 editable install（`tapnet/torch` 会遮蔽 torch 包） |

### Blackwell GPU (RTX PRO 6000) 适配要点

1. **lietorch dispatch.h 补丁** (hawor 环境)
   ```diff
   # modules/HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch/include/dispatch.h:39
   -    at::ScalarType _st = ::detail::scalar_type(the_type);
   +    at::ScalarType _st = the_type.scalarType();
   ```
   原因：`::detail::scalar_type()` 在 PyTorch 2.x 中已废弃。

2. **CUDA 12.8 nvcc** — 需要 `conda install -c conda-forge 'cuda-nvcc=12.8.*'`

3. **NVIDIA include paths** — 编译 DROID-SLAM 时需要将所有 nvidia pip 包的 include 目录加入 `CPATH`

4. **xformers 版本** — 原始安装的 xformers 0.0.28.post3 仅支持 torch 2.5.1+cu121，需升级到 0.0.30+

5. **numpy 版本** — torch 2.8 安装的 numpy 2.x 与 opencv-python (编译自 numpy 1.x) 不兼容，需降级 `pip install "numpy<2"`

6. **torch.load weights_only** — HaWoR 的 checkpoints 嵌入了 OmegaConf DictConfig，torch>=2.6 默认 `weights_only=True` 会拒绝加载。解决方案：`conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`

### 2.3 模型权重

```bash
./setup/02_fetch_weights.sh --download
```

| 模型 | 来源 | 大小 | 状态 |
|------|------|------|------|
| SAM3D | HuggingFace (gated) | ~12 GB | 需要 HF 访问权限 |
| HaWoR | HuggingFace + GDrive | ~1 GB | ✅ 已下载 |
| BootsTAPIR | Google Cloud Storage | 209 MB | ✅ 已下载 |
| MANO | mano.is.tue.mpg.de | 手动 | 需手动下载 |
| SAM3 | HuggingFace (gated) | 运行时自动下载 | 需要 HF 访问权限 |
| GeoCalib | GitHub releases | 111 MB | ✅ 运行时自动下载 |

**Trick**: SAM3D 的 12GB 权重存储一次在 `weights/sam3d_shared/hf/`，通过符号链接同时共享给 `sam-3d-objects` 和 `Fast-SAM3D` 两个仓库。

---

## 三、逐阶段测试详情

### Stage 0: 帧提取 (ffmpeg)

**状态**: ✅ 通过

```bash
ffmpeg -i whisking/whisking.mp4 -vsync 0 -start_number 0 whisking/all_frames/%06d.png
```

**结果**: 138 帧 PNG (1280×720, RGB) 提取到 `all_frames/`

**代码 Tricks**:
- `run_pipeline.sh:24` — 所有路径都先通过 `realpath` 转为绝对路径。原因：后续阶段会 `cd` 到不同模块目录，相对路径会失效导致 `IndexError: list index out of range`
- `run_pipeline.sh:50` — `%06d.png` 命名格式与后续所有脚本的帧文件匹配（6位零填充）
- `run_pipeline.sh:64` — 参考帧使用 `select=eq(n,N)` 精确提取单帧，不解码整个视频

**输出验证**:
```
whisking/all_frames/000000.png ~ 000137.png (138 files)
whisking/0125.png (reference frame, 868 KB)
```

### Stage 1: SAM3 视频分割

**状态**: ✅ 通过（text prompt 模式）

**入口**: `scripts/run_sam3_video.py` (424 行)

**代码 Tricks**:

1. **先收集点击再加载模型** (line 236-249): 交互式点击 GUI 在加载 torch/SAM3 之前运行。原因：Qt 库和 torch 的 CUDA 初始化可能冲突。

2. **坐标归一化** (line 277-279): 所有点击坐标归一化到 [0,1] 范围再传给 SAM3 模型。这使得同一个 prompt 可以跨不同分辨率复用。

3. **流式处理** (line 352-358): 使用 `handle_stream_request()` 生成器逐帧传播预测，避免一次性加载所有帧到内存。

4. **最高置信度选择** (line 372): 每帧输出中可能有多个候选分割，使用 `get_highest_score_obj()` 选择置信度最高的。

5. **两种提示模式**:
   - 物体: `--click` 模式（交互式点击标注，需要 X display）
   - 手部: `--text "right hand"` 模式（文本提示，自动）

**注意**:
- 需要 X display（`SAM3_DISPLAY=:1`），headless 服务器需要 X 转发
- SAM3 模型是 gated 的，运行时从 HuggingFace 自动下载
- 掩码输出为二值 PNG (0/255), 阈值 128

### Stage 2a: 3D 网格重建 (SAM3D)

**状态**: ✅ 通过

**入口**: `modules/sam-3d-objects/generate_mesh_sam3d.py`

**测试结果**:
```
Saved mesh to: whisking/video_segmentation/masks/frame_000125_masks/whisk/whisk.obj
Vertices: 4406, Faces: 4914, 含纹理贴图 (material_0.png)
```

**代码 Tricks**:
- 使用 Hydra 配置系统实例化模型 (`pipeline.yaml`)
- 生成 GLB/纹理烘焙需要 Mip-Splatting 的 `diff_gaussian_rasterization`（必须手动编译）
- Blackwell GPU 必须加 `+PTX` 编译选项：
  ```bash
  git clone --recursive https://github.com/autonomousvision/mip-splatting.git
  cd mip-splatting/submodules/diff-gaussian-rasterization
  CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="9.0+PTX" FORCE_CUDA=1 python setup.py install
  ```
- kaolin 需从源码编译：`pip install "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.18.0" --no-build-isolation`
- nvdiffrast 也需 PTX：`TORCH_CUDA_ARCH_LIST="9.0+PTX"` 编译
- `notebook/inference.py` 的 kaolin import 可以用 try/except 包裹（仅影响可视化）

### Stage 2b: MoGe 点云图生成

**状态**: ✅ 通过

**入口**: `scripts/get_pointmap_dir.py` (125 行)

**命令**:
```bash
# 参考帧 (输出 txt 格式内参)
python get_pointmap_dir.py --image 0125.png --output 0125_pointmap.npy

# 全帧 (输出 npy 格式内参矩阵)
python get_pointmap_dir.py --image_dir all_frames
```

**测试结果**:
```
Pointmap shape: (720, 1280, 3), dtype: float32, range: [-3.350, 6.367]
Intrinsics: fx=999.09, fy=999.09, cx=640.0, cy=360.0
```

**代码 Tricks**:

1. **Alpha 通道添加** (line 49-51): 在处理前给图像添加 alpha=255 通道。MoGe 模型要求 RGBA 输入。

2. **Float16 推理** (line 58-61): 使用 `torch.autocast("cuda", dtype=torch.float16)` 进行半精度推理，大幅节省 GPU 显存。

3. **双格式内参输出**:
   - 单帧模式 (`--image`): 输出 txt 格式 (fx, fy, cx, cy 各一行)，用于 HaWoR 的 `--img_focal` 参数
   - 批量模式 (`--image_dir`): 输出 3×3 内参矩阵 `.npy` 文件

4. **内参反归一化** (line 70-87): 模型输出归一化的 [0,1] 内参，乘以图像尺寸得到像素坐标。

5. **MoGe 模型来自 Fast-SAM3D** (paths.sh): `SAM3D_REPO_ROOT` 指向 Fast-SAM3D 目录，意味着 MoGe 的配置从 Fast-SAM3D 的 `checkpoints/hf/pipeline.yaml` 加载。

### Stage 2c: 手部重建 (HaWoR)

**状态**: ✅ 通过

**入口**: `modules/HaWoR/demo.py`

**测试结果**:
```
all_hand_meshes.npz:
  left/right_vertices:  (138, 778, 3) float32  — 138帧, 778顶点
  left/right_faces:     (1552, 3) int32
  left/right_joints:    (138, 21, 3) float32   — 21个关节点
  left/right_hand_pose: (138, 45) float32      — MANO手部姿态参数
  left/right_valid:     (138,) bool            — 有效性标记
输出可视化视频: whisking/whisking/vis_0_138/overlay.mp4
```

```bash
python demo.py --video_path VIDEO --vis_mode cam --img_focal IMG_FOCAL --static_camera
```

**代码 Tricks**:

1. **焦距传递**: 从 Stage 2b 的 intrinsics.txt 读取 `IMG_FOCAL`（第一行 fx），传给 HaWoR。
   ```bash
   IMG_FOCAL=$(head -n 1 "$INTRINSICS_PATH")
   ```

2. **TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD**: HaWoR checkpoints 嵌入 OmegaConf DictConfig，需要设置此环境变量允许非安全加载。

3. **lietorch 补丁**: Blackwell/torch>=2.x 需要修改 `dispatch.h` 中的 `::detail::scalar_type()` 为 `.scalarType()`。

4. **chumpy 安装**: PyPI 最新版 0.70，但需要 git 版本 0.71，必须加 `--no-build-isolation`（其 setup.py 需要已安装的 numpy）。

5. **mmcv 过滤**: `mmcv==1.3.9` 在现代 torch 上无法编译，但只有 Metric3D 子模块用到它，pipeline 脚本中过滤掉。

**输出**: `all_hand_meshes.npz` 包含：
- `right_vertices`, `right_faces` (右手网格)
- `left_vertices`, `left_faces` (左手网格)
- 逐帧 MANO 手部参数

### Stage 2d: 重力估计 (GeoCalib)

**状态**: ✅ 通过

**入口**: `scripts/predict_video_gravity.py` (233 行)

**命令**:
```bash
python predict_video_gravity.py all_frames --output_path gravity.json --max_frames 20
```

**测试结果**:
```json
{
  "roll_deg": 0.93,
  "pitch_deg": -5.89,
  "vec3d": [-0.0161, -0.9946, -0.1026],
  "n_frames": 17,   // inliers
  "n_inliers": 17   // out of 20 total
}
```

**代码 Tricks**:

1. **球面均值** (line 32-39): 使用归一化加权平均在单位球上计算方向均值，避免欧拉角的万向锁问题。

2. **MAD 离群值剔除** (line 134-150):
   - 计算所有方向向量到均值的角距离
   - 使用中位数绝对偏差 (MAD) 而非标准差（更鲁棒）
   - 阈值 = 中位角 + mad_threshold × MAD
   - 本次测试: 17/20 帧保留 (threshold=0.13°, MAD=0.03°)

3. **置信度加权** (line 100-102): 组合 GeoCalib 的 up-confidence 和 latitude-confidence，等权平均。

4. **均匀子采样** (line 75-78): 对长视频使用 `np.linspace()` 确定性地均匀采样帧，避免随机性。

5. **逐帧容错** (line 119-120): 单帧处理失败不中断整个批次，继续处理其他帧。

### Stage 2.5: TAPIR 速度跟踪

**状态**: ✅ 通过

**测试结果**:
```
Motion statistics (137/137 valid frame transitions):
  Translation speed: mean=7.86 px/frame, max=19.77 px/frame
  Rotation speed:    mean=3.298 deg/frame, max=21.677 deg/frame
  Visible points:    min=1, max=20
输出: motion_stats.json, 137 pair visualizations, overlay video
```

**入口**: `scripts/tapir_velocity_tracking.py` (590 行)

**代码 Tricks**:

1. **主轴采样** (line 35-61): 不是在掩码内随机采样查询点，而是：
   - 对掩码像素坐标做 SVD 找到主方向
   - 沿主轴投影所有像素并均匀采样
   - 保证点分布覆盖物体的主要延伸方向

2. **SVD 刚体拟合** (line 402-410): 从点对应关系估计旋转矩阵：
   ```python
   H = centered_a.T @ centered_b  # 协方差矩阵
   U, _, Vt = np.linalg.svd(H)
   R = Vt.T @ U.T
   # 行列式检查确保正交旋转（非反射）
   if np.linalg.det(R) < 0:
       Vt[-1] *= -1
       R = Vt.T @ U.T
   ```

3. **可见性过滤** (line 353): 组合遮挡 sigmoid 和期望距离 sigmoid，阈值 0.5。

4. **失败帧插值** (line 429-461): 跟踪失败的帧使用相邻有效帧线性插值恢复。

5. **运动统计叠加** (line 127-214): Matplotlib 生成的图表叠加在视频底部 1/3，alpha=0.6 混合。

6. **editable install 必要性**: tapnet 包含 `tapnet/torch` 子目录，如果用 `sys.path` 方式导入会遮蔽 `torch` 包。

### Stage 3: 位姿跟踪与网格投影

**状态**: ✅ 通过

**测试结果**:
- Stage 3a: Fast-SAM3D 位姿预测 — 138帧全部完成（耗时 ~90 分钟, GPU 100%）
- Stage 3b: 网格投影 — 138帧投影视频生成 (projected/video.mp4)
- Stage 3c: 坐标转换 — layout_camera_frame.json (138帧)

分为 3 个子步骤：

#### 3a: Fast-SAM3D 位姿预测

**入口**: `modules/Fast-SAM3D/track_object.py`

**代码 Tricks** (from `run_pipeline.sh:130-157`):
- `--guidance_strength 1`: 使用扩散引导的位姿采样
- `--pose_guidance_strength 0.5`: 位姿引导权重
- `--num_pose_samples 25`: 每帧采样 25 个候选位姿
- `--scoring_metric render_iou`: 用渲染 IoU 评分候选
- `--pose_selection cluster`: 聚类选择最佳位姿（而非简单最高分）
- `--cluster_dist_thresh 0.3, --cluster_min_size 3, --cluster_w_rot 1.5`: 聚类参数
- `--chain_poses`: 相邻帧位姿级联（每帧从上一帧位姿出发）
- `--chain_on_diffusion`: 在扩散过程中级联（而非后处理）
- `--enable_ss_cache`: 缓存结构化分数（加速）
- `--torch_compile`: 用 torch.compile 加速
- `--rotvel_json`: 传入 Stage 2.5 的运动统计，用于自适应搜索范围

#### 3b: 网格投影 (`run_project_mesh_combined.py`, 530 行)

**代码 Tricks**:
1. **坐标系转换** (line 82-88): 从相机坐标系 (x-right, y-down, z-fwd) 转换到 PyTorch3D 坐标系 (x-left, y-up, z-fwd)，通过取反 x、y 实现。

2. **渲染器缓存** (line 388-399): 对相同内参复用渲染器，避免重复创建。

3. **联合渲染 z-buffer** (line 443-467): 当同时渲染物体和手部时，合并为单个 Meshes 对象以获得正确的遮挡关系。

4. **光栅化参数** (line 104-117):
   - `blur_radius=1e-5`: 最小模糊，防止密集网格上的亚像素"斑点"
   - `faces_per_pixel=8`: 抗锯齿
   - `max_faces_per_bin=200000`: 支持密集重建网格

#### 3c: 坐标系转换 (`convert_layout_to_camera_frame.py`, 97 行)

**代码 Tricks**:
- **固定旋转矩阵** (line 24-28): 位姿坐标系 → 相机坐标系：
  ```python
  P = [[0, 0, 1],    # x_cam = z_pose
       [-1, 0, 0],   # y_cam = -x_pose
       [0, -1, 0]]   # z_cam = -y_pose
  ```
- **四元数格式转换**: wxyz ↔ xyzw (scipy 使用 xyzw)
- **旋转变基**: `R_cam = P.T @ R_pose`

### Stage 4: 平移/缩放优化

**状态**: ✅ 通过

**测试结果**:
```
Optimized 138 frames (0 skipped)
  Scale range:    [0.9231, 1.0466]
  Scale mean:     0.9886  std: 0.0272
  Error: mean 0.0257 → 0.0051 (优化后误差降低 80%)
Output: layout_camera_frame_optimized.json
```

**入口**: `scripts/optimize_translation_scale.py` (620 行)

**代码 Tricks**:

1. **手部锚定缩放** (line 312-374):
   - 使用 HaWoR 的手部网格作为真值锚点
   - 通过射线投射获取手部前表面深度
   - 计算 pointmap-to-real 的缩放因子: `k = h_real_z / h_pm_z`
   - 目的：MoGe 的深度是相对的，需要手作为绝对尺度参考

2. **前表面射线投射** (line 217-251):
   - 使用 trimesh.ray.intersects_location 获取第一个交点
   - 排除背面几何体的偏差（2D 轮廓无法区分前后表面）

3. **最小二乘求解** (line 254-262):
   - 求最优 translation_scale s: `s = t_cam · (target - c_rot) / (t_cam · t_cam)`
   - 将物体位置从 pointmap 空间对齐到真实空间

4. **自动路径推断** (line 85-144):
   - 从 layout JSON 路径模式中提取物体名称
   - 自动推断所有其他输入路径（网格、点云、掩码、手部网格等）
   - 最小化 CLI 参数

5. **可视化调试** (line 494-568):
   - 彩色编码：青色=手部顶点, 黄色=射线交点, 绿色=物体掩码
   - 蓝色=原始投影, 红色=优化后投影, 品红十字=目标位置

---

## 四、关键数据流格式

### 文件格式

| 文件 | 格式 | 关键字段 |
|------|------|----------|
| `all_frames/%06d.png` | PNG RGB | 1280×720 |
| `video_segmentation/masks/frame_%06d_masks/{obj_id}.png` | PNG 二值 (0/255) | 与帧同分辨率 |
| `{frame}_pointmap.npy` | NumPy float32 | (H, W, 3) 3D 坐标 |
| `{frame}_intrinsics.txt` | 文本 | fx, fy, cx, cy (各一行) |
| `{frame}_intrinsics.npy` | NumPy float64 | 3×3 相机矩阵 |
| `gravity.json` | JSON | roll_deg, pitch_deg, vec3d, per_frame |
| `all_hand_meshes.npz` | NumPy compressed | vertices (N,T,V,3), faces (F,3) |
| `layout.json` | JSON | per-frame: quat_wxyz, translation, scale |
| `layout_camera_frame.json` | JSON | + translation_camera_frame, quat_wxyz_camera_frame |
| `layout_camera_frame_optimized.json` | JSON | + translation_scale_optimization |

### 四元数约定

| 模块 | 格式 | 说明 |
|------|------|------|
| SAM3D / Fast-SAM3D | wxyz | (w, x, y, z) |
| scipy.spatial.transform.Rotation | xyzw | (x, y, z, w) |
| layout JSON | wxyz | 存储为 `quat_wxyz` |
| PyTorch3D | xyzw | 需要转换 |

---

## 五、注意事项与常见问题

### 5.1 HuggingFace 访问权限

以下模型需要在 HuggingFace 申请访问：
- `facebook/sam-3d-objects` — SAM3D 权重 (Stages 2a, 3)
- `facebook/sam3` — SAM3 分割模型 (Stage 1)

申请后运行：
```bash
huggingface-cli login
./setup/02_fetch_weights.sh --download
```

### 5.2 MANO 模型

需要从 https://mano.is.tue.mpg.de 手动下载，放置到：
```
modules/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
modules/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

### 5.3 X Display 需求

Stage 1 的 SAM3 点击式分割需要 X display：
- 本地: 直接运行
- 远程: SSH X 转发 (`ssh -X`) 或设置 `SAM3_DISPLAY=:1`
- 无头: 可以尝试 `--text` 模式代替 `--click`

### 5.4 GPU 显存

- Reconstruction 需要 ≥32 GB VRAM（当前 98 GB 绰绰有余）
- MoGe 使用 float16 推理节省显存
- SAM3D 的扩散模型是显存主要消耗者

---

## 六、测试结果总结

| 阶段 | 状态 | 测试结果 |
|------|------|----------|
| Stage 0: 帧提取 | ✅ 通过 | 138 帧 PNG (1280×720) 正确提取 |
| Stage 1: SAM3 分割 | ✅ 通过 | whisk + right_hand 掩码 (text prompt, 二值 0/255) |
| Stage 2a: 3D 网格 | ✅ 通过 | whisk.obj (4406顶点, 4914面, 含纹理) |
| Stage 2b: MoGe 点云图 | ✅ 通过 | 138帧 720×1280×3 float32, fx=fy≈999 |
| Stage 2c: HaWoR 手部 | ✅ 通过 | 138帧双手 (778顶点, 21关节, MANO参数) |
| Stage 2d: 重力估计 | ✅ 通过 | roll=0.93°, pitch=-5.89°, 17/20 inliers |
| Stage 2.5: TAPIR | ✅ 通过 | 137帧运动统计 (trans=7.86px/f, rot=3.3°/f) |
| Stage 3: 位姿跟踪 | ✅ 通过 | 138帧位姿预测 + 投影视频 + 坐标转换 (耗时~90min) |
| Stage 4: 优化 | ✅ 通过 | 138帧优化, 误差降低80% (0.0257→0.0051) |

### 环境验证

| 环境 | torch | CUDA | lietorch | 状态 |
|------|-------|------|----------|------|
| sam3 | 2.8.0+cu128 | ✅ | N/A | ✅ |
| sam3d | 2.8.0+cu128 | ✅ | N/A | ✅ |
| hawor | 2.9.0+cu128 | ✅ | ✅ | ✅ |
| tapnet | 2.7.0+cu128 | ✅ | N/A | ✅ |

---

## 七、完整运行命令

当所有权限和模型就绪后：
```bash
cd reconstruction
./run_pipeline.sh whisking/whisking.mp4 125 whisk right
```

或逐阶段运行（参见 `run_pipeline.sh` 中的各阶段命令）。
