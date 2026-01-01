# VedioS - 自然语言视频检索工具

基于Whisper、FAISS和FFmpeg的视频片段自然语言检索工具。

## 功能特性

- **视频转录**: 使用Whisper将视频音频转换为文本
- **智能分块**: 将转录文本按时间段分割成可管理的块
- **向量索引**: 使用FAISS对文本块进行向量索引，支持高效相似度搜索
- **自然语言查询**: 支持中文自然语言查询，检索相关视频片段
- **片段提取**: 可以提取匹配的视频片段

## 安装依赖

```bash
pip install openai whisper faiss-cpu ffmpeg-python aiohttp dotenv
```

## 配置

编辑 `.env` 文件，设置必要的环境变量：

```env
# LLM 配置（用于对话增强）
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=gpt-4

# 视频搜索配置
WHISPER_MODEL=base  # 可选: tiny, base, small, medium, large
EMBEDDING_DIMENSION=1024
INDEX_FILE=video_index.faiss
CHUNK_DURATION=30.0  # 块持续时间（秒）

# 嵌入API密钥（如果使用SiliconFlow）
siliconflow_api_key=your_siliconflow_api_key
```

## 使用方法

### 1. 初始化工具

```python
from video_search_tool import VideoSearchTool

tool = VideoSearchTool()
```

### 2. 索引视频

```python
import asyncio

async def index_video():
    result = await tool.index_video("path/to/your/video.mp4")
    print("索引完成:", result)

asyncio.run(index_video())
```

### 3. 搜索视频片段

```python
async def search_video():
    results = await tool.search_videos("人工智能的发展历程", top_k=5)
    for result in results:
        print(f"分数: {result['score']:.3f}")
        print(f"文本: {result['text'][:100]}...")
        print(f"时间: {result['start_time']:.1f}s - {result['end_time']:.1f}s")

asyncio.run(search_video())
```

### 4. 提取视频片段

```python
tool.extract_segment(
    video_path="input.mp4",
    start_time=120.0,
    end_time=180.0,
    output_path="segment.mp4"
)
```

## 主要组件

### VideoProcessor
- 使用FFmpeg处理视频文件
- 提取音频、获取视频时长、提取片段

### Transcriber
- 使用Whisper进行音频转录
- 支持多种语言和模型大小
- 将转录结果分块处理

### VideoIndexer
- 使用FAISS进行向量索引
- 支持持久化存储索引
- 基于余弦相似度进行搜索

### VideoSearchTool
- 整合所有组件的主工具类
- 提供简化的API接口

## 测试

运行测试脚本：

```bash
python test_video_search.py
```

## 注意事项

1. **依赖安装**: 确保安装了所有必要的Python包
2. **FFmpeg**: 需要系统中安装FFmpeg
3. **模型大小**: Whisper模型大小影响准确性和速度，可根据需要选择
4. **索引持久化**: 索引会自动保存到磁盘，可重复使用
5. **内存使用**: 大视频文件可能需要较多内存

## 扩展功能

- 支持多视频索引
- 实时视频流处理
- 高级搜索过滤器
- 结果可视化界面

## 许可证

MIT License
