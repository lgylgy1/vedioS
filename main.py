from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os
import shutil
import tempfile
import asyncio
from pathlib import Path

from video_search_tool import VideoSearchTool
from configuration import video_config

app = FastAPI(title="VedioS - 视频检索API", description="基于FastAPI的视频上传和检索服务")

# 添加CORS中间件，允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有头部
)

# 全局搜索工具实例
video_tool = VideoSearchTool()

# 确保上传目录存在
UPLOAD_DIR = Path("uploaded_videos")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "VedioS 视频检索API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "上传视频文件并自动处理索引",
            "GET /search": "基于自然语言查询检索视频片段",
            "GET /index-info": "获取索引信息"
        }
    }

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    chunk_duration: float = Query(30.0, description="分块持续时间（秒）"),
    language: Optional[str] = Query(None, description="转录语言（可选，自动检测）")
):
    """
    上传视频文件，自动处理并建立索引

    - **file**: 视频文件（支持mp4、avi、mov等格式）
    - **chunk_duration**: 每个文本块的持续时间（秒）
    - **language**: 转录语言代码（可选）
    """
    # 验证文件类型
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持的格式：{', '.join(allowed_extensions)}"
        )

    # 保存上传的文件
    temp_file_path = None
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # 验证文件大小（限制为500MB）
        file_size = os.path.getsize(temp_file_path)
        if file_size > 500 * 1024 * 1024:  # 500MB
            raise HTTPException(status_code=413, detail="文件过大，最大支持500MB")

        # 移动到上传目录
        final_filename = f"{file.filename}"
        final_path = UPLOAD_DIR / final_filename
        counter = 1
        while final_path.exists():
            name_without_ext = Path(file.filename).stem
            final_filename = f"{name_without_ext}_{counter}{file_extension}"
            final_path = UPLOAD_DIR / final_filename
            counter += 1

        shutil.move(temp_file_path, final_path)
        temp_file_path = None

        # 开始处理视频
        print(f"开始处理视频: {final_path}")
        result = await video_tool.index_video(
            str(final_path),
            chunk_duration=chunk_duration,
            language=language
        )

        return JSONResponse(
            content={
                "status": "success",
                "message": "视频上传并索引成功",
                "data": {
                    "filename": final_filename,
                    "file_path": str(final_path),
                    "file_size": file_size,
                    "indexing_result": result
                }
            },
            status_code=200
        )

    except Exception as e:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"处理视频时出错: {str(e)}")
    finally:
        await file.close()

@app.get("/search")
async def search_videos(
    q: str = Query(..., description="自然语言查询"),
    top_k: int = Query(5, description="返回结果数量", ge=1, le=20)
):
    """
    基于自然语言查询检索相关视频片段

    - **q**: 查询文本
    - **top_k**: 返回的匹配结果数量
    """
    try:
        results = await video_tool.search_videos(q, top_k=top_k)

        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_result = {
                "score": result["score"],
                "text": result["text"],
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "duration": result["end_time"] - result["start_time"],
                "video_path": result["video_path"],
                "chunk_index": result["chunk_index"]
            }
            formatted_results.append(formatted_result)

        return JSONResponse(
            content={
                "status": "success",
                "query": q,
                "total_results": len(formatted_results),
                "results": formatted_results
            },
            status_code=200
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索时出错: {str(e)}")

@app.get("/index-info")
async def get_index_info():
    """获取当前索引的统计信息"""
    try:
        info = video_tool.get_index_info()
        return JSONResponse(
            content={
                "status": "success",
                "index_info": info
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取索引信息时出错: {str(e)}")

@app.post("/extract-segment")
async def extract_segment(
    video_path: str = Query(..., description="视频文件路径"),
    start_time: float = Query(..., description="开始时间（秒）", ge=0),
    end_time: float = Query(..., description="结束时间（秒）", ge=0),
    output_filename: str = Query(..., description="输出文件名")
):
    """
    提取视频片段

    - **video_path**: 源视频文件路径
    - **start_time**: 开始时间（秒）
    - **end_time**: 结束时间（秒）
    - **output_filename**: 输出文件名
    """
    if start_time >= end_time:
        raise HTTPException(status_code=400, detail="开始时间必须小于结束时间")

    # 确保输出目录存在
    output_dir = Path("extracted_segments")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    try:
        video_tool.extract_segment(video_path, start_time, end_time, str(output_path))

        return JSONResponse(
            content={
                "status": "success",
                "message": "视频片段提取成功",
                "data": {
                    "input_video": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "output_path": str(output_path),
                    "output_filename": output_filename
                }
            },
            status_code=200
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提取片段时出错: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8567)
