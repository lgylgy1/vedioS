from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
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
            "GET /video/{filename}": "获取上传的视频文件",
            "GET /index-info": "获取索引信息",
            "POST /extract-segment": "提取视频片段"
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
                "video_filename": Path(result["video_path"]).name,
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

@app.get("/video/{video_filename}")
async def get_video(video_filename: str):
    """
    获取上传的视频文件

    - **video_filename**: 视频文件名
    """
    video_path = UPLOAD_DIR / video_filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")

    return FileResponse(
        path=str(video_path),
        media_type='video/mp4',
        filename=video_filename
    )

@app.post("/extract-segment")
async def extract_segment(
    video_path: str = Form(..., description="视频文件路径"),
    start_time: float = Form(..., description="开始时间（秒）", ge=0),
    end_time: float = Form(..., description="结束时间（秒）", ge=0),
    output_filename: str = Form(..., description="输出文件名")
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

    # 检查视频文件是否存在，如果不是绝对路径则在上传目录中查找
    if not os.path.exists(video_path):
        # 尝试在uploaded_videos目录中查找
        potential_path = UPLOAD_DIR / video_path
        if os.path.exists(str(potential_path)):
            video_path = str(potential_path)
        else:
            # 再次尝试，如果video_path只包含文件名
            filename_only = Path(video_path).name
            potential_path = UPLOAD_DIR / filename_only
            if os.path.exists(str(potential_path)):
                video_path = str(potential_path)
            else:
                raise HTTPException(status_code=404, detail=f"视频文件不存在: {video_path} (尝试过: {potential_path})")

    # 确保输出目录存在
    output_dir = Path("extracted_segments")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    try:
        video_tool.extract_segment(video_path, start_time, end_time, str(output_path))

        # 验证输出文件是否成功创建
        if not os.path.exists(str(output_path)):
            raise HTTPException(status_code=500, detail="输出文件未能成功创建")

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
        # 清理可能创建的输出文件
        if os.path.exists(str(output_path)):
            try:
                os.remove(str(output_path))
            except:
                pass
        raise HTTPException(status_code=500, detail=f"提取片段时出错: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8567)
