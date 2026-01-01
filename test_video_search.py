import asyncio
import os
from video_search_tool import VideoSearchTool
from video_processor import VideoProcessor
from transcriber import Transcriber
from configuration import video_config


async def test_video_search():
    """Test the video search functionality."""
    tool = VideoSearchTool()

    print("=== Video Search Tool Test ===\n")

    # Get index info
    index_info = tool.get_index_info()
    print(f"Current index info: {index_info}\n")

    # Example: Index a video (uncomment and provide actual video path)
    video_path = "vedio\\test2.mp4"
    if os.path.exists(video_path):
        print(f"Indexing video: {video_path}")
        result = await tool.index_video(video_path)
        print(f"Indexing result: {result}\n")
    else:
        print(f"Video file not found: {video_path}\n")

    # Example: Search (uncomment when index has data)
    query = "人工智能的发展历程"
    print(f"Searching for: {query}")
    results = await tool.search_videos(query, top_k=3)
    print("Search results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Text: {result['text'][:200]}...")
        print(f"   Time: {result['start_time']:.1f}s - {result['end_time']:.1f}s")
        print(f"   Video: {result['video_path']}\n")

    print("Test completed. To fully test:")
    print("1. Place a video file in the project directory")
    print("2. Uncomment the indexing code and run")
    print("3. Uncomment the search code and run")


async def test_pipeline_components():
    """Test the three main pipeline components: preprocessing, transcription, and chunking."""
    print("=== Testing Pipeline Components ===\n")

    # Test video path
    video_path = "vedio\\test2.mp4"
    if not os.path.exists(video_path):
        print(f"Test video not found: {video_path}")
        print("Please ensure the test video exists.")
        return

    print(f"Testing with video: {video_path}\n")

    # 1. Test Preprocessing (Audio Extraction)
    print("1. 测试预处理阶段 - 音频提取")
    print("-" * 40)
    try:
        processor = VideoProcessor()
        audio_path = processor.extract_audio(video_path)
        video_duration = processor.get_video_duration(video_path)

        print(f"✓ 视频时长: {video_duration:.2f} 秒")
        print(f"✓ 音频提取成功: {audio_path}")
        print(f"✓ 音频文件大小: {os.path.getsize(audio_path)} bytes")

        # Check if audio file exists and has content
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print("✓ 音频文件验证通过\n")
        else:
            print("✗ 音频文件验证失败\n")

    except Exception as e:
        print(f"✗ 音频提取失败: {e}\n")
        return

    # 2. Test Transcription
    print("2. 测试转录阶段 - Whisper转录")
    print("-" * 40)
    try:
        transcriber = Transcriber(video_config.whisper_model)
        transcription = transcriber.transcribe(audio_path, language="zh")  # Force Chinese for testing

        full_text = transcription.get('text', '').strip()
        segments = transcription.get('segments', [])

        print(f"✓ 转录文本总长度: {len(full_text)} 字符")
        print(f"✓ 识别的段落数量: {len(segments)}")
        print(f"✓ 转录文本预览: {full_text[:200]}...")

        if len(full_text) > 0 and len(segments) > 0:
            print("✓ 转录验证通过\n")
        else:
            print("✗ 转录结果为空\n")

        # Show first few segments
        print("前3个段落详情:")
        for i, segment in enumerate(segments[:3]):
            print(f"  段落{i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
            print(f"    文本: {segment['text'][:100]}...")
        print()

    except Exception as e:
        print(f"✗ 转录失败: {e}\n")
        return

    # 3. Test Chunking
    print("3. 测试分块处理")
    print("-" * 40)
    try:
        chunks = transcriber.split_into_chunks(transcription, chunk_duration=30.0)

        print(f"✓ 生成块数量: {len(chunks)}")
        print("✓ 块详情:")

        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            chunk_duration_actual = chunk['end'] - chunk['start']
            print(f"  块{i+1}: {chunk['start']:.2f}s - {chunk['end']:.2f}s ({chunk_duration_actual:.2f}s)")
            print(f"    文本长度: {len(chunk['text'])} 字符")
            print(f"    包含段落数: {len(chunk['segments'])}")
            print(f"    文本预览: {chunk['text'][:150]}...")
            print()

        # Validate chunking logic
        total_chunk_duration = sum(chunk['end'] - chunk['start'] for chunk in chunks)
        print(f"✓ 总块时长覆盖: {total_chunk_duration:.2f}s (视频总时长: {video_duration:.2f}s)")

        # Check for overlapping or gaps
        sorted_chunks = sorted(chunks, key=lambda x: x['start'])
        has_overlaps = False
        has_gaps = False

        for i in range(1, len(sorted_chunks)):
            prev_end = sorted_chunks[i-1]['end']
            curr_start = sorted_chunks[i]['start']
            if curr_start < prev_end:
                has_overlaps = True
            elif curr_start > prev_end:
                has_gaps = True

        if has_overlaps:
            print("⚠ 检测到块重叠")
        if has_gaps:
            print("⚠ 检测到块间隙")
        if not has_overlaps and not has_gaps:
            print("✓ 块连续且无重叠")

        print("✓ 分块处理验证完成\n")

    except Exception as e:
        print(f"✗ 分块处理失败: {e}\n")
        return

    # Cleanup temp audio file
    try:
        if audio_path != video_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"✓ 清理临时音频文件: {audio_path}")
    except:
        pass

    print("=== 管道组件测试完成 ===")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pipeline":
        asyncio.run(test_pipeline_components())
    else:
        asyncio.run(test_video_search())
