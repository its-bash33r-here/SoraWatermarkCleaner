import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

from sorawm.core import SoraWM


def main():
    st.set_page_config(
        page_title="Sora Watermark Cleaner", page_icon="🎬", layout="centered"
    )

    st.title("🎬 Sora Watermark Cleaner")
    st.markdown("Remove watermarks from Sora-generated videos with ease")

    # Initialize SoraWM
    if "sora_wm" not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.sora_wm = SoraWM()

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Select a video file to remove watermarks",
    )

    if uploaded_file is not None:
        # Display video info
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.video(uploaded_file)

        # Processing mode toggle
        st.markdown("### ⚙️ Processing Options")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            use_parallel = st.toggle(
                "Enable Parallel Pipeline",
                value=False,
                help="Enable parallel detection and cleaning for faster processing (experimental)",
            )
        
        with col2:
            if use_parallel:
                st.markdown("🚀 **Parallel**")
            else:
                st.markdown("🔄 **Serial**")

        if use_parallel:
            st.info(
                "ℹ️ Parallel mode enables overlapping detection and cleaning phases for improved GPU utilization and faster processing."
            )

        st.markdown("---")

        # Process button
        if st.button("🚀 Remove Watermark", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                # Save uploaded file
                input_path = tmp_path / uploaded_file.name
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Process video
                output_path = tmp_path / f"cleaned_{uploaded_file.name}"

                try:
                    # Create progress bar and status text
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    mode_text = st.empty()

                    # Display processing mode
                    mode_indicator = "🚀 Parallel Pipeline" if use_parallel else "🔄 Serial Processing"
                    mode_text.markdown(f"**Processing Mode:** {mode_indicator}")

                    if use_parallel:
                        # 并行模式：使用队列在线程间传递进度
                        progress_queue = queue.Queue()
                        processing_error = []  # 用于捕获处理过程中的错误
                        
                        def update_progress_from_queue():
                            """从队列中读取进度并更新UI"""
                            last_progress = 0
                            while True:
                                try:
                                    progress = progress_queue.get(timeout=0.1)
                                    if progress is None:  # 结束信号
                                        break
                                    
                                    # 更新进度条
                                    last_progress = progress
                                    progress_bar.progress(progress / 100)
                                    
                                    # 更新状态文本
                                    if progress < 50:
                                        status_text.text(f"🔍 Detecting watermarks... {progress}%")
                                    elif progress < 95:
                                        status_text.text(f"🧹 Removing watermarks... {progress}%")
                                    else:
                                        status_text.text(f"🎵 Merging audio... {progress}%")
                                        
                                except queue.Empty:
                                    # 队列为空，继续等待
                                    time.sleep(0.05)
                                    continue
                            
                            return last_progress
                        
                        def run_processing():
                            """在后台线程中运行处理"""
                            try:
                                def progress_callback(progress: int):
                                    """进度回调函数，将进度放入队列"""
                                    progress_queue.put(progress)
                                
                                # 运行水印移除
                                st.session_state.sora_wm.run(
                                    input_path,
                                    output_path,
                                    progress_callback=progress_callback,
                                    overlap_running=True,
                                )
                            except Exception as e:
                                # 捕获错误并存储
                                processing_error.append(e)
                            finally:
                                # 发送结束信号
                                progress_queue.put(None)
                        
                        # 启动处理线程
                        processing_thread = threading.Thread(target=run_processing, daemon=True)
                        processing_thread.start()
                        
                        # 在主线程中更新UI
                        last_progress = 0
                        while processing_thread.is_alive():
                            try:
                                progress = progress_queue.get(timeout=0.1)
                                if progress is None:
                                    break
                                
                                last_progress = progress
                                progress_bar.progress(progress / 100)
                                
                                if progress < 50:
                                    status_text.text(f"🔍 Detecting watermarks... {progress}%")
                                elif progress < 95:
                                    status_text.text(f"🧹 Removing watermarks... {progress}%")
                                else:
                                    status_text.text(f"🎵 Merging audio... {progress}%")
                                    
                            except queue.Empty:
                                time.sleep(0.05)
                                continue
                        
                        # 等待线程完全结束
                        processing_thread.join(timeout=5)
                        
                        # 处理队列中剩余的进度更新
                        while not progress_queue.empty():
                            try:
                                progress = progress_queue.get_nowait()
                                if progress is not None:
                                    last_progress = progress
                                    progress_bar.progress(progress / 100)
                                    if progress < 50:
                                        status_text.text(f"🔍 Detecting watermarks... {progress}%")
                                    elif progress < 95:
                                        status_text.text(f"🧹 Removing watermarks... {progress}%")
                                    else:
                                        status_text.text(f"🎵 Merging audio... {progress}%")
                            except queue.Empty:
                                break
                        
                        # 如果有错误，抛出
                        if processing_error:
                            raise processing_error[0]
                    
                    else:
                        # 串行模式：直接使用回调
                        def update_progress(progress: int):
                            progress_bar.progress(progress / 100)
                            if progress < 50:
                                status_text.text(f"🔍 Detecting watermarks... {progress}%")
                            elif progress < 95:
                                status_text.text(f"🧹 Removing watermarks... {progress}%")
                            else:
                                status_text.text(f"🎵 Merging audio... {progress}%")

                        # Run the watermark removal with progress callback
                        st.session_state.sora_wm.run(
                            input_path,
                            output_path,
                            progress_callback=update_progress,
                            overlap_running=False,
                        )

                    # Complete the progress bar
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    mode_text.empty()

                    st.success("✅ Watermark removed successfully!")

                    # Display result
                    st.markdown("### Result")
                    st.video(str(output_path))

                    # Download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Cleaned Video",
                            data=f,
                            file_name=f"cleaned_{uploaded_file.name}",
                            mime="video/mp4",
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    import traceback

                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())

    else:
        # Show instructions when no file is uploaded
        st.info(
            """
            👆 **Get Started:**
            1. Upload a video file using the file uploader above
            2. Choose your processing mode (Serial or Parallel)
            3. Click 'Remove Watermark' to process
            4. Download your cleaned video
            """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit and AI</p>
            <p><a href='https://github.com/linkedlist771/SoraWatermarkCleaner'>GitHub Repository</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()