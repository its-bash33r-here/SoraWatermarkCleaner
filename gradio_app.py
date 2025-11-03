"""Sora Watermark Cleaner - Gradio App

A web application to remove watermarks from Sora-generated videos.
Built with Gradio framework - Simple and reliable!
"""

import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr

from sorawm.core import SoraWM


class VideoProcessor:
    """Handle video processing logic."""
    
    def __init__(self):
        self.sora_wm: Optional[SoraWM] = None
        
    def get_sora_wm(self) -> SoraWM:
        """Lazy load SoraWM instance."""
        if self.sora_wm is None:
            self.sora_wm = SoraWM()
        return self.sora_wm


# Global processor instance
processor = VideoProcessor()


def process_video(input_video, progress=gr.Progress()):
    """Process uploaded video and remove watermark.
    
    Args:
        input_video: Path to uploaded video file
        progress: Gradio progress tracker
        
    Returns:
        tuple: (output_video_path, status_message)
    """
    if input_video is None:
        return None, "⚠️ Please upload a video first!"
    
    try:
        # Setup paths
        input_path = Path(input_video)
        output_dir = Path('./gradio_outputs')
        output_dir.mkdir(exist_ok=True)
        
        output_filename = f'cleaned_{input_path.name}'
        output_path = output_dir / output_filename
        
        # Progress callback
        def update_progress(pct: int):
            if pct < 50:
                progress(pct / 100, desc=f"🔍 Detecting watermarks... {pct}%")
            elif pct < 95:
                progress(pct / 100, desc=f"🧹 Removing watermarks... {pct}%")
            else:
                progress(pct / 100, desc=f"🎵 Merging audio... {pct}%")
        
        # Process video
        progress(0, desc="🚀 Starting watermark removal...")
        processor.get_sora_wm().run(
            input_path,
            output_path,
            progress_callback=update_progress
        )
        
        progress(1.0, desc="✅ Complete!")
        
        return str(output_path), f"✅ Successfully processed! Saved to: {output_filename}"
        
    except Exception as e:
        error_msg = f"❌ Error processing video: {str(e)}"
        print(error_msg)
        return None, error_msg


# Create Gradio interface
with gr.Blocks(
    title="🎬 Sora Watermark Cleaner",
    theme=gr.themes.Soft(),
) as demo:
    
    # Header
    gr.Markdown(
        """
        # 🎬 Sora Watermark Cleaner
        ### Remove watermarks from Sora-generated videos with ease
        
        Upload your video, click process, and download the cleaned result!
        """
    )
    
    with gr.Row():
        with gr.Column():
            # Input section
            gr.Markdown("### 📤 Upload Video")
            input_video = gr.Video(
                label="Input Video",
                format="mp4",
                height=300,
            )
            
            process_btn = gr.Button(
                "🚀 Remove Watermark",
                variant="primary",
                size="lg",
            )
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
            )
        
        with gr.Column():
            # Output section
            gr.Markdown("### ⬇️ Download Result")
            output_video = gr.Video(
                label="Cleaned Video",
                height=300,
            )
    
    # Examples section (optional)
    gr.Markdown(
        """
        ---
        ### 📝 Instructions
        1. Upload a video file (MP4, AVI, MOV, MKV)
        2. Click "Remove Watermark" button
        3. Wait for processing to complete
        4. Download the cleaned video
        
        **Supported formats:** MP4, AVI, MOV, MKV  
        **Max file size:** 500MB
        """
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        Built with ❤️ using Gradio and AI | 
        [GitHub Repository](https://github.com/linkedlist771/SoraWatermarkCleaner)
        """
    )
    
    # Event handlers
    process_btn.click(
        fn=process_video,
        inputs=[input_video],
        outputs=[output_video, status_text],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=True,  
    )

