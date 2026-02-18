"""COLMAP Frame Curator — Gradio App for interactive frame selection."""

import sys
from pathlib import Path

# Add project root so `vbti.*` imports work when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr

from vbti.image_viewer.backend import (
    probe_video,
    init_work_dir,
    extract_frames,
    get_gallery_data,
    generate_histogram,
    export_accepted,
    find_frame_by_gallery_index,
    FrameData,
)


def make_empty_state():
    return {
        "video_path": "",
        "video_info": {},
        "work_dir": "",
        "frames": [],
        "accepted": {},
        "rejected": {},
        "selected_idx": None,
        "selected_rej_idx": None,
    }


# ── Step 1 handlers ──────────────────────────────────────────────

def on_load_video(path_text, uploaded_video, state):
    """Load video from text path or upload, display info."""
    path = path_text.strip() if path_text.strip() else uploaded_video
    if not path:
        gr.Warning("Please enter a video path or upload a file.")
        return state, "", gr.update()

    try:
        info = probe_video(path)
    except Exception as e:
        gr.Warning(f"Error: {e}")
        return state, "", gr.update()

    state["video_path"] = path
    state["video_info"] = info

    info_text = (
        f"**Resolution:** {info['width']}×{info['height']}  \n"
        f"**FPS:** {info['fps']:.1f}  \n"
        f"**Duration:** {info['duration_sec']}s  \n"
        f"**Total frames:** {info['total_frames']}"
    )
    return state, info_text, gr.update()


def on_k_change(k, state):
    """Update estimated frame count when k slider changes."""
    info = state.get("video_info", {})
    dur = info.get("duration_sec", 0)
    estimate = int(dur * k)
    return f"~{estimate} frames will be extracted"


def on_extract(k, state, progress=gr.Progress()):
    """Extract frames from video, return gallery data and advance to step 2."""
    if not state.get("video_path"):
        gr.Warning("Load a video first.")
        return state, [], None, "", gr.update()

    progress(0, desc="Initializing...")
    work_dir = init_work_dir(state["video_path"])
    state["work_dir"] = work_dir

    def progress_cb(current, total, desc):
        if total > 0:
            progress(current / total, desc=desc)

    frames = extract_frames(state["video_path"], work_dir, k, progress_cb)
    state["frames"] = frames

    # All frames start as accepted
    state["accepted"] = {fd["frame_idx"]: True for fd in frames}
    state["rejected"] = {}
    state["selected_idx"] = None
    state["selected_rej_idx"] = None

    gallery = get_gallery_data(frames, state["accepted"])
    histogram = generate_histogram(frames)
    summary = f"**{len(state['accepted'])} accepted**, 0 rejected"

    return state, gallery, histogram, summary, gr.Walkthrough(selected=2)


# ── Step 2 handlers ──────────────────────────────────────────────

def on_gallery_select(evt: gr.SelectData, state):
    """User clicked a frame in the accepted gallery → show full preview."""
    fd = find_frame_by_gallery_index(state["frames"], state["accepted"], evt.index)
    if fd is None:
        return state, None, ""

    state["selected_idx"] = fd["frame_idx"]
    detail = (
        f"**Frame:** {fd['frame_idx']}  \n"
        f"**Sharpness:** {fd['score']:.1f}  \n"
        f"**Time:** {fd['timestamp_sec']:.1f}s"
    )
    return state, fd["full_path"], detail


def on_reject(state):
    """Move selected frame from accepted → rejected."""
    sel = state.get("selected_idx")
    if sel is None:
        gr.Warning("Select a frame first.")
        return state, [], [], "", None, ""

    if sel in state["accepted"]:
        del state["accepted"][sel]
        state["rejected"][sel] = True

    state["selected_idx"] = None
    acc_gallery = get_gallery_data(state["frames"], state["accepted"])
    rej_gallery = get_gallery_data(state["frames"], state["rejected"])
    summary = f"**{len(state['accepted'])} accepted**, {len(state['rejected'])} rejected"
    return state, acc_gallery, rej_gallery, summary, None, ""


def on_rejected_gallery_select(evt: gr.SelectData, state):
    """User clicked a frame in the rejected gallery."""
    fd = find_frame_by_gallery_index(state["frames"], state["rejected"], evt.index)
    if fd is None:
        return state, None, ""

    state["selected_rej_idx"] = fd["frame_idx"]
    detail = (
        f"**Frame:** {fd['frame_idx']}  \n"
        f"**Sharpness:** {fd['score']:.1f}  \n"
        f"**Time:** {fd['timestamp_sec']:.1f}s"
    )
    return state, fd["full_path"], detail


def on_restore(state):
    """Move selected rejected frame back to accepted."""
    sel = state.get("selected_rej_idx")
    if sel is None:
        gr.Warning("Select a rejected frame first.")
        return state, [], [], "", None, ""

    if sel in state["rejected"]:
        del state["rejected"][sel]
        state["accepted"][sel] = True

    state["selected_rej_idx"] = None
    acc_gallery = get_gallery_data(state["frames"], state["accepted"])
    rej_gallery = get_gallery_data(state["frames"], state["rejected"])
    summary = f"**{len(state['accepted'])} accepted**, {len(state['rejected'])} rejected"
    return state, acc_gallery, rej_gallery, summary, None, ""


# ── Step 3 handlers ──────────────────────────────────────────────

def on_fmt_change(fmt):
    """Show quality slider only for JPEG."""
    return gr.update(visible=(fmt == "jpg"))


def on_advance_to_export(state):
    """Pre-fill export directory and advance to step 3."""
    video_path = state.get("video_path", "")
    stem = Path(video_path).stem if video_path else "output"
    suggested = str(Path(video_path).parent / f"{stem}_colmap") if video_path else "/tmp/colmap_export"
    return suggested, gr.Walkthrough(selected=3)


def on_export(output_dir, fmt, quality, state, progress=gr.Progress()):
    """Export accepted frames with sequential COLMAP naming."""
    if not state.get("frames"):
        gr.Warning("No frames to export.")
        return ""

    def progress_cb(current, total, desc):
        if total > 0:
            progress(current / total, desc=desc)

    path, count = export_accepted(
        state["frames"], state["accepted"], output_dir, fmt, quality, progress_cb
    )
    return f"Exported **{count}** frames to `{path}`"


# ── Layout ────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="COLMAP Frame Curator") as app:
        gr.Markdown("# COLMAP Frame Curator\nExtract, review, and export video frames for 3D reconstruction.")

        state = gr.State(make_empty_state())

        with gr.Walkthrough(selected=1) as walkthrough:

            # ── Step 1: Load Video ────────────────────────────
            with gr.Step("Load Video", id=1):
                with gr.Row():
                    with gr.Column(scale=2):
                        path_input = gr.Textbox(
                            label="Video path (local)",
                            placeholder="/path/to/video.mov",
                        )
                        upload_input = gr.Video(label="Or upload", height=200)
                        load_btn = gr.Button("Load Video", variant="secondary")
                    with gr.Column(scale=1):
                        info_display = gr.Markdown("*No video loaded*")

                k_slider = gr.Slider(
                    minimum=1, maximum=5, value=2, step=1,
                    label="Frames per second (k)",
                )
                estimate_text = gr.Textbox(
                    label="Estimated output", interactive=False, value=""
                )
                extract_btn = gr.Button("Extract Frames", variant="primary", size="lg")

                # Events
                load_btn.click(
                    on_load_video,
                    inputs=[path_input, upload_input, state],
                    outputs=[state, info_display, walkthrough],
                ).then(
                    on_k_change,
                    inputs=[k_slider, state],
                    outputs=[estimate_text],
                )

                k_slider.change(
                    on_k_change,
                    inputs=[k_slider, state],
                    outputs=[estimate_text],
                )

            # ── Step 2: Review & Curate ───────────────────────
            with gr.Step("Review & Curate", id=2):
                summary_md = gr.Markdown("*Extract frames first*")

                with gr.Row():
                    # Main gallery (left)
                    with gr.Column(scale=3):
                        acc_gallery = gr.Gallery(
                            label="Accepted Frames",
                            columns=4,
                            object_fit="contain",
                            height="auto",
                        )
                    # Side panel (right)
                    with gr.Column(scale=1):
                        preview_img = gr.Image(label="Preview", height=400)
                        detail_md = gr.Markdown("")
                        reject_btn = gr.Button("Reject", variant="stop")

                with gr.Accordion("Rejected Frames", open=False):
                    rej_gallery = gr.Gallery(
                        label="Rejected",
                        columns=4,
                        object_fit="contain",
                        height="auto",
                    )
                    with gr.Row():
                        rej_preview = gr.Image(label="Rejected Preview", height=300)
                        with gr.Column():
                            rej_detail_md = gr.Markdown("")
                            restore_btn = gr.Button("Restore", variant="secondary")

                histogram = gr.Plot(label="Sharpness Distribution")

                export_advance_btn = gr.Button("Continue to Export →", variant="primary", size="lg")

                # Events
                acc_gallery.select(
                    on_gallery_select,
                    inputs=[state],
                    outputs=[state, preview_img, detail_md],
                )

                reject_btn.click(
                    on_reject,
                    inputs=[state],
                    outputs=[state, acc_gallery, rej_gallery, summary_md, preview_img, detail_md],
                )

                rej_gallery.select(
                    on_rejected_gallery_select,
                    inputs=[state],
                    outputs=[state, rej_preview, rej_detail_md],
                )

                restore_btn.click(
                    on_restore,
                    inputs=[state],
                    outputs=[state, acc_gallery, rej_gallery, summary_md, rej_preview, rej_detail_md],
                )

            # ── Step 3: Export ────────────────────────────────
            with gr.Step("Export", id=3):
                output_dir_input = gr.Textbox(label="Output directory")
                fmt_radio = gr.Radio(
                    choices=["jpg", "png"], value="jpg", label="Format"
                )
                quality_slider = gr.Slider(
                    minimum=70, maximum=100, value=90, step=1,
                    label="JPEG Quality", visible=True,
                )
                export_btn = gr.Button("Export", variant="primary", size="lg")
                export_result = gr.Markdown("")

                fmt_radio.change(on_fmt_change, inputs=[fmt_radio], outputs=[quality_slider])

                export_btn.click(
                    on_export,
                    inputs=[output_dir_input, fmt_radio, quality_slider, state],
                    outputs=[export_result],
                )

        # ── Cross-step wiring ─────────────────────────────────
        # Extract button: runs extraction, populates step 2, advances walkthrough
        extract_btn.click(
            on_extract,
            inputs=[k_slider, state],
            outputs=[state, acc_gallery, histogram, summary_md, walkthrough],
        )

        # Export advance button: pre-fills output dir, goes to step 3
        export_advance_btn.click(
            on_advance_to_export,
            inputs=[state],
            outputs=[output_dir_input, walkthrough],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(theme=gr.themes.Soft())
