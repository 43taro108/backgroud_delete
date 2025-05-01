# -*- coding: utf-8 -*-
"""
Created on Fri May  2 04:52:58 2025

@author: ktrpt
"""
import streamlit as st
import cv2
import tempfile
import os
from rembg import remove
from PIL import Image
import numpy as np
import zipfile

st.title("rembg × Background Removal App (for ~10 sec videos)")

uploaded_file = st.file_uploader("Upload a video (.mp4)", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(uploaded_file)

    if st.button("Run Background Removal"):
        with st.spinner("Processing... (recommended for ~10 sec videos)"):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_dir = tempfile.mkdtemp()
            out_path = os.path.join(output_dir, "output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), True)

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                removed_pil = remove(frame_pil)
                removed_np = np.array(removed_pil)

                # Replace transparent background with black
                alpha = removed_np[..., 3] / 255.0
                rgb = removed_np[..., :3]
                black_bg = np.zeros_like(rgb)
                combined = (rgb * alpha[..., None] + black_bg * (1 - alpha[..., None])).astype(np.uint8)
                combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

                out.write(combined_bgr)
                frame_idx += 1

                if frame_idx % 10 == 0:
                    st.write(f"Processing frame: {frame_idx}/{frame_count}")

            cap.release()
            out.release()

            # Create ZIP file
            zip_path = os.path.join(output_dir, "output.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(out_path, arcname="output.mp4")

            with open(zip_path, "rb") as f:
                st.download_button("Download processed video (ZIP)", f, "output.zip", "application/zip")

            st.success("Processing completed for all frames!")