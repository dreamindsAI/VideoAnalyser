import streamlit as st
import os
import json
import numpy as np
import tempfile
from pathlib import Path
from utils.metadata_extractor import VideoFile
from utils.description_writer import write_all_frame_descriptions, get_video_description

TEMP_DIR = Path("src/video_analyser/temp")
TEMP_DIR.mkdir(exist_ok=True)
VIDEO_EXTENSIONS = {"mov"}

st.title("Video Description & Retrieval System")

# Tabs
tab1, tab2, tab3 = st.tabs(["Upload Video", "Upload Folder", "Search & Retrieve"])

with tab1:
    st.header("Upload Video, Generate Description, Save to Vector Database")
    uploaded_video = st.file_uploader("Upload a video", type=list(VIDEO_EXTENSIONS))
    
    if uploaded_video:
        file_name = uploaded_video.name.split(".")[0]
        temp_file_directory = TEMP_DIR / file_name
        temp_file_directory.mkdir(exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_file_directory, suffix=f".{uploaded_video.name.split('.')[-1]}")
        temp_file.write(uploaded_video.read())
        temp_video_path = Path(temp_file.name)
        print(temp_video_path)
        
        st.video(temp_video_path)

        video_file = VideoFile(filepath=Path(temp_video_path))
        metadata= video_file.get_metadata()
        video_file.write_frames(output_dir=temp_file_directory)
        video_file.write_metadata(output_dir=temp_file_directory)

        write_all_frame_descriptions(file_directory=temp_file_directory)

        video_description = get_video_description(file_directory=temp_file_directory)

         # Display generated description and metadata
        st.subheader("Generated Description & Metadata")
        description = st.text_area("Edit Description", video_description, height=150)
        
        st.json(metadata)
        
        if st.button("Save Data"):
            metadata["description"] = description
            metadata_path = temp_file_directory / "metdata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            st.success("Metadata saved successfully!")

with tab2:
    st.header("Upload Folder & Process Videos")
    folder_path = st.text_input("Enter folder path containing videos:")
    
    if st.button("Process Folder") and folder_path:
        folder = Path(folder_path)
        if folder.exists() and folder.is_dir():
            video_files = [file for file in folder.iterdir() if file.suffix[1:] in VIDEO_EXTENSIONS]
            
            if video_files:
                for video in video_files:
                    st.write(f"Processing: {video.name}")
                    video_file = VideoFile(filepath=str(video))
                    video_file.get_metadata()
                    video_file.write_frames(output_dir=TEMP_DIR)
                    video_file.write_metadata(output_dir=SAVE_DIR)
                st.success("All videos processed successfully!")
            else:
                st.warning("No valid video files found in the folder.")
        else:
            st.error("Invalid folder path.")

with tab3:
    st.header("Search for Relevant Videos")
    query = st.text_input("Enter a search query")
    
    # if query and st.button("Search"):
    #     query_embedding = model.encode(query).astype(np.float32).reshape(1, -1)
    #     D, I = index.search(query_embedding, k=5)
    #     
    #     results = [list(metadata.keys())[i] for i in I[0] if i < len(metadata)]
    #     
    #     if results:
    #         for vid in results:
    #             st.video(metadata[vid]["path"])
    #             st.write(metadata[vid]["description"])
    #     else:
    #         st.warning("No relevant videos found.")



