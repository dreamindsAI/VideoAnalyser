import ffmpeg
from geopy.geocoders import Nominatim
from pydantic import BaseModel
from typing import Annotated, Optional
from datetime import datetime
from pathlib import Path


class VideoMetaData(BaseModel):
    duration: Annotated[Optional[int], "time in seconds"] = None
    created_date: Annotated[Optional[str], "created date"] = None
    created_time: Annotated[Optional[str], "created time"] = None
    location: Annotated[Optional[str], "location details"] = None
    framerate: Annotated[Optional[float], "frame rate"] = None


class VideoFile:

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def get_metadata(self) -> VideoMetaData:
        probe = ffmpeg.probe(str(self.filepath))
        
        # Initialize metadata fields
        duration = None
        created_date = None
        created_time =None
        location = None
        framerate = None

        # Get video stream information
        video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_info:
            duration = int(float(video_info.get('duration', 0)))
            avg_frame_rate = video_info.get('avg_frame_rate', '0/1')
            if '/' in avg_frame_rate:
                numerator, denominator = map(float, avg_frame_rate.split('/'))
                framerate = int(numerator / denominator) if denominator != 0 else None

        # Get format info
        format_info = probe.get('format', {})
        tags = format_info.get('tags', {})
        if tags:
            # created / modified
            created = self.parse_datetime(tags.get('creation_time'))
            created_date = created.strftime("%Y-%m-%d")
            created_time = created.strftime("%H:%M:%S")

            # location
            iso_location = tags.get('com.apple.quicktime.location.ISO6709')
            if iso_location:
                loc_data = self.parse_location_iso6709(iso_location)
                location = self.get_place_name(loc_data['latitude'], loc_data['longitude'])
        
        return VideoMetaData(
            duration=duration,
            created_date=created_date,
            created_time=created_time,
            location=location,
            framerate=framerate
        )
    
    def write_metadata(self, output_dir: Path) -> None:
        metadata_save_filepath = output_dir / "metadata.json"
        metadata = self.get_metadata()
        with open(metadata_save_filepath, "w") as file:
            file.write(metadata.model_dump_json(indent=4)) 
        print(f"metadata saved to {metadata_save_filepath}")
        return None
    
    def write_frames(self, output_dir: Path):
        output_dir = output_dir / Path("frames")
        output_dir.mkdir(parents=True, exist_ok=True)
        (
            ffmpeg
            .input(str(self.filepath))
            .output(str(output_dir / 'frame_%04d.png'), r=1)
            .global_args('-loglevel', 'error')
            .run()
        )
        print(f"Frames saved to {output_dir}")
        return None

    def parse_datetime(self, dt: Optional[str]) -> Optional[datetime]:
        if not dt:
            return None
        try:
            return datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except Exception:
            return None

    def parse_location_iso6709(self, location_iso: str) -> dict:
        location_iso = location_iso.strip('/')
        lat = float(location_iso[0:8])
        lon = float(location_iso[8:17])
        alt = float(location_iso[17:])
        return {"latitude": lat, "longitude": lon, "altitude_m": alt}

    def get_place_name(self, latitude: float, longitude: float) -> str:
        geolocator = Nominatim(user_agent="geoapi", timeout=10)
        location = geolocator.reverse((latitude, longitude), exactly_one=True, language="en")
        return location.address if location else "Unknown Location"
