import yt_dlp
import pandas as pd
import os

PLAYLIST_URL = "https://youtube.com/playlist?list=PL5HDCGtq1uZtGtAP1eTx34IATGVfHMelL&si=2Tr1T3IOX4lJez9P"
EMOTION = "Neutral"   # change emotion label
CSV_FILE = "youtube_playlist_videos.csv"

ydl_opts = {
    'quiet': True,
    'extract_flat': True,
    'skip_download': True,
}

video_links = []

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(PLAYLIST_URL, download=False)
    for entry in info['entries']:
        if entry:
            video_links.append(f"https://www.youtube.com/watch?v={entry['id']}")

# Create dataframe
df = pd.DataFrame({
    "emotion": [EMOTION] * len(video_links),
    "video_link": video_links
})

# Append to CSV (not overwrite)
file_exists = os.path.isfile(CSV_FILE)

df.to_csv(
    CSV_FILE,
    mode='a',
    header=not file_exists,  # write header only once
    index=False
)

print(f"Appended {len(video_links)} videos under emotion '{EMOTION}'")
