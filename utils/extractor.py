from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from PyPDF2 import PdfReader

def extract_text_from_youtube(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        return None

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_text_from_txt(file):
    return file.read().decode("utf-8")
